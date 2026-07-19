"""Experiment tracking contracts that require no W&B service access."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from marineworld.models.videomae import build_videomae, encode_video, tube_mask
from marineworld.train.experiment import (
    RunIdentity,
    build_run_identity,
    build_wandb_logger,
    metric_name,
    resolved_config,
)
from marineworld.train.module import VideoMAEPretrainingModule


def _tiny_model_config() -> dict[str, int | float | str]:
    return {
        "name": "videomae_tiny",
        "backbone": "tiny",
        "image_size": 16,
        "patch_size": 8,
        "num_frames": 4,
        "tubelet_size": 2,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "decoder_hidden_size": 16,
        "decoder_num_hidden_layers": 1,
        "decoder_num_attention_heads": 4,
        "decoder_intermediate_size": 32,
        "mask_ratio": 0.5,
        "seed": 42,
    }


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.rand(1, 4, 3, 16, 16)}


def _tracking_config(mode: str = "offline") -> DictConfig:
    return OmegaConf.create(
        {
            "tracking": {
                "mode": mode,
                "project": "marineworld-fm",
                "entity": None,
            }
        }
    )


def test_resolved_config_excludes_wandb_key_name_and_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "secret-value")
    cfg = _tracking_config()
    cfg.tracking.api_key = "secret-value"
    cfg.tracking.wandbApiKey = "camel-case-secret"

    serialized = json.dumps(resolved_config(cfg))

    assert "secret-value" not in serialized
    assert "camel-case-secret" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert "api_key" not in serialized
    assert "wandbApiKey" not in serialized


def test_resolved_config_keeps_non_secret_tokenizer_settings() -> None:
    cfg = OmegaConf.create({"model": {"tokenizer": "videomae"}})

    assert resolved_config(cfg) == {"model": {"tokenizer": "videomae"}}


def test_importing_experiment_does_not_mutate_environment() -> None:
    source_dir = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ | {"PYTHONPATH": str(source_dir)}
    program = """
import json
import os
import sys

before = dict(os.environ)
import marineworld.train.experiment
state = [before, dict(os.environ), "pytorch_lightning" in sys.modules]
print(json.dumps(state, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    before, after, lightning_imported = json.loads(result.stdout)
    assert after == before
    assert lightning_imported is False


def test_importing_models_does_not_load_optional_training_dependencies() -> None:
    source_dir = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ | {"PYTHONPATH": str(source_dir)}
    program = """
import json
import sys

import marineworld.models
print(json.dumps({"torch": "torch" in sys.modules, "transformers": "transformers" in sys.modules}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert json.loads(result.stdout) == {"torch": False, "transformers": False}


def test_run_identity_is_stable_across_manifest_order() -> None:
    identity = build_run_identity("videomae", ("sha-a", "sha-b"), seed=42, label_fraction=None)

    assert (
        identity.run_id
        == build_run_identity("videomae", ("sha-b", "sha-a"), seed=42, label_fraction=None).run_id
    )
    assert metric_name("probe", "macro_f1", "fvessel") == "probe/fvessel/macro_f1"


def test_run_identity_keeps_execution_metadata_out_of_run_id() -> None:
    first = build_run_identity(
        "videomae", ("manifest",), seed=42, label_fraction=0.1, git_sha="first", accelerator="l4"
    )
    second = build_run_identity(
        "videomae", ("manifest",), seed=42, label_fraction=0.1, git_sha="second", accelerator="a100"
    )

    assert first.run_id == second.run_id
    assert {"git:first", "accelerator:l4"} <= set(first.tags)


def test_metric_name_omits_dataset_for_non_probe_stages() -> None:
    assert metric_name("pretrain", "loss") == "pretrain/loss"


def test_disabled_tracking_does_not_construct_wandb_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_logger(**_: object) -> None:
        raise AssertionError("disabled tracking must not construct a W&B logger")

    monkeypatch.setattr("marineworld.train.experiment._create_wandb_logger", _unexpected_logger)

    identity = RunIdentity("run", "group", ())
    assert build_wandb_logger(_tracking_config("disabled"), identity) is False


@pytest.mark.parametrize(
    ("mode", "expected_log_model"),
    [("offline", False), ("online", "all")],
)
def test_wandb_logger_configuration_is_offline_safe(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected_log_model: str | bool
) -> None:
    captured: dict[str, object] = {}

    def _fake_logger(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("marineworld.train.experiment._create_wandb_logger", _fake_logger)
    result = build_wandb_logger(
        _tracking_config(mode), RunIdentity("stable-run", "videomae", ("accelerator:l4",))
    )

    assert result is not False
    assert captured["offline"] is (mode == "offline")
    assert captured["log_model"] == expected_log_model
    assert captured["id"] == "stable-run"
    assert "WANDB_API_KEY" not in json.dumps(captured["config"])


@pytest.mark.parametrize(
    ("runtime", "tracking_mode"),
    [("local_smoke", "offline"), ("l4", "online"), ("a100", "online")],
)
def test_runtime_profiles_compose_with_tracking(runtime: str, tracking_mode: str) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[f"runtime={runtime}"])

    assert cfg.runtime.tracking_mode == tracking_mode
    assert cfg.tracking.mode == tracking_mode


def test_tube_mask_has_exact_ratio() -> None:
    mask = tube_mask(
        2,
        sequence_length=80,
        mask_ratio=0.9,
        generator=torch.Generator().manual_seed(42),
    )

    assert mask.dtype == torch.bool
    assert mask.shape == (2, 80)
    assert mask.sum(dim=1).tolist() == [72, 72]


def test_mask_for_step_is_reproducible_after_resume() -> None:
    first = VideoMAEPretrainingModule(_tiny_model_config(), seed=42)
    resumed = VideoMAEPretrainingModule(_tiny_model_config(), seed=42)

    step_three = first.make_mask(1, torch.device("cpu"), step=3)
    resumed_step_three = resumed.make_mask(1, torch.device("cpu"), step=3)
    expected_step_four = tube_mask(1, 8, 0.5, torch.Generator().manual_seed(46))

    assert torch.equal(step_three, resumed_step_three)
    assert torch.equal(first.make_mask(1, torch.device("cpu"), step=4), expected_step_four)


def test_encode_video_preserves_batch_token_and_hidden_layout(
    tiny_batch: dict[str, torch.Tensor],
) -> None:
    model = build_videomae(_tiny_model_config())

    embeddings = encode_video(model, tiny_batch["pixel_values"])

    assert embeddings.shape == (1, 8, 32)


def test_training_step_rejects_non_finite_loss(
    monkeypatch: pytest.MonkeyPatch, tiny_batch: dict[str, torch.Tensor]
) -> None:
    module = VideoMAEPretrainingModule(_tiny_model_config())

    def _nan_loss(**_: torch.Tensor) -> object:
        return type("Output", (), {"loss": torch.tensor(float("nan"))})()

    monkeypatch.setattr(module.model, "forward", _nan_loss)

    with pytest.raises(FloatingPointError, match="non-finite training loss at batch 7"):
        module.training_step(tiny_batch, 7)


def test_tiny_pretraining_step_updates_parameters(tiny_batch: dict[str, torch.Tensor]) -> None:
    module = VideoMAEPretrainingModule(_tiny_model_config(), lr=1e-3, weight_decay=0.0)
    before = next(module.parameters()).detach().clone()
    trainer = Trainer(
        max_steps=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, train_dataloaders=DataLoader([tiny_batch], batch_size=None))

    assert not torch.equal(before, next(module.parameters()).detach())

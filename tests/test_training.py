"""Experiment tracking contracts that require no W&B service access."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from marineworld.train.experiment import (
    RunIdentity,
    build_run_identity,
    build_wandb_logger,
    metric_name,
    resolved_config,
)


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

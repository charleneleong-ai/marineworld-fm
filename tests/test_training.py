"""Experiment tracking contracts that require no W&B service access."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from marineworld.data.adapters import CompositeAdapter
from marineworld.data.clips import AutoVideoDecoder, SyntheticVideoDecoder
from marineworld.data.contracts import DatasetManifest
from marineworld.data.fvessel import FVesselAdapter
from marineworld.data.manifest import manifest_checksum
from marineworld.data.synthetic import SyntheticAdapter
from marineworld.models.videomae import build_videomae, encode_video, tube_mask
from marineworld.train.experiment import (
    RunIdentity,
    build_run_identity,
    build_wandb_logger,
    metric_name,
    resolved_config,
)
from marineworld.train.media import build_media_preview
from marineworld.train.module import VideoMAEPretrainingModule
from marineworld.train.pretrain import (
    BalancedDatasetSampler,
    _BalancedSamplerCheckpoint,
    _build_decoder,
    _prepare_manifest,
    _training_run_identity,
    build_dataloaders,
    run_pretraining,
    write_training_manifest,
)


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


def test_build_media_preview_denormalizes_and_bounds_output() -> None:
    pixels = torch.linspace(-2, 2, steps=24).reshape(1, 2, 3, 2, 2)
    mask = torch.tensor([[True, False, True, False, True, False, True, False]])
    logits = torch.zeros(1, 4, 3)

    preview = build_media_preview(
        pixel_values=pixels,
        bool_masked_pos=mask,
        logits=logits,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        patch_size=(1, 1),
        tubelet_size=1,
        max_frames=2,
        dataset="synthetic",
        source="generated",
    )

    assert preview.input_grid.shape == (2, 4, 3)
    assert preview.reconstruction_panel.shape == (4, 8, 3)
    assert preview.input_grid.min() >= 0
    assert preview.input_grid.max() <= 1
    assert preview.caption == "dataset=synthetic source=generated"


def test_build_media_preview_places_decoder_patches_and_expands_tube_masks() -> None:
    pixels = torch.full((1, 2, 3, 2, 2), 0.1)
    mask = torch.tensor([[True, False, False, True, False, False, False, False]])
    logits = torch.tensor([[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]])

    preview = build_media_preview(
        pixel_values=pixels,
        bool_masked_pos=mask,
        logits=logits,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        patch_size=(1, 1),
        tubelet_size=1,
        max_frames=2,
        dataset="synthetic",
        source="generated",
        norm_pix_loss=False,
    )

    panel = preview.reconstruction_panel
    assert panel[0, 0, 0] == pytest.approx(0.1)  # Original, frame 0, patch 0.
    assert panel[0, 2, 0] == pytest.approx(0.5)  # Expanded mask is gray.
    assert panel[0, 4, 0] == pytest.approx(0.2)  # First decoder patch.
    assert panel[0, 6, 0] == pytest.approx(0.1)  # Absolute error heatmap.
    assert panel[1, 5, 0] == pytest.approx(0.8)  # Non-adjacent decoder patch.
    assert panel[1, 7, 0] == pytest.approx(0.7)  # Its error heatmap.
    assert panel[2, 4, 0] == pytest.approx(0.1)  # Visible frame 1 patch is preserved.


def test_build_media_preview_places_decoder_patches_after_input_denormalization() -> None:
    pixels = torch.full((1, 1, 3, 2, 2), -0.5)
    mask = torch.tensor([[True, False, False, False]])
    logits = torch.full((1, 1, 3), 0.8)

    preview = build_media_preview(
        pixel_values=pixels,
        bool_masked_pos=mask,
        logits=logits,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        patch_size=(1, 1),
        tubelet_size=1,
        max_frames=1,
        dataset="synthetic",
        source="generated",
        norm_pix_loss=False,
    )

    panel = preview.reconstruction_panel
    assert panel[0, 0, 0] == pytest.approx(0.25)
    assert panel[0, 4, 0] == pytest.approx(0.8)
    assert panel[0, 6, 0] == pytest.approx(0.55)


def test_build_media_preview_unnormalizes_patch_normalized_decoder_logits() -> None:
    pixels = torch.tensor(
        [[[[[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]]]]]
    )
    mask = torch.tensor([[True]])
    logits = torch.ones(1, 1, 12)

    preview = build_media_preview(
        pixel_values=pixels,
        bool_masked_pos=mask,
        logits=logits,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        patch_size=(2, 2),
        tubelet_size=1,
        max_frames=1,
        dataset="synthetic",
        source="generated",
        norm_pix_loss=True,
    )

    expected = 0.25 + (1 / 60) ** 0.5 + 1e-6
    assert preview.reconstruction_panel[0, 4, 0] == pytest.approx(expected)
    assert preview.reconstruction_panel[0, 6, 0] == pytest.approx(expected - 0.1)


@pytest.mark.parametrize(
    ("pixels", "mask", "logits", "max_frames", "message"),
    [
        (
            torch.zeros(1, 2, 3, 2, 2),
            torch.zeros(1, 8, dtype=torch.bool),
            torch.zeros(1, 0, 3),
            0,
            "max_frames must be an integer from 1 to 4",
        ),
        (
            torch.zeros(1, 2, 3, 2, 2),
            torch.zeros(1, 7, dtype=torch.bool),
            torch.zeros(1, 0, 3),
            2,
            "bool_masked_pos length",
        ),
        (
            torch.zeros(1, 2, 3, 2, 2),
            torch.zeros(1, 8, dtype=torch.bool),
            torch.zeros(1, 0, 2),
            2,
            "decoder patch width",
        ),
        (
            torch.empty(1, 0, 3, 2, 2),
            torch.zeros(1, 0, dtype=torch.bool),
            torch.zeros(1, 0, 3),
            2,
            "at least one frame",
        ),
    ],
)
def test_build_media_preview_rejects_invalid_shapes(
    pixels: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    max_frames: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_media_preview(
            pixel_values=pixels,
            bool_masked_pos=mask,
            logits=logits,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            patch_size=(1, 1),
            tubelet_size=1,
            max_frames=max_frames,
            dataset="synthetic",
            source="generated",
        )


@pytest.mark.parametrize(
    ("max_frames", "patch_size", "tubelet_size", "message"),
    [
        (5, (1, 1), 1, "max_frames must be an integer from 1 to 4"),
        (True, (1, 1), 1, "max_frames must be an integer from 1 to 4"),
        (1.5, (1, 1), 1, "max_frames must be an integer from 1 to 4"),
        (1, 1, 1, "patch_size must be a two-item sequence"),
        (1, (1,), 1, "patch_size must be a two-item sequence"),
        (1, (1, 1, 1), 1, "patch_size must be a two-item sequence"),
        (1, (0, 1), 1, "patch_size entries must be positive integers"),
        (1, (-1, 1), 1, "patch_size entries must be positive integers"),
        (1, (1.0, 1), 1, "patch_size entries must be positive integers"),
        (1, (True, 1), 1, "patch_size entries must be positive integers"),
        (1, (1, 1), 0, "tubelet_size must be a positive integer"),
        (1, (1, 1), -1, "tubelet_size must be a positive integer"),
        (1, (1, 1), 1.0, "tubelet_size must be a positive integer"),
        (1, (1, 1), True, "tubelet_size must be a positive integer"),
    ],
)
def test_build_media_preview_rejects_invalid_configuration(
    max_frames: object,
    patch_size: object,
    tubelet_size: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_media_preview(
            pixel_values=torch.zeros(1, 1, 3, 2, 2),
            bool_masked_pos=torch.zeros(1, 4, dtype=torch.bool),
            logits=torch.zeros(1, 0, 3),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            patch_size=patch_size,  # type: ignore[arg-type]
            tubelet_size=tubelet_size,  # type: ignore[arg-type]
            max_frames=max_frames,  # type: ignore[arg-type]
            dataset="synthetic",
            source="generated",
        )


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


def _compose_config(*overrides: str) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=list(overrides))


def _smoke_config(tmp_path: Path, *, max_steps: int) -> DictConfig:
    cfg = _compose_config("data=synthetic", "model=videomae_tiny", "runtime=local_smoke")
    return OmegaConf.merge(
        cfg,
        {
            "data": {"root": str(tmp_path / "data")},
            "output_dir": str(tmp_path / "output"),
            "runtime": {"accelerator": "cpu", "max_steps": max_steps},
            "tracking": {"mode": "disabled"},
        },
    )


def _fvessel_manifest(root: Path, *, videos: int = 8) -> tuple[FVesselAdapter, DatasetManifest]:
    for index in range(videos):
        sample = root / f"sample-{index:02d}"
        (sample / "gt").mkdir(parents=True)
        (sample / "clip.mp4").touch()
        (sample / "gt" / "gt.txt").write_text("1,7,1,2,3,4,1,1,1\n")
    adapter = FVesselAdapter(fps=30.0, num_frames=8)
    return adapter, adapter.build_manifest(root)


class _FakeTrainer:
    def __init__(self, checkpoint: object, *, fail: bool) -> None:
        self.checkpoint = checkpoint
        self.fail = fail

    def fit(self, *_: object, **__: object) -> None:
        if self.fail:
            raise RuntimeError("forced trainer failure")
        path = Path(self.checkpoint.dirpath) / "last.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        self.checkpoint.last_model_path = str(path)


class _CheckpointPathTrainer:
    def __init__(self, checkpoint: object, path: str) -> None:
        self.checkpoint = checkpoint
        self.path = path

    def fit(self, *_: object, **__: object) -> None:
        self.checkpoint.last_model_path = self.path


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
    assert identity.run_id == "a82f5fb546450652"
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
    cfg = _compose_config(f"runtime={runtime}")

    assert cfg.runtime.tracking_mode == tracking_mode
    assert cfg.tracking.mode == tracking_mode


def test_single_gpu_profiles_only_tune_hardware_capacity() -> None:
    l4 = _compose_config("runtime=l4")
    a100 = _compose_config("runtime=a100")

    l4_runtime = OmegaConf.to_container(l4.runtime, resolve=True)
    a100_runtime = OmegaConf.to_container(a100.runtime, resolve=True)
    assert isinstance(l4_runtime, dict)
    assert isinstance(a100_runtime, dict)
    assert l4_runtime.keys() == a100_runtime.keys()
    changed = {
        key
        for key in l4_runtime.keys() | a100_runtime.keys()
        if l4_runtime.get(key) != a100_runtime.get(key)
    }

    assert changed == {"precision", "batch_size", "accumulate_grad_batches", "num_workers"}


def test_local_smoke_profile_uses_cpu() -> None:
    cfg = _compose_config("runtime=local_smoke")

    assert cfg.runtime.accelerator == "cpu"


def test_real_data_uses_portable_video_decoder() -> None:
    cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

    assert isinstance(_build_decoder(cfg), AutoVideoDecoder)


def test_annotated_fvessel_batches_omit_non_collatable_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, manifest = _fvessel_manifest(tmp_path / "fvessel", videos=2)
    manifest = replace(
        manifest,
        records=(manifest.records[0], replace(manifest.records[1], split="val")),
    )
    cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")
    monkeypatch.setattr(
        "marineworld.train.pretrain._build_decoder",
        lambda _: SyntheticVideoDecoder(height=16, width=16),
    )
    monkeypatch.setattr(
        FVesselAdapter,
        "load_targets",
        lambda *_: pytest.fail("SSL must not parse supervised annotations"),
    )

    dataloaders = build_dataloaders(cfg, manifest)
    batch = next(iter(dataloaders["train_dataloaders"]))

    assert batch.keys() == {"pixel_values", "dataset", "record_id", "source"}
    assert batch["pixel_values"].shape == (1, 4, 3, 16, 16)


def test_balanced_sampler_equalizes_datasets_and_replays_from_epoch() -> None:
    dataset_ids = ("smd",) * 90 + ("fvessel",) * 10
    first = BalancedDatasetSampler(dataset_ids, weights={"smd": 0.5, "fvessel": 0.5}, seed=7)
    first.set_epoch(3)
    indices = list(first)
    first.position = 10
    state = first.state_dict()
    resumed = BalancedDatasetSampler(dataset_ids, weights={"smd": 0.5, "fvessel": 0.5}, seed=7)
    resumed.load_state_dict(state)

    assert [dataset_ids[index] for index in indices].count("smd") == 50
    assert [dataset_ids[index] for index in indices].count("fvessel") == 50
    assert list(resumed) == indices[10:]


def test_dataloader_uses_distributed_rank_from_lightning_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _smoke_config(tmp_path, max_steps=2)
    adapter = SyntheticAdapter(num_frames=4, num_videos=2)
    manifest = adapter.build_manifest(tmp_path / "data")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    sampler = build_dataloaders(cfg, manifest)["train_dataloaders"].sampler

    assert isinstance(sampler, BalancedDatasetSampler)
    assert (sampler.rank, sampler.replicas) == (1, 2)


def test_balanced_sampler_partitions_one_global_order_across_ranks() -> None:
    dataset_ids = ("smd",) * 8 + ("fvessel",) * 4
    global_order = list(BalancedDatasetSampler(dataset_ids, seed=42))
    rank_zero = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=0, replicas=2))
    rank_one = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=1, replicas=2))

    assert rank_zero == global_order[0::2]
    assert rank_one == global_order[1::2]


def test_balanced_sampler_pads_odd_global_order_equally_across_ranks() -> None:
    dataset_ids = ("smd",) * 7 + ("fvessel",) * 4
    rank_zero = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=0, replicas=2))
    rank_one = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=1, replicas=2))

    assert len(rank_zero) == len(rank_one) == 6
    combined = rank_zero + rank_one
    assert [dataset_ids[index] for index in combined].count("smd") == 6
    assert [dataset_ids[index] for index in combined].count("fvessel") == 6


def test_default_fvessel_manifest_is_split_by_video_without_overlap(tmp_path: Path) -> None:
    adapter, manifest = _fvessel_manifest(tmp_path / "fvessel")
    cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

    first = _prepare_manifest(cfg, manifest)
    second = _prepare_manifest(cfg, adapter.build_manifest(tmp_path / "fvessel"))

    first_splits = {
        split: {record.video_path for record in first.records if record.split == split}
        for split in ("train", "val", "test")
    }
    second_splits = {
        split: {record.video_path for record in second.records if record.split == split}
        for split in ("train", "val", "test")
    }
    assert first_splits == second_splits
    assert first_splits["train"]
    assert first_splits["val"]
    assert not first_splits["train"] & first_splits["val"]
    assert not first_splits["train"] & first_splits["test"]
    assert not first_splits["val"] & first_splits["test"]


@pytest.mark.parametrize("videos", [2, 3])
def test_small_fvessel_manifest_still_has_train_and_validation(tmp_path: Path, videos: int) -> None:
    _, manifest = _fvessel_manifest(tmp_path / "fvessel", videos=videos)
    cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

    prepared = _prepare_manifest(cfg, manifest)

    assert {record.split for record in prepared.records} >= {"train", "val"}


def test_single_video_cannot_form_train_and_validation_splits(tmp_path: Path) -> None:
    _, manifest = _fvessel_manifest(tmp_path / "fvessel", videos=1)
    cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

    with pytest.raises(ValueError, match="at least two videos"):
        _prepare_manifest(cfg, manifest)


def test_empty_data_root_fails_before_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")
    cfg.data.root = str(tmp_path / "empty")
    cfg.data.adapter.fps = 30.0
    cfg.output_dir = str(tmp_path / "output")
    cfg.tracking.mode = "disabled"
    monkeypatch.setattr(
        "marineworld.train.pretrain.build_trainer",
        lambda *_args, **_kwargs: pytest.fail("Trainer must not be built for empty data"),
    )

    with pytest.raises(ValueError, match="manifest contains no records"):
        run_pretraining(cfg)


def test_empty_data_root_cli_exits_without_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    output_dir = tmp_path / "output"
    root.mkdir()
    environment = os.environ | {"KMP_USE_SHM": "0", "TMPDIR": "/tmp"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marineworld.train.pretrain",
            "data=fvessel",
            "model=videomae_tiny",
            "runtime=local_smoke",
            "data.adapter.fps=30",
            f"data.root={root}",
            f"output_dir={output_dir}",
        ],
        capture_output=True,
        cwd=tmp_path,
        env=environment,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert "manifest contains no records" in result.stderr
    assert not (output_dir / "checkpoints").exists()


def test_too_short_records_fail_before_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _smoke_config(tmp_path, max_steps=2)
    cfg.data.adapter.num_frames = 2
    monkeypatch.setattr(
        "marineworld.train.pretrain.build_trainer",
        lambda *_args, **_kwargs: pytest.fail("Trainer must not be built for empty loaders"),
    )

    with pytest.raises(ValueError, match="training split produced no clips"):
        run_pretraining(cfg)


@pytest.mark.parametrize(
    ("last_path", "message"),
    [("", "without a last checkpoint path"), ("missing.ckpt", "last checkpoint does not exist")],
)
def test_run_pretraining_rejects_invalid_last_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    last_path: str,
    message: str,
) -> None:
    def _trainer_factory(
        _cfg: DictConfig, *, logger: object, callbacks: list[object]
    ) -> _CheckpointPathTrainer:
        del logger
        return _CheckpointPathTrainer(callbacks[0], last_path)

    monkeypatch.setattr("marineworld.train.pretrain.build_wandb_logger", lambda *_: False)
    monkeypatch.setattr("marineworld.train.pretrain.build_trainer", _trainer_factory)
    monkeypatch.setattr("marineworld.train.pretrain._build_module", lambda _: object())

    with pytest.raises(RuntimeError, match=message):
        run_pretraining(_smoke_config(tmp_path, max_steps=2))


def test_run_pretraining_finishes_wandb_before_starting_another_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    active = False
    finishes = 0
    failures = iter((False, True))

    def _finish() -> None:
        nonlocal active, finishes
        active = False
        finishes += 1

    def _logger_factory(*_: object) -> object:
        nonlocal active
        assert not active, "the prior W&B run leaked into the next training call"
        active = True
        return SimpleNamespace(experiment=SimpleNamespace(finish=_finish))

    def _trainer_factory(
        _cfg: DictConfig, *, logger: object, callbacks: list[object]
    ) -> _FakeTrainer:
        del logger
        return _FakeTrainer(callbacks[0], fail=next(failures))

    monkeypatch.setattr("marineworld.train.pretrain.build_wandb_logger", _logger_factory)
    monkeypatch.setattr("marineworld.train.pretrain.build_trainer", _trainer_factory)
    monkeypatch.setattr("marineworld.train.pretrain._build_module", lambda _: object())

    assert run_pretraining(_smoke_config(tmp_path / "success", max_steps=2)).is_file()
    assert not active

    with pytest.raises(RuntimeError, match="forced trainer failure"):
        run_pretraining(_smoke_config(tmp_path / "failure", max_steps=2))

    assert not active
    assert finishes == 2


def test_local_smoke_saves_best_and_last_then_resumes(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path, max_steps=2)

    first_checkpoint = run_pretraining(cfg)

    assert first_checkpoint == tmp_path / "output" / "checkpoints" / "last.ckpt"
    assert first_checkpoint.exists()
    assert any(path.name != "last.ckpt" for path in first_checkpoint.parent.glob("*.ckpt"))
    first_state = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
    assert first_state["global_step"] == 2

    resumed = OmegaConf.merge(
        cfg,
        {"runtime": {"max_steps": 3, "ckpt_path": str(first_checkpoint)}},
    )
    second_checkpoint = run_pretraining(resumed)

    state = torch.load(second_checkpoint, map_location="cpu", weights_only=False)
    assert state["global_step"] == 3


def test_local_smoke_saves_last_checkpoint_on_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_training_step = VideoMAEPretrainingModule.training_step

    def _fail_after_first_step(
        module: VideoMAEPretrainingModule,
        batch: dict[str, object],
        batch_idx: int,
    ) -> torch.Tensor:
        if module.global_step == 1:
            raise RuntimeError("forced training failure")
        return original_training_step(module, batch, batch_idx)

    monkeypatch.setattr(VideoMAEPretrainingModule, "training_step", _fail_after_first_step)
    cfg = _smoke_config(tmp_path, max_steps=2)

    with pytest.raises(RuntimeError, match="forced training failure"):
        run_pretraining(cfg)

    checkpoint = tmp_path / "output" / "checkpoints" / "last.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["global_step"] == 1


def test_offline_smoke_creates_local_wandb_run_without_network(
    tmp_path: Path,
) -> None:
    wandb_dir = tmp_path / "wandb-data"
    wandb_dir.mkdir()
    output_dir = tmp_path / "output"
    data_root = tmp_path / "data"
    environment = os.environ | {
        "KMP_USE_SHM": "0",
        "TMPDIR": "/tmp",
        "WANDB_BASE_URL": "http://127.0.0.1:9",
        "WANDB_DIR": str(wandb_dir),
        "WANDB_MODE": "offline",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marineworld.train.pretrain",
            "data=synthetic",
            "model=videomae_tiny",
            "runtime=local_smoke",
            f"data.root={data_root}",
            f"output_dir={output_dir}",
        ],
        check=True,
        capture_output=True,
        cwd=wandb_dir,
        env=environment,
        text=True,
        timeout=30,
    )

    assert (output_dir / "checkpoints" / "last.ckpt").exists()
    assert any(wandb_dir.rglob("offline-run-*"))
    assert "W&B syncing is set to `offline`" in result.stderr


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


def test_masks_are_unique_per_accumulated_microbatch() -> None:
    module = VideoMAEPretrainingModule(_tiny_model_config(), seed=42)

    first = module.make_mask(1, torch.device("cpu"), step=3, microbatch=0)
    second = module.make_mask(1, torch.device("cpu"), step=3, microbatch=1)

    assert not torch.equal(first, second)


def test_optimizer_uses_warmup_cosine_scheduler() -> None:
    module = VideoMAEPretrainingModule(_tiny_model_config(), lr=1e-3, warmup_epochs=1, max_epochs=4)
    configured = module.configure_optimizers()

    assert configured["lr_scheduler"]["interval"] == "step"
    scheduler = configured["lr_scheduler"]["scheduler"]
    initial = scheduler.get_last_lr()[0]
    values = []
    for _ in range(8):
        scheduler.optimizer.step()
        scheduler.step()
        values.append(scheduler.get_last_lr()[0])
    assert initial < values[0]
    assert values[-1] < values[2]

    resumed_module = VideoMAEPretrainingModule(
        _tiny_model_config(), lr=1e-3, warmup_epochs=1, max_epochs=4
    )
    resumed = resumed_module.configure_optimizers()["lr_scheduler"]["scheduler"]
    resumed.load_state_dict(scheduler.state_dict())
    assert resumed.get_last_lr() == scheduler.get_last_lr()


def test_scheduler_uses_trainer_optimizer_step_estimate() -> None:
    module = VideoMAEPretrainingModule(_tiny_model_config(), lr=1e-3, warmup_epochs=1, max_epochs=4)
    module._trainer = SimpleNamespace(estimated_stepping_batches=4)
    scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

    for _ in range(4):
        scheduler.optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr() == [0.0]


def test_training_manifest_records_exact_splits_without_raw_paths(tmp_path: Path) -> None:
    _, manifest = _fvessel_manifest(tmp_path / "restricted", videos=2)

    path = write_training_manifest(manifest, tmp_path / "artifacts")
    payload = json.loads(path.read_text())

    assert payload["manifest_checksum"] == manifest_checksum(manifest)
    assert {record["id"] for record in payload["records"]} == {
        record.id for record in manifest.records
    }
    assert "video_path" not in path.read_text()
    assert str(tmp_path / "restricted") not in path.read_text()


@pytest.mark.parametrize(
    ("update", "value"),
    [
        ("data.sampling_weights.smd", 0.9),
        ("data.transforms.train.color_jitter", 0.2),
        ("train.warmup_epochs", 20),
    ],
)
def test_training_identity_hashes_material_scientific_config(
    tmp_path: Path, update: str, value: object
) -> None:
    cfg = _compose_config("data=joint_synthetic", "model=videomae_tiny")
    first = SyntheticAdapter(num_frames=4, num_videos=2).build_manifest(tmp_path / "data")
    baseline = _training_run_identity(cfg, first)
    OmegaConf.update(cfg, update, value)

    assert _training_run_identity(cfg, first).run_id != baseline.run_id


def test_training_identity_ignores_execution_hardware(tmp_path: Path) -> None:
    manifest = SyntheticAdapter(num_frames=4, num_videos=2).build_manifest(tmp_path / "data")
    l4 = _compose_config("data=joint_synthetic", "model=videomae_tiny", "runtime=l4")
    a100 = _compose_config("data=joint_synthetic", "model=videomae_tiny", "runtime=a100")
    a100.runtime.batch_size = l4.runtime.batch_size
    a100.runtime.accumulate_grad_batches = l4.runtime.accumulate_grad_batches

    assert (
        _training_run_identity(l4, manifest).run_id == _training_run_identity(a100, manifest).run_id
    )


def test_training_identity_is_stable_across_resume_horizon(tmp_path: Path) -> None:
    manifest = SyntheticAdapter(num_frames=4, num_videos=2).build_manifest(tmp_path / "data")
    first = _smoke_config(tmp_path, max_steps=2)
    resumed = OmegaConf.merge(
        first,
        {
            "runtime": {
                "max_steps": 3,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "ckpt_path": str(tmp_path / "last.ckpt"),
            }
        },
    )

    assert (
        _training_run_identity(first, manifest).run_id
        == _training_run_identity(resumed, manifest, "sha256:resume").run_id
    )


def test_sampler_callback_advances_relative_to_restored_position() -> None:
    sampler = BalancedDatasetSampler(("smd",) * 4 + ("fvessel",) * 4, seed=42)
    callback = _BalancedSamplerCheckpoint(sampler, batch_size=2)
    callback.load_state_dict({"epoch": 0, "position": 4})

    callback.on_train_batch_end(None, None, None, None, batch_idx=0)  # type: ignore[arg-type]
    assert sampler.position == 6
    callback.on_train_batch_end(None, None, None, None, batch_idx=1)  # type: ignore[arg-type]

    assert sampler.position == 8


def test_build_dataloaders_accepts_legacy_adapter_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = SyntheticAdapter(num_frames=4, num_videos=2)
    manifest = adapter.build_manifest(tmp_path / "data")
    cfg = _smoke_config(tmp_path, max_steps=1)

    loaders = build_dataloaders(cfg, manifest, adapter)

    assert loaders["train_dataloaders"]


def test_joint_split_preserves_each_component_in_training(tmp_path: Path) -> None:
    composite = CompositeAdapter(
        components={
            "smd": SyntheticAdapter(num_frames=4, num_videos=2),
            "fvessel": SyntheticAdapter(num_frames=4, num_videos=2),
        },
        roots={"smd": tmp_path / "smd", "fvessel": tmp_path / "fvessel"},
    )
    manifest = composite.build_manifest(tmp_path)
    manifest = replace(
        manifest,
        records=tuple(replace(record, split="train") for record in manifest.records),
    )
    cfg = _compose_config("data=joint_synthetic", "model=videomae_tiny")
    cfg.seed = 5

    prepared = _prepare_manifest(cfg, manifest)

    assert {record.dataset for record in prepared.records if record.split == "train"} == {
        "smd",
        "fvessel",
    }


def test_draw_tokens_vary_by_epoch_and_replay_after_resume(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path, max_steps=2)
    adapter = SyntheticAdapter(num_frames=4, num_videos=4)
    manifest = adapter.build_manifest(tmp_path / "data")
    loader = build_dataloaders(cfg, manifest)["train_dataloaders"]
    sampler = loader.sampler
    assert isinstance(sampler, BalancedDatasetSampler)
    dataset = loader.dataset

    first_draw = list(sampler)[0]
    sampler.set_epoch(1)
    second_draw = list(sampler)[0]
    assert not torch.equal(
        dataset[first_draw]["pixel_values"], dataset[second_draw]["pixel_values"]
    )

    sampler.set_epoch(3)
    sampler.position = 1
    state = sampler.state_dict()
    remaining = list(sampler)
    replay = BalancedDatasetSampler(
        tuple("synthetic" for _ in range(len(dataset))),
        seed=int(cfg.seed),
        draw_tokens=True,
    )
    replay.load_state_dict(state)
    replayed = list(replay)
    assert replayed == remaining
    assert all(
        torch.equal(dataset[left]["pixel_values"], dataset[right]["pixel_values"])
        for left, right in zip(remaining, replayed, strict=True)
    )

    val_dataset = build_dataloaders(cfg, manifest)["val_dataloaders"].dataset
    assert torch.equal(val_dataset[0]["pixel_values"], val_dataset[0]["pixel_values"])


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

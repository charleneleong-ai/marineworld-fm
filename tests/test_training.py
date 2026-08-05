"""Experiment tracking contracts that require no W&B service access."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import marineworld.train.experiment as experiment_module
from marineworld.data.adapters import CompositeAdapter
from marineworld.data.clips import AutoVideoDecoder, SyntheticVideoDecoder, build_clip_index
from marineworld.data.contracts import DatasetManifest
from marineworld.data.fvessel import FVesselAdapter
from marineworld.data.manifest import manifest_checksum
from marineworld.data.synthetic import SyntheticAdapter
from marineworld.models.videomae import build_videomae, encode_video, tube_mask
from marineworld.train.experiment import (
    RunIdentity,
    build_run_identity,
    build_wandb_logger,
    resolved_config,
)
from marineworld.train.media import WandbMediaCallback, build_media_preview
from marineworld.train.module import VideoMAEPretrainingModule
from marineworld.train.naming import metric_name
from marineworld.train.pretrain import (
    BalancedDatasetSampler,
    BalancedSamplerCheckpoint,
    build_dataloaders,
    build_decoder,
    build_module,
    prepare_training_manifest,
    run_pretraining,
    training_run_identity,
    validate_media_config,
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


class TestBuildMediaPreview:
    """Reconstruction preview rendering and shape validation."""

    def test_build_media_preview_denormalizes_and_bounds_output(self) -> None:
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

    def test_build_media_preview_places_decoder_patches_and_expands_tube_masks(self) -> None:
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

    def test_build_media_preview_places_decoder_patches_after_input_denormalization(self) -> None:
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

    def test_build_media_preview_unnormalizes_patch_normalized_decoder_logits(self) -> None:
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
        self,
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
        self,
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


class FakeExperiment:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int | None]] = []
        self.explicit_steps: list[int] = []
        self.history: dict[str, object] = {}
        self.local_step = 0
        self.finished = False

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        if step is not None:
            self.explicit_steps.append(step)
            if step < self.local_step:
                raise ValueError("non-monotonic explicit W&B step")
            self.local_step = step
        self.calls.append((payload, step))
        self.history.update(payload)
        self.local_step += 1

    def finish(self) -> None:
        self.finished = True


class WandbLogger:
    __module__ = "pytorch_lightning.loggers.wandb"

    def __init__(self, experiment: FakeExperiment) -> None:
        self.experiment = experiment

    def log_metrics(self, metrics: dict[str, object], step: int | None = None) -> None:
        payload = dict(metrics)
        if step is not None:
            payload["trainer/global_step"] = step
        self.experiment.log(payload)


class FakeImage:
    def __init__(self, data: object, caption: str) -> None:
        self.data = data
        self.caption = caption


def _media_batch() -> dict[str, object]:
    return {
        "pixel_values": torch.stack((torch.zeros(4, 3, 16, 16), torch.ones(4, 3, 16, 16))),
        "dataset": ("synthetic", "other"),
        "source": ("generated/clip-0", "generated/clip-1"),
        "record_id": ("clip-0", "clip-1"),
    }


def _media_trainer(logger: object, *, rank: int = 0, epoch: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        logger=logger,
        global_rank=rank,
        current_epoch=epoch,
        global_step=7,
    )


def _media_callback(
    checkpoint: object, *, enabled: bool = True, every_n_epochs: int = 2
) -> WandbMediaCallback:
    return WandbMediaCallback(
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        enabled=enabled,
        every_n_epochs=every_n_epochs,
        max_frames=2,
        checkpoint_callback=checkpoint,
    )


class TestWandbMediaCallback:
    """Validation, best-checkpoint and config behaviour of the media callback."""

    @pytest.mark.parametrize(
        ("enabled", "rank", "epoch", "expected_calls"),
        [(False, 0, 0, 0), (True, 1, 0, 0), (True, 0, 1, 0), (True, 0, 2, 1)],
    )
    def test_media_callback_gates_validation_logging(
        self,
        monkeypatch: pytest.MonkeyPatch,
        enabled: bool,
        rank: int,
        epoch: int,
        expected_calls: int,
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        experiment = FakeExperiment()
        callback = _media_callback(SimpleNamespace(best_model_path=""), enabled=enabled)
        module = VideoMAEPretrainingModule(_tiny_model_config())

        callback.on_validation_batch_end(
            _media_trainer(WandbLogger(experiment), rank=rank, epoch=epoch),
            module,
            None,
            _media_batch(),
            0,
        )

        assert len(experiment.calls) == expected_calls
        if expected_calls:
            payload, step = experiment.calls[0]
            assert set(payload) == {
                "val/synthetic/media/inputs",
                "val/synthetic/media/reconstruction",
                "trainer/global_step",
            }
            assert step is None
            assert payload["trainer/global_step"] == 7
            media = {key: image for key, image in payload.items() if "/media/" in key}
            assert all(isinstance(image, FakeImage) for image in media.values())
            assert all(
                image.caption == "dataset=synthetic source=generated/clip-0"
                for image in media.values()
            )
            assert payload["val/synthetic/media/inputs"].data.shape == (16, 32, 3)

    @pytest.mark.parametrize(
        ("hook", "stage", "dataset"),
        [
            ("on_train_batch_end", "pretrain", "synthetic"),
            ("on_validation_batch_end", "val", "fvessel"),
            ("on_validation_batch_end", "val", "smd"),
        ],
    )
    def test_media_keys_are_namespaced_by_stage_and_dataset(
        self, monkeypatch: pytest.MonkeyPatch, hook: str, stage: str, dataset: str
    ) -> None:
        """Joint runs must not overwrite one dataset's panel with another's."""
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        experiment = FakeExperiment()
        callback = _media_callback(SimpleNamespace(best_model_path=""))

        getattr(callback, hook)(
            _media_trainer(WandbLogger(experiment), rank=0, epoch=2),
            VideoMAEPretrainingModule(_tiny_model_config()),
            None,
            _media_batch() | {"dataset": (dataset, "other")},
            0,
        )

        payload, _ = experiment.calls[0]
        assert {key for key in payload if "/media/" in key} == {
            f"{stage}/{dataset}/media/inputs",
            f"{stage}/{dataset}/media/reconstruction",
        }

    @pytest.mark.parametrize(
        ("enabled", "logger"),
        [
            (False, WandbLogger(FakeExperiment())),
            (True, False),
            (True, SimpleNamespace(experiment=FakeExperiment())),
        ],
    )
    def test_media_callback_skips_inference_and_wandb_import_without_active_wandb(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool, logger: object
    ) -> None:
        monkeypatch.delitem(sys.modules, "wandb", raising=False)
        module = VideoMAEPretrainingModule(_tiny_model_config())
        monkeypatch.setattr(
            module.model,
            "forward",
            lambda **_: pytest.fail("media inference must be gated before model execution"),
        )
        callback = _media_callback(SimpleNamespace(best_model_path=""), enabled=enabled)

        callback.on_validation_batch_end(
            _media_trainer(logger, epoch=2), module, None, _media_batch(), 0
        )

        assert "wandb" not in sys.modules

    def test_media_callback_skips_secondary_validation_loaders(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        experiment = FakeExperiment()
        callback = _media_callback(SimpleNamespace(best_model_path=""))

        callback.on_validation_batch_end(
            _media_trainer(WandbLogger(experiment), epoch=2),
            VideoMAEPretrainingModule(_tiny_model_config()),
            None,
            _media_batch(),
            0,
            dataloader_idx=1,
        )

        assert experiment.calls == []

    def test_best_checkpoint_preview_uses_best_and_preserves_live_module(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        model_config = _tiny_model_config() | {"norm_pix_loss": False}
        best = VideoMAEPretrainingModule(model_config)
        live = VideoMAEPretrainingModule(model_config)
        with torch.no_grad():
            best.model.decoder.head.weight.zero_()
            best.model.decoder.head.bias.fill_(0.1)
            live.model.decoder.head.weight.zero_()
            live.model.decoder.head.bias.fill_(0.9)
        best_path = tmp_path / "best.ckpt"
        last_path = tmp_path / "last.ckpt"
        torch.save({"state_dict": best.state_dict()}, best_path)
        torch.save({"state_dict": live.state_dict()}, last_path)
        checkpoint = SimpleNamespace(best_model_path=str(best_path), last_model_path=str(last_path))
        experiment = FakeExperiment()
        trainer = _media_trainer(WandbLogger(experiment), epoch=2)
        callback = _media_callback(checkpoint)
        callback.on_validation_batch_end(trainer, live, None, _media_batch(), 0)
        experiment.calls.clear()
        before = {
            name: value.detach().cpu().numpy().tobytes() for name, value in live.named_parameters()
        }

        callback.on_fit_end(trainer, live)

        assert len(experiment.calls) == 1
        payload, step = experiment.calls[0]
        assert set(payload) == {
            "best/synthetic/media/inputs",
            "best/synthetic/media/reconstruction",
            "trainer/global_step",
        }
        assert step is None
        assert payload["trainer/global_step"] == 7
        reconstruction = payload["best/synthetic/media/reconstruction"]
        assert isinstance(reconstruction, FakeImage)
        assert np.isclose(reconstruction.data[:, 32:48], 0.1).any()
        assert not np.isclose(reconstruction.data[:, 32:48], 0.9).any()
        assert all(
            value.detach().cpu().numpy().tobytes() == before[name]
            for name, value in live.named_parameters()
        )

    def test_best_checkpoint_preview_warns_and_skips_when_path_is_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        experiment = FakeExperiment()
        trainer = _media_trainer(WandbLogger(experiment), epoch=2)
        module = VideoMAEPretrainingModule(_tiny_model_config())
        callback = _media_callback(SimpleNamespace(best_model_path=""))
        callback.on_validation_batch_end(trainer, module, None, _media_batch(), 0)
        experiment.calls.clear()

        with pytest.warns(UserWarning, match="best checkpoint"):
            callback.on_fit_end(trainer, module)

        assert experiment.calls == []

    def test_best_checkpoint_preview_is_rank_zero_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        module = VideoMAEPretrainingModule(_tiny_model_config())
        checkpoint_path = tmp_path / "best.ckpt"
        torch.save({"state_dict": module.state_dict()}, checkpoint_path)
        experiment = FakeExperiment()
        trainer = _media_trainer(WandbLogger(experiment), epoch=2)
        callback = _media_callback(SimpleNamespace(best_model_path=str(checkpoint_path)))
        callback.on_validation_batch_end(trainer, module, None, _media_batch(), 0)
        experiment.calls.clear()
        trainer.global_rank = 1

        callback.on_fit_end(trainer, module)

        assert experiment.calls == []

    def test_best_checkpoint_preview_uses_sample_from_non_periodic_epoch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        module = VideoMAEPretrainingModule(_tiny_model_config())
        checkpoint_path = tmp_path / "best.ckpt"
        torch.save({"state_dict": module.state_dict()}, checkpoint_path)
        experiment = FakeExperiment()
        trainer = _media_trainer(WandbLogger(experiment), epoch=1)
        callback = _media_callback(SimpleNamespace(best_model_path=str(checkpoint_path)))

        callback.on_validation_batch_end(trainer, module, None, _media_batch(), 0)
        assert experiment.calls == []
        callback.on_fit_end(trainer, module)

        assert len(experiment.calls) == 1
        payload, _ = experiment.calls[0]
        assert set(payload) == {
            "best/synthetic/media/inputs",
            "best/synthetic/media/reconstruction",
            "trainer/global_step",
        }

    def test_media_callback_uses_lightning_logger_without_explicit_wandb_steps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        module = VideoMAEPretrainingModule(_tiny_model_config())
        checkpoint_path = tmp_path / "best.ckpt"
        torch.save({"state_dict": module.state_dict()}, checkpoint_path)
        experiment = FakeExperiment()
        logger = WandbLogger(experiment)
        trainer = _media_trainer(logger, epoch=2)
        callback = _media_callback(SimpleNamespace(best_model_path=str(checkpoint_path)))
        for step in range(8):
            logger.log_metrics({"train/loss": float(step)}, step=step)

        callback.on_validation_batch_end(trainer, module, None, _media_batch(), 0)
        callback.on_fit_end(trainer, module)

        assert {
            "val/synthetic/media/inputs",
            "val/synthetic/media/reconstruction",
            "best/synthetic/media/inputs",
            "best/synthetic/media/reconstruction",
        } <= experiment.history.keys()
        assert experiment.explicit_steps == []
        assert experiment.finished is False

    def test_best_checkpoint_preview_loads_checkpoint_with_safe_torch_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        module = VideoMAEPretrainingModule(_tiny_model_config())
        checkpoint_path = tmp_path / "best.ckpt"
        calls: list[tuple[Path, dict[str, object]]] = []

        def _safe_load(path: Path, **kwargs: object) -> dict[str, object]:
            calls.append((path, kwargs))
            return {"state_dict": module.state_dict()}

        monkeypatch.setattr(torch, "load", _safe_load)
        experiment = FakeExperiment()
        trainer = _media_trainer(WandbLogger(experiment), epoch=2)
        callback = _media_callback(SimpleNamespace(best_model_path=str(checkpoint_path)))
        callback.on_validation_batch_end(trainer, module, None, _media_batch(), 0)

        callback.on_fit_end(trainer, module)

        assert calls == [
            (
                checkpoint_path,
                {"map_location": "cpu", "weights_only": True, "mmap": True},
            )
        ]

    @pytest.mark.parametrize(
        ("case", "message"),
        [
            ("missing", "checkpoint could not be loaded"),
            ("corrupt", "checkpoint could not be loaded"),
            ("incompatible", "checkpoint state is incompatible"),
        ],
    )
    def test_best_checkpoint_preview_warnings_do_not_leak_paths_or_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        case: str,
        message: str,
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        sentinel = "PRIVATE-CHECKPOINT-PATH-SENTINEL"
        checkpoint_path = tmp_path / sentinel / f"{case}.ckpt"
        checkpoint_path.parent.mkdir()
        if case == "corrupt":
            checkpoint_path.write_bytes(b"not a pytorch archive")
        elif case == "incompatible":
            torch.save({"state_dict": {sentinel: torch.tensor(1)}}, checkpoint_path)
        module = VideoMAEPretrainingModule(_tiny_model_config())
        experiment = FakeExperiment()
        trainer = _media_trainer(WandbLogger(experiment), epoch=2)
        callback = _media_callback(SimpleNamespace(best_model_path=str(checkpoint_path)))
        callback.on_validation_batch_end(trainer, module, None, _media_batch(), 0)
        experiment.calls.clear()

        with pytest.warns(UserWarning) as warnings:
            callback.on_fit_end(trainer, module)

        assert experiment.calls == []
        assert [str(warning.message) for warning in warnings] == [
            f"best checkpoint media skipped: {message}"
        ]
        assert sentinel not in str(warnings[0].message)

    def test_best_checkpoint_preview_loads_real_lightning_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
        last_checkpoint = run_pretraining(_smoke_config(tmp_path, max_steps=2))
        best_checkpoint = next(
            path for path in last_checkpoint.parent.glob("*.ckpt") if path != last_checkpoint
        )
        module = VideoMAEPretrainingModule(_tiny_model_config())
        experiment = FakeExperiment()
        trainer = _media_trainer(WandbLogger(experiment), epoch=2)
        callback = _media_callback(SimpleNamespace(best_model_path=str(best_checkpoint)))
        callback.on_validation_batch_end(trainer, module, None, _media_batch(), 0)
        experiment.calls.clear()

        callback.on_fit_end(trainer, module)

        payload, step = experiment.calls[0]
        assert set(payload) >= {
            "best/synthetic/media/inputs",
            "best/synthetic/media/reconstruction",
        }
        assert step is None

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            *[
                (
                    "media_log_every_n_epochs",
                    value,
                    "tracking.media_log_every_n_epochs must be positive",
                )
                for value in (0, -1, "1", 1.5)
            ],
            *[
                ("media_max_frames", value, "tracking.media_max_frames must be positive")
                for value in (0, -1, 5, "4", 1.5)
            ],
            *[
                ("log_media", value, "tracking.log_media must be boolean")
                for value in ("false", "off", 1, 0, None)
            ],
        ],
    )
    def test_media_config_rejects_invalid_values_before_construction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str, value: object, message: str
    ) -> None:
        cfg = _smoke_config(tmp_path, max_steps=1)
        cfg.tracking[key] = value
        monkeypatch.setattr(
            "marineworld.train.pretrain.build_data_adapter",
            lambda *_: pytest.fail("config must fail before adapter construction"),
        )
        monkeypatch.setattr(
            "marineworld.train.pretrain.build_wandb_logger",
            lambda *_: pytest.fail("config must fail before logger construction"),
        )

        with pytest.raises(ValueError, match=message):
            run_pretraining(cfg)

    @pytest.mark.parametrize("value", [True, False])
    def test_media_config_accepts_boolean_log_media(self, tmp_path: Path, value: bool) -> None:
        cfg = _smoke_config(tmp_path, max_steps=1)
        cfg.tracking.log_media = value

        assert validate_media_config(cfg) is None

    def test_wandb_tracking_enables_media_by_default(self) -> None:
        cfg = _compose_config("data=synthetic", "model=videomae_tiny", "runtime=local_smoke")

        assert cfg.tracking.log_media is True
        assert cfg.tracking.media_log_every_n_epochs == 1
        assert cfg.tracking.media_max_frames == 4


def _fvessel_manifest(root: Path, *, videos: int = 8) -> tuple[FVesselAdapter, DatasetManifest]:
    for index in range(videos):
        sample = root / f"sample-{index:02d}"
        (sample / "gt").mkdir(parents=True)
        (sample / "clip.mp4").touch()
        (sample / "gt" / "gt.txt").write_text("1,7,1,2,3,4,1,1,1\n")
    adapter = FVesselAdapter(fps=30.0, num_frames=8)
    return adapter, adapter.build_manifest(root)


def _write_reference_checkpoint(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"videomae"}')
    (root / "model.safetensors").write_bytes(b"weights")


def _write_reference_file(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_bytes(b"same checkpoint")


class FakeTrainer:
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


class CheckpointPathTrainer:
    def __init__(self, checkpoint: object, path: str) -> None:
        self.checkpoint = checkpoint
        self.path = path

    def fit(self, *_: object, **__: object) -> None:
        self.checkpoint.last_model_path = self.path


class TestResolvedConfig:
    """Config redaction, path allowlisting and reference-checkpoint content identity."""

    def test_resolved_config_excludes_wandb_key_name_and_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_resolved_config_keeps_non_secret_tokenizer_settings(self) -> None:
        cfg = OmegaConf.create({"model": {"tokenizer": "videomae"}})

        assert resolved_config(cfg) == {"model": {"tokenizer": "videomae"}}

    def test_resolved_config_keeps_scientific_runtime_settings(self) -> None:
        cfg = _compose_config("runtime=real_smoke")

        runtime = resolved_config(cfg)["runtime"]

        assert runtime == {
            "precision": "32-true",
            "batch_size": 1,
            "accumulate_grad_batches": 1,
            "max_steps": 2,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
        }
        assert "accelerator" not in runtime
        assert "devices" not in runtime
        assert "num_workers" not in runtime

    @pytest.mark.parametrize(
        ("machine_a", "machine_b", "make_checkpoint"),
        [
            ("machine-a/reference.ckpt", "machine-b/renamed.ckpt", _write_reference_file),
            ("machine-a/saved-model", "machine-b/renamed-model", _write_reference_checkpoint),
        ],
    )
    def test_reference_checkpoint_uses_portable_content_identity(
        self,
        tmp_path: Path,
        machine_a: str,
        machine_b: str,
        make_checkpoint: Callable[[Path], None],
    ) -> None:
        first, second = tmp_path / machine_a, tmp_path / machine_b
        make_checkpoint(first)
        make_checkpoint(second)

        first_resolved = resolved_config(
            OmegaConf.create({"model": {"reference_checkpoint": str(first)}})
        )
        second_resolved = resolved_config(
            OmegaConf.create({"model": {"reference_checkpoint": str(second)}})
        )
        first_identity = build_run_identity(
            "videomae", ("manifest",), 42, None, logical_dimensions={"config": first_resolved}
        )
        second_identity = build_run_identity(
            "videomae", ("manifest",), 42, None, logical_dimensions={"config": second_resolved}
        )

        assert first_resolved == second_resolved
        assert first_identity.condition_id == second_identity.condition_id
        assert str(tmp_path) not in json.dumps(first_resolved)

    def test_reference_checkpoint_content_changes_condition(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "reference.ckpt"
        cfg = OmegaConf.create({"model": {"reference_checkpoint": str(checkpoint)}})
        checkpoint.write_bytes(b"first checkpoint")
        first = resolved_config(cfg)

        checkpoint.write_bytes(b"different checkpoint")
        second = resolved_config(cfg)

        assert first != second

    @pytest.mark.parametrize(
        ("filename", "replacement"),
        [("config.json", b'{"model_type":"changed"}'), ("model.safetensors", b"new weights")],
    )
    def test_reference_checkpoint_directory_content_changes_condition(
        self, tmp_path: Path, filename: str, replacement: bytes
    ) -> None:
        checkpoint = tmp_path / "saved-model"
        _write_reference_checkpoint(checkpoint)
        cfg = OmegaConf.create({"model": {"reference_checkpoint": str(checkpoint)}})
        first = resolved_config(cfg)

        (checkpoint / filename).write_bytes(replacement)
        second = resolved_config(cfg)

        assert first != second

    @pytest.mark.parametrize("data_config", ["fvessel", "synthetic", "joint_synthetic"])
    def test_resolved_config_drops_component_roots_for_every_data_config(
        self, data_config: str
    ) -> None:
        """The allowlist must survive joint configs, whose roots nest under data.components."""
        configs = []
        for sentinel in ("PRIVATE-PATH-A", "PRIVATE-PATH-B"):
            cfg = _compose_config(
                f"data={data_config}", "model=videomae_tiny", "runtime=local_smoke"
            )
            cfg.output_dir = f"/{sentinel}/outputs"
            cfg.runtime.ckpt_path = f"/{sentinel}/checkpoint.ckpt"
            cfg.data.root = f"/{sentinel}/data"
            for component in cfg.data.get("components", {}).values():
                component.root = f"/{sentinel}/{component.adapter._target_}"
            OmegaConf.update(cfg, "tracking.api_key", f"{sentinel}-secret", force_add=True)
            configs.append(cfg)

        sanitized = resolved_config(configs[0])
        serialized = json.dumps(sanitized)

        assert resolved_config(configs[1]) == sanitized
        assert "PRIVATE-PATH" not in serialized
        assert "root" not in sanitized["data"]
        assert all(
            "root" not in component
            for component in sanitized["data"].get("components", {}).values()
        )
        for dropped in ("output_dir", "tracking"):
            assert dropped not in sanitized
        assert "ckpt_path" not in sanitized["runtime"]
        assert sanitized["model"]["name"] == "videomae_tiny"
        assert sanitized["runtime"]["batch_size"] == 1


class TestRunIdentity:
    """Run and training identity determinism and metadata isolation."""

    def test_run_identity_is_stable_across_manifest_order(self) -> None:
        identity = build_run_identity("videomae", ("sha-a", "sha-b"), seed=42, label_fraction=None)

        assert (
            identity.condition_id
            == build_run_identity(
                "videomae", ("sha-b", "sha-a"), seed=42, label_fraction=None
            ).condition_id
        )
        assert metric_name("probe", "macro_f1", "fvessel") == "probe/fvessel/macro_f1"

    def test_run_identity_keeps_execution_metadata_out_of_run_id(self) -> None:
        first = build_run_identity(
            "videomae",
            ("manifest",),
            seed=42,
            label_fraction=0.1,
            git_sha="first",
            accelerator="l4",
        )
        second = build_run_identity(
            "videomae",
            ("manifest",),
            seed=42,
            label_fraction=0.1,
            git_sha="second",
            accelerator="a100",
        )

        assert first.condition_id == second.condition_id
        assert first.run_id != second.run_id
        assert {"git:first", "accelerator:l4"} <= set(first.tags)

    def test_run_identity_changes_with_scientific_configuration(self) -> None:
        baseline = build_run_identity(
            "videomae",
            ("manifest",),
            seed=42,
            label_fraction=None,
            logical_dimensions={"config": {"model": {"mask_ratio": 0.9}, "train": {"lr": 1e-4}}},
        )
        changed = build_run_identity(
            "videomae",
            ("manifest",),
            seed=42,
            label_fraction=None,
            logical_dimensions={"config": {"model": {"mask_ratio": 0.75}, "train": {"lr": 1e-4}}},
        )

        assert baseline.condition_id != changed.condition_id
        assert f"condition:{baseline.condition_id}" in baseline.tags

    def test_run_identity_does_not_log_checkpoint_paths(self) -> None:
        identity = build_run_identity(
            "videomae",
            ("manifest",),
            seed=42,
            label_fraction=None,
            checkpoint_provenance="/private/checkpoints/last.ckpt",
        )

        assert "checkpoint:last.ckpt" in identity.tags
        assert "/private" not in json.dumps(identity.tags)

    def test_metric_name_omits_absent_dataset(self) -> None:
        assert metric_name("pretrain", "loss") == "pretrain/loss"

    @pytest.mark.parametrize(
        ("update", "value"),
        [
            ("data.sampling_weights.smd", 0.9),
            ("data.transforms.train.color_jitter", 0.2),
            ("train.warmup_epochs", 20),
        ],
    )
    def test_training_identity_hashes_material_scientific_config(
        self, tmp_path: Path, update: str, value: object
    ) -> None:
        cfg = _compose_config("data=joint_synthetic", "model=videomae_tiny")
        first = SyntheticAdapter(num_frames=4, num_videos=2).build_manifest(tmp_path / "data")
        baseline = training_run_identity(cfg, first)
        OmegaConf.update(cfg, update, value)

        assert training_run_identity(cfg, first).condition_id != baseline.condition_id

    def test_training_identity_ignores_execution_hardware(self, tmp_path: Path) -> None:
        manifest = SyntheticAdapter(num_frames=4, num_videos=2).build_manifest(tmp_path / "data")
        l4 = _compose_config("data=joint_synthetic", "model=videomae_tiny", "runtime=l4")
        a100 = _compose_config("data=joint_synthetic", "model=videomae_tiny", "runtime=a100")
        a100.runtime.batch_size = l4.runtime.batch_size
        a100.runtime.accumulate_grad_batches = l4.runtime.accumulate_grad_batches

        assert (
            training_run_identity(l4, manifest).condition_id
            == training_run_identity(a100, manifest).condition_id
        )

    def test_training_identity_is_stable_across_resume_horizon(self, tmp_path: Path) -> None:
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
            training_run_identity(first, manifest).condition_id
            == training_run_identity(resumed, manifest, "sha256:resume").condition_id
        )


class TestWandbLoggerBuild:
    """W&B logger construction and resolved-config reuse."""

    def test_disabled_tracking_does_not_construct_wandb_logger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        self, monkeypatch: pytest.MonkeyPatch, mode: str, expected_log_model: str | bool
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
        assert captured["resume"] == "never"
        assert "WANDB_API_KEY" not in json.dumps(captured["config"])

    def test_shared_resolved_config_is_not_recomputed_by_the_logger(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """resolved_config touches the filesystem, so the run must resolve it once."""
        calls: list[int] = []
        real = experiment_module.resolved_config

        def counting(cfg: DictConfig) -> dict[str, object]:
            calls.append(1)
            return real(cfg)

        monkeypatch.setattr(experiment_module, "resolved_config", counting)
        cfg = _compose_config("data=synthetic", "model=videomae_tiny", "runtime=local_smoke")
        identity = build_run_identity("videomae", ("manifest",), 42, None)

        build_wandb_logger(cfg, identity, real(cfg))

        assert calls == []


class TestRuntimeProfiles:
    """Hydra runtime profile composition and decoder selection."""

    @pytest.mark.parametrize(
        ("runtime", "tracking_mode"),
        [("local_smoke", "offline"), ("l4", "online"), ("a100", "online")],
    )
    def test_runtime_profiles_compose_with_tracking(self, runtime: str, tracking_mode: str) -> None:
        cfg = _compose_config(f"runtime={runtime}")

        assert cfg.runtime.tracking_mode == tracking_mode
        assert cfg.tracking.mode == tracking_mode

    def test_single_gpu_profiles_only_tune_hardware_capacity(self) -> None:
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

    def test_local_smoke_profile_uses_cpu(self) -> None:
        cfg = _compose_config("runtime=local_smoke")

        assert cfg.runtime.accelerator == "cpu"

    def test_real_smoke_profile_is_bounded_and_online(self) -> None:
        cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=real_smoke")

        assert cfg.runtime.max_steps == 2
        assert cfg.runtime.limit_train_batches == 2
        assert cfg.runtime.limit_val_batches == 1
        assert cfg.runtime.num_workers == 0
        assert cfg.runtime.tracking_mode == "online"

    def test_real_data_uses_portable_video_decoder(self) -> None:
        cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

        assert isinstance(build_decoder(cfg), AutoVideoDecoder)


class TestBalancedSampler:
    """Balanced sampling, resume, distributed sharding and draw tokens."""

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            ({"smd": 0.5, "fvessel": 0.5}, r"absent from the corpus: smd\. Present: fvessel"),
            ({"fvessel": 1.0, "nir": 1.0}, r"absent from the corpus: nir"),
            ({"fvessel": 0.0}, r"must be positive: fvessel"),
        ],
    )
    def test_balanced_sampler_names_the_offending_dataset(
        self, weights: dict[str, float], message: str
    ) -> None:
        """A joint config pointed at a missing corpus must say which one is missing."""
        with pytest.raises(ValueError, match=message):
            BalancedDatasetSampler(("fvessel", "fvessel"), weights=weights, seed=7)

    def test_clip_index_reuses_a_supplied_fingerprint(self, tmp_path: Path) -> None:
        """Both split datasets share one corpus hash instead of each rehashing it."""
        manifest = SyntheticAdapter(num_frames=4, num_videos=12).build_manifest(tmp_path / "data")
        index = functools.partial(
            build_clip_index, manifest, split="train", frames=2, stride=1, seed=42
        )

        assert index(fingerprint=manifest_checksum(manifest)) == index()
        # ordering is fingerprint-derived, so a supplied value must actually be used
        assert index(fingerprint="a-different-corpus") != index()

    def test_balanced_sampler_equalizes_datasets_and_replays_from_epoch(self) -> None:
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = _smoke_config(tmp_path, max_steps=2)
        adapter = SyntheticAdapter(num_frames=4, num_videos=2)
        manifest = adapter.build_manifest(tmp_path / "data")
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")

        sampler = build_dataloaders(cfg, manifest)["train_dataloaders"].sampler

        assert isinstance(sampler, BalancedDatasetSampler)
        assert (sampler.rank, sampler.replicas) == (1, 2)

    def test_balanced_sampler_partitions_one_global_order_across_ranks(self) -> None:
        dataset_ids = ("smd",) * 8 + ("fvessel",) * 4
        global_order = list(BalancedDatasetSampler(dataset_ids, seed=42))
        rank_zero = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=0, replicas=2))
        rank_one = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=1, replicas=2))

        assert rank_zero == global_order[0::2]
        assert rank_one == global_order[1::2]

    def test_balanced_sampler_pads_odd_global_order_equally_across_ranks(self) -> None:
        dataset_ids = ("smd",) * 7 + ("fvessel",) * 4
        rank_zero = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=0, replicas=2))
        rank_one = list(BalancedDatasetSampler(dataset_ids, seed=42, rank=1, replicas=2))

        assert len(rank_zero) == len(rank_one) == 6
        combined = rank_zero + rank_one
        assert [dataset_ids[index] for index in combined].count("smd") == 6
        assert [dataset_ids[index] for index in combined].count("fvessel") == 6

    def test_sampler_callback_advances_relative_to_restored_position(self) -> None:
        sampler = BalancedDatasetSampler(("smd",) * 4 + ("fvessel",) * 4, seed=42)
        callback = BalancedSamplerCheckpoint(sampler, batch_size=2)
        callback.load_state_dict({"epoch": 0, "position": 4})

        callback.on_train_batch_end(None, None, None, None, batch_idx=0)  # type: ignore[arg-type]
        assert sampler.position == 6
        callback.on_train_batch_end(None, None, None, None, batch_idx=1)  # type: ignore[arg-type]

        assert sampler.position == 8

    def test_draw_tokens_vary_by_epoch_and_replay_after_resume(self, tmp_path: Path) -> None:
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


class TestTrainingManifest:
    """Manifest splitting, path redaction and batch collation."""

    def test_annotated_fvessel_batches_omit_non_collatable_targets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        adapter, manifest = _fvessel_manifest(tmp_path / "fvessel", videos=2)
        manifest = replace(
            manifest,
            records=(manifest.records[0], replace(manifest.records[1], split="val")),
        )
        cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")
        monkeypatch.setattr(
            "marineworld.train.pretrain.build_decoder",
            lambda _: SyntheticVideoDecoder(height=16, width=16),
        )
        monkeypatch.setattr(
            FVesselAdapter,
            "load_targets",
            lambda *_: pytest.fail("SSL must not parse supervised annotations"),
        )

        dataloaders = build_dataloaders(cfg, manifest)
        batch = next(iter(dataloaders["train_dataloaders"]))

        assert batch.keys() == {
            "pixel_values",
            "dataset",
            "record_id",
            "source",
            "ais_features",
            "ais_mask",
        }
        assert batch["pixel_values"].shape == (1, 4, 3, 16, 16)

    def test_default_fvessel_manifest_is_split_by_video_without_overlap(
        self, tmp_path: Path
    ) -> None:
        adapter, manifest = _fvessel_manifest(tmp_path / "fvessel")
        cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

        first = prepare_training_manifest(cfg, manifest)
        second = prepare_training_manifest(cfg, adapter.build_manifest(tmp_path / "fvessel"))

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
    def test_small_fvessel_manifest_still_has_train_and_validation(
        self, tmp_path: Path, videos: int
    ) -> None:
        _, manifest = _fvessel_manifest(tmp_path / "fvessel", videos=videos)
        cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

        prepared = prepare_training_manifest(cfg, manifest)

        assert {record.split for record in prepared.records} >= {"train", "val"}

    def test_single_video_cannot_form_train_and_validation_splits(self, tmp_path: Path) -> None:
        _, manifest = _fvessel_manifest(tmp_path / "fvessel", videos=1)
        cfg = _compose_config("data=fvessel", "model=videomae_tiny", "runtime=local_smoke")

        with pytest.raises(ValueError, match="at least two videos"):
            prepare_training_manifest(cfg, manifest)

    def test_training_manifest_records_exact_splits_without_raw_paths(self, tmp_path: Path) -> None:
        _, manifest = _fvessel_manifest(tmp_path / "restricted", videos=2)

        path = write_training_manifest(manifest, tmp_path / "artifacts")
        payload = json.loads(path.read_text())

        assert payload["manifest_checksum"] == manifest_checksum(manifest)
        assert {record["id"] for record in payload["records"]} == {
            record.id for record in manifest.records
        }
        assert "video_path" not in path.read_text()
        assert str(tmp_path / "restricted") not in path.read_text()

    def test_joint_split_preserves_each_component_in_training(self, tmp_path: Path) -> None:
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

        prepared = prepare_training_manifest(cfg, manifest)

        assert {record.dataset for record in prepared.records if record.split == "train"} == {
            "smd",
            "fvessel",
        }


class TestPretrainingRun:
    """run_pretraining lifecycle, checkpointing and W&B teardown."""

    def test_empty_data_root_fails_before_trainer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

    def test_too_short_records_fail_before_trainer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
        [
            ("", "without a last checkpoint path"),
            ("missing.ckpt", "last checkpoint does not exist"),
        ],
    )
    def test_run_pretraining_rejects_invalid_last_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        last_path: str,
        message: str,
    ) -> None:
        def _trainer_factory(
            _cfg: DictConfig, *, logger: object, callbacks: list[object]
        ) -> CheckpointPathTrainer:
            del logger
            return CheckpointPathTrainer(callbacks[0], last_path)

        monkeypatch.setattr("marineworld.train.pretrain.build_wandb_logger", lambda *_: False)
        monkeypatch.setattr("marineworld.train.pretrain.build_trainer", _trainer_factory)
        monkeypatch.setattr("marineworld.train.pretrain.build_module", lambda _: object())

        with pytest.raises(RuntimeError, match=message):
            run_pretraining(_smoke_config(tmp_path, max_steps=2))

    def test_run_pretraining_finishes_wandb_before_starting_another_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        active = False
        exit_codes: list[int] = []
        failures = iter((False, True))

        def _finish(*, exit_code: int) -> None:
            nonlocal active
            active = False
            exit_codes.append(exit_code)

        def _logger_factory(*_: object) -> object:
            nonlocal active
            assert not active, "the prior W&B run leaked into the next training call"
            active = True
            return SimpleNamespace(experiment=SimpleNamespace(finish=_finish))

        def _trainer_factory(
            _cfg: DictConfig, *, logger: object, callbacks: list[object]
        ) -> FakeTrainer:
            del logger
            return FakeTrainer(callbacks[0], fail=next(failures))

        monkeypatch.setattr("marineworld.train.pretrain.build_wandb_logger", _logger_factory)
        monkeypatch.setattr("marineworld.train.pretrain.build_trainer", _trainer_factory)
        monkeypatch.setattr("marineworld.train.pretrain.build_module", lambda _: object())

        assert run_pretraining(_smoke_config(tmp_path / "success", max_steps=2)).is_file()
        assert not active

        with pytest.raises(RuntimeError, match="forced trainer failure"):
            run_pretraining(_smoke_config(tmp_path / "failure", max_steps=2))

        assert not active
        assert exit_codes == [0, 1]

    def test_wandb_cleanup_does_not_mask_training_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        logger = SimpleNamespace(
            experiment=SimpleNamespace(
                finish=lambda **_: (_ for _ in ()).throw(RuntimeError("cleanup failure"))
            )
        )

        def _trainer_factory(
            _cfg: DictConfig, *, logger: object, callbacks: list[object]
        ) -> FakeTrainer:
            del logger
            return FakeTrainer(callbacks[0], fail=True)

        monkeypatch.setattr("marineworld.train.pretrain.build_wandb_logger", lambda *_: logger)
        monkeypatch.setattr("marineworld.train.pretrain.build_trainer", _trainer_factory)
        monkeypatch.setattr("marineworld.train.pretrain.build_module", lambda _: object())

        with pytest.raises(RuntimeError, match="forced trainer failure"):
            run_pretraining(_smoke_config(tmp_path, max_steps=2))

    def test_local_smoke_saves_best_and_last_then_resumes(self, tmp_path: Path) -> None:
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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


class TestCli:
    """Subprocess entrypoints and import-time hygiene."""

    def test_importing_experiment_does_not_mutate_environment(self) -> None:
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

    def test_importing_models_does_not_load_optional_training_dependencies(self) -> None:
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

    def test_empty_data_root_cli_exits_without_checkpoint(self, tmp_path: Path) -> None:
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
            timeout=180,
        )

        assert result.returncode != 0
        assert "manifest contains no records" in result.stderr
        assert not (output_dir / "checkpoints").exists()

    def test_offline_smoke_creates_local_wandb_run_without_network(
        self,
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
            timeout=180,
        )

        assert (output_dir / "checkpoints" / "last.ckpt").exists()
        assert any(wandb_dir.rglob("offline-run-*"))
        assert "W&B syncing is set to `offline`" in result.stderr


class TestPretrainingModule:
    """VideoMAE masking, loss, parameter updates and warmup config."""

    def test_tube_mask_has_exact_ratio(self) -> None:
        mask = tube_mask(
            2,
            sequence_length=80,
            mask_ratio=0.9,
            generator=torch.Generator().manual_seed(42),
        )

        assert mask.dtype == torch.bool
        assert mask.shape == (2, 80)
        assert mask.sum(dim=1).tolist() == [72, 72]

    def test_mask_for_step_is_reproducible_after_resume(self) -> None:
        first = VideoMAEPretrainingModule(_tiny_model_config(), seed=42)
        resumed = VideoMAEPretrainingModule(_tiny_model_config(), seed=42)

        step_three = first.make_mask(1, torch.device("cpu"), step=3)
        resumed_step_three = resumed.make_mask(1, torch.device("cpu"), step=3)
        expected_step_four = tube_mask(1, 8, 0.5, torch.Generator().manual_seed(46))

        assert torch.equal(step_three, resumed_step_three)
        assert torch.equal(first.make_mask(1, torch.device("cpu"), step=4), expected_step_four)

    def test_masks_are_unique_per_accumulated_microbatch(self) -> None:
        module = VideoMAEPretrainingModule(_tiny_model_config(), seed=42)

        first = module.make_mask(1, torch.device("cpu"), step=3, microbatch=0)
        second = module.make_mask(1, torch.device("cpu"), step=3, microbatch=1)

        assert not torch.equal(first, second)

    def test_training_logs_lr_and_grad_norm_under_the_pretrain_stage(
        self, tmp_path: Path, tiny_batch: dict[str, torch.Tensor]
    ) -> None:
        """The cosine schedule is only observable if the LR it produces is logged."""
        logged: dict[str, float] = {}
        module = VideoMAEPretrainingModule(_tiny_model_config(), warmup_epochs=0, max_epochs=1)
        module.log = lambda name, value, **_: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
        trainer = Trainer(
            max_steps=1,
            limit_train_batches=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="cpu",
            default_root_dir=str(tmp_path),
        )

        trainer.fit(module, train_dataloaders=DataLoader([tiny_batch], batch_size=None))

        assert logged["pretrain/lr"] > 0
        assert logged["pretrain/grad_norm"] > 0

    def test_encode_video_preserves_batch_token_and_hidden_layout(
        self,
        tiny_batch: dict[str, torch.Tensor],
    ) -> None:
        model = build_videomae(_tiny_model_config())

        embeddings = encode_video(model, tiny_batch["pixel_values"])

        assert embeddings.shape == (1, 8, 32)

    def test_training_step_rejects_non_finite_loss(
        self, monkeypatch: pytest.MonkeyPatch, tiny_batch: dict[str, torch.Tensor]
    ) -> None:
        module = VideoMAEPretrainingModule(_tiny_model_config())

        def _nan_loss(**_: torch.Tensor) -> object:
            return type("Output", (), {"loss": torch.tensor(float("nan"))})()

        monkeypatch.setattr(module.model, "forward", _nan_loss)

        with pytest.raises(FloatingPointError, match="non-finite training loss at batch 7"):
            module.training_step(tiny_batch, 7)

    def test_tiny_pretraining_step_updates_parameters(
        self, tiny_batch: dict[str, torch.Tensor]
    ) -> None:
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

    def test_pretraining_module_receives_configured_warmup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _compose_config("model=videomae_tiny")
        captured: dict[str, object] = {}

        def _module_factory(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("marineworld.train.pretrain.VideoMAEPretrainingModule", _module_factory)
        build_module(cfg)

        assert captured["warmup_epochs"] == cfg.train.warmup_epochs

    @pytest.mark.parametrize("warmup_epochs", [-1, 101])
    def test_pretraining_module_rejects_invalid_warmup(self, warmup_epochs: int) -> None:
        cfg = _compose_config("model=videomae_tiny", f"train.warmup_epochs={warmup_epochs}")

        with pytest.raises(ValueError, match="warmup_epochs"):
            build_module(cfg)

    def test_bounded_run_takes_its_step_budget_from_max_steps(self, tmp_path: Path) -> None:
        # A fixed max_steps must reach the scheduler as total_steps so it never
        # reads trainer.estimated_stepping_batches, which would iterate the whole
        # train dataloader (a full decode pass) before the first step.
        module = build_module(_smoke_config(tmp_path, max_steps=300))
        assert module.total_steps == 300
        assert module.configure_optimizers()["lr_scheduler"]["scheduler"] is not None

    def test_unbounded_run_defers_the_step_budget_to_the_trainer(self, tmp_path: Path) -> None:
        module = build_module(_smoke_config(tmp_path, max_steps=-1))
        assert module.total_steps is None


class TestScheduler:
    """Warmup-cosine LR schedule shape and step budget."""

    def test_scheduler_without_trainer_requires_an_explicit_step_budget(self) -> None:
        module = VideoMAEPretrainingModule(_tiny_model_config(), warmup_epochs=1, max_epochs=4)

        with pytest.raises(RuntimeError, match="attach a Trainer or pass total_steps"):
            module.configure_optimizers()

    def test_optimizer_uses_warmup_cosine_scheduler(self) -> None:
        module = VideoMAEPretrainingModule(
            _tiny_model_config(), lr=1e-3, warmup_epochs=1, max_epochs=4, total_steps=8
        )
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
            _tiny_model_config(), lr=1e-3, warmup_epochs=1, max_epochs=4, total_steps=8
        )
        resumed = resumed_module.configure_optimizers()["lr_scheduler"]["scheduler"]
        resumed.load_state_dict(scheduler.state_dict())
        assert resumed.get_last_lr() == scheduler.get_last_lr()

    def test_scheduler_uses_trainer_optimizer_step_estimate(self) -> None:
        module = VideoMAEPretrainingModule(
            _tiny_model_config(), lr=1e-3, warmup_epochs=1, max_epochs=4
        )
        module._trainer = SimpleNamespace(estimated_stepping_batches=4)
        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        for _ in range(4):
            scheduler.optimizer.step()
            scheduler.step()

        assert scheduler.get_last_lr() == [0.0]

"""Frozen encoder and reproducible probe behavior."""

from __future__ import annotations

import json
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from httpx import ConnectError as HttpxConnectError
from httpx import Request, Response
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RevisionNotFoundError,
)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from requests import ConnectionError as RequestsConnectionError
from safetensors.torch import save_file
from transformers import (
    VideoMAEImageProcessor,
    VJEPA2Config,
    VJEPA2Model,
    VJEPA2VideoProcessor,
)

from marineworld.data.clips import SpatialTransform, build_clip_index
from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.fvessel import FVesselAdapter
from marineworld.data.synthetic import SyntheticAdapter
from marineworld.eval.encoders import (
    MODEL_CONDITIONS,
    DINOv3FrozenEncoder,
    EncoderFeatures,
    FrozenVideoEncoder,
    ResourceUnavailableError,
    VideoMAEFrozenEncoder,
    VJEPAFrozenEncoder,
    load_frozen_encoder,
)
from marineworld.eval.probes import (
    LabelsUnavailableError,
    ProbeResult,
    aggregate_probe_results,
    evaluate_dense_probe_streaming,
    evaluate_probe,
    fit_dense_probe,
    fit_linear_probe,
    sample_labelled_records,
)
from marineworld.eval.run_probes import (
    ProbeRun,
    _checkpoint_identity,
    _count_bin,
    _dense_batch_factory,
    _dense_token_labels,
    _has_supervised_labels,
    _labelled_training_records,
    _log_selection_manifest,
    _prepare_probe_manifest,
    _probe_run_identity,
    _sample_label,
    _supervised_scalar_label,
    log_probe_results,
    run_evaluation,
    run_probe_condition,
)
from marineworld.train.experiment import build_run_identity


class FakeEncoder(torch.nn.Module):
    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures:
        pooled = pixel_values.mean(dim=(1, 3, 4))
        return EncoderFeatures(self.projection(pooled))


def _frames_96(_: Path) -> int:
    return 96


def _probe_config(tmp_path: Path, *overrides: str) -> Any:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=random",
                "eval=probes",
                "tracking.mode=disabled",
                "eval.label_fractions=[1.0]",
                "eval.seeds=[42]",
                f"data.root={tmp_path / 'data'}",
                f"output_dir={tmp_path / 'outputs'}",
                *overrides,
            ],
        )


def _unlabelled_smd_probe_config(tmp_path: Path) -> Any:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "data=smd",
                "model=random",
                "eval=probes",
                "tracking.mode=disabled",
                "eval.label_fractions=[1.0]",
                "eval.seeds=[42]",
                f"data.root={tmp_path / 'smd'}",
                f"output_dir={tmp_path / 'outputs'}",
            ],
        )


def _wrapped_os_error(cause: Exception, message: str = "transformers wrapper") -> OSError:
    try:
        raise cause
    except Exception as error:
        try:
            raise OSError(message) from error
        except OSError as wrapped:
            return wrapped


def _os_error_with_context(context: Exception, message: str) -> OSError:
    try:
        raise context
    except Exception:
        try:
            raise OSError(message)
        except OSError as wrapped:
            return wrapped


def _write_hf_config(checkpoint: Path, model_type: str = "videomae") -> None:
    (checkpoint / "config.json").write_text(json.dumps({"model_type": model_type}))


def _set_local_reference(cfg: Any, checkpoint: Path) -> None:
    OmegaConf.update(cfg, "model.condition", "generic_videomae")
    OmegaConf.update(cfg, "eval.checkpoint", str(checkpoint))
    OmegaConf.update(cfg, "model.revision", None, force_add=True)


def _hub_error(kind: str) -> Exception:
    if kind == "cache":
        return LocalEntryNotFoundError("cache miss")
    status = 403 if kind == "gated" else 503
    response = Response(status, request=Request("GET", "https://huggingface.co/org/reference"))
    if kind == "gated":
        return GatedRepoError("access denied", response=response)
    return HfHubHTTPError("service unavailable", response=response)


def _optional_reference_run() -> ProbeRun:
    return ProbeRun(
        condition="generic_videomae",
        checkpoint="org/reference@revision",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=1.0,
        seed=42,
        metric="macro_f1",
        model="videomae-base",
        device="cpu",
        optional=True,
    )


class TinyVideoBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.config = SimpleNamespace(
            num_frames=2,
            tubelet_size=1,
            image_size=4,
            patch_size=2,
        )
        self.pixel_values: torch.Tensor | None = None

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.pixel_values = pixel_values
        batch_size = pixel_values.shape[0]
        tokens = self.scale * torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
        return SimpleNamespace(last_hidden_state=tokens.expand(batch_size, -1, -1))


class BatchTrackingEncoder(torch.nn.Module):
    name = "batch-tracking"

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures:
        self.batch_sizes.append(len(pixel_values))
        features = pixel_values.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return EncoderFeatures(features)


class TinyVJEPABackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            frames_per_clip=2,
            tubelet_size=1,
            crop_size=4,
            patch_size=2,
        )
        self.input_shape: tuple[int, ...] | None = None

    def forward(
        self, pixel_values_videos: torch.Tensor, *, skip_predictor: bool
    ) -> SimpleNamespace:
        assert skip_predictor
        self.input_shape = tuple(pixel_values_videos.shape)
        batch_size = pixel_values_videos.shape[0]
        tokens = self.scale * torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
        return SimpleNamespace(last_hidden_state=tokens.expand(batch_size, -1, -1))


class DenseFakeEncoder(torch.nn.Module):
    name = "dense-fake"

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures:
        batch_size = len(pixel_values)
        spatial = torch.zeros(batch_size, 2, 2, 2, 2)
        spatial[..., 1] = 1.0
        spatial[:, :, 0, 0, 0] = self.scale.detach()
        return EncoderFeatures(spatial.mean(dim=(1, 2, 3)), spatial)


@dataclass(frozen=True)
class ProbeBatch:
    features: torch.Tensor
    labels: torch.Tensor


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def probe_batch() -> ProbeBatch:
    return ProbeBatch(
        features=torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        labels=torch.tensor([0, 0, 1, 1]),
    )


@pytest.fixture
def records(tmp_path: Path) -> tuple[VideoRecord, ...]:
    return tuple(
        VideoRecord(
            id=f"train-{index}",
            dataset="fixture",
            video_path=tmp_path / f"train-{index}.mp4",
            split="train",
            source="fixture",
            fps=10.0,
            num_frames=4,
        )
        for index in range(20)
    ) + (
        VideoRecord(
            id="test-0",
            dataset="fixture",
            video_path=tmp_path / "test-0.mp4",
            split="test",
            source="fixture",
            fps=10.0,
            num_frames=4,
        ),
    )


def test_encoder_remains_frozen_during_probe(
    fake_encoder: FakeEncoder, probe_batch: ProbeBatch
) -> None:
    before = {name: value.clone() for name, value in fake_encoder.state_dict().items()}

    fit_linear_probe(
        fake_encoder,
        probe_batch.features,
        probe_batch.labels,
        task="classification",
    )

    assert all(
        torch.equal(before[name], value) for name, value in fake_encoder.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in fake_encoder.parameters())
    assert all(parameter.grad is None for parameter in fake_encoder.parameters())


@pytest.mark.parametrize("fraction, expected", [(0.01, 1), (0.05, 1), (0.1, 2), (1.0, 20)])
def test_label_subsampling_is_video_level_and_nonempty(
    records: tuple[VideoRecord, ...], fraction: float, expected: int
) -> None:
    selected = sample_labelled_records(records, fraction=fraction, seed=42)

    assert len(selected) == expected
    assert selected == sample_labelled_records(records, fraction=fraction, seed=42)
    assert all(record_id.startswith("train-") for record_id in selected)


def test_probe_reports_random_and_pretrained_conditions() -> None:
    results = [
        ProbeResult(
            condition=condition,
            checkpoint=f"{condition}-checkpoint",
            manifest_checksum="fixture-checksum",
            dataset="fixture",
            task="classification",
            fraction=0.1,
            seed=seed,
            metric="macro_f1",
            value=0.5,
            status="COMPLETED",
        )
        for condition, seed in zip(
            ("random", "generic_videomae", "maritime_videomae"),
            (40, 41, 42),
            strict=True,
        )
    ]

    table = aggregate_probe_results(results)

    assert set(table["condition"]) == {
        "random",
        "generic_videomae",
        "maritime_videomae",
    }
    assert set(table["seed"]) == {40, 41, 42}


def test_videomae_adapter_returns_detached_global_and_spatial_features() -> None:
    encoder = VideoMAEFrozenEncoder(TinyVideoBackbone(), name="local-videomae")

    features = encoder.encode(torch.zeros(2, 2, 3, 4, 4, requires_grad=True))

    assert isinstance(encoder, FrozenVideoEncoder)
    assert features.global_features.shape == (2, 3)
    assert features.spatial_features is not None
    assert features.spatial_features.shape == (2, 2, 2, 2, 3)
    assert not features.global_features.requires_grad
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_pretrained_adapter_applies_processor_normalization() -> None:
    model = TinyVideoBackbone()
    processor = SimpleNamespace(image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5))
    encoder = VideoMAEFrozenEncoder(
        model,
        name="pretrained",
        processor=processor,
    )

    encoder.encode(torch.zeros(1, 2, 3, 4, 4))

    assert model.pixel_values is not None
    assert torch.equal(model.pixel_values, -torch.ones_like(model.pixel_values))


def test_videomae_processor_preserves_non_square_resize_crop_geometry() -> None:
    model = TinyVideoBackbone()
    processor = VideoMAEImageProcessor(
        size={"shortest_edge": 4},
        crop_size={"height": 4, "width": 4},
    )
    encoder = VideoMAEFrozenEncoder(model, name="pretrained", processor=processor)

    encoder.encode(torch.zeros(1, 2, 3, 4, 8))
    transform = encoder.spatial_transform((4, 8))

    assert model.pixel_values is not None
    assert model.pixel_values.shape == (1, 2, 3, 4, 4)
    assert transform == SpatialTransform(
        source_size=(4, 8),
        output_size=(4, 4),
        scale=(1.0, 1.0),
        offset=(0.0, -2.0),
    )


def test_vjepa_processor_preserves_non_square_resize_crop_geometry() -> None:
    model = TinyVJEPABackbone()
    processor = VJEPA2VideoProcessor(
        size={"shortest_edge": 4},
        crop_size={"height": 4, "width": 4},
    )
    encoder = VJEPAFrozenEncoder(model, name="pretrained", device="cpu", processor=processor)

    encoder.encode(torch.zeros(1, 2, 3, 4, 8))
    transform = encoder.spatial_transform((4, 8))

    assert model.input_shape == (1, 2, 3, 4, 4)
    assert transform.offset == (0.0, -2.0)


def test_dinov3_adapter_applies_imagenet_processor_normalization() -> None:
    model = TinyVideoBackbone()
    processor = SimpleNamespace(
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    )
    encoder = DINOv3FrozenEncoder(model, name="pretrained", device="cpu", processor=processor)

    encoder.encode(torch.zeros(1, 2, 3, 4, 4))

    assert model.pixel_values is not None
    expected = -torch.tensor(processor.image_mean) / torch.tensor(processor.image_std)
    assert torch.allclose(model.pixel_values[0, :, 0, 0], expected)


def test_random_encoder_loader_builds_local_model_without_checkpoint() -> None:
    encoder = load_frozen_encoder(
        "random",
        model_config={
            "name": "test-random",
            "image_size": 4,
            "patch_size": 2,
            "num_frames": 2,
            "tubelet_size": 1,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "decoder_hidden_size": 4,
            "decoder_num_hidden_layers": 1,
            "decoder_num_attention_heads": 2,
            "decoder_intermediate_size": 8,
            "mask_ratio": 0.5,
        },
    )

    features = encoder.encode(torch.zeros(1, 2, 3, 4, 4))

    assert encoder.name == "random"
    assert features.global_features.shape == (1, 8)
    assert MODEL_CONDITIONS == (
        "random",
        "generic_videomae",
        "maritime_videomae",
        "dinov3",
        "vjepa",
    )


def test_vjepa_adapter_uses_current_huggingface_shape_fields() -> None:
    model = TinyVJEPABackbone()
    encoder = VJEPAFrozenEncoder(model, name="local-vjepa", device="cpu")

    features = encoder.encode(torch.zeros(1, 2, 3, 4, 4))

    assert model.input_shape == (1, 2, 3, 4, 4)
    assert features.global_features.shape == (1, 3)
    assert features.spatial_features is not None
    assert features.spatial_features.shape == (1, 2, 2, 2, 3)


def test_vjepa_adapter_accepts_tiny_real_huggingface_model() -> None:
    model = VJEPA2Model(
        VJEPA2Config(
            crop_size=4,
            frames_per_clip=2,
            tubelet_size=1,
            patch_size=2,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            mlp_ratio=2,
            pred_hidden_size=4,
            pred_num_attention_heads=1,
            pred_num_hidden_layers=1,
        )
    )
    encoder = VJEPAFrozenEncoder(model, name="local-vjepa", device="cpu")

    features = encoder.encode(torch.zeros(1, 2, 3, 4, 4))

    assert features.global_features.shape == (1, 8)
    assert features.spatial_features is not None
    assert features.spatial_features.shape == (1, 2, 2, 2, 8)


@pytest.mark.parametrize("task", ["classification", "count"])
def test_linear_probe_uses_sklearn_and_reports_classification_metric(
    fake_encoder: FakeEncoder, probe_batch: ProbeBatch, task: str
) -> None:
    probe = fit_linear_probe(fake_encoder, probe_batch.features, probe_batch.labels, task=task)

    metric, value = evaluate_probe(
        probe,
        probe_batch.features,
        probe_batch.labels,
        task=task,
    )

    assert probe.__class__.__module__.startswith("sklearn.")
    assert metric == "macro_f1"
    assert value == pytest.approx(1.0)


def test_dense_probe_trains_only_a_single_linear_spatial_head(fake_encoder: FakeEncoder) -> None:
    before = {name: value.clone() for name, value in fake_encoder.state_dict().items()}
    spatial_features = torch.tensor([[[[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]]]])
    labels = torch.tensor([[[[0, 0], [1, 1]]]])

    head = fit_dense_probe(
        fake_encoder,
        spatial_features,
        labels,
        num_classes=2,
        epochs=20,
        lr=0.2,
        seed=42,
    )

    assert isinstance(head, torch.nn.Linear)
    predictions = head(spatial_features.reshape(-1, 2)).argmax(dim=1)
    assert torch.equal(predictions, labels.reshape(-1))
    assert all(
        torch.equal(before[name], value) for name, value in fake_encoder.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in fake_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in fake_encoder.parameters())


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("too large"),
        ResourceUnavailableError("CUDA is not available"),
    ],
)
def test_optional_resource_failure_produces_metadata_complete_skip(error: Exception) -> None:
    run = ProbeRun(
        condition="vjepa",
        checkpoint="facebook/vjepa2-vitg-fpc64-256",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=0.1,
        seed=42,
        metric="macro_f1",
        model="vjepa2-vitg",
        device="cpu",
        optional=True,
    )

    def unavailable() -> FrozenVideoEncoder:
        raise error

    results = run_probe_condition(run, loader=unavailable, evaluator=lambda _: [])

    assert results == (
        ProbeResult(
            condition="vjepa",
            checkpoint="facebook/vjepa2-vitg-fpc64-256",
            manifest_checksum="fixture-checksum",
            dataset="fixture",
            task="classification",
            fraction=0.1,
            seed=42,
            metric="macro_f1",
            value=None,
            status="SKIPPED_RESOURCE",
            model="vjepa2-vitg",
            device="cpu",
        ),
    )
    table = aggregate_probe_results(results)
    assert table.loc[0, "value"] is None
    assert table.loc[0, "status"] == "SKIPPED_RESOURCE"
    assert list(table.columns[:10]) == [
        "condition",
        "checkpoint",
        "manifest_checksum",
        "dataset",
        "task",
        "fraction",
        "seed",
        "metric",
        "value",
        "status",
    ]


def test_optional_resource_failure_during_evaluation_is_skipped(
    fake_encoder: FakeEncoder,
) -> None:
    run = ProbeRun(
        condition="dinov3",
        checkpoint="cached/reference",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=0.1,
        seed=42,
        metric="macro_f1",
        model="dinov3",
        device="cuda",
        optional=True,
    )

    def out_of_memory(_: FrozenVideoEncoder) -> list[ProbeResult]:
        raise torch.OutOfMemoryError("evaluation batch too large")

    results = run_probe_condition(
        run,
        loader=lambda: fake_encoder,
        evaluator=out_of_memory,
    )

    assert results[0].status == "SKIPPED_RESOURCE"
    assert results[0].value is None


def test_optional_reference_does_not_swallow_arbitrary_runtime_errors(
    fake_encoder: FakeEncoder,
) -> None:
    run = ProbeRun(
        condition="dinov3",
        checkpoint="cached/reference",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=0.1,
        seed=42,
        metric="macro_f1",
        model="dinov3",
        device="cpu",
        optional=True,
    )

    with pytest.raises(RuntimeError, match="model implementation bug"):
        run_probe_condition(
            run,
            loader=lambda: fake_encoder,
            evaluator=lambda _: (_ for _ in ()).throw(RuntimeError("model implementation bug")),
        )


def test_optional_reference_does_not_swallow_evaluation_os_errors(
    fake_encoder: FakeEncoder,
) -> None:
    run = ProbeRun(
        condition="dinov3",
        checkpoint="cached/reference",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=0.1,
        seed=42,
        metric="macro_f1",
        model="dinov3",
        device="cpu",
        optional=True,
    )

    with pytest.raises(PermissionError, match="annotation denied"):
        run_probe_condition(
            run,
            loader=lambda: fake_encoder,
            evaluator=lambda _: (_ for _ in ()).throw(PermissionError("annotation denied")),
        )


def test_pretrained_loading_is_cache_only_unless_download_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def local_model(_: str, **kwargs: Any) -> TinyVideoBackbone:
        calls.append(kwargs)
        return TinyVideoBackbone()

    monkeypatch.setattr("transformers.VideoMAEModel.from_pretrained", local_model)
    monkeypatch.setattr(
        "transformers.AutoVideoProcessor.from_pretrained",
        lambda *_, **__: SimpleNamespace(image_mean=(0.0, 0.0, 0.0), image_std=(1.0, 1.0, 1.0)),
    )

    cached = load_frozen_encoder("generic_videomae", checkpoint="cached/checkpoint")
    remote = load_frozen_encoder(
        "generic_videomae",
        checkpoint="remote/checkpoint",
        allow_download=True,
    )

    assert isinstance(cached, FrozenVideoEncoder)
    assert isinstance(remote, FrozenVideoEncoder)
    assert calls == [{"local_files_only": True}, {"local_files_only": False}]


def test_missing_cached_reference_is_a_narrow_resource_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_: Any, **__: Any) -> TinyVideoBackbone:
        raise OSError("not cached")

    monkeypatch.setattr("transformers.VideoMAEModel.from_pretrained", missing)

    with pytest.raises(ResourceUnavailableError, match="local cache"):
        load_frozen_encoder("generic_videomae", checkpoint="org/reference")


@pytest.mark.parametrize(
    "error",
    [
        LocalEntryNotFoundError("cache miss"),
        HttpxConnectError("connection refused"),
        RequestsConnectionError("connection refused"),
    ],
)
def test_known_hub_cache_and_network_failures_are_resource_unavailable(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*_: Any, **__: Any) -> TinyVideoBackbone:
        raise error

    monkeypatch.setattr("transformers.VideoMAEModel.from_pretrained", unavailable)

    with pytest.raises(ResourceUnavailableError, match="checkpoint is unavailable"):
        load_frozen_encoder(
            "generic_videomae",
            checkpoint="org/reference",
            allow_download=True,
        )


def test_invalid_hub_revision_is_not_a_resource_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Response(404, request=Request("GET", "https://huggingface.co/org/reference"))

    def invalid_revision(*_: Any, **__: Any) -> TinyVideoBackbone:
        raise RevisionNotFoundError("unknown revision", response=response)

    monkeypatch.setattr("transformers.VideoMAEModel.from_pretrained", invalid_revision)

    with pytest.raises(RevisionNotFoundError, match="unknown revision"):
        load_frozen_encoder(
            "generic_videomae",
            checkpoint="org/reference",
            allow_download=True,
        )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (
            "gated",
            "You are trying to access a gated repo.\n"
            "Make sure to have access to it at https://huggingface.co/org/reference.",
        ),
        (
            "cache",
            "We couldn't connect to 'https://huggingface.co' to load the files, and "
            "couldn't find them in the cached files.",
        ),
        (
            "transient",
            "There was a specific connection error when trying to load org/reference.",
        ),
    ],
)
def test_wrapped_hub_resource_failures_are_optional_skips(
    kind: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped = _wrapped_os_error(_hub_error(kind), message)

    def unavailable(*_: Any, **__: Any) -> TinyVideoBackbone:
        raise wrapped

    monkeypatch.setattr("transformers.VideoMAEModel.from_pretrained", unavailable)

    results = run_probe_condition(
        _optional_reference_run(),
        loader=lambda: load_frozen_encoder(
            "generic_videomae",
            checkpoint="org/reference",
            allow_download=True,
        ),
        evaluator=lambda _: (),
    )

    assert results[0].status == "SKIPPED_RESOURCE"


@pytest.mark.parametrize("message", ["invalid config", "failed to load invalid config"])
def test_hard_outer_os_error_is_not_downgraded_by_resource_context(
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped = _os_error_with_context(_hub_error("gated"), message)

    def invalid(*_: Any, **__: Any) -> TinyVideoBackbone:
        raise wrapped

    monkeypatch.setattr("transformers.VideoMAEModel.from_pretrained", invalid)

    with pytest.raises(OSError, match="invalid config") as raised:
        run_probe_condition(
            _optional_reference_run(),
            loader=lambda: load_frozen_encoder(
                "generic_videomae",
                checkpoint="org/reference",
                allow_download=True,
            ),
            evaluator=lambda _: (),
        )

    assert isinstance(raised.value.__context__, GatedRepoError)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("kind", ["revision", "not_found", "corrupt_config"])
def test_wrapped_invalid_hub_references_remain_hard_failures(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = Response(404, request=Request("GET", "https://huggingface.co/org/reference"))
    if kind == "revision":
        cause: Exception | None = RevisionNotFoundError("bad revision", response=response)
    elif kind == "not_found":
        cause = HfHubHTTPError("not found", response=response)
    else:
        cause = None
    wrapped = (
        _wrapped_os_error(cause, "invalid config")
        if cause is not None
        else OSError("invalid config")
    )

    def invalid(*_: Any, **__: Any) -> TinyVideoBackbone:
        raise wrapped

    monkeypatch.setattr("transformers.VideoMAEModel.from_pretrained", invalid)

    with pytest.raises(OSError, match="invalid config|transformers wrapper") as raised:
        run_probe_condition(
            _optional_reference_run(),
            loader=lambda: load_frozen_encoder(
                "generic_videomae",
                checkpoint="org/reference",
                allow_download=kind != "corrupt_config",
            ),
            evaluator=lambda _: (),
        )

    assert not isinstance(raised.value, ResourceUnavailableError)


def test_missing_local_checkpoint_fails_validation_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local checkpoint does not exist"):
        load_frozen_encoder(
            "maritime_videomae",
            checkpoint=str(tmp_path / "missing.ckpt"),
            model_config={},
        )


def test_corrupt_local_checkpoint_os_error_is_not_a_resource_skip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marineworld.train.module import VideoMAEPretrainingModule

    checkpoint = tmp_path / "corrupt.ckpt"
    checkpoint.write_bytes(b"not a checkpoint")

    def corrupt(*_: Any, **__: Any) -> None:
        raise OSError("checkpoint cannot be decoded")

    monkeypatch.setattr(VideoMAEPretrainingModule, "load_from_checkpoint", corrupt)

    with pytest.raises(OSError, match="cannot be decoded"):
        load_frozen_encoder(
            "maritime_videomae",
            checkpoint=str(checkpoint),
            model_config={},
        )


@pytest.mark.parametrize("device", ["xpu", "meta"])
def test_probe_rejects_unsupported_device_types(device: str) -> None:
    with pytest.raises(ValueError, match="unsupported probe device"):
        load_frozen_encoder("random", model_config={}, device=device)


def test_probe_rejects_unavailable_cuda_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ResourceUnavailableError, match="index is unavailable"):
        load_frozen_encoder("random", model_config={}, device="cuda:3")


def test_probe_config_composes_and_disabled_synthetic_run_is_local(tmp_path: Path) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=random",
                "eval=probes",
                "tracking.mode=disabled",
                f"data.root={tmp_path / 'data'}",
                f"output_dir={tmp_path / 'outputs'}",
            ],
        )

    results = run_evaluation(cfg)

    assert len(results) == 12
    assert {result.fraction for result in results} == {0.01, 0.05, 0.1, 1.0}
    assert {result.seed for result in results} == {40, 41, 42}
    assert {result.condition for result in results} == {"random"}
    assert {result.status for result in results} == {"COMPLETED"}
    assert not list(tmp_path.rglob("wandb-*"))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("eval.task", "tracking", "eval.task"),
        ("eval.seeds", None, "eval.seeds"),
        ("eval.seeds", [], "eval.seeds"),
        ("eval.seeds", [42, 42], "eval.seeds"),
        ("eval.seeds", [42.5], "eval.seeds"),
        ("eval.seeds", [-1], "eval.seeds"),
        ("eval.seeds", [2**32], "eval.seeds"),
        ("eval.label_fractions", None, "eval.label_fractions"),
        ("eval.label_fractions", [], "eval.label_fractions"),
        ("eval.label_fractions", [0.0], "eval.label_fractions"),
        ("eval.label_fractions", [1.1], "eval.label_fractions"),
        ("eval.label_fractions", [0.1, 0.1], "eval.label_fractions"),
        ("eval.batch_size", 0, "eval.batch_size"),
        ("eval.dense_epochs", 0, "eval.dense_epochs"),
        ("eval.dense_lr", 0.0, "eval.dense_lr"),
        ("seed", -1, "seed"),
        ("seed", 2**32, "seed"),
    ],
)
def test_probe_config_validation_precedes_logging_and_data_work(
    path: str,
    value: Any,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _probe_config(tmp_path)
    OmegaConf.update(cfg, path, value)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before config validation"),
    )
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_wandb_logger",
        lambda *_: pytest.fail("logger built before config validation"),
    )

    with pytest.raises(ValueError, match=message):
        run_evaluation(cfg)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            (
                ("model.condition", "bogus"),
                ("eval.checkpoint", None),
                ("model.revision", None),
            ),
            "model.condition",
        ),
        (
            (
                ("eval.device", "xpu"),
                ("model.revision", None),
                ("eval.checkpoint", None),
            ),
            "unsupported probe device",
        ),
        (
            (
                ("model.condition", "generic_videomae"),
                ("eval.checkpoint", ""),
                ("model.revision", None),
            ),
            "requires a non-null checkpoint",
        ),
        (
            (
                ("model.condition", "generic_videomae"),
                ("eval.checkpoint", "not a repo"),
                ("model.revision", "a" * 40),
            ),
            "Hub checkpoint",
        ),
        (
            (
                ("model.condition", "generic_videomae"),
                ("eval.checkpoint", "org/reference"),
                ("model.revision", "main"),
            ),
            "model.revision",
        ),
    ],
)
def test_encoder_preflight_rejects_invalid_requests_before_unlabelled_smd(
    updates: tuple[tuple[str, Any], ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    for path, value in updates:
        OmegaConf.update(cfg, path, value, force_add=True)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before encoder request validation"),
    )
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_wandb_logger",
        lambda *_: pytest.fail("logger built for invalid encoder request"),
    )

    with pytest.raises(ValueError, match=message):
        run_evaluation(cfg)


def test_encoder_preflight_rejects_missing_local_checkpoint_before_unlabelled_smd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    missing = tmp_path / "missing.ckpt"
    OmegaConf.update(cfg, "model.condition", "maritime_videomae")
    OmegaConf.update(cfg, "eval.checkpoint", str(missing))
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before checkpoint validation"),
    )

    with pytest.raises(ValueError, match="local checkpoint does not exist"):
        run_evaluation(cfg)


def test_encoder_preflight_rejects_incomplete_local_hf_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "incomplete-save-pretrained"
    checkpoint.mkdir()
    OmegaConf.update(cfg, "model.condition", "generic_videomae")
    OmegaConf.update(cfg, "eval.checkpoint", str(checkpoint))
    OmegaConf.update(cfg, "model.revision", None, force_add=True)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before local checkpoint validation"),
    )

    with pytest.raises(ValueError, match="config.json and model weights"):
        run_evaluation(cfg)


def test_encoder_preflight_rejects_invalid_config_json_before_unavailable_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "invalid-config"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{invalid")
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")
    _set_local_reference(cfg, checkpoint)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before config validation"),
    )

    with pytest.raises(ValueError, match="invalid config.json"):
        run_evaluation(cfg)


def test_encoder_preflight_rejects_incompatible_model_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "wrong-model-type"
    checkpoint.mkdir()
    _write_hf_config(checkpoint, model_type="vjepa2")
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")
    _set_local_reference(cfg, checkpoint)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before config validation"),
    )

    with pytest.raises(ValueError, match="model_type.*videomae"):
        run_evaluation(cfg)


@pytest.mark.parametrize(
    ("weight_name", "contents", "message"),
    [
        ("model.safetensors", b"", "invalid safetensors"),
        ("model.safetensors", b"truncated", "invalid safetensors"),
        ("pytorch_model.bin", b"not a zip archive", "invalid PyTorch weight archive"),
    ],
)
def test_encoder_preflight_rejects_corrupt_weight_container(
    weight_name: str,
    contents: bytes,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / f"corrupt-{weight_name.replace('.', '-')}"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    (checkpoint / weight_name).write_bytes(contents)
    _set_local_reference(cfg, checkpoint)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before weight validation"),
    )

    with pytest.raises(ValueError, match=message):
        run_evaluation(cfg)


def test_encoder_preflight_rejects_junk_pytorch_zip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "junk-pytorch-zip"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    with zipfile.ZipFile(checkpoint / "pytorch_model.bin", "w") as archive:
        archive.writestr("archive/data.pkl", b"x")
        archive.writestr("archive/data/0", b"x")
    _set_local_reference(cfg, checkpoint)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before weight validation"),
    )

    with pytest.raises(ValueError, match="invalid PyTorch weight archive"):
        run_evaluation(cfg)


def test_encoder_preflight_accepts_minimal_torch_state_dict(tmp_path: Path) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "valid-pytorch-state-dict"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    torch.save({"weight": torch.ones(1)}, checkpoint / "pytorch_model.bin")
    _set_local_reference(cfg, checkpoint)

    results = run_evaluation(cfg)

    assert results[0].status == "SKIPPED_UNAVAILABLE_LABELS"


def test_encoder_preflight_rejects_malformed_weight_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "malformed-index"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    (checkpoint / "model.safetensors.index.json").write_text("{invalid")
    _set_local_reference(cfg, checkpoint)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before index validation"),
    )

    with pytest.raises(ValueError, match="invalid Hugging Face weight index"):
        run_evaluation(cfg)


def test_encoder_preflight_accepts_minimal_valid_local_hf_fixture(tmp_path: Path) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "valid-save-pretrained"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")
    _set_local_reference(cfg, checkpoint)

    results = run_evaluation(cfg)

    assert results[0].status == "SKIPPED_UNAVAILABLE_LABELS"


def test_encoder_preflight_rejects_unrelated_safetensors_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "optimizer-only"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    (checkpoint / "optimizer.safetensors").write_bytes(b"optimizer")
    OmegaConf.update(cfg, "model.condition", "generic_videomae")
    OmegaConf.update(cfg, "eval.checkpoint", str(checkpoint))
    OmegaConf.update(cfg, "model.revision", None, force_add=True)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before local checkpoint validation"),
    )

    with pytest.raises(ValueError, match="recognized model weights"):
        run_evaluation(cfg)


def test_encoder_preflight_rejects_sharded_index_with_missing_weight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "missing-shard"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}})
    )
    OmegaConf.update(cfg, "model.condition", "generic_videomae")
    OmegaConf.update(cfg, "eval.checkpoint", str(checkpoint))
    OmegaConf.update(cfg, "model.revision", None, force_add=True)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before local checkpoint validation"),
    )

    with pytest.raises(ValueError, match="missing referenced weight shard"):
        run_evaluation(cfg)


@pytest.mark.parametrize("weight_name", ["tf_model.h5", "flax_model.msgpack"])
def test_encoder_preflight_rejects_non_pytorch_weight_formats(
    weight_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "unsupported-format"
    checkpoint.mkdir()
    _write_hf_config(checkpoint)
    (checkpoint / weight_name).write_bytes(b"weights")
    OmegaConf.update(cfg, "model.condition", "generic_videomae")
    OmegaConf.update(cfg, "eval.checkpoint", str(checkpoint))
    OmegaConf.update(cfg, "model.revision", None, force_add=True)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before local checkpoint validation"),
    )

    with pytest.raises(ValueError, match="recognized model weights"):
        run_evaluation(cfg)


def test_encoder_preflight_accepts_symlinked_sharded_hf_snapshot(tmp_path: Path) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    checkpoint = tmp_path / "snapshots" / "revision"
    blob = tmp_path / "blobs" / "shard"
    checkpoint.mkdir(parents=True)
    blob.parent.mkdir()
    save_file({"weight": torch.ones(1)}, blob)
    _write_hf_config(checkpoint)
    shard_name = "model-00001-of-00001.safetensors"
    (checkpoint / shard_name).symlink_to(blob)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard_name}})
    )
    OmegaConf.update(cfg, "model.condition", "generic_videomae")
    OmegaConf.update(cfg, "eval.checkpoint", str(checkpoint))
    OmegaConf.update(cfg, "model.revision", None, force_add=True)

    results = run_evaluation(cfg)

    assert results[0].status == "SKIPPED_UNAVAILABLE_LABELS"


def test_encoder_preflight_rejects_unavailable_device_before_unlabelled_smd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)
    OmegaConf.update(cfg, "eval.device", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_adapter",
        lambda *_: pytest.fail("adapter built before device validation"),
    )

    with pytest.raises(ResourceUnavailableError, match="CUDA is not available"):
        run_evaluation(cfg)


def test_runner_identity_covers_batch_size_and_complete_model_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed_ids: list[str] = []

    def capture_identity(_: Any, identity: Any) -> bool:
        observed_ids.append(identity.run_id)
        return False

    def unavailable(*_: Any, **__: Any) -> FrozenVideoEncoder:
        raise ResourceUnavailableError("fixture")

    monkeypatch.setattr("marineworld.eval.run_probes.build_wandb_logger", capture_identity)
    monkeypatch.setattr("marineworld.eval.run_probes.load_frozen_encoder", unavailable)
    configs = (
        _probe_config(tmp_path, "eval.optional=true"),
        _probe_config(tmp_path, "eval.optional=true"),
        _probe_config(tmp_path, "eval.optional=true", "eval.batch_size=4"),
        _probe_config(tmp_path, "eval.optional=true", "model.hidden_size=64"),
    )

    results = [run_evaluation(cfg) for cfg in configs]

    assert observed_ids[0] == observed_ids[1]
    assert observed_ids[0] != observed_ids[2]
    assert observed_ids[0] != observed_ids[3]
    assert all(result[0].status == "SKIPPED_RESOURCE" for result in results)
    assert all(result[0].label_subset_checksum for result in results)
    assert all(result[0].labelled_record_ids for result in results)
    assert all(
        (Path(cfg.output_dir) / "probe_selection_manifest.json").is_file() for cfg in configs
    )


def test_selection_manifest_logs_as_wandb_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    added: list[tuple[str, str]] = []
    logged: list[Any] = []

    class Artifact:
        def __init__(self, *, name: str, type: str) -> None:
            assert name == "probe-selection-run-123"
            assert type == "dataset"

        def add_file(self, path: str, *, name: str) -> None:
            added.append((path, name))

    path = tmp_path / "probe_selection_manifest.json"
    path.write_text("{}")
    monkeypatch.setattr(
        "marineworld.eval.run_probes._wandb_artifact_type",
        lambda: Artifact,
    )
    logger = SimpleNamespace(experiment=SimpleNamespace(log_artifact=logged.append))

    _log_selection_manifest(path, "run-123", logger)

    assert added == [(str(path), "probe_selection_manifest.json")]
    assert len(logged) == 1


def test_logger_finishes_when_selection_artifact_logging_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    finished: list[bool] = []
    logger = SimpleNamespace(experiment=SimpleNamespace(finish=lambda: finished.append(True)))
    monkeypatch.setattr("marineworld.eval.run_probes.build_wandb_logger", lambda *_: logger)
    monkeypatch.setattr(
        "marineworld.eval.run_probes._log_selection_manifest",
        lambda *_: (_ for _ in ()).throw(RuntimeError("artifact upload failed")),
    )

    with pytest.raises(RuntimeError, match="artifact upload failed"):
        run_evaluation(_probe_config(tmp_path))

    assert finished == [True]


def test_probe_results_log_as_one_table(monkeypatch: pytest.MonkeyPatch) -> None:
    result = ProbeResult(
        condition="random",
        checkpoint="random-init",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=1.0,
        seed=42,
        metric="macro_f1",
        value=1.0,
        status="COMPLETED",
    )
    logged: list[dict[str, object]] = []
    logger = SimpleNamespace(experiment=SimpleNamespace(log=logged.append))
    table = object()
    monkeypatch.setattr("marineworld.eval.run_probes._wandb_table", lambda _: table)

    log_probe_results((result,), logger)

    assert logged == [{"probe/results": table}]


def test_mixed_result_table_preserves_skips_as_nonnumeric_none() -> None:
    completed = ProbeResult(
        condition="random",
        checkpoint="random-init",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=1.0,
        seed=42,
        metric="macro_f1",
        value=1.0,
        status="COMPLETED",
    )
    skipped = ProbeResult(
        condition="vjepa",
        checkpoint="cached/reference",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=1.0,
        seed=42,
        metric="macro_f1",
        value=None,
        status="SKIPPED_RESOURCE",
        model="vjepa",
        device="cpu",
    )

    table = aggregate_probe_results((completed, skipped))

    assert table.loc[1, "value"] is None


def test_feature_extraction_honors_configured_batch_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=random",
                "eval=probes",
                "tracking.mode=disabled",
                f"data.root={tmp_path / 'data'}",
                "eval.batch_size=2",
            ],
        )
    encoder = BatchTrackingEncoder()
    monkeypatch.setattr(
        "marineworld.eval.run_probes.load_frozen_encoder",
        lambda *_, **__: encoder,
    )

    run_evaluation(cfg)

    assert encoder.batch_sizes
    assert max(encoder.batch_sizes) == 2


def test_classification_ignores_frames_without_class_annotations() -> None:
    sample = {
        "dataset": "fixture",
        "frame_indices": torch.tensor([0]),
        "targets": (
            FrameTargets(
                frame_index=0,
                boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                class_ids=np.empty(0, dtype=np.int64),
            ),
        ),
    }

    assert _sample_label(sample, "classification") is None


def test_count_probe_excludes_unlabelled_real_clips() -> None:
    sample = {
        "dataset": "fixture",
        "frame_indices": torch.tensor([0]),
        "targets": (),
        "is_labelled": False,
    }

    assert _supervised_scalar_label(sample, "count") is None
    assert _supervised_scalar_label(sample | {"is_labelled": True}, "count") == 0


@pytest.mark.parametrize(("count", "expected"), [(0, 0), (1, 1), (2, 2), (3, 3), (12, 3)])
def test_count_probe_uses_documented_three_plus_bin(count: int, expected: int) -> None:
    assert _count_bin(count) == expected


def test_probe_manifest_splits_real_train_only_records_by_video(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"video-{index}.mp4").touch()
        (tmp_path / f"video-{index}.txt").touch()
    records = tuple(
        VideoRecord(
            id=f"video-{index}",
            dataset="fixture",
            video_path=tmp_path / f"video-{index}.mp4",
            split="train",
            source="fixture",
            fps=10.0,
            num_frames=16,
            annotation_path=tmp_path / f"video-{index}.txt",
        )
        for index in range(5)
    )
    manifest = DatasetManifest("fixture", "v1", "test", records)
    cfg = SimpleNamespace(
        seed=42,
        data={"split": SimpleNamespace(val_frac=0.2, test_frac=0.2)},
    )

    prepared = _prepare_probe_manifest(cfg, manifest)
    repeated = _prepare_probe_manifest(cfg, manifest)

    assert prepared == repeated
    assert {record.split for record in prepared.records} >= {"train", "val"}
    assert {record.id for record in prepared.records if record.split == "train"}.isdisjoint(
        record.id for record in prepared.records if record.split == "val"
    )


def test_empty_dense_validation_is_never_reported_as_numeric() -> None:
    head = torch.nn.Linear(2, 2)

    with pytest.raises(LabelsUnavailableError, match="validation"):
        evaluate_dense_probe_streaming(head, lambda: iter(()))


def test_dense_batches_exclude_unlabelled_real_clips() -> None:
    sample = {
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "is_labelled": False,
        "dataset": "fixture",
        "frame_indices": torch.tensor([0, 1]),
        "targets": (),
        "spatial_transform": SpatialTransform((4, 4), (4, 4), (1.0, 1.0)),
    }
    dataset = SimpleNamespace(
        clips=(SimpleNamespace(record_id="unlabelled"),),
        __len__=lambda self: 1,
        __getitem__=lambda self, index: sample,
    )

    class DatasetStub:
        clips = dataset.clips

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, Any]:
            return sample

    assert list(_dense_batch_factory(DatasetStub(), DenseFakeEncoder(), batch_size=1)()) == []


def test_result_table_preserves_label_subset_membership() -> None:
    result = ProbeResult(
        condition="random",
        checkpoint="random-init",
        manifest_checksum="fixture-checksum",
        dataset="fixture",
        task="classification",
        fraction=0.1,
        seed=42,
        metric="macro_f1",
        value=0.5,
        status="COMPLETED",
        label_subset_checksum="abc123",
        labelled_record_ids='["video-1"]',
    )

    table = aggregate_probe_results((result,))

    assert table.loc[0, "label_subset_checksum"] == "abc123"
    assert table.loc[0, "labelled_record_ids"] == '["video-1"]'


def test_wandb_mode_environment_disables_probe_logger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=random",
                "eval=probes",
                f"data.root={tmp_path / 'data'}",
            ],
        )
    observed_modes: list[str] = []

    def disabled_logger(config: Any, _: Any) -> bool:
        observed_modes.append(str(config.tracking.mode))
        return False

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr("marineworld.eval.run_probes.build_wandb_logger", disabled_logger)

    run_evaluation(cfg)

    assert observed_modes == ["disabled"]


def test_single_class_subset_is_reported_without_numeric_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=random",
                "eval=probes",
                "tracking.mode=disabled",
                f"data.root={tmp_path / 'data'}",
            ],
        )
    monkeypatch.setattr("marineworld.eval.run_probes._sample_label", lambda *_: 0)

    results = run_evaluation(cfg)

    assert {result.status for result in results} == {"SKIPPED_DEGENERATE_LABELS"}
    assert all(result.value is None for result in results)
    assert all(result.label_subset_checksum for result in results)
    assert all(result.labelled_record_ids for result in results)


@pytest.mark.parametrize(
    ("model", "condition", "image_size", "num_frames"),
    [
        ("generic_videomae", "generic_videomae", 224, 16),
        ("maritime_videomae", "maritime_videomae", 224, 16),
        ("dinov3", "dinov3", 224, 16),
        ("vjepa", "vjepa", 256, 64),
    ],
)
def test_reference_model_configs_compose_with_declared_input_contract(
    model: str, condition: str, image_size: int, num_frames: int
) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[f"model={model}", "eval=probes"],
        )

    assert cfg.model.condition == condition
    assert cfg.model.image_size == image_size
    assert cfg.model.num_frames == num_frames
    if model == "vjepa":
        assert cfg.data.adapter.num_frames is None
    if model != "maritime_videomae":
        assert len(cfg.model.revision) == 40


def test_vjepa_real_data_config_builds_a_nonempty_clip_index(tmp_path: Path) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["data=fvessel", "model=vjepa"])
    sample = tmp_path / "clip"
    (sample / "gt").mkdir(parents=True)
    (sample / "clip.mp4").touch()
    (sample / "gt" / "gt.txt").touch()
    adapter = FVesselAdapter(
        fps=30.0,
        num_frames=cfg.data.adapter.num_frames,
        frame_count_probe=_frames_96,
    )
    manifest = adapter.build_manifest(tmp_path)

    clips = build_clip_index(
        manifest,
        split="train",
        frames=int(cfg.model.num_frames),
        stride=1,
        seed=42,
    )

    assert len(clips) == 1
    assert len(clips[0].frame_indices) == 64


def test_random_vit_small_control_matches_maritime_architecture() -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        random_cfg = compose(
            config_name="config", overrides=["model=random_vit_small", "eval=probes"]
        )
        maritime_cfg = compose(
            config_name="config", overrides=["model=maritime_videomae", "eval=probes"]
        )

    architecture = (
        "image_size",
        "patch_size",
        "num_frames",
        "tubelet_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
    )
    assert random_cfg.model.condition == "random"
    assert all(random_cfg.model[key] == maritime_cfg.model[key] for key in architecture)


def test_tiny_and_matched_random_controls_have_distinct_run_ids(tmp_path: Path) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        tiny = compose(
            config_name="config", overrides=["data=synthetic", "model=random", "eval=probes"]
        )
        matched = compose(
            config_name="config",
            overrides=["data=synthetic", "model=random_vit_small", "eval=probes"],
        )
    manifest = SyntheticAdapter().build_manifest(tmp_path)

    tiny_id = _probe_run_identity(tiny, manifest, "random-init", None)
    matched_id = _probe_run_identity(matched, manifest, "random-init", None)

    assert tiny_id.run_id != matched_id.run_id


def test_probe_identity_changes_with_logical_matrix_and_checkpoint_content(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"first")
    first_checkpoint = _checkpoint_identity(str(checkpoint))
    checkpoint.write_bytes(b"second")
    second_checkpoint = _checkpoint_identity(str(checkpoint))
    dimensions = {
        "task": "classification",
        "fractions": [0.1],
        "seeds": [42],
        "checkpoint": first_checkpoint,
    }
    variants = (
        dimensions,
        dimensions | {"task": "dense"},
        dimensions | {"fractions": [1.0]},
        dimensions | {"seeds": [41]},
        dimensions | {"checkpoint": second_checkpoint},
    )
    run_ids = {
        build_run_identity(
            "random",
            ("manifest",),
            42,
            None,
            logical_dimensions=variant,
        ).run_id
        for variant in variants
    }

    assert first_checkpoint != second_checkpoint
    assert len(run_ids) == len(variants)


def test_local_huggingface_directory_identity_hashes_recursive_contents(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "save_pretrained"
    nested = checkpoint / "processor"
    nested.mkdir(parents=True)
    config = checkpoint / "config.json"
    weights = checkpoint / "model.safetensors"
    processor = nested / "preprocessor_config.json"
    config.write_text('{"hidden_size": 32}')
    weights.write_bytes(b"weights-v1")
    processor.write_text('{"size": 224}')

    initial = _checkpoint_identity(str(checkpoint))

    assert initial == _checkpoint_identity(str(checkpoint))
    config.write_text('{"hidden_size": 64}')
    config_changed = _checkpoint_identity(str(checkpoint))
    weights.write_bytes(b"weights-v2")
    weights_changed = _checkpoint_identity(str(checkpoint))

    assert initial.startswith("sha256-tree:")
    assert len({initial, config_changed, weights_changed}) == 3


def test_local_huggingface_snapshot_identity_hashes_symlink_target_contents(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshots" / "revision"
    blob = tmp_path / "blobs" / "weight"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    blob.write_bytes(b"weights-v1")
    (snapshot / "model.safetensors").symlink_to(blob)

    initial = _checkpoint_identity(str(snapshot))
    blob.write_bytes(b"weights-v2")

    assert _checkpoint_identity(str(snapshot)) != initial


def test_unlabelled_smd_manifest_is_explicitly_unavailable(tmp_path: Path) -> None:
    records = tuple(
        VideoRecord(
            id=split,
            dataset="smd",
            video_path=tmp_path / f"{split}.mp4",
            split=split,
            source="onshore",
            fps=30.0,
            num_frames=64,
        )
        for split in ("train", "val")
    )

    assert not _has_supervised_labels(DatasetManifest("smd", "v1", "research", records))


def test_unlabelled_smd_runner_records_known_empty_selection_and_artifact(
    tmp_path: Path,
) -> None:
    cfg = _unlabelled_smd_probe_config(tmp_path)

    results = run_evaluation(cfg)

    assert len(results) == 1
    assert results[0].status == "SKIPPED_UNAVAILABLE_LABELS"
    assert results[0].labelled_record_ids == "[]"
    assert results[0].label_subset_checksum is not None
    payload = json.loads((Path(cfg.output_dir) / "probe_selection_manifest.json").read_text())
    subset = payload["label_subsets"][0]
    assert subset["record_ids"] == []
    assert subset["checksum"] == results[0].label_subset_checksum


def test_label_fraction_population_excludes_videos_too_short_for_encoder(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "labels.txt"
    annotation.touch()
    records = tuple(
        VideoRecord(
            id=record_id,
            dataset="fixture",
            video_path=tmp_path / f"{record_id}.mp4",
            split="train",
            source="fixture",
            fps=30.0,
            num_frames=num_frames,
            annotation_path=annotation,
        )
        for record_id, num_frames in (("short", 32), ("eligible", 64))
    )

    eligible = _labelled_training_records(
        DatasetManifest("fixture", "v1", "test", records), min_frames=64
    )

    assert [record.id for record in eligible] == ["eligible"]


def test_transformers_floor_supports_vjepa2_and_auto_video_processor() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependency = next(
        value
        for value in project["project"]["optional-dependencies"]["train"]
        if value.startswith("transformers")
    )

    assert dependency.startswith("transformers>=5.0")


def test_missing_maritime_checkpoint_fails_before_logger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=maritime_videomae",
                "eval=probes",
                f"data.root={tmp_path / 'data'}",
            ],
        )
    monkeypatch.setattr(
        "marineworld.eval.run_probes.build_wandb_logger",
        lambda *_: pytest.fail("logger constructed before checkpoint validation"),
    )

    with pytest.raises(ValueError, match="requires a non-null checkpoint"):
        run_evaluation(cfg)


def test_dense_labels_map_source_boxes_to_spatiotemporal_tokens() -> None:
    sample = {
        "dataset": "fixture",
        "frame_indices": torch.tensor([0, 1]),
        "targets": (
            FrameTargets(
                frame_index=0,
                boxes_xyxy=np.array([[4.0, 2.0, 20.0, 10.0]], dtype=np.float32),
                class_ids=np.array([3], dtype=np.int64),
            ),
        ),
        "spatial_transform": SpatialTransform(
            source_size=(20, 40),
            output_size=(10, 10),
            scale=(0.5, 0.25),
        ),
    }

    labels = _dense_token_labels(sample, spatial_shape=(2, 2, 2))

    assert labels.tolist() == [
        [[1, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]


def test_dense_probe_runs_through_result_matrix_without_encoder_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=random",
                "eval=probes",
                "eval.task=dense",
                "eval.batch_size=1",
                "tracking.mode=disabled",
                f"data.root={tmp_path / 'data'}",
            ],
        )
    encoder = DenseFakeEncoder()
    before = {name: value.clone() for name, value in encoder.state_dict().items()}
    torch_cat = torch.cat

    def reject_dense_concatenation(tensors: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        values = tuple(tensors)
        if values and values[0].ndim == 5:
            raise AssertionError("dense feature batches must not be accumulated")
        return torch_cat(values, *args, **kwargs)

    monkeypatch.setattr(
        "marineworld.eval.run_probes.load_frozen_encoder",
        lambda *_, **__: encoder,
    )
    monkeypatch.setattr("torch.cat", reject_dense_concatenation)

    results = run_evaluation(cfg)

    assert len(results) == 12
    assert {result.task for result in results} == {"dense"}
    assert {result.status for result in results} == {"COMPLETED"}
    assert all(result.value is not None for result in results)
    assert all(torch.equal(before[name], value) for name, value in encoder.state_dict().items())
    assert all(parameter.grad is None for parameter in encoder.parameters())

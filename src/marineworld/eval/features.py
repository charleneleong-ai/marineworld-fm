"""Frozen-encoder feature extraction for representation probes."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from marineworld.data.adapters import DatasetAdapter
from marineworld.data.clips import (
    AutoVideoDecoder,
    MaritimeClipDataset,
    SyntheticVideoDecoder,
)
from marineworld.data.contracts import DatasetManifest
from marineworld.eval.encoders import EncoderFeatures, FrozenVideoEncoder
from marineworld.eval.probes import (
    LabelsUnavailableError,
    dense_token_labels,
    supervised_scalar_label,
)

__all__ = [
    "FeatureSet",
    "build_probe_dataset",
    "dense_batch_factory",
    "encode_samples",
    "extract_features",
    "sample_with_encoder_geometry",
]


@dataclass(frozen=True)
class FeatureSet:
    features: torch.Tensor
    labels: torch.Tensor
    record_ids: tuple[str, ...]
    sources: tuple[str, ...]
    datasets: tuple[str, ...]


def extract_features(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
    *,
    split: str,
) -> FeatureSet:
    dataset = build_probe_dataset(cfg, manifest, adapter, encoder, split=split)
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    record_ids: list[str] = []
    sources: list[str] = []
    datasets: list[str] = []
    batch_size = int(cfg.eval.batch_size)
    if batch_size <= 0:
        raise ValueError("probe batch_size must be positive")
    for start in range(0, len(dataset), batch_size):
        decoded = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        samples = [sample for sample in decoded if sample is not None]
        labelled = [
            (sample, label)
            for sample in samples
            if (label := supervised_scalar_label(sample, str(cfg.eval.task))) is not None
        ]
        if not labelled:
            continue
        samples, scalar_labels = zip(*labelled, strict=True)
        if not samples:
            continue
        encoded = encode_samples(encoder, samples)
        feature_batches.append(encoded.global_features.cpu())
        label_batches.append(torch.tensor(scalar_labels, dtype=torch.long))
        record_ids.extend(str(sample["record_id"]) for sample in samples)
        sources.extend(str(sample["source"]) for sample in samples)
        datasets.extend(str(sample["dataset"]) for sample in samples)
    if not feature_batches:
        raise LabelsUnavailableError(f"{split} split produced no labelled probe clips")
    return FeatureSet(
        torch.cat(feature_batches),
        torch.cat(label_batches),
        tuple(record_ids),
        tuple(sources),
        tuple(datasets),
    )


def build_probe_dataset(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
    *,
    split: str,
) -> MaritimeClipDataset:
    size = int(cfg.model.image_size)
    processor_owns_geometry = getattr(encoder, "processor", None) is not None
    decoder = (
        SyntheticVideoDecoder(size, size) if manifest.name == "synthetic" else AutoVideoDecoder()
    )
    return MaritimeClipDataset(
        manifest,
        decoder,
        split=split,
        frames=int(cfg.model.num_frames),
        stride=1,
        image_size=None if processor_owns_geometry else size,
        seed=int(cfg.seed),
        target_loader=adapter.load_targets,
        normalization_mean=(
            tuple(cfg.data.transforms.normalization.mean) if not processor_owns_geometry else None
        ),
        normalization_std=(
            tuple(cfg.data.transforms.normalization.std) if not processor_owns_geometry else None
        ),
    )


def dense_batch_factory(
    dataset: MaritimeClipDataset,
    encoder: FrozenVideoEncoder,
    *,
    batch_size: int,
    record_ids: set[str] | None = None,
) -> Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    if batch_size <= 0:
        raise ValueError("probe batch_size must be positive")

    def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(dataset), batch_size):
            samples = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
                if record_ids is None or dataset.clips[index].record_id in record_ids
            ]
            samples = [sample for sample in samples if bool(sample["is_labelled"])]
            if not samples:
                continue
            with torch.inference_mode():
                spatial = encode_samples(encoder, samples).spatial_features
            if spatial is None:
                raise ValueError("dense probe requires encoder spatial features")
            spatial = spatial.detach().cpu().clone()
            labels = torch.stack(
                [
                    dense_token_labels(
                        sample_with_encoder_geometry(sample, encoder),
                        spatial_shape=spatial.shape[1:4],
                    )
                    for sample in samples
                ]
            )
            yield spatial, labels

    return batches


def encode_samples(
    encoder: FrozenVideoEncoder,
    samples: Sequence[dict[str, Any]],
) -> EncoderFeatures:
    if getattr(encoder, "processor", None) is None:
        return encoder.encode(torch.stack([sample["pixel_values"] for sample in samples]))
    encoded = [encoder.encode(sample["pixel_values"].unsqueeze(0)) for sample in samples]
    global_features = torch.cat([features.global_features for features in encoded])
    spatial_batches = [features.spatial_features for features in encoded]
    spatial = None
    if all(features is not None for features in spatial_batches):
        spatial = torch.cat([features for features in spatial_batches if features is not None])
    return EncoderFeatures(global_features, spatial)


def sample_with_encoder_geometry(
    sample: dict[str, Any], encoder: FrozenVideoEncoder
) -> dict[str, Any]:
    transform = sample["spatial_transform"]
    spatial_transform = getattr(encoder, "spatial_transform", None)
    if not callable(spatial_transform) or getattr(encoder, "processor", None) is None:
        return sample
    return {**sample, "spatial_transform": spatial_transform(transform.source_size)}

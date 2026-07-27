"""Leakage-safe clip indexing and decoders for canonical video tensors."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.manifest import (
    manifest_checksum,
    validate_frame_targets,
    validate_manifest,
)
from marineworld.data.video import decode_video

Split = Literal["train", "val", "test"]
TargetLoader = Callable[[VideoRecord], tuple[FrameTargets, ...]]


class VideoDecoder(Protocol):
    """Decode selected source frames to a ``[T, C, H, W]`` tensor."""

    def decode(self, record: VideoRecord, frame_indices: Sequence[int]) -> torch.Tensor: ...


@dataclass(frozen=True)
class ClipIndex:
    """A deterministic set of frame positions from one video record."""

    record_id: str
    start: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class SampleDraw:
    """One clip draw with a stateless, replayable augmentation seed."""

    index: int
    augmentation_seed: int


@dataclass(frozen=True)
class SpatialTransform:
    """Map source-pixel coordinates into a decoded sample's output canvas."""

    source_size: tuple[int, int]
    output_size: tuple[int, int]
    scale: tuple[float, float]
    offset: tuple[float, float] = (0.0, 0.0)

    def apply_boxes_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Apply resize/crop geometry to source ``[x1,y1,x2,y2]`` boxes."""
        transformed = np.asarray(boxes, dtype=np.float32).copy()
        if transformed.ndim != 2 or transformed.shape[1] != 4:
            raise ValueError("boxes must be shaped [N, 4]")
        scale_y, scale_x = self.scale
        offset_y, offset_x = self.offset
        transformed[:, (0, 2)] = transformed[:, (0, 2)] * scale_x + offset_x
        transformed[:, (1, 3)] = transformed[:, (1, 3)] * scale_y + offset_y
        return transformed


class DecordVideoDecoder:
    """Decode RGB video frames with the optional Decord training dependency."""

    def decode(self, record: VideoRecord, frame_indices: Sequence[int]) -> torch.Tensor:
        try:
            # Decord is optional and expensive to import, so core data contracts stay usable.
            from decord import VideoReader
        except ImportError as error:
            raise RuntimeError(
                "DecordVideoDecoder requires optional dependency 'decord'; run on a supported "
                "platform or inject another VideoDecoder"
            ) from error

        try:
            frames = VideoReader(str(record.video_path)).get_batch(list(frame_indices)).asnumpy()
        except Exception as error:
            raise RuntimeError(f"could not decode frames from {record.video_path}") from error
        return torch.from_numpy(frames).permute(0, 3, 1, 2)


@dataclass(frozen=True)
class ClipDecodeFailure:
    """One clip that could not be decoded, retained so losses can be reported."""

    record_id: str
    frame_indices: tuple[int, ...]
    reason: str


def drop_undecodable(samples: Sequence[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Filter clips that failed to decode, refusing a batch with nothing left."""
    kept = [sample for sample in samples if sample is not None]
    if not kept:
        raise ValueError("every clip in the batch failed to decode")
    return kept


class AutoVideoDecoder:
    """Decode frames with the first available supported backend."""

    def decode(self, record: VideoRecord, frame_indices: Sequence[int]) -> torch.Tensor:
        return decode_video(record, tuple(frame_indices))


@dataclass(frozen=True)
class SyntheticVideoDecoder:
    """Generate deterministic RGB frames for local smoke tests."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("synthetic frame dimensions must be positive")

    def decode(self, record: VideoRecord, frame_indices: Sequence[int]) -> torch.Tensor:
        del record
        return torch.stack(
            [
                torch.full((3, self.height, self.width), index % 256, dtype=torch.uint8)
                for index in frame_indices
            ]
        )


def build_clip_index(
    manifest: DatasetManifest,
    *,
    split: Split,
    frames: int,
    stride: int,
    seed: int,
    fingerprint: str | None = None,
) -> tuple[ClipIndex, ...]:
    """Build deterministic non-overlapping clips from videos in one split."""
    validate_manifest(manifest)
    if frames <= 0:
        raise ValueError("frames must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    span = 1 + (frames - 1) * stride
    fingerprint = manifest_checksum(manifest) if fingerprint is None else fingerprint
    records = sorted(
        (record for record in manifest.records if record.split == split),
        key=lambda record: _record_order_key(record, fingerprint, seed),
    )
    return tuple(
        ClipIndex(
            record_id=record.id,
            start=start,
            frame_indices=tuple(start + offset * stride for offset in range(frames)),
        )
        for record in records
        for start in range(0, record.num_frames - span + 1, span)
    )


def _record_order_key(record: VideoRecord, fingerprint: str, seed: int) -> bytes:
    payload = f"{fingerprint}:{seed}:{record.id}".encode()
    return hashlib.sha256(payload).digest()


class MaritimeClipDataset(Dataset[dict[str, Any]]):
    """Present injected video decoders through a single model-ready contract."""

    def __init__(
        self,
        manifest: DatasetManifest,
        decoder: VideoDecoder,
        *,
        split: Split,
        frames: int,
        stride: int,
        image_size: int | None,
        seed: int,
        target_loader: TargetLoader | None = None,
        normalization_mean: tuple[float, float, float] | None = None,
        normalization_std: tuple[float, float, float] | None = None,
        color_jitter: float = 0.0,
        fingerprint: str | None = None,
    ) -> None:
        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive")
        self.records = {record.id: record for record in manifest.records if record.split == split}
        self.clips = build_clip_index(
            manifest, split=split, frames=frames, stride=stride, seed=seed, fingerprint=fingerprint
        )
        self.decoder = decoder
        self.image_size = image_size
        self.target_loader = target_loader or _empty_targets
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("normalization mean and std must be configured together")
        if color_jitter < 0:
            raise ValueError("color jitter must be non-negative")
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.color_jitter = color_jitter if split == "train" else 0.0
        self.seed = seed
        self.decode_failures: list[ClipDecodeFailure] = []
        self._warned_records: set[str] = set()
        self._targets_by_record: dict[str, tuple[FrameTargets, ...]] = {}
        self._validated_target_records: set[str] = set()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int | SampleDraw) -> dict[str, Any] | None:
        augmentation_seed = index.augmentation_seed if isinstance(index, SampleDraw) else None
        index = index.index if isinstance(index, SampleDraw) else index
        clip = self.clips[index]
        record = self.records[clip.record_id]
        try:
            frames = self.decoder.decode(record, clip.frame_indices)
        except (RuntimeError, ValueError) as error:
            self._record_decode_failure(clip, str(error))
            return None
        expected_frames = len(clip.frame_indices)
        actual_frames = frames.shape[0] if frames.ndim else 0
        if actual_frames != expected_frames:
            raise ValueError(
                f"record {record.id}: expected {expected_frames} decoded frames, "
                f"got {actual_frames}"
            )
        transform = self.spatial_transform(frames)
        targets = self.targets_for(record, clip.frame_indices, source_size=frames.shape[-2:])
        return {
            "pixel_values": self.transform(
                frames, sample_index=index, augmentation_seed=augmentation_seed
            ),
            "frame_indices": torch.tensor(clip.frame_indices, dtype=torch.long),
            "record_id": record.id,
            "dataset": record.dataset,
            "source": record.source,
            "targets": targets,
            "is_labelled": record.dataset == "synthetic" or record.annotation_path is not None,
            "spatial_transform": transform,
        }

    def _record_decode_failure(self, clip: ClipIndex, reason: str) -> None:
        self.decode_failures.append(
            ClipDecodeFailure(
                record_id=clip.record_id, frame_indices=clip.frame_indices, reason=reason
            )
        )
        if clip.record_id not in self._warned_records:
            self._warned_records.add(clip.record_id)
            warnings.warn(
                f"dropping undecodable clips from {clip.record_id}: {reason}", stacklevel=2
            )

    def spatial_transform(self, frames: torch.Tensor) -> SpatialTransform:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("decoder must return frames shaped [T, 3, H, W]")
        source_height, source_width = frames.shape[-2:]
        if self.image_size is None:
            return SpatialTransform(
                source_size=(source_height, source_width),
                output_size=(source_height, source_width),
                scale=(1.0, 1.0),
            )
        output_size = (self.image_size, self.image_size)
        return SpatialTransform(
            source_size=(source_height, source_width),
            output_size=output_size,
            scale=(self.image_size / source_height, self.image_size / source_width),
        )

    def targets_for(
        self,
        record: VideoRecord,
        frame_indices: Sequence[int],
        *,
        source_size: tuple[int, int] | None = None,
    ) -> tuple[FrameTargets, ...]:
        selected_indices = set(frame_indices)
        targets = self._targets_by_record.get(record.id)
        if targets is None:
            targets = tuple(
                sorted(self.target_loader(record), key=lambda target: target.frame_index)
            )
            self._targets_by_record[record.id] = targets
        if source_size is not None and record.id not in self._validated_target_records:
            validate_frame_targets(record, targets, source_size=source_size)
            self._validated_target_records.add(record.id)
        return tuple(target for target in targets if target.frame_index in selected_indices)

    def transform(
        self,
        frames: torch.Tensor,
        *,
        sample_index: int = 0,
        augmentation_seed: int | None = None,
    ) -> torch.Tensor:
        """Resize decoded RGB frames and normalize byte tensors to floats."""
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("decoder must return frames shaped [T, 3, H, W]")

        pixel_values = frames.to(dtype=torch.float32)
        if not torch.is_floating_point(frames):
            pixel_values = pixel_values / 255.0
        if self.image_size is not None:
            pixel_values = functional.interpolate(
                pixel_values,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.color_jitter:
            generator = torch.Generator().manual_seed(
                self.seed + sample_index if augmentation_seed is None else augmentation_seed
            )
            brightness = 1 + self.color_jitter * (2 * torch.rand((), generator=generator) - 1)
            pixel_values = (pixel_values * brightness).clamp(0, 1)
        if self.normalization_mean is not None and self.normalization_std is not None:
            mean = pixel_values.new_tensor(self.normalization_mean).view(1, 3, 1, 1)
            std = pixel_values.new_tensor(self.normalization_std).view(1, 3, 1, 1)
            pixel_values = (pixel_values - mean) / std
        return pixel_values


def _empty_targets(_: VideoRecord) -> tuple[FrameTargets, ...]:
    return ()

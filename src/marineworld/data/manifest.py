"""Validation and deterministic checksums for dataset manifests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord

_SPLITS = frozenset({"train", "val", "test"})


def _canonical_path(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_manifest(manifest: DatasetManifest) -> None:
    """Raise ``ValueError`` when a manifest violates dataset invariants."""
    ids: set[str] = set()
    video_splits: dict[Path, str] = {}

    for record in manifest.records:
        _validate_record(record, ids)
        video_path = _canonical_path(record.video_path)
        prior_split = video_splits.setdefault(video_path, record.split)
        if prior_split != record.split:
            raise ValueError("video appears in multiple splits")


def _validate_record(record: VideoRecord, ids: set[str]) -> None:
    if not record.id:
        raise ValueError("record ID must be non-empty")
    if record.id in ids:
        raise ValueError(f"duplicate record ID: {record.id}")
    ids.add(record.id)

    if record.split not in _SPLITS:
        raise ValueError(f"invalid split: {record.split}")
    if not record.video_path.is_file():
        raise ValueError(f"video path does not exist: {record.video_path}")
    if record.annotation_path is not None and not record.annotation_path.is_file():
        raise ValueError(f"annotation path does not exist: {record.annotation_path}")
    if not math.isfinite(record.fps) or record.fps <= 0:
        raise ValueError("FPS must be positive")
    if record.num_frames <= 0:
        raise ValueError("frame count must be positive")


def manifest_checksum(manifest: DatasetManifest) -> str:
    """Return an order-independent SHA-256 checksum of manifest contents."""
    payload = {
        "name": manifest.name,
        "version": manifest.version,
        "license": manifest.license,
        "access": manifest.access,
        "label_mapping": dict(manifest.label_mapping),
        "native_labels": sorted(manifest.native_labels),
        "component_checksums": dict(manifest.component_checksums),
        "components": dict(manifest.components),
        "records": [
            _serialize_record(record) for record in sorted(manifest.records, key=lambda r: r.id)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _serialize_record(record: VideoRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "dataset": record.dataset,
        "video_fingerprint": _file_fingerprint(record.video_path),
        "split": record.split,
        "source": record.source,
        "fps": record.fps,
        "num_frames": record.num_frames,
        "annotation_fingerprint": (
            _file_fingerprint(record.annotation_path)
            if record.annotation_path is not None
            else None
        ),
        "metadata": dict(record.metadata),
    }


def _file_fingerprint(path: Path) -> str:
    """Hash every byte with bounded memory for portable content identity."""
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    chunk_size = 64 * 1024
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame_targets(
    record: VideoRecord,
    targets: tuple[FrameTargets, ...],
    *,
    source_size: tuple[int, int],
) -> None:
    """Validate canonical annotations against video length and source geometry."""
    height, width = source_size
    if height <= 0 or width <= 0:
        raise ValueError("source dimensions must be positive")
    seen: set[int] = set()
    for target in targets:
        if target.frame_index in seen or not 0 <= target.frame_index < record.num_frames:
            raise ValueError(f"record {record.id}: frame index is duplicate or out of range")
        seen.add(target.frame_index)
        boxes = np.asarray(target.boxes_xyxy)
        classes = np.asarray(target.class_ids)
        tracks = None if target.track_ids is None else np.asarray(target.track_ids)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError(f"record {record.id}: boxes must be shaped [N, 4]")
        if classes.ndim != 1 or len(classes) != len(boxes):
            raise ValueError(f"record {record.id}: box and class lengths must align")
        if not np.issubdtype(classes.dtype, np.integer) or not np.isfinite(classes).all():
            raise ValueError(f"record {record.id}: class IDs must be finite integers")
        if np.any(classes < 0):
            raise ValueError(f"record {record.id}: class IDs must be non-negative")
        if tracks is not None and (tracks.ndim != 1 or len(tracks) != len(boxes)):
            raise ValueError(f"record {record.id}: box and track lengths must align")
        if tracks is not None and (
            not np.issubdtype(tracks.dtype, np.integer) or not np.isfinite(tracks).all()
        ):
            raise ValueError(f"record {record.id}: track IDs must be finite integers")
        if not np.isfinite(boxes).all():
            raise ValueError(f"record {record.id}: boxes must be finite")
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError(f"record {record.id}: boxes must have positive area")
        if (
            np.any(boxes < 0)
            or np.any(boxes[:, (0, 2)] > width)
            or np.any(boxes[:, (1, 3)] > height)
        ):
            raise ValueError(f"record {record.id}: boxes exceed source bounds")

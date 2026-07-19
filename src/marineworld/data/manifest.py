"""Validation and deterministic checksums for dataset manifests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from marineworld.data.contracts import DatasetManifest, VideoRecord

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
        "video_path": str(_canonical_path(record.video_path)),
        "split": record.split,
        "source": record.source,
        "fps": record.fps,
        "num_frames": record.num_frames,
        "annotation_path": (
            str(_canonical_path(record.annotation_path))
            if record.annotation_path is not None
            else None
        ),
        "metadata": dict(record.metadata),
    }

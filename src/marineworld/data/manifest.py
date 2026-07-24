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
        "split": record.split,
        "source": record.source,
        "fps": record.fps,
        "num_frames": record.num_frames,
        "video_checksum": file_checksum(record.video_path),
        "annotation_checksum": (
            file_checksum(record.annotation_path) if record.annotation_path is not None else None
        ),
        "metadata": dict(record.metadata),
    }


def content_identity(path: Path) -> str | None:
    """Return a content digest tag for an existing file or directory, else None."""
    if path.is_file():
        return f"sha256:{file_checksum(path)}"
    if path.is_dir():
        return f"sha256-directory:{directory_checksum(path)}"
    return None


def file_checksum(path: Path) -> str:
    """Return a content digest that is independent of the file's local path.

    Not memoised on (size, mtime): those collide on a same-size rewrite within the
    filesystem's mtime resolution, which would return a stale digest for changed
    content -- wrong for a function whose callers use it to detect that change.
    Manifest builds avoid re-hashing by sharing one checksum, not by caching here.
    """
    with _canonical_path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def directory_checksum(root: Path) -> str:
    """Hash a directory tree using portable relative names and file contents."""
    canonical = _canonical_path(root)
    digest = hashlib.sha256(b"marineworld-directory-v1\0")
    for path in sorted(candidate for candidate in canonical.rglob("*") if candidate.is_file()):
        relative_name = path.relative_to(canonical).as_posix().encode()
        content_digest = bytes.fromhex(file_checksum(path))
        digest.update(b"file\0")
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        digest.update(len(content_digest).to_bytes(8, "big"))
        digest.update(content_digest)
    return digest.hexdigest()


def validate_boxes(record_id: str, boxes: np.ndarray, source_size: tuple[int, int]) -> None:
    """Require boxes to be finite, positively sized and inside the source frame."""
    height, width = source_size
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"record {record_id}: boxes must be shaped [N, 4]")
    if not np.isfinite(boxes).all():
        raise ValueError(f"record {record_id}: boxes must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError(f"record {record_id}: boxes must have positive area")
    if np.any(boxes < 0) or np.any(boxes[:, (0, 2)] > width) or np.any(boxes[:, (1, 3)] > height):
        raise ValueError(f"record {record_id}: boxes exceed source bounds")


def validate_class_ids(record_id: str, classes: np.ndarray, count: int) -> None:
    """Require one non-negative integer class per box."""
    if classes.ndim != 1 or len(classes) != count:
        raise ValueError(f"record {record_id}: box and class lengths must align")
    if not np.issubdtype(classes.dtype, np.integer):
        raise ValueError(f"record {record_id}: class IDs must be finite integers")
    if np.any(classes < 0):
        raise ValueError(f"record {record_id}: class IDs must be non-negative")


def validate_track_ids(record_id: str, tracks: np.ndarray | None, count: int) -> None:
    """Require one integer track ID per box when tracking annotations are present."""
    if tracks is None:
        return
    if tracks.ndim != 1 or len(tracks) != count:
        raise ValueError(f"record {record_id}: box and track lengths must align")
    if not np.issubdtype(tracks.dtype, np.integer):
        raise ValueError(f"record {record_id}: track IDs must be finite integers")


def validate_frame_targets(
    record: VideoRecord,
    targets: tuple[FrameTargets, ...],
    *,
    source_size: tuple[int, int],
) -> None:
    """Validate canonical annotations against video length and source geometry."""
    if source_size[0] <= 0 or source_size[1] <= 0:
        raise ValueError("source dimensions must be positive")
    seen: set[int] = set()
    for target in targets:
        if target.frame_index in seen or not 0 <= target.frame_index < record.num_frames:
            raise ValueError(f"record {record.id}: frame index is duplicate or out of range")
        seen.add(target.frame_index)
        boxes = np.asarray(target.boxes_xyxy)
        validate_boxes(record.id, boxes, source_size)
        validate_class_ids(record.id, np.asarray(target.class_ids), len(boxes))
        validate_track_ids(
            record.id,
            None if target.track_ids is None else np.asarray(target.track_ids),
            len(boxes),
        )

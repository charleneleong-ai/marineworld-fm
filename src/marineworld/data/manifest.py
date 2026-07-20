"""Validation and deterministic checksums for dataset manifests."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
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


def file_checksum(path: Path) -> str:
    """Return a content digest that is independent of the file's local path."""
    canonical = _canonical_path(path)
    stat = canonical.stat()
    return cached_file_checksum(canonical, stat.st_size, stat.st_mtime_ns)


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


@lru_cache(maxsize=512)
def cached_file_checksum(path: Path, size: int, modified_ns: int) -> str:
    """Cache content digests while file size and modification time are unchanged."""
    del size, modified_ns
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

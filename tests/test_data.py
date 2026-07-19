"""Tests for immutable dataset records and leakage-safe manifests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from marineworld.data.contracts import DatasetManifest, VideoRecord
from marineworld.data.manifest import manifest_checksum, validate_manifest


def _record(video: Path, *, record_id: str, split: str) -> VideoRecord:
    return VideoRecord(
        id=record_id,
        dataset="demo",
        video_path=video,
        split=split,  # type: ignore[arg-type]
        source="unit-test",
        fps=30.0,
        num_frames=10,
    )


def _manifest(*records: VideoRecord) -> DatasetManifest:
    return DatasetManifest("demo", "1", "MIT", records)


def test_manifest_rejects_video_leakage(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.touch()
    records = (
        _record(video, record_id="train/clip", split="train"),
        _record(video, record_id="test/clip", split="test"),
    )
    with pytest.raises(ValueError, match="video appears in multiple splits"):
        validate_manifest(DatasetManifest("demo", "1", "MIT", records))


def test_manifest_checksum_is_order_independent(tmp_path: Path):
    first = _record(tmp_path / "a.mp4", record_id="a", split="train")
    second = _record(tmp_path / "b.mp4", record_id="b", split="val")
    assert manifest_checksum(_manifest(first, second)) == manifest_checksum(
        _manifest(second, first)
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty_id", "record ID must be non-empty"),
        ("duplicate_id", "duplicate record ID: clip"),
        ("invalid_split", "invalid split: holdout"),
        ("missing_video", "video path does not exist"),
        ("missing_annotation", "annotation path does not exist"),
        ("non_positive_fps", "FPS must be positive"),
        ("non_positive_frame_count", "frame count must be positive"),
    ],
)
def test_manifest_rejects_invalid_records(tmp_path: Path, case: str, message: str) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    record = _record(video, record_id="clip", split="train")

    if case == "empty_id":
        records = (replace(record, id=""),)
    elif case == "duplicate_id":
        records = (record, replace(record, split="val"))
    elif case == "invalid_split":
        records = (replace(record, split="holdout"),)
    elif case == "missing_video":
        records = (replace(record, video_path=tmp_path / "missing.mp4"),)
    elif case == "missing_annotation":
        records = (replace(record, annotation_path=tmp_path / "missing.json"),)
    elif case == "non_positive_fps":
        records = (replace(record, fps=0),)
    else:
        records = (replace(record, num_frames=0),)

    with pytest.raises(ValueError, match=message):
        validate_manifest(_manifest(*records))


def test_manifest_converts_record_membership_to_an_immutable_tuple(tmp_path: Path) -> None:
    records = [_record(tmp_path / "clip.mp4", record_id="clip", split="train")]
    manifest = DatasetManifest("demo", "1", "MIT", records)

    assert isinstance(manifest.records, tuple)
    records.clear()

    assert manifest.records == (manifest.records[0],)

"""Tests for immutable dataset records and leakage-safe manifests."""

from __future__ import annotations

import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
import torch
import yaml
from typer.testing import CliRunner

from marineworld.data import download as download_module
from marineworld.data import video
from marineworld.data.adapters import DatasetAdapter, build_adapter
from marineworld.data.clips import (
    AutoVideoDecoder,
    DecordVideoDecoder,
    MaritimeClipDataset,
    SyntheticVideoDecoder,
    build_clip_index,
)
from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.download import app as download_app
from marineworld.data.download import safe_extract_zip
from marineworld.data.fvessel import FVesselAdapter
from marineworld.data.manifest import manifest_checksum, validate_manifest
from marineworld.data.smd import SMDAdapter
from marineworld.data.synthetic import SyntheticAdapter
from marineworld.data.video import VideoBackendUnavailable, VideoMetadata, probe_video


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


def _synthetic_adapter(tmp_path: Path) -> tuple[DatasetAdapter, Path]:
    return SyntheticAdapter(version="fixture", num_videos=2), tmp_path / "synthetic"


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> DatasetManifest:
    adapter, root = _synthetic_adapter(tmp_path)
    return adapter.build_manifest(root)


def _fps_25(_: Path) -> float:
    return 25.0


def _never_probe(_: Path) -> float:
    raise AssertionError("explicit FPS must bypass probing")


def _fps_nan(_: Path) -> float:
    return float("nan")


def _backend_unavailable(*_args: object) -> NoReturn:
    raise VideoBackendUnavailable("unavailable")


def _target(frame_index: int) -> FrameTargets:
    return FrameTargets(
        frame_index=frame_index,
        boxes_xyxy=np.empty((0, 4), dtype=np.float32),
        class_ids=np.empty(0, dtype=np.int64),
    )


class _WrongLengthDecoder:
    def decode(self, record: VideoRecord, frame_indices: tuple[int, ...]) -> torch.Tensor:
        del record
        return torch.zeros((len(frame_indices) - 1, 3, 32, 32))


def _fvessel_adapter(tmp_path: Path) -> tuple[DatasetAdapter, Path]:
    root = tmp_path / "fvessel"
    sample = root / "sample-01"
    (sample / "gt").mkdir(parents=True)
    (sample / "ais").mkdir()
    (sample / "sample.mp4").touch()
    (sample / "gt" / "gt.txt").write_text("1,7,10,20,30,40,1,1,1\n")
    return FVesselAdapter(version="fixture", fps=30.0, num_frames=16), root


def _smd_adapter(tmp_path: Path) -> tuple[DatasetAdapter, Path]:
    root = tmp_path / "smd"
    video = root / "VIS_Onshore" / "onshore-01.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    return SMDAdapter(version="fixture"), root


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
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mp4").touch()
    first = _record(tmp_path / "a.mp4", record_id="a", split="train")
    second = _record(tmp_path / "b.mp4", record_id="b", split="val")
    assert manifest_checksum(_manifest(first, second)) == manifest_checksum(
        _manifest(second, first)
    )


def test_manifest_checksum_is_portable_across_data_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "machine-a"
    second_root = tmp_path / "machine-b"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "clip.mp4").touch()
    (second_root / "clip.mp4").touch()
    first = VideoRecord("clip", "demo", first_root / "clip.mp4", "train", "demo", 25.0, 8)
    second = replace(first, video_path=second_root / "clip.mp4")

    assert manifest_checksum(_manifest(first)) == manifest_checksum(_manifest(second))


def test_manifest_checksum_changes_with_video_content(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"first")
    record = VideoRecord("clip", "demo", video, "train", "demo", 25.0, 8)
    first = manifest_checksum(_manifest(record))

    video.write_bytes(b"second")

    assert manifest_checksum(_manifest(record)) != first


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


@pytest.mark.parametrize("adapter_factory", [_synthetic_adapter, _fvessel_adapter, _smd_adapter])
def test_adapter_produces_valid_manifest(adapter_factory, tmp_path: Path) -> None:
    adapter, root = adapter_factory(tmp_path)

    manifest = adapter.build_manifest(root)

    validate_manifest(manifest)
    assert manifest.records
    assert all(record.dataset == manifest.name for record in manifest.records)


def test_fvessel_parses_mot_targets(tmp_path: Path) -> None:
    adapter, root = _fvessel_adapter(tmp_path)
    record = adapter.build_manifest(root).records[0]

    targets = adapter.load_targets(record)

    assert targets[0].boxes_xyxy.tolist() == [[10.0, 20.0, 40.0, 60.0]]
    assert targets[0].track_ids.tolist() == [7]


def test_fvessel_probes_fps_when_not_explicit(tmp_path: Path) -> None:
    _, root = _fvessel_adapter(tmp_path)
    adapter = FVesselAdapter(version="fixture", fps=None, num_frames=16, fps_probe=_fps_25)

    record = adapter.build_manifest(root).records[0]

    assert record.fps == 25.0


def test_fvessel_explicit_fps_bypasses_probe(tmp_path: Path) -> None:
    _, root = _fvessel_adapter(tmp_path)
    adapter = FVesselAdapter(version="fixture", fps=20.0, num_frames=16, fps_probe=_never_probe)

    record = adapter.build_manifest(root).records[0]

    assert record.fps == 20.0


def test_fvessel_rejects_invalid_probed_fps(tmp_path: Path) -> None:
    _, root = _fvessel_adapter(tmp_path)
    adapter = FVesselAdapter(version="fixture", fps=None, num_frames=16, fps_probe=_fps_nan)

    with pytest.raises(ValueError, match="FPS must be positive"):
        adapter.build_manifest(root)


def test_build_adapter_instantiates_configured_target() -> None:
    adapter = build_adapter(
        {
            "_target_": "marineworld.data.synthetic.SyntheticAdapter",
            "version": "fixture",
            "num_videos": 1,
        }
    )

    assert isinstance(adapter, SyntheticAdapter)


@pytest.mark.parametrize(
    ("config_name", "adapter_type"),
    [
        ("fvessel", FVesselAdapter),
        ("smd", SMDAdapter),
        ("synthetic", SyntheticAdapter),
    ],
)
def test_data_config_constructs_its_adapter(
    config_name: str, adapter_type: type[DatasetAdapter]
) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "data" / f"{config_name}.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert isinstance(build_adapter(config["adapter"]), adapter_type)


def test_fvessel_data_config_passes_fps_to_adapter() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "data" / "fvessel.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["adapter"]["fps"] is None
    config["adapter"]["fps"] = 20.0

    adapter = build_adapter(config["adapter"])

    assert isinstance(adapter, FVesselAdapter)
    assert adapter.fps == 20.0


def test_clip_index_is_deterministic_and_split_scoped(
    synthetic_manifest: DatasetManifest,
) -> None:
    first = build_clip_index(synthetic_manifest, split="train", frames=4, stride=2, seed=42)
    second = build_clip_index(synthetic_manifest, split="train", frames=4, stride=2, seed=42)

    assert first == second
    assert {clip.record_id for clip in first} == {"train-0"}
    assert [(clip.start, clip.frame_indices) for clip in first] == [
        (0, (0, 2, 4, 6)),
        (7, (7, 9, 11, 13)),
    ]


def test_clip_dataset_returns_canonical_tensor(synthetic_manifest: DatasetManifest) -> None:
    dataset = MaritimeClipDataset(
        synthetic_manifest,
        SyntheticVideoDecoder(height=32, width=32),
        split="train",
        frames=4,
        stride=1,
        image_size=16,
        seed=42,
    )

    sample = dataset[0]

    assert sample["pixel_values"].shape == (4, 3, 16, 16)
    assert sample["pixel_values"].dtype == torch.float32
    assert sample["frame_indices"].tolist() == [0, 1, 2, 3]


def test_clip_dataset_rejects_wrong_decoder_length(
    synthetic_manifest: DatasetManifest,
) -> None:
    dataset = MaritimeClipDataset(
        synthetic_manifest,
        _WrongLengthDecoder(),
        split="train",
        frames=4,
        stride=1,
        image_size=16,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match=r"record train-0: expected 4 decoded frames, got 3",
    ):
        dataset[0]


def test_decord_decoder_missing_dependency_explains_supported_fallback(
    synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "decord", None)
    record = synthetic_manifest.records[0]

    with pytest.raises(RuntimeError, match="supported platform or inject another VideoDecoder"):
        DecordVideoDecoder().decode(record, (0,))


def test_auto_decoder_prefers_decord(
    synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    frames = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
    monkeypatch.setattr(video, "_decord_decode", lambda *_: calls.append("decord") or frames)
    monkeypatch.setattr(video, "_pyav_decode", lambda *_: calls.append("pyav") or frames)

    AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))

    assert calls == ["decord"]


def test_auto_decoder_falls_back_to_pyav(
    synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = torch.tensor([2, 0, 2], dtype=torch.uint8).view(3, 1, 1, 1).expand(-1, 3, 2, 2)
    monkeypatch.setattr(video, "_decord_decode", _backend_unavailable)
    monkeypatch.setattr(video, "_pyav_decode", lambda *_: frames)

    decoded = AutoVideoDecoder().decode(synthetic_manifest.records[0], (2, 0, 2))

    assert decoded[:, 0, 0, 0].tolist() == [2, 0, 2]


def test_auto_decoder_recovers_from_decord_decode_failure(
    synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)

    def fail_decord(*_: object) -> torch.Tensor:
        raise video.VideoDecodeError("Decord rejected malformed H.264 packets")

    monkeypatch.setattr(video, "_decord_decode", fail_decord)
    monkeypatch.setattr(video, "_pyav_decode", lambda *_: frames)

    assert AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,)) is frames


def test_auto_decoder_does_not_mask_programmer_errors(
    synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = RuntimeError("unexpected tensor contract bug")

    def fail_unexpectedly(*_: object) -> torch.Tensor:
        raise error

    monkeypatch.setattr(video, "_decord_decode", fail_unexpectedly)

    with pytest.raises(RuntimeError) as caught:
        AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))

    assert caught.value is error


def test_auto_decoder_names_both_missing_backends(
    synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video, "_decord_decode", _backend_unavailable)
    monkeypatch.setattr(video, "_pyav_decode", _backend_unavailable)

    with pytest.raises(RuntimeError, match="decord: unavailable; PyAV: unavailable") as caught:
        AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))

    assert isinstance(caught.value.__cause__, VideoBackendUnavailable)


def test_video_probe_prefers_decord(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    expected = VideoMetadata(frame_count=12, fps=25.0)
    monkeypatch.setattr(video, "_decord_probe", lambda *_: calls.append("decord") or expected)
    monkeypatch.setattr(video, "_pyav_probe", lambda *_: calls.append("pyav") or expected)

    assert probe_video(tmp_path / "clip.mp4") == expected
    assert calls == ["decord"]


def test_video_probe_falls_back_to_pyav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = VideoMetadata(frame_count=12, fps=25.0)
    monkeypatch.setattr(video, "_decord_probe", _backend_unavailable)
    monkeypatch.setattr(video, "_pyav_probe", lambda *_: expected)

    assert probe_video(tmp_path / "clip.mp4") == expected


@pytest.mark.parametrize(
    "metadata",
    [VideoMetadata(frame_count=0, fps=25.0), VideoMetadata(frame_count=12, fps=0.0)],
)
def test_video_probe_rejects_non_positive_metadata(
    metadata: VideoMetadata, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(video, "_decord_probe", lambda *_: metadata)

    with pytest.raises(ValueError, match="positive"):
        probe_video(tmp_path / "private" / "clip.mp4")


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "nested/../../escape.txt"])
def test_safe_extract_zip_rejects_members_outside_destination(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, "unsafe")

    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "data")


def test_safe_extract_zip_preserves_nested_files(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sample/video.mp4", "video")

    destination = tmp_path / "data"
    safe_extract_zip(archive, destination)

    assert (destination / "sample" / "video.mp4").read_text() == "video"


def test_fvessel_downloader_exposes_named_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(download_module, "download_fvessel_clip10", lambda output: output)
    result = CliRunner().invoke(
        download_app, ["fvessel-clip10", "--output", str(tmp_path / "fvessel")]
    )

    assert result.exit_code == 0


def test_clip_dataset_caches_and_filters_adapter_targets(
    synthetic_manifest: DatasetManifest,
) -> None:
    calls = 0

    def load_targets(_: VideoRecord) -> tuple[FrameTargets, ...]:
        nonlocal calls
        calls += 1
        return (_target(4), _target(2), _target(0))

    dataset = MaritimeClipDataset(
        synthetic_manifest,
        SyntheticVideoDecoder(height=32, width=32),
        split="train",
        frames=4,
        stride=1,
        image_size=16,
        seed=42,
        target_loader=load_targets,
    )

    first = dataset[0]
    second = dataset[1]

    assert first["record_id"] == "train-0"
    assert first["dataset"] == "synthetic"
    assert [target.frame_index for target in first["targets"]] == [0, 2]
    assert [target.frame_index for target in second["targets"]] == [4]
    assert calls == 1

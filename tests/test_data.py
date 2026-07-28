"""Tests for immutable dataset records and leakage-safe manifests."""

from __future__ import annotations

import sys
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
import torch
import yaml
from scipy.io import savemat
from typer.testing import CliRunner

from marineworld.data import download as download_module
from marineworld.data import video as video_module
from marineworld.data.adapters import (
    CompositeAdapter,
    DatasetAdapter,
    FrameCountCache,
    build_adapter,
    resolve_frame_count,
)
from marineworld.data.clips import (
    AutoVideoDecoder,
    DecordVideoDecoder,
    MaritimeClipDataset,
    SpatialTransform,
    SyntheticVideoDecoder,
    build_clip_index,
    drop_undecodable,
)
from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.download import app as download_app
from marineworld.data.download import safe_extract_zip
from marineworld.data.fvessel import FVesselAdapter
from marineworld.data.manifest import manifest_checksum, validate_frame_targets, validate_manifest
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


def _frames_96(_: Path) -> int:
    return 96


def _target(frame_index: int) -> FrameTargets:
    return FrameTargets(
        frame_index=frame_index,
        boxes_xyxy=np.empty((0, 4), dtype=np.float32),
        class_ids=np.empty(0, dtype=np.int64),
    )


class WrongLengthDecoder:
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
    return SMDAdapter(version="fixture", num_frames=16), root


def _write_smd_objectgt(
    path: Path,
    frames: list[tuple[list[list[float]], list[int]]],
) -> None:
    """Write the native SMD `structXML` shape used by the public converter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    struct_xml = np.empty(
        (1, len(frames)),
        dtype=[("BB", object), ("Object", object), ("Motion", object), ("Distance", object)],
    )
    for index, (boxes, classes) in enumerate(frames):
        struct_xml["BB"][0, index] = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        struct_xml["Object"][0, index] = np.asarray(classes, dtype=np.float64).reshape(-1, 1)
        struct_xml["Motion"][0, index] = np.empty((0, 1), dtype=np.float64)
        struct_xml["Distance"][0, index] = np.empty((0, 1), dtype=np.float64)
    savemat(path, {"structXML": struct_xml})


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


def test_manifest_checksum_is_portable_and_content_sensitive(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    manifests = []
    for root in roots:
        video = root / "nested" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"same-video-content")
        manifests.append(_manifest(_record(video, record_id="nested/clip", split="train")))

    assert manifest_checksum(manifests[0]) == manifest_checksum(manifests[1])

    roots[1].joinpath("nested/clip.mp4").write_bytes(b"changed-video-content")
    assert manifest_checksum(manifests[0]) != manifest_checksum(manifests[1])


def test_file_checksum_reflects_a_same_size_rewrite_at_one_path(tmp_path: Path) -> None:
    """A same-size rewrite within the mtime tick must not return a stale digest."""
    from marineworld.data.manifest import file_checksum

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"content-v1")
    first = file_checksum(video)
    video.write_bytes(b"content-v2")  # identical length, immediately

    assert file_checksum(video) != first


def test_manifest_checksum_detects_same_size_large_media_mutation(tmp_path: Path) -> None:
    payload = bytearray(b"a" * 1024 * 1024)
    videos = [tmp_path / name / "clip.mp4" for name in ("first", "second")]
    for video in videos:
        video.parent.mkdir()
        video.write_bytes(payload)
    manifests = [_manifest(_record(video, record_id="clip", split="train")) for video in videos]
    assert manifest_checksum(manifests[0]) == manifest_checksum(manifests[1])

    payload[len(payload) // 4] = ord("b")
    videos[1].write_bytes(payload)
    assert manifest_checksum(manifests[0]) != manifest_checksum(manifests[1])


def test_manifest_carries_access_and_label_provenance(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        "demo",
        "1",
        "MIT",
        (_record(tmp_path / "clip.mp4", record_id="clip", split="train"),),
        access="public",
        label_mapping={"1": "vessel"},
        native_labels=("native:1",),
    )

    assert manifest.access == "public"
    assert manifest.label_mapping == {"1": "vessel"}
    assert manifest.native_labels == ("native:1",)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (_target(-1), "frame index"),
        (
            FrameTargets(0, np.asarray([[1, 2, 1, 4]], dtype=np.float32), np.asarray([1])),
            "positive area",
        ),
        (
            FrameTargets(0, np.asarray([[1, 2, 20, 4]], dtype=np.float32), np.asarray([1])),
            "bounds",
        ),
        (
            FrameTargets(
                0,
                np.asarray([[1, 2, 3, 4]], dtype=np.float32),
                np.asarray([float("nan")]),
            ),
            "class IDs",
        ),
    ],
)
def test_frame_target_validator_rejects_malformed_geometry(
    tmp_path: Path, target: FrameTargets, message: str
) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    record = _record(video, record_id="clip", split="train")

    with pytest.raises(ValueError, match=message):
        validate_frame_targets(record, (target,), source_size=(10, 10))


def test_clip_dataset_validates_targets_outside_current_clip(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    record = replace(_record(video, record_id="clip", split="train"), num_frames=4)
    dataset = MaritimeClipDataset(
        _manifest(record),
        SyntheticVideoDecoder(10, 10),
        split="train",
        frames=2,
        stride=1,
        image_size=10,
        seed=42,
        target_loader=lambda _: (
            FrameTargets(99, np.empty((0, 4), np.float32), np.empty(0, np.int64)),
        ),
    )

    with pytest.raises(ValueError, match="frame index"):
        dataset[0]


def test_composite_adapter_names_the_component_that_resolved_to_nothing(tmp_path: Path) -> None:
    """A missing component must not silently degrade the mixture to a smaller corpus."""
    adapter = CompositeAdapter(
        components={
            "smd": SMDAdapter(version="fixture", num_frames=4),
            "fvessel": SyntheticAdapter(version="one", num_videos=1, num_frames=4),
        },
        roots={"smd": tmp_path / "missing-smd", "fvessel": tmp_path / "fvessel"},
    )

    with pytest.raises(ValueError, match=r"zero videos.*smd at .*missing-smd") as caught:
        adapter.build_manifest(tmp_path)

    assert "fvessel" not in str(caught.value)


def test_composite_adapter_preserves_dataset_identity_and_provenance(tmp_path: Path) -> None:
    first = SyntheticAdapter(version="one", num_videos=1, num_frames=4)
    second = SyntheticAdapter(version="two", num_videos=1, num_frames=4)
    adapter = CompositeAdapter(
        components={"smd": first, "fvessel": second},
        roots={"smd": tmp_path / "smd", "fvessel": tmp_path / "fvessel"},
    )

    manifest = adapter.build_manifest(tmp_path)

    assert {record.dataset for record in manifest.records} == {"smd", "fvessel"}
    assert set(manifest.component_checksums) == {"smd", "fvessel"}
    assert manifest.components["smd"]["version"] == "one"
    assert {record.id.split(":", 1)[0] for record in manifest.records} == {"smd", "fvessel"}


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


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("1,7,10\n", "MOT row 1"),
        ("1,1.5,10,20,30,40,1,1\n", "finite non-negative integer"),
        ("1,nan,10,20,30,40,1,1\n", "finite non-negative integer"),
        ("1,-1,10,20,30,40,1,1\n", "finite non-negative integer"),
        ("0,1,10,20,30,40,1,1\n", "positive integer"),
    ],
)
def test_fvessel_rejects_malformed_mot_rows(tmp_path: Path, row: str, message: str) -> None:
    adapter, root = _fvessel_adapter(tmp_path)
    record = adapter.build_manifest(root).records[0]
    assert record.annotation_path is not None
    record.annotation_path.write_text(row)

    with pytest.raises(ValueError, match=message):
        adapter.load_targets(record)


def test_smd_pairs_native_objectgt_and_parses_vessel_targets(tmp_path: Path) -> None:
    root = tmp_path / "smd"
    video = root / "VIS_Onshore" / "MVI_1478_VIS.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    annotation = video.parent / "ObjectGT" / "MVI_1478_VIS_ObjectGT.mat"
    _write_smd_objectgt(
        annotation,
        [
            (
                [
                    [10, 20, 30, 40],
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [0, 0, 0, 0],
                ],
                [3, 8, 7, 2, 0],
            ),
            ([], []),
        ],
    )
    adapter = SMDAdapter(num_frames=2, annotation_format="smd_objectgt_mat")

    record = adapter.build_manifest(root).records[0]
    targets = adapter.load_targets(record)

    assert record.annotation_path == annotation
    assert record.metadata["annotation_format"] == "smd_objectgt_mat"
    assert [target.frame_index for target in targets] == [0, 1]
    assert targets[0].boxes_xyxy.tolist() == [[10.0, 20.0, 40.0, 60.0], [5.0, 6.0, 12.0, 14.0]]
    assert targets[0].class_ids.tolist() == [3, 7]
    assert targets[1].boxes_xyxy.shape == (0, 4)
    assert targets[1].class_ids.shape == (0,)


def test_smd_rejects_orphan_objectgt_filename(tmp_path: Path) -> None:
    root = tmp_path / "smd"
    video = root / "NIR" / "MVI_1_NIR.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    _write_smd_objectgt(video.parent / "ObjectGT" / "MVI_2_NIR_ObjectGT.mat", [([], [])])

    with pytest.raises(ValueError, match="does not pair with an SMD video"):
        SMDAdapter(num_frames=1).build_manifest(root)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"unexpected": np.ones(1)}, "required root 'structXML'"),
        (
            {"structXML": np.empty((1, 1), dtype=[("BB", object)])},
            "required fields",
        ),
    ],
)
def test_smd_rejects_malformed_objectgt_schema(
    tmp_path: Path, payload: dict[str, np.ndarray], message: str
) -> None:
    root = tmp_path / "smd"
    video = root / "VIS_Onboard" / "MVI_1_VIS.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    annotation = video.parent / "ObjectGT" / "MVI_1_VIS_ObjectGT.mat"
    annotation.parent.mkdir()
    if "structXML" in payload:
        payload["structXML"]["BB"][0, 0] = np.empty((0, 4))
    savemat(annotation, payload)
    record = SMDAdapter(num_frames=1).build_manifest(root).records[0]

    with pytest.raises(ValueError, match=message):
        SMDAdapter(num_frames=1).load_targets(record)


def test_smd_rejects_objectgt_box_class_length_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "smd"
    video = root / "VIS_Onboard" / "MVI_1_VIS.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    annotation = video.parent / "ObjectGT" / "MVI_1_VIS_ObjectGT.mat"
    _write_smd_objectgt(annotation, [([[1, 2, 3, 4], [5, 6, 7, 8]], [3])])
    record = SMDAdapter(num_frames=1).build_manifest(root).records[0]

    with pytest.raises(ValueError, match="frame 0 has 2 boxes but 1 class labels"):
        SMDAdapter(num_frames=1).load_targets(record)


def test_smd_rejects_objectgt_frame_count_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "smd"
    video = root / "NIR" / "MVI_1_NIR.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    _write_smd_objectgt(video.parent / "ObjectGT" / "MVI_1_NIR_ObjectGT.mat", [([], [])])
    adapter = SMDAdapter(num_frames=2)
    record = adapter.build_manifest(root).records[0]

    with pytest.raises(ValueError, match="contains 1 frames but video manifest declares 2"):
        adapter.load_targets(record)


def test_smd_rejects_unknown_annotation_format(tmp_path: Path) -> None:
    _, root = _smd_adapter(tmp_path)

    with pytest.raises(ValueError, match="unsupported SMD annotation_format"):
        SMDAdapter(num_frames=1, annotation_format="auto").build_manifest(root)


def test_real_adapter_defaults_probe_source_frame_counts() -> None:
    assert FVesselAdapter().num_frames is None
    assert SMDAdapter().num_frames is None


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


@pytest.mark.parametrize("dataset", ["fvessel", "smd"])
def test_real_adapters_probe_source_frame_counts(dataset: str, tmp_path: Path) -> None:
    if dataset == "fvessel":
        _, root = _fvessel_adapter(tmp_path)
        adapter = FVesselAdapter(
            version="fixture",
            fps=30.0,
            num_frames=None,
            frame_count_probe=_frames_96,
        )
    else:
        _, root = _smd_adapter(tmp_path)
        adapter = SMDAdapter(
            version="fixture",
            num_frames=None,
            frame_count_probe=_frames_96,
        )

    record = adapter.build_manifest(root).records[0]

    assert record.num_frames == 96


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


def test_clip_dataset_maps_non_square_source_boxes_to_square_output(
    synthetic_manifest: DatasetManifest,
) -> None:
    dataset = MaritimeClipDataset(
        synthetic_manifest,
        SyntheticVideoDecoder(height=20, width=40),
        split="train",
        frames=4,
        stride=1,
        image_size=10,
        seed=42,
    )

    transform = dataset[0]["spatial_transform"]
    boxes = np.array([[4.0, 2.0, 20.0, 10.0]], dtype=np.float32)

    assert transform == SpatialTransform(
        source_size=(20, 40),
        output_size=(10, 10),
        scale=(0.5, 0.25),
        offset=(0.0, 0.0),
    )
    assert transform.apply_boxes_xyxy(boxes).tolist() == [[1.0, 1.0, 5.0, 5.0]]


def test_clip_dataset_rejects_wrong_decoder_length(
    synthetic_manifest: DatasetManifest,
) -> None:
    dataset = MaritimeClipDataset(
        synthetic_manifest,
        WrongLengthDecoder(),
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


def test_frame_count_reflects_frames_that_actually_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Container metadata overreports damaged H.264 tails; clips must not span them."""
    video = tmp_path / "clip.mp4"
    video.touch()
    monkeypatch.setattr(
        video_module, "probe_video", lambda _: video_module.VideoMetadata(frame_count=100, fps=25.0)
    )
    # the reported tail does not decode, so the stream is counted for real
    monkeypatch.setattr(video_module, "_decord_can_seek", lambda _path, index: index < 90)
    monkeypatch.setattr(video_module, "count_decodable_frames", lambda _path: 90)

    assert video_module.probe_video_frame_count(video) == 90


def test_frame_count_trusts_metadata_when_the_tail_decodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An intact video must cost exactly one verification decode, not a bisect."""
    video = tmp_path / "clip.mp4"
    video.touch()
    checked: list[int] = []
    monkeypatch.setattr(
        video_module, "probe_video", lambda _: video_module.VideoMetadata(frame_count=100, fps=25.0)
    )
    monkeypatch.setattr(
        video_module, "_decord_can_seek", lambda _p, index: checked.append(index) or True
    )

    assert video_module.probe_video_frame_count(video) == 100
    assert checked == [99]


class HoleDecoder:
    """Decode everything except an interior region, as damaged H.264 streams do."""

    def __init__(self, dead: range) -> None:
        self.dead = dead

    def decode(self, record: VideoRecord, frame_indices: Sequence[int]) -> torch.Tensor:
        if any(index in self.dead for index in frame_indices):
            raise RuntimeError(f"could not decode frames from {record.video_path}")
        return torch.zeros((len(frame_indices), 3, 8, 8))


def test_dataset_drops_clips_it_cannot_decode_instead_of_failing_the_run(
    synthetic_manifest: DatasetManifest,
) -> None:
    """Damage is not always a suffix, so a run must survive an interior hole."""
    dataset = MaritimeClipDataset(
        synthetic_manifest,
        HoleDecoder(range(4, 8)),
        split="train",
        frames=2,
        stride=1,
        image_size=8,
        seed=42,
    )

    samples = [dataset[index] for index in range(len(dataset))]

    assert any(sample is None for sample in samples), "expected the damaged clips to drop"
    assert any(sample is not None for sample in samples), "expected intact clips to survive"
    assert dataset.decode_failures
    assert all(failure.record_id for failure in dataset.decode_failures)


def test_collate_drops_undecodable_samples_but_refuses_an_empty_batch() -> None:
    """Dropping is data loss, so a batch with nothing left must be loud."""
    good = {
        "pixel_values": torch.zeros((2, 3, 8, 8)),
        "dataset": "demo",
        "record_id": "a",
        "source": "s",
    }

    assert drop_undecodable([good, None, good]) == [good, good]
    with pytest.raises(ValueError, match="every clip in the batch failed to decode"):
        drop_undecodable([None, None])


class TestAutoVideoDecoder:
    """Backend selection and fallback for frame decoding."""

    def test_auto_decoder_prefers_decord(
        self, synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        frames = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
        monkeypatch.setattr(
            video_module, "_decord_decode", lambda *_: calls.append("decord") or frames
        )
        monkeypatch.setattr(video_module, "_pyav_decode", lambda *_: calls.append("pyav") or frames)

        AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))

        assert calls == ["decord"]

    def test_auto_decoder_falls_back_to_pyav(
        self, synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = torch.tensor([2, 0, 2], dtype=torch.uint8).view(3, 1, 1, 1).expand(-1, 3, 2, 2)
        monkeypatch.setattr(video_module, "_decord_decode", _backend_unavailable)
        monkeypatch.setattr(video_module, "_pyav_decode", lambda *_: frames)

        decoded = AutoVideoDecoder().decode(synthetic_manifest.records[0], (2, 0, 2))

        assert decoded[:, 0, 0, 0].tolist() == [2, 0, 2]

    def test_auto_decoder_recovers_from_decord_decode_failure(
        self, synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)

        def fail_decord(*_: object) -> torch.Tensor:
            raise video_module.VideoDecodeError("Decord rejected malformed H.264 packets")

        monkeypatch.setattr(video_module, "_decord_decode", fail_decord)
        monkeypatch.setattr(video_module, "_pyav_decode", lambda *_: frames)

        assert AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,)) is frames

    def test_auto_decoder_does_not_mask_programmer_errors(
        self, synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = RuntimeError("unexpected tensor contract bug")

        def fail_unexpectedly(*_: object) -> torch.Tensor:
            raise error

        monkeypatch.setattr(video_module, "_decord_decode", fail_unexpectedly)

        with pytest.raises(RuntimeError) as caught:
            AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))

        assert caught.value is error

    def test_auto_decoder_names_both_missing_backends(
        self, synthetic_manifest: DatasetManifest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(video_module, "_decord_decode", _backend_unavailable)
        monkeypatch.setattr(video_module, "_pyav_decode", _backend_unavailable)

        with pytest.raises(RuntimeError, match="decord: unavailable; PyAV: unavailable") as caught:
            AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))

        assert isinstance(caught.value.__cause__, VideoBackendUnavailable)


class TestVideoProbe:
    """Metadata probing across decord and PyAV backends."""

    def test_video_probe_prefers_decord(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []
        expected = VideoMetadata(frame_count=12, fps=25.0)
        monkeypatch.setattr(
            video_module, "_decord_probe", lambda *_: calls.append("decord") or expected
        )
        monkeypatch.setattr(
            video_module, "_pyav_probe", lambda *_: calls.append("pyav") or expected
        )

        assert probe_video(tmp_path / "clip.mp4") == expected
        assert calls == ["decord"]

    def test_video_probe_falls_back_to_pyav(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        expected = VideoMetadata(frame_count=12, fps=25.0)
        monkeypatch.setattr(video_module, "_decord_probe", _backend_unavailable)
        monkeypatch.setattr(video_module, "_pyav_probe", lambda *_: expected)

        assert probe_video(tmp_path / "clip.mp4") == expected

    @pytest.mark.parametrize(
        "metadata",
        [VideoMetadata(frame_count=0, fps=25.0), VideoMetadata(frame_count=12, fps=0.0)],
    )
    def test_video_probe_rejects_non_positive_metadata(
        self, metadata: VideoMetadata, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(video_module, "_decord_probe", lambda *_: metadata)

        with pytest.raises(ValueError, match="positive"):
            probe_video(tmp_path / "private" / "clip.mp4")


class TestSafeExtractZip:
    """Path-traversal-safe zip extraction."""

    @pytest.mark.parametrize(
        "member", ["../escape.txt", "/absolute.txt", "nested/../../escape.txt"]
    )
    def test_safe_extract_zip_rejects_members_outside_destination(
        self, tmp_path: Path, member: str
    ) -> None:
        archive = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(member, "unsafe")

        with pytest.raises(ValueError, match="unsafe archive member"):
            safe_extract_zip(archive, tmp_path / "data")

    def test_safe_extract_zip_preserves_nested_files(self, tmp_path: Path) -> None:
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


class CountingProbe:
    """A frame-count probe that records how many videos it was asked to decode."""

    def __init__(self, value: int = 96) -> None:
        self.value = value
        self.calls: list[Path] = []

    def __call__(self, video: Path) -> int:
        self.calls.append(video)
        return self.value


class TestFrameCountCache:
    """Persisted frame counts so a damaged corpus is decoded once, not per run."""

    def test_second_build_reuses_the_persisted_count(self, tmp_path: Path) -> None:
        root = tmp_path / "fvessel"
        (root / "sample-01").mkdir(parents=True)
        (root / "sample-01" / "clip.mp4").write_bytes(b"video")
        probe = CountingProbe()
        adapter = FVesselAdapter(version="fixture", fps=30.0, frame_count_probe=probe)

        first = adapter.build_manifest(root)
        second = adapter.build_manifest(root)

        assert first.records[0].num_frames == 96
        assert second.records[0].num_frames == 96
        assert len(probe.calls) == 1  # decoded once, reused on the second build
        assert (root / ".marineworld_frame_counts.json").is_file()

    def test_a_changed_video_is_recounted(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"first")
        probe = CountingProbe()

        resolve_frame_count(None, video, probe, "T", FrameCountCache.for_root(tmp_path))
        video.write_bytes(b"second-and-longer")  # different size invalidates the entry
        resolve_frame_count(None, video, probe, "T", FrameCountCache.for_root(tmp_path))

        assert len(probe.calls) == 2

    def test_a_read_only_data_dir_still_builds(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video")
        probe = CountingProbe(42)
        cache = FrameCountCache(tmp_path / "nonexistent-subdir" / "cache.json")
        cache.path.parent.mkdir()
        cache.path.parent.chmod(0o500)  # unwritable
        try:
            assert resolve_frame_count(None, video, probe, "T", cache) == 42
        finally:
            cache.path.parent.chmod(0o700)

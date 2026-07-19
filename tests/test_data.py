"""Tests for immutable dataset records and leakage-safe manifests."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import savemat

from marineworld.data.adapters import DatasetAdapter, build_adapter
from marineworld.data.clips import (
    DecordVideoDecoder,
    MaritimeClipDataset,
    SpatialTransform,
    SyntheticVideoDecoder,
    build_clip_index,
)
from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.fvessel import FVesselAdapter
from marineworld.data.manifest import manifest_checksum, validate_manifest
from marineworld.data.smd import SMDAdapter
from marineworld.data.synthetic import SyntheticAdapter


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


def _frames_96(_: Path) -> int:
    return 96


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
    adapter = FVesselAdapter(version="fixture", fps=None, fps_probe=_fps_nan)

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


def test_fvessel_default_probe_reports_missing_decord(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _fvessel_adapter(tmp_path)
    monkeypatch.setitem(sys.modules, "decord", None)

    with pytest.raises(RuntimeError, match="supported platform or configure a positive fps"):
        FVesselAdapter(version="fixture").build_manifest(root)


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

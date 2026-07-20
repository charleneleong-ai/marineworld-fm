"""FVessel dataset adapter (video + time-synchronised AIS).

FVessel layout (per video sample):
    <sample>/
        ais/                         # many CSVs, one per received window
        <clip>.mp4
        camera_para.txt
        gt/                          # MOT-format detection/tracking/fusion GT

This adapter is intentionally thin for v0: it resolves the AIS tracks and
exposes per-frame AIS alignment via `marineworld.data.alignment`. Video frame
decoding (decord) and the torch `Dataset` wrapper are added in v1/v2; the AIS
alignment plumbing it depends on is implemented and tested now.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from marineworld.data.alignment import (
    AISRecord,
    align_tracks_to_frames,
    frame_timestamps_ms,
    load_ais_tracks,
)
from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.video import probe_video_fps, probe_video_frame_count

__all__ = ["FVesselAdapter", "FVesselSample", "load_sample"]


@dataclass(frozen=True)
class FVesselAdapter:
    """Adapt FVessel video samples and their MOT targets to common records."""

    version: str = "v1"
    fps: float | None = None
    num_frames: int | None = None
    fps_probe: Callable[[Path], float] = probe_video_fps
    frame_count_probe: Callable[[Path], int] = probe_video_frame_count

    def build_manifest(self, root: Path) -> DatasetManifest:
        records = tuple(self._record(root, video) for video in sorted(root.rglob("*.mp4")))
        return DatasetManifest("fvessel", self.version, "MIT", records)

    def load_targets(self, record: VideoRecord) -> tuple[FrameTargets, ...]:
        if record.annotation_path is None:
            return ()

        frames: dict[int, list[tuple[list[float], int, int]]] = {}
        with record.annotation_path.open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 8:
                    continue
                frame, track_id, left, top, width, height, _, class_id = row[:8]
                frame_index = int(float(frame)) - 1
                frames.setdefault(frame_index, []).append(
                    (
                        [
                            float(left),
                            float(top),
                            float(left) + float(width),
                            float(top) + float(height),
                        ],
                        int(float(class_id)),
                        int(float(track_id)),
                    )
                )

        return tuple(
            FrameTargets(
                frame_index=frame_index,
                boxes_xyxy=np.asarray([box for box, _, _ in entries], dtype=np.float32),
                class_ids=np.asarray([class_id for _, class_id, _ in entries], dtype=np.int64),
                track_ids=np.asarray([track_id for _, _, track_id in entries], dtype=np.int64),
            )
            for frame_index, entries in sorted(frames.items())
        )

    def _record(self, root: Path, video: Path) -> VideoRecord:
        annotation = _mot_path(video.parent)
        return VideoRecord(
            id=video.relative_to(root).with_suffix("").as_posix(),
            dataset="fvessel",
            video_path=video,
            split="train",
            source=video.parent.name,
            fps=self._source_fps(video),
            num_frames=self._source_num_frames(video),
            annotation_path=annotation,
            metadata={"has_ais": (video.parent / "ais").is_dir()},
        )

    def _source_fps(self, video: Path) -> float:
        fps = self.fps if self.fps is not None else self.fps_probe(video)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"FVessel FPS must be positive for {video}, got {fps}")
        return fps

    def _source_num_frames(self, video: Path) -> int:
        count = self.num_frames if self.num_frames is not None else self.frame_count_probe(video)
        if count <= 0:
            raise ValueError(f"FVessel frame count must be positive, got {count}")
        return count


def _mot_path(sample_root: Path) -> Path | None:
    targets = sorted((sample_root / "gt").glob("*.txt"))
    return targets[0] if targets else None


@dataclass
class FVesselSample:
    """A single FVessel video sample with resolved AIS tracks."""

    root: Path
    video_path: Path
    ais_dir: Path
    start_ms: float
    fps: float

    def aligned_ais(
        self,
        num_frames: int,
        *,
        method: str = "linear",
        tolerance_ms: float = 500.0,
        max_gap_ms: float | None = None,
    ) -> list[dict[int, AISRecord]]:
        """Return per-frame AIS state for the first `num_frames` frames."""
        tracks = load_ais_tracks(self.ais_dir)
        frame_ts = frame_timestamps_ms(self.start_ms, num_frames, self.fps)
        return align_tracks_to_frames(
            tracks,
            frame_ts,
            method=method,
            tolerance_ms=tolerance_ms,
            max_gap_ms=max_gap_ms,
        )


def load_sample(root: str | Path, *, start_ms: float, fps: float) -> FVesselSample:
    """Resolve an FVessel sample directory into an `FVesselSample`.

    Args:
        root: Path to the sample directory (contains `ais/` and a `.mp4`).
        start_ms: Epoch timestamp (ms) of the first video frame. FVessel encodes
            the start time in the clip filename; parsing it is left to the caller
            for now (v1 will parse it automatically).
        fps: Source video frame rate.
    """
    root = Path(root)
    ais_dir = root / "ais"
    if not ais_dir.is_dir():
        raise FileNotFoundError(f"expected AIS directory at {ais_dir}")
    videos = sorted(root.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 found under {root}")
    return FVesselSample(
        root=root,
        video_path=videos[0],
        ais_dir=ais_dir,
        start_ms=float(start_ms),
        fps=float(fps),
    )

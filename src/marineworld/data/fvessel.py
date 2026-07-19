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

from dataclasses import dataclass
from pathlib import Path

from marineworld.data.alignment import (
    AISRecord,
    align_tracks_to_frames,
    frame_timestamps_ms,
    load_ais_tracks,
)

__all__ = ["FVesselSample", "load_sample"]


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

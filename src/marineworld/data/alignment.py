"""Temporal alignment of asynchronous AIS records to video frames (FVessel).

FVessel stores video and AIS on independent clocks: video frames are implied by
a start time + FPS, while AIS arrives as irregularly-timestamped records (one
CSV row per received message, millisecond epoch timestamps). To fuse the two we
must, for each video frame timestamp, estimate each vessel's AIS state.

This module provides:
  * `load_ais_tracks`  - parse FVessel AIS CSV(s) into per-MMSI tracks.
  * `AISTrack.query`   - estimate a vessel's state at an arbitrary timestamp via
                         nearest-neighbour or linear interpolation, honouring a
                         tolerance window and a max-gap (dropout) guard.
  * `align_tracks_to_frames` - map a whole clip's frame timestamps to per-vessel
                         AIS states, omitting vessels with no valid estimate.
  * `frame_timestamps_ms` - derive frame epoch timestamps from start + FPS.

Design choices worth noting:
  * Longitude/latitude/speed are interpolated linearly.
  * Course/heading are angular (degrees) and interpolated on the circle, so a
    350 deg -> 10 deg transition yields 0 deg, not 180 deg.
  * AIS heading uses 511 as the "not available" sentinel; such values are kept
    as NaN and never interpolated into a spurious angle.
  * A dropout longer than `max_gap_ms` between bracketing samples yields no
    estimate (None) rather than interpolating across a data hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "AISRecord",
    "AISTrack",
    "load_ais_tracks",
    "align_tracks_to_frames",
    "frame_timestamps_ms",
]

# AIS sentinel for "heading not available" (ITU-R M.1371).
HEADING_NA = 511.0

# FVessel AIS CSV columns.
_AIS_COLUMNS = [
    "Number",
    "MMSI",
    "Lon",
    "Lat",
    "Speed",
    "Course",
    "Heading",
    "Type",
    "Timestamp",
]


@dataclass(frozen=True)
class AISRecord:
    """A single (possibly interpolated) AIS state for one vessel.

    Angles are in degrees in [0, 360). `heading` may be NaN when unavailable.
    `interpolated` flags whether this record was synthesised between samples.
    """

    mmsi: int
    timestamp_ms: float
    lon: float
    lat: float
    speed: float
    course: float
    heading: float
    vtype: int
    interpolated: bool = False


def _circular_interp(a0: float, a1: float, w: float) -> float:
    """Interpolate between two angles (degrees) along the shortest arc.

    Args:
        a0: Angle at weight 0, in degrees.
        a1: Angle at weight 1, in degrees.
        w: Interpolation weight in [0, 1].

    Returns:
        Interpolated angle in [0, 360). Returns NaN if either input is NaN.
    """
    if np.isnan(a0) or np.isnan(a1):
        return float("nan")
    r0 = np.radians(a0)
    r1 = np.radians(a1)
    # Interpolate on the unit circle to respect wraparound.
    x = (1.0 - w) * np.cos(r0) + w * np.cos(r1)
    y = (1.0 - w) * np.sin(r0) + w * np.sin(r1)
    ang = np.degrees(np.arctan2(y, x))
    return float(ang % 360.0)


class AISTrack:
    """Time-ordered AIS samples for a single MMSI, supporting temporal query."""

    def __init__(self, mmsi: int, frame: pd.DataFrame) -> None:
        """Build a track from a per-MMSI DataFrame.

        Args:
            mmsi: The vessel MMSI identifier.
            frame: Rows for this MMSI with the FVessel AIS columns. Rows are
                sorted by timestamp and de-duplicated on timestamp (last wins).
        """
        self.mmsi = int(mmsi)
        df = frame.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
        self._ts = df["Timestamp"].to_numpy(dtype=np.float64)
        self._lon = df["Lon"].to_numpy(dtype=np.float64)
        self._lat = df["Lat"].to_numpy(dtype=np.float64)
        self._speed = df["Speed"].to_numpy(dtype=np.float64)
        self._course = df["Course"].to_numpy(dtype=np.float64)
        heading = df["Heading"].to_numpy(dtype=np.float64).copy()
        heading[heading == HEADING_NA] = np.nan
        self._heading = heading
        self._vtype = df["Type"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return int(self._ts.shape[0])

    @property
    def timestamps_ms(self) -> np.ndarray:
        """Sorted sample timestamps (epoch milliseconds)."""
        return self._ts

    def _record_at_index(self, i: int, t: float, *, interpolated: bool) -> AISRecord:
        return AISRecord(
            mmsi=self.mmsi,
            timestamp_ms=float(t),
            lon=float(self._lon[i]),
            lat=float(self._lat[i]),
            speed=float(self._speed[i]),
            course=float(self._course[i]),
            heading=float(self._heading[i]),
            vtype=int(self._vtype[i]),
            interpolated=interpolated,
        )

    def query(
        self,
        t_ms: float,
        *,
        method: str = "linear",
        tolerance_ms: float = 500.0,
        max_gap_ms: float | None = None,
    ) -> AISRecord | None:
        """Estimate the vessel state at time `t_ms`.

        Args:
            t_ms: Query timestamp (epoch milliseconds).
            method: "nearest" or "linear".
            tolerance_ms: For "nearest", the max allowed |t - sample|. For
                "linear" outside the sample range, the max allowed extrapolation
                distance to the nearest endpoint before returning None.
            max_gap_ms: For "linear" strictly between samples, the max allowed
                gap between the two bracketing samples. A larger gap is treated
                as a dropout and yields None. Defaults to `4 * tolerance_ms`.

        Returns:
            An `AISRecord`, or None if no valid estimate exists at `t_ms`.
        """
        if len(self) == 0:
            return None
        if method not in ("nearest", "linear"):
            raise ValueError(f"unknown method: {method!r}")

        ts = self._ts
        # Index of the first sample >= t.
        idx = int(np.searchsorted(ts, t_ms, side="left"))

        # Candidate neighbours bracketing t.
        left = idx - 1 if idx > 0 else None
        right = idx if idx < len(ts) else None
        # Exact hit: searchsorted 'left' places equal values at idx.
        if right is not None and ts[right] == t_ms:
            return self._record_at_index(right, t_ms, interpolated=False)

        if method == "nearest":
            best = self._nearest_index(t_ms, left, right)
            if best is None or abs(ts[best] - t_ms) > tolerance_ms:
                return None
            return self._record_at_index(best, t_ms, interpolated=False)

        # method == "linear"
        if left is None or right is None:
            # Outside sample range: allow small extrapolation to the endpoint.
            endpoint = right if left is None else left
            if abs(ts[endpoint] - t_ms) > tolerance_ms:
                return None
            return self._record_at_index(endpoint, t_ms, interpolated=False)

        t0, t1 = ts[left], ts[right]
        gap = t1 - t0
        limit = 4.0 * tolerance_ms if max_gap_ms is None else max_gap_ms
        if gap > limit:
            return None  # dropout: do not interpolate across a hole
        if gap <= 0:
            return self._record_at_index(left, t_ms, interpolated=False)

        w = (t_ms - t0) / gap
        return AISRecord(
            mmsi=self.mmsi,
            timestamp_ms=float(t_ms),
            lon=float((1 - w) * self._lon[left] + w * self._lon[right]),
            lat=float((1 - w) * self._lat[left] + w * self._lat[right]),
            speed=float((1 - w) * self._speed[left] + w * self._speed[right]),
            course=_circular_interp(self._course[left], self._course[right], w),
            heading=_circular_interp(self._heading[left], self._heading[right], w),
            # Vessel type is categorical; snap to the nearer sample.
            vtype=int(self._vtype[left] if w < 0.5 else self._vtype[right]),
            interpolated=True,
        )

    def _nearest_index(self, t_ms: float, left: int | None, right: int | None) -> int | None:
        if left is None:
            return right
        if right is None:
            return left
        return left if (t_ms - self._ts[left]) <= (self._ts[right] - t_ms) else right


def load_ais_tracks(path: str | Path) -> dict[int, AISTrack]:
    """Load FVessel AIS data into per-MMSI tracks.

    Args:
        path: A single AIS CSV file, or a directory containing many CSV files
            (FVessel stores one CSV per received window). All rows are pooled
            and grouped by MMSI.

    Returns:
        Mapping from MMSI to `AISTrack`.
    """
    path = Path(path)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"no CSV files found under {path}")
        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)

    missing = set(_AIS_COLUMNS) - set(df.columns)
    if missing:
        # Try case-insensitive rename (FVessel CSVs use lowercase).
        col_map = {c.lower(): c for c in _AIS_COLUMNS if c != "Number"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        # Synthesize "Number" as row index if absent.
        if "Number" not in df.columns:
            df["Number"] = range(len(df))
        missing = set(_AIS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"AIS data missing columns: {sorted(missing)}")

    tracks: dict[int, AISTrack] = {}
    for mmsi, group in df.groupby("MMSI"):
        tracks[int(mmsi)] = AISTrack(int(mmsi), group)
    return tracks


def frame_timestamps_ms(start_ms: float, num_frames: int, fps: float) -> np.ndarray:
    """Compute epoch-millisecond timestamps for a clip's frames.

    Args:
        start_ms: Epoch timestamp of frame 0, in milliseconds.
        num_frames: Number of frames.
        fps: Frames per second of the source video.

    Returns:
        Array of shape (num_frames,) of frame timestamps in milliseconds.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")
    return start_ms + (np.arange(num_frames, dtype=np.float64) * (1000.0 / fps))


def align_tracks_to_frames(
    tracks: dict[int, AISTrack],
    frame_ts_ms: np.ndarray,
    *,
    method: str = "linear",
    tolerance_ms: float = 500.0,
    max_gap_ms: float | None = None,
) -> list[dict[int, AISRecord]]:
    """Align AIS tracks to a sequence of frame timestamps.

    Args:
        tracks: Per-MMSI tracks (see `load_ais_tracks`).
        frame_ts_ms: Frame timestamps (epoch milliseconds).
        method: "nearest" or "linear" (passed to `AISTrack.query`).
        tolerance_ms: Tolerance window (passed to `AISTrack.query`).
        max_gap_ms: Max interpolation gap (passed to `AISTrack.query`).

    Returns:
        One dict per frame mapping MMSI -> `AISRecord` for every vessel that has
        a valid estimate at that frame. Vessels without an estimate are omitted,
        so the dict is empty when no vessel is observable at that frame.
    """
    aligned: list[dict[int, AISRecord]] = []
    for t in np.asarray(frame_ts_ms, dtype=np.float64):
        frame_records: dict[int, AISRecord] = {}
        for mmsi, track in tracks.items():
            rec = track.query(
                float(t),
                method=method,
                tolerance_ms=tolerance_ms,
                max_gap_ms=max_gap_ms,
            )
            if rec is not None:
                frame_records[mmsi] = rec
        aligned.append(frame_records)
    return aligned

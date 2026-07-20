"""Behavioural tests for FVessel AIS<->frame temporal alignment.

These assert observable behaviour (chosen samples, interpolated values, angular
wraparound, tolerance/gap rejection) rather than merely that code runs.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from marineworld.data.alignment import (
    HEADING_NA,
    AISTrack,
    align_tracks_to_frames,
    frame_timestamps_ms,
    load_ais_tracks,
)


def _track(rows: list[dict], mmsi: int = 111) -> AISTrack:
    """Build an AISTrack from a list of partial row dicts (sensible defaults)."""
    defaults = {
        "Number": 0,
        "MMSI": mmsi,
        "Lon": 0.0,
        "Lat": 0.0,
        "Speed": 0.0,
        "Course": 0.0,
        "Heading": 0.0,
        "Type": 70,
    }
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    return AISTrack(mmsi, df)


def test_exact_timestamp_returns_uninterpolated_sample():
    track = _track(
        [
            {"Timestamp": 1000, "Lon": 10.0},
            {"Timestamp": 2000, "Lon": 20.0},
        ]
    )
    rec = track.query(1000, method="linear")
    assert rec is not None
    assert rec.lon == pytest.approx(10.0)
    assert rec.interpolated is False


def test_nearest_picks_closer_sample():
    track = _track(
        [
            {"Timestamp": 1000, "Lon": 10.0},
            {"Timestamp": 2000, "Lon": 20.0},
        ]
    )
    # 1300 is closer to 1000 than to 2000.
    rec = track.query(1300, method="nearest", tolerance_ms=1000)
    assert rec is not None
    assert rec.lon == pytest.approx(10.0)
    # 1700 is closer to 2000.
    rec = track.query(1700, method="nearest", tolerance_ms=1000)
    assert rec.lon == pytest.approx(20.0)


def test_nearest_respects_tolerance():
    track = _track([{"Timestamp": 1000, "Lon": 10.0}])
    assert track.query(1600, method="nearest", tolerance_ms=500) is None
    assert track.query(1400, method="nearest", tolerance_ms=500) is not None


def test_linear_interpolates_midpoint():
    track = _track(
        [
            {"Timestamp": 1000, "Lon": 10.0, "Lat": 0.0, "Speed": 4.0},
            {"Timestamp": 2000, "Lon": 20.0, "Lat": 4.0, "Speed": 8.0},
        ]
    )
    rec = track.query(1500, method="linear")
    assert rec is not None
    assert rec.interpolated is True
    assert rec.lon == pytest.approx(15.0)
    assert rec.lat == pytest.approx(2.0)
    assert rec.speed == pytest.approx(6.0)


def test_linear_interpolation_is_weighted_not_just_midpoint():
    track = _track(
        [
            {"Timestamp": 1000, "Lon": 0.0},
            {"Timestamp": 2000, "Lon": 100.0},
        ]
    )
    rec = track.query(1250, method="linear")
    assert rec.lon == pytest.approx(25.0)


def test_course_interpolation_wraps_around_zero():
    # 350 deg -> 10 deg shortest arc crosses 0, midpoint should be 0 (=360).
    track = _track(
        [
            {"Timestamp": 1000, "Course": 350.0},
            {"Timestamp": 2000, "Course": 10.0},
        ]
    )
    rec = track.query(1500, method="linear")
    ang = rec.course
    # Accept 0 or 360 representation.
    assert min(ang, 360.0 - ang) == pytest.approx(0.0, abs=1e-6)


def test_naive_average_would_be_wrong_for_angles():
    # Guard against a linear-average regression: naive mean of 350 and 10 is 180.
    track = _track(
        [
            {"Timestamp": 1000, "Course": 350.0},
            {"Timestamp": 2000, "Course": 10.0},
        ]
    )
    rec = track.query(1500, method="linear")
    assert not (170.0 < rec.course < 190.0)


def test_heading_na_sentinel_becomes_nan():
    track = _track(
        [
            {"Timestamp": 1000, "Heading": HEADING_NA},
            {"Timestamp": 2000, "Heading": HEADING_NA},
        ]
    )
    rec = track.query(1500, method="linear")
    assert math.isnan(rec.heading)


def test_gap_larger_than_limit_returns_none():
    track = _track(
        [
            {"Timestamp": 1000, "Lon": 10.0},
            {"Timestamp": 9000, "Lon": 90.0},
        ]
    )
    # 8000ms gap exceeds max_gap_ms; should refuse to interpolate.
    assert track.query(5000, method="linear", max_gap_ms=2000) is None
    # With a permissive gap it interpolates.
    rec = track.query(5000, method="linear", max_gap_ms=10000)
    assert rec is not None
    assert rec.lon == pytest.approx(50.0)


def test_extrapolation_beyond_range_bounded_by_tolerance():
    track = _track([{"Timestamp": 1000, "Lon": 10.0}])
    # Before first sample, within tolerance -> clamp to endpoint.
    assert track.query(700, method="linear", tolerance_ms=500) is not None
    # Beyond tolerance -> None.
    assert track.query(200, method="linear", tolerance_ms=500) is None


def test_empty_track_returns_none():
    df = pd.DataFrame(
        columns=[
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
    )
    track = AISTrack(999, df)
    assert len(track) == 0
    assert track.query(1000) is None


def test_unsorted_input_is_sorted_internally():
    track = _track(
        [
            {"Timestamp": 2000, "Lon": 20.0},
            {"Timestamp": 1000, "Lon": 10.0},
        ]
    )
    assert list(track.timestamps_ms) == [1000.0, 2000.0]
    assert track.query(1500, method="linear").lon == pytest.approx(15.0)


def test_invalid_method_raises():
    track = _track([{"Timestamp": 1000}])
    with pytest.raises(ValueError):
        track.query(1000, method="cubic")


def test_frame_timestamps_ms_spacing():
    ts = frame_timestamps_ms(start_ms=1000, num_frames=4, fps=10.0)
    assert list(ts) == [1000.0, 1100.0, 1200.0, 1300.0]


def test_frame_timestamps_ms_rejects_bad_fps():
    with pytest.raises(ValueError):
        frame_timestamps_ms(0, 4, fps=0.0)


def test_load_and_align_end_to_end(tmp_path):
    # Two vessels, one dropping out mid-clip.
    df = pd.DataFrame(
        [
            {
                "Number": 0,
                "MMSI": 111,
                "Lon": 10.0,
                "Lat": 0.0,
                "Speed": 5.0,
                "Course": 90.0,
                "Heading": 90.0,
                "Type": 70,
                "Timestamp": 1000,
            },
            {
                "Number": 1,
                "MMSI": 111,
                "Lon": 12.0,
                "Lat": 0.0,
                "Speed": 5.0,
                "Course": 90.0,
                "Heading": 90.0,
                "Type": 70,
                "Timestamp": 2000,
            },
            {
                "Number": 2,
                "MMSI": 222,
                "Lon": 30.0,
                "Lat": 1.0,
                "Speed": 3.0,
                "Course": 180.0,
                "Heading": 511.0,
                "Type": 60,
                "Timestamp": 1000,
            },
        ]
    )
    csv = tmp_path / "ais.csv"
    df.to_csv(csv, index=False)

    tracks = load_ais_tracks(csv)
    assert set(tracks) == {111, 222}

    frame_ts = frame_timestamps_ms(start_ms=1000, num_frames=3, fps=2.0)  # 1000,1500,2000
    aligned = align_tracks_to_frames(tracks, frame_ts, method="linear", tolerance_ms=500)

    assert len(aligned) == 3
    # Vessel 111 present at all frames; interpolated lon at t=1500 is 11.0.
    assert aligned[1][111].lon == pytest.approx(11.0)
    # Vessel 222 only has one sample at t=1000; at t=2000 it is out of tolerance.
    assert 222 in aligned[0]
    assert 222 not in aligned[2]


def test_load_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"MMSI": 1, "Lon": 0.0}]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_ais_tracks(bad)

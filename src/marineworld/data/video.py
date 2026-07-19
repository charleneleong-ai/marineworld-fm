"""Lazy video metadata probes shared by dataset adapters."""

from __future__ import annotations

from pathlib import Path


def probe_video_frame_count(video: Path) -> int:
    """Read a source video's real frame count with optional Decord."""
    try:
        from decord import VideoReader
    except ImportError as error:
        raise RuntimeError(
            "frame-count probing requires optional dependency 'decord'; run on a supported "
            "platform or configure an explicit num_frames override"
        ) from error
    try:
        count = len(VideoReader(str(video)))
    except Exception as error:
        raise RuntimeError(f"could not probe frame count for {video}") from error
    if count <= 0:
        raise ValueError(f"video frame count must be positive for {video}, got {count}")
    return count

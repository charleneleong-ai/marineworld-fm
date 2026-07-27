"""Lazy, backend-neutral video metadata probes and frame decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marineworld.data.contracts import VideoRecord


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata required to construct canonical video records."""

    frame_count: int
    fps: float


class VideoBackendUnavailable(RuntimeError):
    """Raised when an optional video backend is not installed."""


class VideoDecodeError(RuntimeError):
    """Raised when a video backend cannot decode a requested frame set."""


def probe_video(path: Path) -> VideoMetadata:
    """Probe video metadata with Decord first and PyAV second."""
    for probe in (_decord_probe, _pyav_probe):
        try:
            return _validate_metadata(probe(path))
        except VideoBackendUnavailable:
            continue
    raise RuntimeError("video probing requires optional dependency 'decord' or 'PyAV'")


def count_decodable_frames(path: Path) -> int:
    """Count frames by decoding the stream once, ignoring container metadata."""
    stat = path.stat()
    return _cached_decodable_frames(path, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=None)
def _cached_decodable_frames(path: Path, size: int, modified_ns: int) -> int:
    del size, modified_ns
    av = _import_pyav()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for _ in container.decode(stream))


def _decord_can_seek(path: Path, index: int) -> bool:
    """Seek one frame with Decord, which is O(1) rather than a sequential walk."""
    try:
        from decord import VideoReader
    except ImportError:
        return False
    try:
        VideoReader(str(path)).get_batch([index])
    except Exception:
        return False
    return True


def probe_video_frame_count(path: Path) -> int:
    """Return the number of frames that actually decode.

    Container metadata overreports damaged or truncated H.264 streams, so trusting
    it yields clips over frames that do not exist.

    The reported tail is checked with a Decord seek, which is cheap. Falling back to
    PyAV here would cost a full sequential walk *before* counting, so a damaged video
    would pay two passes instead of one. A Decord failure therefore only means "count
    it properly", never "this frame is unreachable" -- the count is the authority.
    """
    reported = probe_video(path).frame_count
    if _decord_can_seek(path, reported - 1):
        return reported
    return count_decodable_frames(path)


def probe_video_fps(path: Path) -> float:
    """Return a source video's positive average FPS."""
    return probe_video(path).fps


def decode_video(record: VideoRecord, frame_indices: tuple[int, ...]) -> torch.Tensor:
    """Decode indexed RGB frames with Decord first and PyAV second."""
    failures: list[tuple[str, RuntimeError]] = []
    for backend, decode in (("decord", _decord_decode), ("PyAV", _pyav_decode)):
        try:
            return decode(record, frame_indices)
        except (VideoBackendUnavailable, VideoDecodeError) as error:
            failures.append((backend, error))
    details = "; ".join(f"{backend}: {error}" for backend, error in failures)
    raise RuntimeError(
        f"video decoding failed with all supported backends ({details})"
    ) from failures[-1][1]


def _validate_metadata(metadata: VideoMetadata) -> VideoMetadata:
    if metadata.frame_count <= 0 or not math.isfinite(metadata.fps) or metadata.fps <= 0:
        raise ValueError("video frame count and FPS must be positive")
    return metadata


def _decord_probe(path: Path) -> VideoMetadata:
    try:
        from decord import VideoReader
    except ImportError as error:
        raise VideoBackendUnavailable("decord is unavailable") from error
    try:
        reader = VideoReader(str(path))
        return VideoMetadata(frame_count=len(reader), fps=float(reader.get_avg_fps()))
    except Exception as error:
        raise RuntimeError("could not probe video metadata") from error


def _decord_decode(record: VideoRecord, frame_indices: tuple[int, ...]) -> torch.Tensor:
    try:
        from decord import VideoReader
    except ImportError as error:
        raise VideoBackendUnavailable("decord is unavailable") from error
    try:
        frames = VideoReader(str(record.video_path)).get_batch(list(frame_indices)).asnumpy()
    except Exception as error:
        raise VideoDecodeError(f"could not decode frames for record {record.id}") from error
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


def _pyav_probe(path: Path) -> VideoMetadata:
    av = _import_pyav()
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
            frame_count = int(stream.frames or 0)
            if frame_count <= 0:
                frame_count = sum(1 for _ in container.decode(stream))
    except Exception as error:
        raise RuntimeError("could not probe video metadata") from error
    return VideoMetadata(frame_count=frame_count, fps=fps)


def _pyav_decode(record: VideoRecord, frame_indices: tuple[int, ...]) -> torch.Tensor:
    if not frame_indices:
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8)
    if min(frame_indices) < 0:
        raise ValueError("frame indices must be non-negative")
    av = _import_pyav()
    requested = set(frame_indices)
    decoded: dict[int, torch.Tensor] = {}
    try:
        with av.open(str(record.video_path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in requested:
                    array = np.asarray(frame.to_ndarray(format="rgb24"))
                    decoded[index] = torch.from_numpy(array.copy()).permute(2, 0, 1)
                if index >= max(requested):
                    break
    except Exception as error:
        raise VideoDecodeError(f"could not decode frames for record {record.id}") from error
    missing = requested.difference(decoded)
    if missing:
        raise VideoDecodeError(f"record {record.id} is missing requested frames")
    return torch.stack([decoded[index] for index in frame_indices])


def _import_pyav() -> Any:
    try:
        import av
    except ImportError as error:
        raise VideoBackendUnavailable("PyAV is unavailable") from error
    return av

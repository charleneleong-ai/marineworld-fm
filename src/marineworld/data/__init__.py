"""Dataset adapters and multimodal alignment utilities."""

from marineworld.data.alignment import (
    AISRecord,
    AISTrack,
    align_tracks_to_frames,
    frame_timestamps_ms,
    load_ais_tracks,
)
from marineworld.data.contracts import DatasetManifest, FrameTargets, JSONValue, VideoRecord
from marineworld.data.manifest import manifest_checksum, validate_manifest

__all__ = [
    "AISRecord",
    "AISTrack",
    "DatasetManifest",
    "FrameTargets",
    "JSONValue",
    "VideoRecord",
    "align_tracks_to_frames",
    "frame_timestamps_ms",
    "load_ais_tracks",
    "manifest_checksum",
    "validate_manifest",
]

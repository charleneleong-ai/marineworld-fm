"""Deterministic local-only adapter used by smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord

_SPLITS: tuple[Literal["train", "val", "test"], ...] = ("train", "val")


@dataclass(frozen=True)
class SyntheticAdapter:
    """Materialize placeholder videos for decoders that generate frames in memory."""

    version: str = "v1"
    fps: float = 10.0
    num_frames: int = 16
    num_videos: int = 2

    def build_manifest(self, root: Path) -> DatasetManifest:
        root.mkdir(parents=True, exist_ok=True)
        records = tuple(
            self._record(root, _SPLITS[index % len(_SPLITS)], index // len(_SPLITS))
            for index in range(self.num_videos)
        )
        return DatasetManifest("synthetic", self.version, "MIT", records, access="public")

    def load_targets(self, record: VideoRecord) -> tuple[FrameTargets, ...]:
        return ()

    def _record(
        self, root: Path, split: Literal["train", "val", "test"], index: int
    ) -> VideoRecord:
        record_id = f"{split}-{index}"
        marker = root / f"{record_id}.mp4"
        marker.touch(exist_ok=True)
        return VideoRecord(
            id=record_id,
            dataset="synthetic",
            video_path=marker,
            split=split,
            source="generated",
            fps=self.fps,
            num_frames=self.num_frames,
        )

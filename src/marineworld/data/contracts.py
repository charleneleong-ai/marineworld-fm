"""Immutable data contracts shared by dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

import numpy as np

JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class VideoRecord:
    id: str
    dataset: str
    video_path: Path
    split: Literal["train", "val", "test"]
    source: str
    fps: float
    num_frames: int
    annotation_path: Path | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameTargets:
    frame_index: int
    boxes_xyxy: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray | None = None


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    license: str
    records: tuple[VideoRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))

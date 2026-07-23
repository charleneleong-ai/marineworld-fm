"""Singapore Maritime Dataset (SMD / SMD-Plus) video and ObjectGT adapter.

The explicit ``smd_objectgt_mat`` format follows the original distribution's
``ObjectGT/<video_stem>_ObjectGT.mat`` convention. Its ``structXML`` array has
one element per zero-based frame, with ``BB`` rows in ``x, y, width, height``
form and matching integer ``Object`` labels. Vessel classes 1 and 3--7 are
retained; invalid class 0, buoy class 2, and non-vessel/other classes 8--10 are
excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from marineworld.data.adapters import resolve_frame_count
from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.video import probe_video_frame_count

__all__ = ["SMDAdapter", "SMDVideo"]

SMDAnnotationFormat = Literal["smd_objectgt_mat"]
SMD_VESSEL_CLASSES = {
    1: "ferry",
    3: "vessel_ship",
    4: "speed_boat",
    5: "boat",
    6: "kayak",
    7: "sail_boat",
}
OBJECTGT_SUFFIX = "_ObjectGT.mat"
OBJECTGT_FIELDS = frozenset({"BB", "Object", "Motion", "Distance"})


@dataclass(frozen=True)
class SMDVideo:
    """An SMD video with its source split (onshore / onboard / nir)."""

    path: Path
    source: str  # one of: "onshore", "onboard", "nir"


@dataclass(frozen=True)
class SMDAdapter:
    """Adapt SMD videos while retaining each source subset as native metadata."""

    version: str = "v1"
    fps: float = 30.0
    num_frames: int | None = None
    annotation_format: SMDAnnotationFormat = "smd_objectgt_mat"
    frame_count_probe: Callable[[Path], int] = probe_video_frame_count

    def build_manifest(self, root: Path) -> DatasetManifest:
        self._validate_annotation_format()
        videos = SMDAdapter.list_videos(root)
        SMDAdapter.validate_objectgt_pairing(videos)
        records = tuple(
            VideoRecord(
                id=video.path.relative_to(root).with_suffix("").as_posix(),
                dataset="smd",
                video_path=video.path,
                split="train",
                source=video.source,
                fps=self.fps,
                num_frames=resolve_frame_count(
                    self.num_frames, video.path, self.frame_count_probe, "SMD"
                ),
                annotation_path=SMDAdapter.objectgt_path(video.path),
                metadata={
                    "native_source": video.source,
                    "annotation_format": self.annotation_format,
                },
            )
            for video in videos
        )
        return DatasetManifest(
            "smd",
            self.version,
            "research-only; see dataset terms",
            records,
            access="restricted",
            label_mapping={str(key): value for key, value in SMD_VESSEL_CLASSES.items()},
            native_labels=tuple(f"{key}:{value}" for key, value in SMD_VESSEL_CLASSES.items()),
        )

    def load_targets(self, record: VideoRecord) -> tuple[FrameTargets, ...]:
        self._validate_annotation_format()
        if record.annotation_path is None:
            return ()
        targets = SMDAdapter.load_objectgt(record.annotation_path)
        if len(targets) != record.num_frames:
            raise ValueError(
                f"SMD ObjectGT {record.annotation_path} contains {len(targets)} frames "
                f"but video manifest declares {record.num_frames}"
            )
        return targets

    def _validate_annotation_format(self) -> None:
        if self.annotation_format != "smd_objectgt_mat":
            raise ValueError(
                "unsupported SMD annotation_format "
                f"{self.annotation_format!r}; expected 'smd_objectgt_mat'"
            )

    @staticmethod
    def list_videos(root: str | Path) -> list[SMDVideo]:
        """Enumerate SMD videos under `root`, tagging each with its source subset.

        The SMD distribution groups videos by capture condition; we preserve that
        grouping because it defines the domain-shift eval splits in the plan.
        """
        root = Path(root)
        sources = {
            "onshore": ["VIS_Onshore", "onshore"],
            "onboard": ["VIS_Onboard", "onboard"],
            "nir": ["NIR", "nir"],
        }
        videos: list[SMDVideo] = []
        for source, dirnames in sources.items():
            for dirname in dirnames:
                for path in (
                    sorted((root / dirname).glob("*.avi")) if (root / dirname).is_dir() else []
                ):
                    videos.append(SMDVideo(path=path, source=source))
        return videos

    @staticmethod
    def objectgt_path(video: Path) -> Path | None:
        annotation = video.parent / "ObjectGT" / f"{video.stem}{OBJECTGT_SUFFIX}"
        return annotation if annotation.is_file() else None

    @staticmethod
    def validate_objectgt_pairing(videos: list[SMDVideo]) -> None:
        for directory in {video.path.parent for video in videos}:
            video_stems = {video.path.stem for video in videos if video.path.parent == directory}
            for annotation in sorted((directory / "ObjectGT").glob(f"*{OBJECTGT_SUFFIX}")):
                stem = annotation.name.removesuffix(OBJECTGT_SUFFIX)
                if stem not in video_stems:
                    raise ValueError(
                        f"SMD annotation {annotation} does not pair with an SMD video named "
                        f"{stem}.avi in {directory}"
                    )

    @staticmethod
    def load_objectgt(path: Path) -> tuple[FrameTargets, ...]:
        try:
            from scipy.io import loadmat
        except ImportError as error:
            raise RuntimeError(
                "SMD ObjectGT parsing requires scipy; install the 'train' dependencies"
            ) from error

        try:
            payload = loadmat(path)
        except NotImplementedError as error:
            raise ValueError(
                f"unsupported SMD ObjectGT MAT encoding in {path}; expected MATLAB v5/v7"
            ) from error
        if "structXML" not in payload:
            raise ValueError(f"SMD ObjectGT {path} is missing required root 'structXML'")
        frames = np.asarray(payload["structXML"])
        fields = set(frames.dtype.names or ())
        if not OBJECTGT_FIELDS.issubset(fields):
            missing = sorted(OBJECTGT_FIELDS - fields)
            raise ValueError(f"SMD ObjectGT {path} is missing required fields: {missing}")
        if frames.ndim != 2 or frames.shape[0] != 1:
            raise ValueError(
                f"SMD ObjectGT {path} structXML must be shaped [1, frames], got {frames.shape}"
            )
        return tuple(
            SMDAdapter.parse_objectgt_frame(path, index, frame)
            for index, frame in enumerate(frames.reshape(-1))
        )

    @staticmethod
    def parse_objectgt_frame(path: Path, frame_index: int, frame: np.void) -> FrameTargets:
        boxes_xywh = np.asarray(frame["BB"], dtype=np.float64)
        classes_raw = np.asarray(frame["Object"], dtype=np.float64).reshape(-1)
        if boxes_xywh.size == 0:
            boxes_xywh = np.empty((0, 4), dtype=np.float64)
        if boxes_xywh.ndim != 2 or boxes_xywh.shape[1] != 4:
            raise ValueError(f"SMD ObjectGT {path} frame {frame_index} BB must have four columns")
        if len(boxes_xywh) != len(classes_raw):
            raise ValueError(
                f"SMD ObjectGT {path} frame {frame_index} has {len(boxes_xywh)} boxes "
                f"but {len(classes_raw)} class labels"
            )
        if not np.isfinite(boxes_xywh).all():
            raise ValueError(f"SMD ObjectGT {path} frame {frame_index} has non-finite boxes")
        if (
            not np.isfinite(classes_raw).all()
            or not np.equal(classes_raw, classes_raw.astype(int)).all()
        ):
            raise ValueError(
                f"SMD ObjectGT {path} frame {frame_index} has non-integer class labels"
            )
        classes = classes_raw.astype(np.int64)
        if np.any((classes < 0) | (classes > 10)):
            raise ValueError(f"SMD ObjectGT {path} frame {frame_index} has class outside 0..10")

        keep = np.isin(classes, tuple(SMD_VESSEL_CLASSES))
        boxes_xywh = boxes_xywh[keep]
        if np.any(boxes_xywh[:, 2:] <= 0):
            raise ValueError(
                f"SMD ObjectGT {path} frame {frame_index} has a non-positive vessel box"
            )
        boxes_xyxy = boxes_xywh.astype(np.float32, copy=True)
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
        return FrameTargets(
            frame_index=frame_index,
            boxes_xyxy=boxes_xyxy,
            class_ids=classes[keep],
        )

"""Singapore Maritime Dataset (SMD / SMD-Plus) adapter.

Placeholder for v1: SMD is video-only (no AIS), used for self-supervised video
pretraining and few-shot detection/tracking evaluation. Frame extraction and the
torch `Dataset` wrapper land in v1; this module currently only defines the
common sample contract so `splits.py` can reason about SMD videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["SMDVideo", "list_videos"]


@dataclass(frozen=True)
class SMDVideo:
    """An SMD video with its source split (onshore / onboard / nir)."""

    path: Path
    source: str  # one of: "onshore", "onboard", "nir"


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
            for path in sorted((root / dirname).glob("*.avi")) if (root / dirname).is_dir() else []:
                videos.append(SMDVideo(path=path, source=source))
    return videos

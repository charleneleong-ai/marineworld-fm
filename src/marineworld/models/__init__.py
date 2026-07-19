"""Model components.

v1 adds:
  * `video_vit.py`  - plain ViT video encoder.
  * `videomae.py`   - VideoMAE tube-masking reconstruction head for SSL.
v2 adds an AIS encoder + fusion transformer. Kept empty in v0 by design.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marineworld.models.videomae import build_videomae, encode_video, tube_mask

__all__ = ["build_videomae", "encode_video", "tube_mask"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from marineworld.models import videomae

    return getattr(videomae, name)

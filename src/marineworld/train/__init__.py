"""Training entrypoints.

v1 adds a Lightning module for VideoMAE-style masked pretraining plus a
few-shot fine-tuning entrypoint. Every entrypoint must call
`marineworld.utils.seed.seed_everything()` before constructing models/data.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marineworld.train.pretrain import build_trainer, run_pretraining

__all__ = ["build_trainer", "run_pretraining"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from marineworld.train import pretrain

    return getattr(pretrain, name)

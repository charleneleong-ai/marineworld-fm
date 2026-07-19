"""Reproducibility helpers.

Per project rule, all random seeds default to 42 across torch, cuda, numpy and
the stdlib `random` module. `torch` is imported lazily so this module (and the
data/alignment utilities that transitively import `marineworld.utils`) works in
a core-only install without the heavy ML stack.
"""

from __future__ import annotations

import os
import random

import numpy as np

from marineworld import SEED

__all__ = ["seed_everything"]


def seed_everything(seed: int = SEED, *, deterministic: bool = True) -> int:
    """Seed all RNGs used in the project.

    Args:
        seed: The seed value. Defaults to the project-wide ``SEED`` (42).
        deterministic: If True, request deterministic cuDNN/torch algorithms.

    Returns:
        The seed that was set (useful for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

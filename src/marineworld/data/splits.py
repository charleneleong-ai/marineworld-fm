"""Deterministic dataset splitting.

Splits are made **by video**, never by frame, so that frames from one video
cannot leak across train/val/test. Splitting is seeded (42) and stable given the
same input identifiers, and the resulting split is meant to be committed to the
repo for reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from marineworld import SEED

__all__ = ["Split", "split_by_video"]


@dataclass(frozen=True)
class Split:
    """A train/val/test partition of video identifiers."""

    train: list[str]
    val: list[str]
    test: list[str]


def split_by_video(
    video_ids: list[str],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> Split:
    """Partition video identifiers into train/val/test by whole video.

    Args:
        video_ids: Unique video identifiers.
        val_frac: Fraction assigned to validation.
        test_frac: Fraction assigned to test.
        seed: RNG seed (defaults to project SEED=42).

    Returns:
        A `Split`. Fractions are applied to the sorted, de-duplicated id list so
        the result is deterministic and independent of input ordering.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1.0:
        raise ValueError("val_frac and test_frac must be >= 0 and sum to < 1.0")

    ids = sorted(set(video_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = ids[:n_test]
    val = ids[n_test : n_test + n_val]
    train = ids[n_test + n_val :]
    return Split(train=sorted(train), val=sorted(val), test=sorted(test))

"""Deterministic dataset splitting.

Splits are made **by video**, never by frame, so that frames from one video
cannot leak across train/val/test. Splitting is seeded (42) and stable given the
same input identifiers, and the resulting split is meant to be committed to the
repo for reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from marineworld import SEED
from marineworld.data.contracts import DatasetManifest
from marineworld.data.manifest import validate_manifest

__all__ = ["Split", "SplitUnavailableError", "prepare_manifest_splits", "split_by_video"]


class SplitUnavailableError(ValueError):
    """A manifest cannot provide disjoint training and validation videos."""


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


def prepare_manifest_splits(
    manifest: DatasetManifest,
    *,
    val_frac: float | None,
    test_frac: float | None,
    seed: int,
) -> DatasetManifest:
    """Create deterministic whole-video splits and require train/validation data."""
    if not manifest.records:
        raise SplitUnavailableError("manifest contains no records")
    validate_manifest(manifest)

    if all(record.split == "train" for record in manifest.records):
        if val_frac is None or test_frac is None:
            raise SplitUnavailableError("manifest has no validation split or split policy")
        if len(manifest.records) < 2:
            raise SplitUnavailableError(
                "at least two videos are required for train and validation splits"
            )
        partition = split_by_video(
            [record.id for record in manifest.records],
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
        train_ids = list(partition.train)
        val_ids = list(partition.val)
        test_ids = list(partition.test)
        if not val_ids:
            source = train_ids if len(train_ids) > 1 else test_ids
            val_ids.append(source.pop())
        if not train_ids:
            source = test_ids if test_ids else val_ids
            train_ids.append(source.pop())
        split_by_id = {
            record_id: split
            for split, record_ids in (
                ("train", train_ids),
                ("val", val_ids),
                ("test", test_ids),
            )
            for record_id in record_ids
        }
        manifest = replace(
            manifest,
            records=tuple(
                replace(record, split=split_by_id[record.id]) for record in manifest.records
            ),
        )
        validate_manifest(manifest)

    splits = {record.split for record in manifest.records}
    if "train" not in splits:
        raise SplitUnavailableError("manifest contains no training records")
    if "val" not in splits:
        raise SplitUnavailableError("manifest contains no validation records")
    return manifest

"""Tests for deterministic by-video splitting and seeding."""

from __future__ import annotations

import pytest

from marineworld.data.splits import split_by_video
from marineworld.utils.seed import seed_everything


def test_split_is_deterministic_across_calls():
    ids = [f"vid_{i}" for i in range(20)]
    a = split_by_video(ids)
    b = split_by_video(ids)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_split_independent_of_input_order():
    ids = [f"vid_{i}" for i in range(20)]
    a = split_by_video(ids)
    b = split_by_video(list(reversed(ids)))
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_split_is_a_partition_with_no_leakage():
    ids = [f"vid_{i}" for i in range(20)]
    s = split_by_video(ids, val_frac=0.15, test_frac=0.15)
    combined = set(s.train) | set(s.val) | set(s.test)
    assert combined == set(ids)
    # No video appears in more than one split.
    assert len(s.train) + len(s.val) + len(s.test) == len(ids)
    assert not (set(s.train) & set(s.val))
    assert not (set(s.val) & set(s.test))
    assert not (set(s.train) & set(s.test))


def test_split_fraction_sizes_are_reasonable():
    ids = [f"vid_{i}" for i in range(100)]
    s = split_by_video(ids, val_frac=0.2, test_frac=0.1)
    assert len(s.val) == 20
    assert len(s.test) == 10
    assert len(s.train) == 70


def test_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        split_by_video(["a", "b"], val_frac=0.6, test_frac=0.6)


def test_seed_everything_returns_seed_and_seeds_random():
    import random

    assert seed_everything(42) == 42
    first = [random.random() for _ in range(3)]
    seed_everything(42)
    second = [random.random() for _ in range(3)]
    assert first == second

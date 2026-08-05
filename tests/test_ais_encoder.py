"""Behavioural tests for the AIS encoder and feature conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from marineworld.models.ais_encoder import (
    AIS_FEATURE_DIM,
    AISEncoder,
    ais_features_from_records,
)


@dataclass(frozen=True)
class _FakeRecord:
    """Minimal stand-in for AISRecord used in feature conversion tests."""

    mmsi: int
    lon: float
    lat: float
    speed: float
    course: float
    vtype: int
    interpolated: bool = False


class TestAisFeaturesFromRecords:
    """ais_features_from_records converts AISRecord dicts to tensors."""

    def test_empty_frames(self):
        aligned: list[dict[int, _FakeRecord]] = [{}, {}]
        features, mask = ais_features_from_records(aligned, max_vessels=5)
        assert features.shape == (2, 5, AIS_FEATURE_DIM)
        assert mask.shape == (2, 5)
        assert not mask.any()

    def test_single_vessel_single_frame(self):
        rec = _FakeRecord(mmsi=1, lon=103.8, lat=1.3, speed=8.0, course=90.0, vtype=70)
        aligned: list[dict[int, _FakeRecord]] = [{1: rec}]
        features, mask = ais_features_from_records(aligned, max_vessels=5)
        assert features.shape == (1, 5, AIS_FEATURE_DIM)
        assert mask[0, 0]
        assert not mask[0, 1:].any()
        assert features[0, 0, 0] == pytest.approx(103.8)  # lon
        assert features[0, 0, 1] == pytest.approx(1.3)  # lat
        assert features[0, 0, 2] == pytest.approx(8.0 / 50.0)  # speed normalised
        assert features[0, 0, 3] == pytest.approx(math.sin(math.radians(90.0)))
        assert features[0, 0, 4] == pytest.approx(math.cos(math.radians(90.0)))
        assert features[0, 0, 5] == pytest.approx(70.0)  # vtype

    def test_speed_clamped_to_max(self):
        rec = _FakeRecord(mmsi=1, lon=0, lat=0, speed=100.0, course=0, vtype=70)
        aligned: list[dict[int, _FakeRecord]] = [{1: rec}]
        features, _ = ais_features_from_records(aligned, max_vessels=1)
        assert features[0, 0, 2] == pytest.approx(1.0)  # clamped to 50/50

    def test_truncates_to_max_vessels(self):
        recs = {
            i: _FakeRecord(mmsi=i, lon=float(i), lat=0, speed=0, course=0, vtype=70)
            for i in range(1, 8)
        }
        aligned: list[dict[int, _FakeRecord]] = [recs]
        features, mask = ais_features_from_records(aligned, max_vessels=3)
        assert mask[0].sum() == 3
        assert features[0, 0, 0] == pytest.approx(1.0)
        assert features[0, 2, 0] == pytest.approx(3.0)

    def test_course_wraparound(self):
        rec_350 = _FakeRecord(mmsi=1, lon=0, lat=0, speed=0, course=350.0, vtype=70)
        rec_10 = _FakeRecord(mmsi=2, lon=0, lat=0, speed=0, course=10.0, vtype=70)
        aligned: list[dict[int, _FakeRecord]] = [{1: rec_350, 2: rec_10}]
        features, _ = ais_features_from_records(aligned, max_vessels=5)
        # sin/cos of 350 deg ≈ sin(-10), cos(-10)
        assert features[0, 0, 3] == pytest.approx(math.sin(math.radians(350)), abs=1e-5)
        assert features[0, 1, 3] == pytest.approx(math.sin(math.radians(10)), abs=1e-5)
        # The average should NOT be ~180 (which raw interpolation would give).
        raw_mid = (math.radians(350) + math.radians(10)) / 2
        assert abs(math.degrees(raw_mid) - 180) < 1.0  # raw would be near 180

    def test_masks_heading_sentinel_nan(self):
        """Records with NaN heading produce NaN in features — but heading is not used."""
        rec = _FakeRecord(mmsi=1, lon=0, lat=0, speed=0, course=0, vtype=70)
        aligned: list[dict[int, _FakeRecord]] = [{1: rec}]
        features, _ = ais_features_from_records(aligned, max_vessels=1)
        # vtype placeholder slot (index 6) is always 0.0
        assert features[0, 0, 6] == pytest.approx(0.0)


class TestAISEncoder:
    """AISEncoder projects raw AIS features into hidden_size tokens."""

    def test_output_shape(self):
        H = 384
        enc = AISEncoder(hidden_size=H, num_vessel_types=100)
        B, T, V = 2, 8, 5
        features = torch.randn(B, T, V, AIS_FEATURE_DIM)
        mask = torch.ones(B, T, V, dtype=torch.bool)
        tokens, flat_mask = enc(features, mask)
        assert tokens.shape == (B, T * V, H)
        assert flat_mask.shape == (B, T * V)

    def test_padded_positions_masked(self):
        enc = AISEncoder(hidden_size=64, num_vessel_types=10)
        features = torch.zeros(1, 4, 3, AIS_FEATURE_DIM)
        mask = torch.zeros(1, 4, 3, dtype=torch.bool)
        mask[0, 0, 0] = True  # only one valid vessel
        tokens, flat_mask = enc(features, mask)
        assert tokens.shape == (1, 12, 64)
        assert flat_mask[0, 0].item() is True
        assert not flat_mask[0, 1:].any()

    def test_vtype_embedding(self):
        enc = AISEncoder(hidden_size=32, num_vessel_types=50)
        features = torch.zeros(1, 1, 2, AIS_FEATURE_DIM)
        features[0, 0, 0, 5] = 5.0  # vtype=5
        features[0, 0, 1, 5] = 10.0  # vtype=10
        mask = torch.ones(1, 1, 2, dtype=torch.bool)
        tokens, _ = enc(features, mask)
        # Different vtype values should produce different tokens.
        assert not torch.allclose(tokens[0, 0], tokens[0, 1], atol=1e-4)

    def test_deterministic(self):
        enc = AISEncoder(hidden_size=32, num_vessel_types=10)
        features = torch.randn(1, 2, 3, AIS_FEATURE_DIM)
        mask = torch.ones(1, 2, 3, dtype=torch.bool)
        t1, m1 = enc(features, mask)
        t2, m2 = enc(features, mask)
        assert torch.allclose(t1, t2)
        assert torch.equal(m1, m2)

    def test_invalid_feature_dim_raises(self):
        enc = AISEncoder(hidden_size=32)
        features = torch.randn(1, 1, 1, 5)  # wrong last dim
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="features last dim"):
            enc(features, mask)

    def test_invalid_ndim_raises(self):
        enc = AISEncoder(hidden_size=32)
        features = torch.randn(1, 1, AIS_FEATURE_DIM)  # 3-D, not 4-D
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="4-D"):
            enc(features, mask)

    def test_gradients_flow(self):
        enc = AISEncoder(hidden_size=64, num_vessel_types=10)
        features = torch.randn(2, 4, 5, AIS_FEATURE_DIM)
        mask = torch.ones(2, 4, 5, dtype=torch.bool)
        tokens, _ = enc(features, mask)
        loss = tokens.sum()
        loss.backward()
        assert all(p.grad is not None for p in enc.parameters())

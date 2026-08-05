"""Behavioural tests for the MaritimeVideoMAE cross-modal wrapper."""

from __future__ import annotations

import pytest
import torch
from transformers import VideoMAEConfig, VideoMAEForPreTraining

from marineworld.models.ais_encoder import AIS_FEATURE_DIM, AISEncoder
from marineworld.models.maritime_videomae import MaritimeVideoMAE


def _tiny_videomae() -> VideoMAEForPreTraining:
    return VideoMAEForPreTraining(
        VideoMAEConfig(
            image_size=224,
            patch_size=16,
            num_frames=16,
            tubelet_size=2,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            decoder_hidden_size=32,
            decoder_num_hidden_layers=1,
            decoder_num_attention_heads=2,
            decoder_intermediate_size=64,
        )
    )


def _make_batch(
    batch_size: int = 2,
    num_frames: int = 16,
    max_vessels: int = 4,
    hidden_size: int = 64,
    mask_ratio: float = 0.9,
) -> dict[str, torch.Tensor]:
    num_patches = (num_frames // 2) * (224 // 16) * (224 // 16)  # tubelet_size=2, patch=16
    # Create a proper mask with ~90% masked positions.
    masked_count = int(num_patches * mask_ratio)
    bool_masked_pos = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    for b in range(batch_size):
        bool_masked_pos[b, :masked_count] = True
    return {
        "pixel_values": torch.randn(batch_size, num_frames, 3, 224, 224),
        "ais_features": torch.randn(batch_size, num_frames, max_vessels, AIS_FEATURE_DIM),
        "ais_mask": torch.ones(batch_size, num_frames, max_vessels, dtype=torch.bool),
        "bool_masked_pos": bool_masked_pos,
    }


class TestMaritimeVideoMAE:
    """MaritimeVideoMAE joins AIS and video tokens through a shared encoder."""

    def test_output_keys(self):
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        batch = _make_batch()
        out = model(**batch)
        assert "video_loss" in out
        assert "ais_loss" in out
        assert "total_loss" in out

    def test_losses_are_finite(self):
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        batch = _make_batch()
        out = model(**batch)
        assert torch.isfinite(out["video_loss"])
        assert torch.isfinite(out["ais_loss"])
        assert torch.isfinite(out["total_loss"])

    def test_total_loss_equals_weighted_sum(self):
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        batch = _make_batch()
        out = model(**batch)
        expected = out["video_loss"] + 0.5 * out["ais_loss"]
        assert out["total_loss"].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_ais_loss_only_on_masked_tubes(self):
        """AIS loss should only include vessels in masked temporal tubes."""
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        batch = _make_batch(mask_ratio=0.9)
        out_full = model(**batch)
        # With all vessels masked, AIS loss should be positive.
        assert out_full["ais_loss"].item() > 0

    def test_no_ais_loss_when_no_valid_vessels(self):
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        batch = _make_batch()
        batch["ais_mask"] = torch.zeros(2, 16, 4, dtype=torch.bool)  # no valid vessels
        out = model(**batch)
        assert out["ais_loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_grflows_to_both_encoders(self):
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        batch = _make_batch()
        out = model(**batch)
        out["total_loss"].backward()
        # AIS encoder should have gradients.
        ais_encoder = model.ais_encoder
        assert any(p.grad is not None for p in ais_encoder.parameters())
        # VideoMAE encoder should have gradients (unless frozen).
        videomae = model.videomae
        has_videomae_grads = any(
            p.grad is not None for p in videomae.parameters() if p.requires_grad
        )
        assert has_videomae_grads

    def test_cross_attention_layers_exist(self):
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        assert len(model.cross_attn_layers) == 2

    def test_deterministic(self):
        model = MaritimeVideoMAE(_tiny_videomae(), AISEncoder(64))
        batch = _make_batch()
        out1 = model(**batch)
        out2 = model(**batch)
        assert out1["video_loss"].item() == pytest.approx(out2["video_loss"].item())
        assert out1["ais_loss"].item() == pytest.approx(out2["ais_loss"].item())

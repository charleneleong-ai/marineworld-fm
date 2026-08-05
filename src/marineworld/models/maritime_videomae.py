"""Maritime VideoMAE: cross-modal video + AIS pretraining wrapper.

Wraps ``VideoMAEForPreTraining`` and adds AIS cross-attention after the
video encoder.  The video encoder runs its standard masked-reconstruction
forward pass unchanged.  AIS tokens attend to the encoded video tokens via
a lightweight cross-attention layer, then an AIS decoder predicts the
original features.

This design preserves VideoMAE's reconstruction objective exactly while
allowing AIS context to influence the learned representations.

Architecture::

    pixel_values ──► VideoMAE (encoder + decoder) ──► video_loss
                         │
                    encoder hidden states
                         │
    ais_features ──► AISEncoder ──► cross-attention(video_hidden) ──► ais_loss
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import VideoMAEForPreTraining

from marineworld.models.ais_encoder import AISEncoder

__all__ = ["MaritimeVideoMAE"]


class CrossAttentionBlock(nn.Module):
    """Single cross-attention layer: AIS queries attend to video keys/values."""

    def __init__(self, hidden_size: int, num_heads: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, ais_tokens: torch.Tensor, video_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ais_tokens: ``[B, T*V, H]`` AIS token embeddings.
            video_hidden: ``[B, N, H]`` video encoder hidden states.

        Returns:
            AIS tokens updated with video context, ``[B, T*V, H]``.
        """
        # Cross-attention: AIS queries, video keys/values.
        residual = ais_tokens
        ais_norm = self.norm1(ais_tokens)
        video_norm = self.norm2(video_hidden)
        attn_out, _ = self.cross_attn(ais_norm, video_norm, video_norm)
        ais_tokens = residual + attn_out
        # FFN.
        ais_tokens = ais_tokens + self.ffn(self.norm1(ais_tokens))
        return ais_tokens


class MaritimeVideoMAE(nn.Module):
    """VideoMAE with AIS cross-attention for cross-modal pretraining."""

    def __init__(
        self,
        videomae: VideoMAEForPreTraining,
        ais_encoder: AISEncoder,
        *,
        num_cross_attn_layers: int = 2,
        cross_attn_heads: int = 2,
    ) -> None:
        super().__init__()
        self.videomae = videomae
        self.ais_encoder = ais_encoder
        hidden_size = videomae.config.hidden_size

        # Cross-attention: AIS queries attend to video encoder output.
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_size, cross_attn_heads)
                for _ in range(num_cross_attn_layers)
            ]
        )

        # AIS reconstruction head: predicts original 6 features.
        self.ais_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 6),  # lon, lat, speed, course_sin, course_cos, vtype
        )

    @property
    def config(self) -> Any:
        return self.videomae.config

    def forward(
        self,
        pixel_values: torch.Tensor,
        ais_features: torch.Tensor,
        ais_mask: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with joint video + AIS reconstruction.

        Args:
            pixel_values: ``[B, T, C, H, W]`` video frames.
            ais_features: ``[B, T, V, 7]`` raw AIS features.
            ais_mask: ``[B, T, V]`` bool — True for valid vessels.
            bool_masked_pos: ``[B, num_video_tokens]`` bool — masked positions.

        Returns:
            dict with ``video_loss``, ``ais_loss``, ``total_loss``.
        """
        # 1. Run VideoMAE's standard forward (encoder + decoder + loss).
        videomae_out = self.videomae(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
        )
        video_loss = videomae_out.loss

        # 2. Get encoder hidden states for cross-attention.
        #    Re-run just the encoder to get the full hidden state sequence.
        video_embeds = self.videomae.videomae.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos
        )
        encoder_out = self.videomae.videomae.encoder(video_embeds)
        video_hidden = encoder_out.last_hidden_state  # [B, N, H]

        # 3. Encode AIS features.
        ais_tokens, flat_ais_mask = self.ais_encoder(ais_features, ais_mask)

        # 4. Cross-attention: AIS queries attend to video context.
        for layer in self.cross_attn_layers:
            ais_tokens = layer(ais_tokens, video_hidden)

        # 5. AIS reconstruction loss.
        ais_loss = self._ais_reconstruction_loss(
            ais_tokens, ais_features, flat_ais_mask, bool_masked_pos
        )

        return {
            "video_loss": video_loss,
            "ais_loss": ais_loss,
            "total_loss": video_loss + 0.5 * ais_loss,
        }

    def _ais_reconstruction_loss(
        self,
        ais_hidden: torch.Tensor,
        ais_features: torch.Tensor,
        flat_ais_mask: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss on AIS tokens for vessels in masked temporal tubes.

        Vessels in visible tubelets are excluded — the model should only
        reconstruct AIS when the corresponding video context is masked.
        """
        batch_size = ais_hidden.shape[0]
        num_frames = ais_features.shape[1]
        max_vessels = ais_features.shape[2]

        # Predict AIS features.
        pred = self.ais_decoder(ais_hidden)  # [B, T*V, 6]

        # Reshape targets.
        targets = ais_features[..., :6].view(batch_size, -1, 6)

        # Map tubelet mask to per-frame mask.
        tubelet_size = self.config.tubelet_size
        frame_mask = bool_masked_pos.repeat_interleave(tubelet_size, dim=1)
        frame_mask = frame_mask[:, :num_frames]

        # Expand to per-vessel: [B, T] → [B, T*V]
        frame_mask_expanded = frame_mask.unsqueeze(2).expand(-1, -1, max_vessels)
        frame_mask_flat = frame_mask_expanded.reshape(batch_size, -1)

        # Combined mask.
        combined_mask = flat_ais_mask & frame_mask_flat
        mask_weight = combined_mask.unsqueeze(-1).float()

        diff = (pred - targets) ** 2
        loss = (diff * mask_weight).sum() / mask_weight.sum().clamp(min=1)
        return loss

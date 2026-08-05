"""Encode per-frame AIS records into transformer-compatible tokens.

Each video frame may have zero or more vessels with AIS telemetry (lon, lat,
speed, course, vessel type).  This module maps variable-length per-vessel
features into fixed-size token embeddings that can be concatenated with
VideoMAE video tokens for cross-modal pretraining.

Design choices:
  * Course (angular) is represented as sin/cos to preserve circular continuity.
  * Heading is excluded for v2 — it is often the sentinel 511 (NaN).
  * lon/lat are normalised to per-clip bounds at the dataset level.
  * speed is clipped to [0, 50] knots and divided by 50.
  * vessel type uses a learned embedding projected to 1 dim.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

__all__ = ["AIS_FEATURE_DIM", "AISEncoder", "ais_features_from_records"]

# Features: lon, lat, speed, course_sin, course_cos, vtype_embed(1)
AIS_FEATURE_DIM = 7

# AIS fields extracted from alignment records.
AIS_FIELDS = ("lon", "lat", "speed", "course", "vtype")

# Speed upper bound (knots) — SOLAS max for all vessel classes.
_SPEED_MAX = 50.0


def ais_features_from_records(
    aligned: list[dict[int, Any]],
    *,
    max_vessels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert aligned AIS records to padded feature and mask tensors.

    Args:
        aligned: Per-frame dicts of MMSI -> AISRecord (from
            ``align_tracks_to_frames``).
        max_vessels: Maximum vessels per frame. Frames with more vessels are
            truncated; fewer vessels are zero-padded.

    Returns:
        features: ``[num_frames, max_vessels, AIS_FEATURE_DIM]`` float tensor.
        mask: ``[num_frames, max_vessels]`` bool tensor (True = valid vessel).
    """
    num_frames = len(aligned)
    features = torch.zeros(num_frames, max_vessels, AIS_FEATURE_DIM)
    mask = torch.zeros(num_frames, max_vessels, dtype=torch.bool)

    for frame_idx, frame_records in enumerate(aligned):
        for vessel_idx, (_, rec) in enumerate(sorted(frame_records.items())[:max_vessels]):
            course_rad = math.radians(rec.course)
            speed_norm = min(rec.speed, _SPEED_MAX) / _SPEED_MAX
            features[frame_idx, vessel_idx] = torch.tensor(
                [
                    rec.lon,
                    rec.lat,
                    speed_norm,
                    math.sin(course_rad),
                    math.cos(course_rad),
                    float(rec.vtype),
                    0.0,  # placeholder — vtype embedding is handled by the encoder
                ],
                dtype=torch.float32,
            )
            mask[frame_idx, vessel_idx] = True

    return features, mask


class AISEncoder(nn.Module):
    """Encode per-frame AIS records into transformer-compatible tokens.

    The encoder takes raw AIS features (lon, lat, speed, course_sin/cos,
    vtype) and projects them into the same hidden space as the VideoMAE
    video tokens, enabling direct concatenation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_vessel_types: int = 100,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            hidden_size: Must match the VideoMAE encoder hidden size.
            num_vessel_types: Number of distinct AIS vessel types to embed.
            dropout: Dropout rate in the MLP.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vessel_types = num_vessel_types
        # Vessel type embedding — +1 for padding (type 0 reserved).
        self.vtype_embedding = nn.Embedding(num_vessel_types + 1, 1)
        # Input: lon, lat, speed, course_sin, course_cos, vtype_embed
        self.feature_mlp = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.feature_mlp[0].weight)
        nn.init.zeros_(self.feature_mlp[0].bias)
        nn.init.xavier_uniform_(self.feature_mlp[3].weight)
        nn.init.zeros_(self.feature_mlp[3].bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode AIS features into tokens.

        Args:
            features: ``[batch, num_frames, max_vessels, AIS_FEATURE_DIM]``
                raw features (lon, lat, speed, course_sin, course_cos, vtype,
                placeholder).
            mask: ``[batch, num_frames, max_vessels]`` bool — True for valid
                vessels.

        Returns:
            tokens: ``[batch, num_frames * max_vessels, hidden_size]`` token
                embeddings.  Padded positions receive zeros.
            flat_mask: ``[batch, num_frames * max_vessels]`` bool — flattened
                validity mask.
        """
        if features.ndim != 4:
            raise ValueError(f"features must be 4-D, got {features.ndim}-D")
        if features.shape[-1] != AIS_FEATURE_DIM:
            raise ValueError(
                f"features last dim must be {AIS_FEATURE_DIM}, got {features.shape[-1]}"
            )
        batch, num_frames, max_vessels, _ = features.shape

        # Embed vessel type and project to 1-D, then concat with raw features.
        vtype = features[..., 5].long().clamp(0, self.num_vessel_types)
        vtype_emb = self.vtype_embedding(vtype)  # [B, T, V, 1]

        # Replace the placeholder slot with the vtype embedding.
        proj_input = torch.cat([features[..., :5], vtype_emb], dim=-1)  # [B, T, V, 6]

        tokens = self.feature_mlp(proj_input)  # [B, T, V, H]

        # Flatten spatial+temporal dims.
        tokens = tokens.view(batch, num_frames * max_vessels, self.hidden_size)
        flat_mask = mask.view(batch, num_frames * max_vessels)

        return tokens, flat_mask

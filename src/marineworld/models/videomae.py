"""VideoMAE construction, masking, and encoder extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from transformers import VideoMAEConfig, VideoMAEForPreTraining

__all__ = ["build_videomae", "encode_video", "pair", "tube_mask"]


def pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalise a scalar-or-pair config field to an explicit (height, width)."""
    return (value, value) if isinstance(value, int) else (value[0], value[1])


def tube_mask(
    batch_size: int,
    sequence_length: int,
    mask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return independently shuffled masks with an exact masked-token count."""
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    if not 0 <= mask_ratio <= 1:
        raise ValueError("mask_ratio must be between 0 and 1")

    masked = round(sequence_length * mask_ratio)
    noise = torch.rand(batch_size, sequence_length, generator=generator)
    order = noise.argsort(dim=1)
    mask = torch.zeros_like(noise, dtype=torch.bool)
    return mask.scatter(1, order[:, :masked], True)


def build_videomae(config: Mapping[str, Any]) -> VideoMAEForPreTraining:
    """Build a random VideoMAE or load an explicitly declared reference checkpoint."""
    values = dict(config)
    reference_checkpoint = values.pop("reference_checkpoint", None)
    if reference_checkpoint:
        return VideoMAEForPreTraining.from_pretrained(str(reference_checkpoint))

    return VideoMAEForPreTraining(VideoMAEConfig(**_model_kwargs(values)))


def encode_video(model: VideoMAEForPreTraining, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return encoder tokens shaped ``(batch, tokens, hidden_size)``."""
    return model.videomae(pixel_values=pixel_values).last_hidden_state


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"backbone", "mask_ratio", "name", "reference_checkpoint", "seed"}
    }

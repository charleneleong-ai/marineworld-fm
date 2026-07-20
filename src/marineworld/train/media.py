"""Tracking-independent VideoMAE reconstruction preview helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MediaPreview:
    """Display-ready VideoMAE input and reconstruction images."""

    input_grid: np.ndarray
    reconstruction_panel: np.ndarray
    caption: str


def build_media_preview(
    *,
    pixel_values: torch.Tensor,
    bool_masked_pos: torch.Tensor,
    logits: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    patch_size: Sequence[int],
    tubelet_size: int,
    max_frames: int,
    dataset: str,
    source: str,
    norm_pix_loss: bool = False,
) -> MediaPreview:
    """Build bounded RGB previews from one normalized VideoMAE batch.

    ``norm_pix_loss`` must match the VideoMAE model configuration.
    """
    patch_size, tubelet_size, max_frames = _validate_preview_config(
        patch_size, tubelet_size, max_frames
    )

    original = denormalize_video(pixel_values, mean, std)
    reconstruction, pixel_mask = _complete_reconstruction(
        original, bool_masked_pos, logits, patch_size, tubelet_size, norm_pix_loss
    )
    completed = reconstruction.clamp(0, 1)
    masked = original.masked_fill(pixel_mask, 0.5)
    error = (original - completed).abs().mean(dim=2, keepdim=True).expand_as(original)

    indices = _frame_indices(original.shape[1], max_frames).to(original.device)
    originals = original[0, indices]
    input_grid = _to_hwc(_horizontal_grid(originals))
    rows = torch.cat(
        (originals, masked[0, indices], completed[0, indices], error[0, indices]), dim=-1
    )
    panel = _to_hwc(torch.cat(tuple(rows), dim=-2))
    return MediaPreview(input_grid, panel, f"dataset={dataset} source={source}")


def _validate_preview_config(
    patch_size: object, tubelet_size: object, max_frames: object
) -> tuple[tuple[int, int], int, int]:
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or not 1 <= max_frames <= 4:
        raise ValueError("max_frames must be an integer from 1 to 4")
    if (
        not isinstance(patch_size, Sequence)
        or isinstance(patch_size, (str, bytes))
        or len(patch_size) != 2
    ):
        raise ValueError("patch_size must be a two-item sequence")
    patch_height, patch_width = patch_size
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in patch_size
    ):
        raise ValueError("patch_size entries must be positive integers")
    if not isinstance(tubelet_size, int) or isinstance(tubelet_size, bool) or tubelet_size < 1:
        raise ValueError("tubelet_size must be a positive integer")
    return (patch_height, patch_width), tubelet_size, max_frames


def denormalize_video(
    pixel_values: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Convert normalized video values to clamped display-space RGB."""
    _validate_normalization(pixel_values, mean, std)
    mean_tensor = pixel_values.new_tensor(mean).view(1, 1, -1, 1, 1)
    std_tensor = pixel_values.new_tensor(std).view(1, 1, -1, 1, 1)
    return (pixel_values * std_tensor + mean_tensor).clamp(0, 1)


def _complete_reconstruction(
    pixel_values: torch.Tensor,
    bool_masked_pos: torch.Tensor,
    logits: torch.Tensor,
    patch_size: tuple[int, int],
    tubelet_size: int,
    norm_pix_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill display-space masked patches and expand their token mask to pixels."""
    batch, frames, channels, height, width = _validate_shapes(
        pixel_values, bool_masked_pos, logits, patch_size, tubelet_size
    )
    patch_height, patch_width = patch_size
    patchified = _patchify(pixel_values, patch_size, tubelet_size)
    completed = patchified.clone()
    decoded = (
        _restore_patch_statistics(
            logits, patchified[bool_masked_pos].reshape_as(logits), tubelet_size, patch_size
        )
        if norm_pix_loss
        else logits
    )
    for batch_index, mask in enumerate(bool_masked_pos):
        completed[batch_index, mask] = decoded[batch_index]

    reconstruction = _unpatchify(
        completed, batch, frames, channels, height, width, patch_size, tubelet_size
    )
    token_mask = bool_masked_pos.reshape(
        batch, frames // tubelet_size, height // patch_height, width // patch_width
    )
    pixel_mask = (
        token_mask.repeat_interleave(tubelet_size, dim=1)
        .repeat_interleave(patch_height, dim=2)
        .repeat_interleave(patch_width, dim=3)
        .unsqueeze(2)
    )
    return reconstruction, pixel_mask


def _restore_patch_statistics(
    logits: torch.Tensor,
    original_patches: torch.Tensor,
    tubelet_size: int,
    patch_size: tuple[int, int],
) -> torch.Tensor:
    """Map VideoMAE's patch-normalized decoder output back to display RGB."""
    patch_height, patch_width = patch_size
    channels = original_patches.shape[-1] // (tubelet_size * patch_height * patch_width)
    elements = tubelet_size * patch_height * patch_width
    patches = original_patches.reshape(*original_patches.shape[:2], elements, channels)
    mean = patches.mean(dim=-2, keepdim=True)
    std = patches.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
    return (logits.reshape(*logits.shape[:2], elements, channels) * std + mean).reshape_as(logits)


def _validate_shapes(
    pixel_values: torch.Tensor,
    bool_masked_pos: torch.Tensor,
    logits: torch.Tensor,
    patch_size: tuple[int, int],
    tubelet_size: int,
) -> tuple[int, int, int, int, int]:
    if pixel_values.ndim != 5:
        raise ValueError("pixel_values must have shape [B, T, C, H, W]")
    batch, frames, channels, height, width = pixel_values.shape
    if min(batch, channels, height, width) < 1 or frames < 1:
        raise ValueError("pixel_values must contain at least one frame and non-empty dimensions")
    patch_height, patch_width = patch_size
    if frames % tubelet_size or height % patch_height or width % patch_width:
        raise ValueError("video dimensions must divide exactly into tubelets and patches")

    expected_tokens = (frames // tubelet_size) * (height // patch_height) * (width // patch_width)
    if bool_masked_pos.shape != (batch, expected_tokens):
        raise ValueError(f"bool_masked_pos length must be {expected_tokens} per video")
    if bool_masked_pos.dtype is not torch.bool:
        raise ValueError("bool_masked_pos must be boolean")
    expected_patch_width = tubelet_size * patch_height * patch_width * channels
    if logits.ndim != 3 or logits.shape[0] != batch or logits.shape[2] != expected_patch_width:
        raise ValueError(f"decoder patch width must be {expected_patch_width}")
    if not torch.equal(
        bool_masked_pos.sum(dim=1),
        torch.full((batch,), logits.shape[1], device=bool_masked_pos.device),
    ):
        raise ValueError("decoder logits must provide one patch for each masked token")
    return batch, frames, channels, height, width


def _validate_normalization(
    pixel_values: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> None:
    if pixel_values.ndim != 5:
        raise ValueError("pixel_values must have shape [B, T, C, H, W]")
    if len(mean) != pixel_values.shape[2] or len(std) != pixel_values.shape[2]:
        raise ValueError("mean and std must match the video channel count")


def _patchify(video: torch.Tensor, patch_size: tuple[int, int], tubelet_size: int) -> torch.Tensor:
    batch, frames, channels, height, width = video.shape
    patch_height, patch_width = patch_size
    return (
        video.reshape(
            batch,
            frames // tubelet_size,
            tubelet_size,
            channels,
            height // patch_height,
            patch_height,
            width // patch_width,
            patch_width,
        )
        .permute(0, 1, 4, 6, 2, 5, 7, 3)
        .reshape(batch, -1, tubelet_size * patch_height * patch_width * channels)
    )


def _unpatchify(
    patches: torch.Tensor,
    batch: int,
    frames: int,
    channels: int,
    height: int,
    width: int,
    patch_size: tuple[int, int],
    tubelet_size: int,
) -> torch.Tensor:
    patch_height, patch_width = patch_size
    return (
        patches.reshape(
            batch,
            frames // tubelet_size,
            height // patch_height,
            width // patch_width,
            tubelet_size,
            patch_height,
            patch_width,
            channels,
        )
        .permute(0, 1, 4, 7, 2, 5, 3, 6)
        .reshape(batch, frames, channels, height, width)
    )


def _frame_indices(total: int, maximum: int) -> torch.Tensor:
    """Return at most ``maximum`` evenly-spaced frame indices."""
    return torch.linspace(0, total - 1, steps=min(total, maximum), dtype=torch.long)


def _horizontal_grid(frames: torch.Tensor) -> torch.Tensor:
    """Place ``[F, C, H, W]`` frames side by side."""
    return torch.cat(tuple(frames), dim=-1)


def _to_hwc(array: torch.Tensor) -> np.ndarray:
    """Detach one ``[C, H, W]`` image as a NumPy HWC array."""
    return array.detach().cpu().permute(1, 2, 0).numpy()

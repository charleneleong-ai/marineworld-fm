"""Tracking-independent VideoMAE reconstruction preview helpers."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from marineworld.models.videomae import pair
from marineworld.train.module import VideoMAEPretrainingModule
from marineworld.train.naming import metric_name

__all__ = ["MediaPreview", "WandbMediaCallback", "build_media_preview"]


@dataclass(frozen=True)
class MediaPreview:
    """Display-ready VideoMAE input and reconstruction images."""

    input_grid: np.ndarray
    reconstruction_panel: np.ndarray
    caption: str
    dataset: str


class WandbMediaCallback(Callback):
    """Log bounded validation and best-checkpoint previews to W&B."""

    def __init__(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        enabled: bool,
        every_n_epochs: int,
        max_frames: int,
        checkpoint_callback: ModelCheckpoint,
    ) -> None:
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.enabled = enabled
        self.every_n_epochs = every_n_epochs
        self.max_frames = max_frames
        self.checkpoint_callback = checkpoint_callback
        self._sample: dict[str, Any] | None = None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs
        logger = self._preview_logger(trainer, batch_idx, dataloader_idx)
        if logger is None:
            return
        if not isinstance(pl_module, VideoMAEPretrainingModule):
            warnings.warn("media preview requires a VideoMAE pretraining module", stacklevel=2)
            return

        self._sample = self._first_sample(batch)
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        self._emit(
            logger, pl_module, self._sample, stage="val", trainer=trainer, microbatch=batch_idx
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        sample, self._sample = self._sample, None
        logger = (
            self._wandb_logger(trainer.logger)
            if self.enabled and trainer.global_rank == 0
            else None
        )
        if logger is None or sample is None:
            if logger is not None and sample is None:
                warnings.warn("best checkpoint media skipped: no validation sample", stacklevel=2)
            return
        if not isinstance(pl_module, VideoMAEPretrainingModule):
            warnings.warn(
                "best checkpoint media requires a VideoMAE pretraining module", stacklevel=2
            )
            return

        best_model_path = self.checkpoint_callback.best_model_path
        if not best_model_path:
            warnings.warn("best checkpoint media skipped: no best checkpoint", stacklevel=2)
            return

        restored = self._restore_module(pl_module, Path(best_model_path))
        if restored is None:
            return
        self._emit(logger, restored, sample, stage="best", trainer=trainer, microbatch=0)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> None:
        """Preview the live train reconstruction so it can be read against validation."""
        del outputs
        logger = self._preview_logger(trainer, batch_idx)
        if logger is None or trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(pl_module, VideoMAEPretrainingModule):
            warnings.warn("media preview requires a VideoMAE pretraining module", stacklevel=2)
            return

        self._emit(
            logger,
            pl_module,
            self._first_sample(batch, to_cpu=False),
            stage="pretrain",
            trainer=trainer,
            microbatch=batch_idx,
        )

    def _emit(
        self,
        logger: Any,
        module: VideoMAEPretrainingModule,
        sample: Mapping[str, Any],
        *,
        stage: str,
        trainer: Trainer,
        microbatch: int,
    ) -> None:
        """Render one preview from the live weights and log it under the given stage."""
        pixel_values = sample["pixel_values"].to(module.device)
        mask = module.make_mask(
            pixel_values.shape[0],
            pixel_values.device,
            step=trainer.global_step,
            microbatch=microbatch,
        )
        was_training = module.training
        module.eval()
        try:
            with torch.inference_mode():
                logits = module.model(pixel_values=pixel_values, bool_masked_pos=mask).logits
        finally:
            module.train(was_training)
        if logits is None:
            warnings.warn(
                "media preview skipped: model returned no reconstruction logits", stacklevel=2
            )
            return

        config = module.model.config
        preview = build_media_preview(
            pixel_values=pixel_values,
            bool_masked_pos=mask,
            logits=logits,
            mean=self.mean,
            std=self.std,
            patch_size=pair(config.patch_size),
            tubelet_size=int(config.tubelet_size),
            max_frames=self.max_frames,
            dataset=str(sample["dataset"]),
            source=str(sample["source"]),
            norm_pix_loss=bool(config.norm_pix_loss),
        )
        self._log_preview(logger, preview, stage, trainer.global_step)

    def _preview_logger(
        self, trainer: Trainer, batch_idx: int, dataloader_idx: int = 0
    ) -> Any | None:
        if batch_idx != 0 or dataloader_idx != 0 or not self.enabled or trainer.global_rank != 0:
            return None
        return self._wandb_logger(trainer.logger)

    @staticmethod
    def _wandb_logger(logger: object) -> Any | None:
        """Return a Lightning W&B logger without importing W&B or its logger."""
        logger_type = type(logger)
        if logger_type.__name__ != "WandbLogger" or not logger_type.__module__.endswith(
            ".loggers.wandb"
        ):
            return None
        return logger if callable(getattr(logger, "log_metrics", None)) else None

    @staticmethod
    def _first_sample(batch: Mapping[str, Any], *, to_cpu: bool = True) -> dict[str, Any]:
        """Copy the first clip; retained samples go to CPU, inline ones stay on device."""
        pixel_values = batch["pixel_values"][:1].detach()
        return {
            "pixel_values": pixel_values.cpu().clone() if to_cpu else pixel_values,
            "dataset": WandbMediaCallback._first_metadata(batch, "dataset"),
            "source": WandbMediaCallback._first_metadata(batch, "source"),
        }

    @staticmethod
    def _first_metadata(batch: Mapping[str, Any], key: str) -> str:
        value = batch.get(key, "unknown")
        if isinstance(value, str):
            return value
        return str(value[0]) if value else "unknown"

    @staticmethod
    def _restore_module(
        live_module: VideoMAEPretrainingModule, checkpoint_path: Path
    ) -> VideoMAEPretrainingModule | None:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError):
            warnings.warn(
                "best checkpoint media skipped: checkpoint could not be loaded",
                stacklevel=2,
            )
            return None
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("state_dict"), Mapping
        ):
            warnings.warn(
                "best checkpoint media skipped: checkpoint has no state_dict", stacklevel=2
            )
            return None

        restored = VideoMAEPretrainingModule(
            dict(live_module.model_config),
            lr=live_module.lr,
            weight_decay=live_module.weight_decay,
            seed=live_module.seed,
            warmup_epochs=live_module.warmup_epochs,
            max_epochs=live_module.max_epochs,
        )
        try:
            restored.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            warnings.warn(
                "best checkpoint media skipped: checkpoint state is incompatible",
                stacklevel=2,
            )
            return None
        restored.eval()
        return restored

    @staticmethod
    def _log_preview(logger: Any, preview: MediaPreview, stage: str, step: int) -> None:
        """Log previews under the stage that produced them, split per dataset."""
        import wandb

        logger.log_metrics(
            {
                metric_name(stage, "media/inputs", preview.dataset): wandb.Image(
                    preview.input_grid, caption=preview.caption
                ),
                metric_name(stage, "media/reconstruction", preview.dataset): wandb.Image(
                    preview.reconstruction_panel, caption=preview.caption
                ),
            },
            step=step,
        )


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

    indices = torch.arange(min(original.shape[1], max_frames)).to(original.device)
    originals = original[0, indices]
    input_grid = _to_hwc(torch.cat(tuple(originals), dim=-1))
    rows = torch.cat(
        (originals, masked[0, indices], completed[0, indices], error[0, indices]), dim=-1
    )
    panel = _to_hwc(torch.cat(tuple(rows), dim=-2))
    return MediaPreview(input_grid, panel, f"dataset={dataset} source={source}", dataset)


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


def _to_hwc(array: torch.Tensor) -> np.ndarray:
    """Detach one ``[C, H, W]`` image as a NumPy HWC array."""
    return array.detach().cpu().permute(1, 2, 0).numpy()

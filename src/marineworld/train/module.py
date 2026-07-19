"""Lightning module for VideoMAE masked-reconstruction pretraining."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from pytorch_lightning import LightningModule

from marineworld.models.videomae import build_videomae, tube_mask

__all__ = ["VideoMAEPretrainingModule"]


class VideoMAEPretrainingModule(LightningModule):
    """Optimize VideoMAE reconstruction with deterministic per-step masking."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        *,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        seed: int | None = None,
        warmup_epochs: int = 0,
        max_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.model_config = dict(model_config)
        self.model = build_videomae(self.model_config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.mask_ratio = float(self.model_config["mask_ratio"])
        self.seed = int(self.model_config.get("seed", 42) if seed is None else seed)
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

    def make_mask(
        self,
        batch_size: int,
        device: torch.device,
        *,
        step: int | None = None,
        microbatch: int = 0,
    ) -> torch.Tensor:
        """Create the mask for a global step, preserving resume determinism."""
        generator = torch.Generator().manual_seed(
            self.seed + (self.global_step if step is None else step) + 1_000_003 * microbatch
        )
        return tube_mask(
            batch_size,
            _sequence_length(self.model.config),
            self.mask_ratio,
            generator,
        ).to(device)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._reconstruction_loss(batch, batch_idx, stage="training")
        self.log(
            "pretrain/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(batch["pixel_values"]),
        )
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._reconstruction_loss(batch, batch_idx, stage="validation")
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(batch["pixel_values"]),
        )
        return loss

    def _reconstruction_loss(
        self, batch: Mapping[str, Any], batch_idx: int, *, stage: str
    ) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        rank = int(getattr(self, "global_rank", 0))
        mask = self.make_mask(
            pixel_values.shape[0], pixel_values.device, microbatch=batch_idx + 10_000 * rank
        )
        loss = self.model(pixel_values=pixel_values, bool_masked_pos=mask).loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {stage} loss at batch {batch_idx}")
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        trainer = getattr(self, "_trainer", None)
        total_steps = max(
            1,
            int(trainer.estimated_stepping_batches) if trainer is not None else self.max_epochs * 2,
        )
        warmup_steps = round(total_steps * self.warmup_epochs / max(1, self.max_epochs))

        def scale(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def _sequence_length(config: Any) -> int:
    image_height, image_width = _pair(config.image_size)
    patch_height, patch_width = _pair(config.patch_size)
    return (
        (config.num_frames // config.tubelet_size)
        * (image_height // patch_height)
        * (image_width // patch_width)
    )


def _pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else (value[0], value[1])

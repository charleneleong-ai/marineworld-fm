"""Lightning module for VideoMAE masked-reconstruction pretraining."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from pytorch_lightning import LightningModule

from marineworld.models.videomae import build_videomae, pair, tube_mask
from marineworld.train.naming import metric_name

__all__ = ["VideoMAEPretrainingModule", "MaritimePretrainingModule"]


class VideoMAEPretrainingModule(LightningModule):
    """Optimize VideoMAE reconstruction with deterministic per-step masking."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        *,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 0,
        max_epochs: int = 1,
        total_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.model_config = dict(model_config)
        self.model = build_videomae(self.model_config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.total_steps = total_steps
        self.mask_ratio = float(self.model_config["mask_ratio"])
        self.seed = int(self.model_config.get("seed", 42) if seed is None else seed)

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
            metric_name("pretrain", "loss"),
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(batch["pixel_values"]),
        )
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Record what the schedule is doing and how large the update is."""
        self.log(
            metric_name("pretrain", "lr"),
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        grads = [
            parameter.grad.detach() for parameter in self.parameters() if parameter.grad is not None
        ]
        if grads:
            self.log(
                metric_name("pretrain", "grad_norm"),
                torch.linalg.vector_norm(torch.stack([grad.norm(2) for grad in grads]), 2),
                on_step=True,
                on_epoch=False,
            )

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._reconstruction_loss(batch, batch_idx, stage="validation")
        self.log(
            metric_name("val", "loss"),
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

    def _schedule_length(self) -> int:
        """Total optimizer steps the cosine schedule spans."""
        if self.total_steps is not None:
            return self.total_steps
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            raise RuntimeError(
                "cosine schedule needs a step budget: attach a Trainer or pass total_steps"
            )
        return int(trainer.estimated_stepping_batches)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = max(1, self._schedule_length())
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
    image_height, image_width = pair(config.image_size)
    patch_height, patch_width = pair(config.patch_size)
    return (
        (config.num_frames // config.tubelet_size)
        * (image_height // patch_height)
        * (image_width // patch_width)
    )


class MaritimePretrainingModule(LightningModule):
    """Video + AIS cross-modal pretraining with joint reconstruction losses."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        *,
        ais_loss_weight: float = 0.5,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 0,
        max_epochs: int = 1,
        total_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model_config = dict(model_config)
        self.ais_loss_weight = ais_loss_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.total_steps = total_steps
        self.mask_ratio = float(self.model_config["mask_ratio"])
        self.seed = int(self.model_config.get("seed", 42) if seed is None else seed)

        from marineworld.models.ais_encoder import AISEncoder
        from marineworld.models.maritime_videomae import MaritimeVideoMAE

        videomae = build_videomae(self.model_config)
        hidden_size = videomae.config.hidden_size
        ais_encoder = AISEncoder(
            hidden_size=hidden_size,
            num_vessel_types=int(self.model_config.get("num_vessel_types", 100)),
        )
        self.model = MaritimeVideoMAE(
            videomae,
            ais_encoder,
            num_cross_attn_layers=int(self.model_config.get("num_cross_attn_layers", 2)),
            cross_attn_heads=int(self.model_config.get("cross_attn_heads", 2)),
        )

    def make_mask(
        self,
        batch_size: int,
        device: torch.device,
        *,
        step: int | None = None,
        microbatch: int = 0,
    ) -> torch.Tensor:
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
        pixel_values = batch["pixel_values"]
        rank = int(getattr(self, "global_rank", 0))
        mask = self.make_mask(
            pixel_values.shape[0], pixel_values.device, microbatch=batch_idx + 10_000 * rank
        )
        out = self.model(
            pixel_values=pixel_values,
            ais_features=batch["ais_features"],
            ais_mask=batch["ais_mask"],
            bool_masked_pos=mask,
        )
        loss = out["total_loss"]
        self.log(
            metric_name("pretrain", "loss"),
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(pixel_values),
        )
        self.log(
            metric_name("pretrain", "video_loss"),
            out["video_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(pixel_values),
        )
        self.log(
            metric_name("pretrain", "ais_loss"),
            out["ais_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(pixel_values),
        )
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        self.log(
            metric_name("pretrain", "lr"),
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        grads = [p.grad.detach() for p in self.parameters() if p.grad is not None]
        if grads:
            self.log(
                metric_name("pretrain", "grad_norm"),
                torch.linalg.vector_norm(torch.stack([g.norm(2) for g in grads]), 2),
                on_step=True,
                on_epoch=False,
            )

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        mask = self.make_mask(pixel_values.shape[0], pixel_values.device, microbatch=batch_idx)
        out = self.model(
            pixel_values=pixel_values,
            ais_features=batch["ais_features"],
            ais_mask=batch["ais_mask"],
            bool_masked_pos=mask,
        )
        self.log(
            metric_name("val", "loss"),
            out["total_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(pixel_values),
        )
        return out["total_loss"]

    def _schedule_length(self) -> int:
        if self.total_steps is not None:
            return self.total_steps
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            raise RuntimeError("cosine schedule needs a step budget")
        return int(trainer.estimated_stepping_batches)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = max(1, self._schedule_length())
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

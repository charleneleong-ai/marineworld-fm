"""Hydra entrypoint for VideoMAE pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader

from marineworld.data.adapters import DatasetAdapter, build_adapter
from marineworld.data.clips import (
    AutoVideoDecoder,
    MaritimeClipDataset,
    SyntheticVideoDecoder,
    VideoDecoder,
)
from marineworld.data.contracts import DatasetManifest
from marineworld.data.manifest import manifest_checksum
from marineworld.data.splits import prepare_manifest_splits
from marineworld.train.experiment import build_run_identity, build_wandb_logger
from marineworld.train.module import VideoMAEPretrainingModule
from marineworld.utils.seed import seed_everything

__all__ = ["build_dataloaders", "build_trainer", "run_pretraining"]


class _ExceptionSafeModelCheckpoint(ModelCheckpoint):
    """Backport Lightning 2.5's ``save_on_exception`` checkpoint behavior."""

    def __init__(self, *, save_on_exception: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_on_exception = save_on_exception

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module, exception
        if self.save_on_exception:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


def build_dataloaders(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
) -> dict[str, DataLoader[dict[str, Any]]]:
    """Build train and validation loaders over the shared clip contract."""
    decoder = _build_decoder(cfg)
    common = {
        "manifest": manifest,
        "decoder": decoder,
        "frames": int(cfg.model.num_frames),
        "stride": 1,
        "image_size": int(cfg.model.image_size),
        "seed": int(cfg.seed),
    }
    loader = {
        "batch_size": int(cfg.runtime.batch_size),
        "num_workers": int(cfg.runtime.num_workers),
        "persistent_workers": int(cfg.runtime.num_workers) > 0,
    }
    train_dataset = MaritimeClipDataset(split="train", **common)
    val_dataset = MaritimeClipDataset(split="val", **common)
    if not train_dataset:
        raise ValueError("training split produced no clips")
    if not val_dataset:
        raise ValueError("validation split produced no clips")
    return {
        "train_dataloaders": DataLoader(
            train_dataset,
            collate_fn=_pretraining_collate,
            **loader,
        ),
        "val_dataloaders": DataLoader(
            val_dataset,
            collate_fn=_pretraining_collate,
            **loader,
        ),
    }


def build_trainer(
    cfg: DictConfig,
    *,
    logger: Logger | bool = False,
    callbacks: list[Callback] | None = None,
) -> Trainer:
    """Construct a Lightning trainer from a composed runtime profile."""
    return Trainer(
        accelerator=str(cfg.runtime.accelerator),
        devices=int(cfg.runtime.devices),
        precision=str(cfg.runtime.precision),
        max_epochs=int(cfg.train.epochs),
        max_steps=int(cfg.runtime.max_steps),
        accumulate_grad_batches=int(cfg.runtime.accumulate_grad_batches),
        limit_train_batches=cfg.runtime.limit_train_batches,
        limit_val_batches=cfg.runtime.limit_val_batches,
        deterministic=True,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def run_pretraining(cfg: DictConfig) -> Path:
    """Train from a composed config and return the last checkpoint path."""
    seed_everything(int(cfg.seed))
    adapter = build_adapter(cfg.data.adapter)
    manifest = _prepare_manifest(cfg, adapter.build_manifest(Path(cfg.data.root)))
    dataloaders = build_dataloaders(cfg, manifest, adapter)
    identity = build_run_identity(
        str(cfg.model.name),
        (manifest_checksum(manifest),),
        int(cfg.seed),
        None,
        accelerator=str(cfg.runtime.accelerator),
    )
    logger = build_wandb_logger(cfg, identity)
    try:
        checkpoint = _ExceptionSafeModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=1,
            save_on_exception=True,
        )
        trainer = build_trainer(cfg, logger=logger, callbacks=[checkpoint])
        trainer.fit(
            _build_module(cfg),
            **dataloaders,
            ckpt_path=cfg.runtime.ckpt_path,
        )
    finally:
        if logger is not False:
            logger.experiment.finish()

    if not checkpoint.last_model_path:
        raise RuntimeError("training completed without a last checkpoint path")
    last_checkpoint = Path(checkpoint.last_model_path)
    if not last_checkpoint.is_file():
        raise RuntimeError(f"last checkpoint does not exist: {last_checkpoint}")
    return last_checkpoint


def _prepare_manifest(cfg: DictConfig, manifest: DatasetManifest) -> DatasetManifest:
    split = cfg.data.get("split")
    return prepare_manifest_splits(
        manifest,
        val_frac=float(split.val_frac) if split else None,
        test_frac=float(split.test_frac) if split else None,
        seed=int(cfg.seed),
    )


def _pretraining_collate(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack only model inputs; annotations are not part of SSL pretraining."""
    return {"pixel_values": torch.stack([sample["pixel_values"] for sample in samples])}


def _build_decoder(cfg: DictConfig) -> VideoDecoder:
    if str(cfg.data.name) == "synthetic":
        size = int(cfg.model.image_size)
        return SyntheticVideoDecoder(size, size)
    return AutoVideoDecoder()


def _build_module(cfg: DictConfig) -> VideoMAEPretrainingModule:
    model_config = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    if not isinstance(model_config, dict):
        raise TypeError("model config must resolve to a mapping")
    return VideoMAEPretrainingModule(
        model_config,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        seed=int(cfg.seed),
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run pretraining from Hydra CLI overrides."""
    checkpoint = run_pretraining(cfg)
    print(checkpoint)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()

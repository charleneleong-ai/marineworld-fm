"""Hydra entrypoint for VideoMAE pretraining."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, Sampler

from marineworld.data.adapters import build_data_adapter
from marineworld.data.clips import (
    AutoVideoDecoder,
    MaritimeClipDataset,
    SampleDraw,
    SyntheticVideoDecoder,
    VideoDecoder,
    drop_undecodable,
)
from marineworld.data.contracts import DatasetManifest
from marineworld.data.manifest import file_checksum, manifest_checksum
from marineworld.data.splits import prepare_manifest_splits
from marineworld.train.experiment import (
    RunIdentity,
    build_run_identity,
    build_wandb_logger,
    resolved_config,
)
from marineworld.train.media import WandbMediaCallback
from marineworld.train.module import VideoMAEPretrainingModule
from marineworld.train.naming import metric_name
from marineworld.utils.seed import seed_everything

__all__ = [
    "BalancedDatasetSampler",
    "build_dataloaders",
    "build_trainer",
    "run_pretraining",
    "write_training_manifest",
]


class BalancedDatasetSampler(Sampler[int]):
    """Deterministically sample configurable dataset mass and shard by rank."""

    def __init__(
        self,
        dataset_ids: Sequence[str],
        *,
        weights: Mapping[str, float] | None = None,
        seed: int,
        rank: int = 0,
        replicas: int = 1,
        draw_tokens: bool = False,
    ) -> None:
        if not dataset_ids:
            raise ValueError("balanced sampler requires at least one sample")
        if replicas <= 0 or not 0 <= rank < replicas:
            raise ValueError("rank must be within a positive replica count")
        self.dataset_ids = tuple(dataset_ids)
        self.groups: dict[str, tuple[int, ...]] = {
            name: tuple(index for index, value in enumerate(dataset_ids) if value == name)
            for name in sorted(set(dataset_ids))
        }
        configured = dict(weights or {name: 1.0 for name in self.groups})
        if absent := sorted(set(configured) - set(self.groups)):
            raise ValueError(
                f"sampler weights name dataset(s) absent from the corpus: {', '.join(absent)}. "
                f"Present: {', '.join(sorted(self.groups)) or 'none'}."
            )
        if unweighted := sorted(set(self.groups) - set(configured)):
            raise ValueError(f"sampler weights omit dataset(s): {', '.join(unweighted)}")
        if nonpositive := sorted(name for name, value in configured.items() if value <= 0):
            raise ValueError(f"sampler weights must be positive: {', '.join(nonpositive)}")
        total = sum(configured.values())
        self.weights = {name: value / total for name, value in configured.items()}
        self.seed = seed
        self.rank = rank
        self.replicas = replicas
        self.epoch = 0
        self.position = 0
        self.draw_tokens = draw_tokens

    def __len__(self) -> int:
        return math.ceil(len(self.dataset_ids) / self.replicas)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self.position = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.set_epoch(state["epoch"])
        self.position = int(state.get("position", 0))

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        count = math.ceil(len(self.dataset_ids) / self.replicas) * self.replicas
        exact = {name: count * weight for name, weight in self.weights.items()}
        counts = {name: math.floor(value) for name, value in exact.items()}
        for name in sorted(
            self.groups, key=lambda item: (exact[item] - counts[item], item), reverse=True
        )[: count - sum(counts.values())]:
            counts[name] += 1
        sampled: list[int] = []
        for name, group in self.groups.items():
            repeats = math.ceil(counts[name] / len(group))
            selected: list[int] = []
            for _ in range(repeats):
                order = torch.randperm(len(group), generator=generator).tolist()
                selected.extend(group[index] for index in order)
            sampled.extend(selected[: counts[name]])
        order = torch.randperm(len(sampled), generator=generator).tolist()
        global_indices = [sampled[index] for index in order]
        padding = (-len(global_indices)) % self.replicas
        global_indices.extend(global_indices[:padding])
        local_indices = global_indices[self.rank :: self.replicas]
        draws = (
            SampleDraw(
                index=index,
                augmentation_seed=self.seed
                + 1_000_003 * self.epoch
                + 10_007 * self.rank
                + position,
            )
            if self.draw_tokens
            else index
            for position, index in enumerate(local_indices)
        )
        return iter(tuple(draws)[self.position :])


class _BalancedSamplerCheckpoint(Callback):
    """Checkpoint completed samples without observing prefetched iterator yields."""

    def __init__(self, sampler: BalancedDatasetSampler, batch_size: int) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.base_position = sampler.position

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self.base_position = self.sampler.position

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del trainer, pl_module, outputs, batch
        self.sampler.position = max(
            self.sampler.position,
            min(
                len(self.sampler),
                self.base_position + (batch_idx + 1) * self.batch_size,
            ),
        )

    def state_dict(self) -> dict[str, int]:
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.sampler.load_state_dict(state_dict)
        self.base_position = self.sampler.position


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
) -> dict[str, DataLoader[dict[str, Any]]]:
    """Build train and validation loaders over the shared clip contract."""
    decoder = _build_decoder(cfg)
    common = {
        "manifest": manifest,
        "fingerprint": manifest_checksum(manifest),
        "decoder": decoder,
        "frames": int(cfg.model.num_frames),
        "stride": 1,
        "image_size": int(cfg.model.image_size),
        "seed": int(cfg.seed),
        "normalization_mean": tuple(cfg.data.transforms.normalization.mean),
        "normalization_std": tuple(cfg.data.transforms.normalization.std),
    }
    loader = {
        "batch_size": int(cfg.runtime.batch_size),
        "num_workers": int(cfg.runtime.num_workers),
        "persistent_workers": int(cfg.runtime.num_workers) > 0,
    }
    train_dataset = MaritimeClipDataset(
        split="train", color_jitter=float(cfg.data.transforms.train.color_jitter), **common
    )
    val_dataset = MaritimeClipDataset(split="val", **common)
    if not train_dataset:
        raise ValueError("training split produced no clips")
    if not val_dataset:
        raise ValueError("validation split produced no clips")
    dataset_ids = tuple(
        train_dataset.records[clip.record_id].dataset for clip in train_dataset.clips
    )
    weights = cfg.data.get("sampling_weights")
    rank, replicas = _distributed_context()
    sampler = BalancedDatasetSampler(
        dataset_ids,
        weights=dict(weights) if weights else None,
        seed=int(cfg.seed),
        rank=rank,
        replicas=replicas,
        draw_tokens=True,
    )
    return {
        "train_dataloaders": DataLoader(
            train_dataset,
            collate_fn=_pretraining_collate,
            sampler=sampler,
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
    _validate_media_config(cfg)
    seed_everything(int(cfg.seed))
    adapter = build_data_adapter(cfg.data)
    manifest = _prepare_manifest(cfg, adapter.build_manifest(Path(cfg.data.root)))
    dataloaders = build_dataloaders(cfg, manifest)
    checkpoint_provenance = _checkpoint_provenance(cfg.runtime.ckpt_path)
    resolved = resolved_config(cfg)
    identity = _training_run_identity(cfg, manifest, checkpoint_provenance, resolved)
    return _run_with_identity(cfg, manifest, dataloaders, identity, resolved)


def _training_run_identity(
    cfg: DictConfig,
    manifest: DatasetManifest,
    checkpoint_provenance: str | None = None,
    resolved: dict[str, Any] | None = None,
) -> RunIdentity:
    checksums = tuple(manifest.component_checksums.values()) or (manifest_checksum(manifest),)
    resolved = resolved_config(cfg) if resolved is None else resolved
    material = {
        "data": resolved["data"],
        "model": resolved["model"],
        "train": resolved["train"],
        "runtime": {
            key: resolved["runtime"][key]
            for key in (
                "batch_size",
                "accumulate_grad_batches",
            )
        },
    }
    return build_run_identity(
        str(cfg.model.name),
        checksums,
        int(cfg.seed),
        None,
        git_sha=_git_sha(),
        accelerator=str(cfg.runtime.accelerator),
        checkpoint_provenance=checkpoint_provenance,
        logical_dimensions={"training": material},
    )


def _run_with_identity(
    cfg: DictConfig,
    manifest: DatasetManifest,
    dataloaders: dict[str, DataLoader[dict[str, Any]]],
    identity: RunIdentity,
    resolved: dict[str, Any] | None = None,
) -> Path:
    logger = build_wandb_logger(cfg, identity, resolved)
    checkpoint = _ExceptionSafeModelCheckpoint(
        dirpath=Path(cfg.output_dir) / "checkpoints",
        monitor=metric_name("val", "loss"),
        mode="min",
        save_last=True,
        save_top_k=1,
        save_on_exception=True,
    )
    try:
        manifest_path = write_training_manifest(manifest, Path(cfg.output_dir) / "artifacts")
        _log_training_manifest(manifest_path, identity.run_id, logger)
        train_loader = dataloaders["train_dataloaders"]
        sampler = train_loader.sampler
        media = WandbMediaCallback(
            mean=tuple(cfg.data.transforms.normalization.mean),
            std=tuple(cfg.data.transforms.normalization.std),
            enabled=cfg.tracking.log_media,
            every_n_epochs=int(cfg.tracking.media_log_every_n_epochs),
            max_frames=int(cfg.tracking.media_max_frames),
            checkpoint_callback=checkpoint,
        )
        callbacks: list[Callback] = [checkpoint, media]
        if isinstance(sampler, BalancedDatasetSampler):
            callbacks.append(_BalancedSamplerCheckpoint(sampler, int(cfg.runtime.batch_size)))
        trainer = build_trainer(cfg, logger=logger, callbacks=callbacks)
        module = _build_module(cfg)
        trainer.fit(
            module,
            **dataloaders,
            ckpt_path=cfg.runtime.ckpt_path,
        )
        if not checkpoint.last_model_path:
            raise RuntimeError("training completed without a last checkpoint path")
        last_checkpoint = Path(checkpoint.last_model_path)
        if not last_checkpoint.is_file():
            raise RuntimeError(f"last checkpoint does not exist: {last_checkpoint}")
    except BaseException as primary_error:
        try:
            _finish_wandb(logger, exit_code=1)
        except BaseException as cleanup_error:
            primary_error.add_note(f"W&B cleanup also failed: {cleanup_error}")
        raise
    _finish_wandb(logger, exit_code=0)
    return last_checkpoint


def _validate_media_config(cfg: DictConfig) -> None:
    if type(cfg.tracking.log_media) is not bool:
        raise ValueError("tracking.log_media must be boolean")
    for key in ("media_log_every_n_epochs", "media_max_frames"):
        value = cfg.tracking[key]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or (key == "media_max_frames" and value > 4)
        ):
            constraint = (
                "positive integer between 1 and 4"
                if key == "media_max_frames"
                else "positive integer"
            )
            raise ValueError(f"tracking.{key} must be {constraint}")


def write_training_manifest(manifest: DatasetManifest, output_dir: Path) -> Path:
    """Persist exact portable membership/provenance without restricted media paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": manifest.name,
        "version": manifest.version,
        "license": manifest.license,
        "access": manifest.access,
        "label_mapping": dict(manifest.label_mapping),
        "native_labels": list(manifest.native_labels),
        "manifest_checksum": manifest_checksum(manifest),
        "component_checksums": dict(manifest.component_checksums),
        "components": dict(manifest.components),
        "records": [
            {
                "id": record.id,
                "dataset": record.dataset,
                "split": record.split,
                "source": record.source,
                "fps": record.fps,
                "num_frames": record.num_frames,
                "has_annotation": record.annotation_path is not None,
            }
            for record in sorted(manifest.records, key=lambda item: item.id)
        ],
    }
    path = output_dir / "training_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _log_training_manifest(path: Path, run_id: str, logger: Logger | bool) -> None:
    if logger is False or not hasattr(logger.experiment, "log_artifact"):
        return
    from wandb import Artifact

    artifact = Artifact(name=f"training-manifest-{run_id}", type="dataset")
    artifact.add_file(str(path), name=path.name)
    logger.experiment.log_artifact(artifact)


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() or None


def _checkpoint_provenance(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.is_file():
        return str(checkpoint)
    return f"sha256:{file_checksum(path)}"


def _distributed_context() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _prepare_manifest(cfg: DictConfig, manifest: DatasetManifest) -> DatasetManifest:
    split = cfg.data.get("split")
    if manifest.components and all(record.split == "train" for record in manifest.records):
        grouped: dict[tuple[str, str | None], list[Any]] = {}
        smd_source_counts: dict[str, int] = {}
        for record in manifest.records:
            if record.dataset == "smd":
                smd_source_counts[record.source] = smd_source_counts.get(record.source, 0) + 1
        stratify_smd_source = bool(smd_source_counts) and all(
            count >= 2 for count in smd_source_counts.values()
        )
        for record in manifest.records:
            source = record.source if record.dataset == "smd" and stratify_smd_source else None
            grouped.setdefault((record.dataset, source), []).append(record)
        prepared = []
        for records in grouped.values():
            component = DatasetManifest(
                manifest.name,
                manifest.version,
                manifest.license,
                tuple(records),
            )
            prepared.extend(
                prepare_manifest_splits(
                    component,
                    val_frac=float(split.val_frac),
                    test_frac=float(split.test_frac),
                    seed=int(cfg.seed),
                ).records
            )
        return replace(manifest, records=tuple(prepared))
    return prepare_manifest_splits(
        manifest,
        val_frac=float(split.val_frac) if split else None,
        test_frac=float(split.test_frac) if split else None,
        seed=int(cfg.seed),
    )


def _pretraining_collate(samples: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Stack only model inputs; annotations are not part of SSL pretraining."""
    samples = drop_undecodable(samples)
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "dataset": tuple(sample["dataset"] for sample in samples),
        "record_id": tuple(sample["record_id"] for sample in samples),
        "source": tuple(sample["source"] for sample in samples),
    }


def _build_decoder(cfg: DictConfig) -> VideoDecoder:
    if str(cfg.data.name) == "synthetic" or cfg.data.get("decoder") == "synthetic":
        size = int(cfg.model.image_size)
        return SyntheticVideoDecoder(size, size)
    return AutoVideoDecoder()


def _build_module(cfg: DictConfig) -> VideoMAEPretrainingModule:
    model_config = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    if not isinstance(model_config, dict):
        raise TypeError("model config must resolve to a mapping")
    warmup_epochs = int(cfg.train.warmup_epochs)
    epochs = int(cfg.train.epochs)
    if not 0 <= warmup_epochs <= epochs:
        raise ValueError(f"warmup_epochs must be between 0 and epochs ({epochs})")
    return VideoMAEPretrainingModule(
        model_config,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        seed=int(cfg.seed),
    )


def _finish_wandb(logger: Logger | bool, *, exit_code: int) -> None:
    if logger is not False:
        logger.experiment.finish(exit_code=exit_code)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run pretraining from Hydra CLI overrides."""
    checkpoint = run_pretraining(cfg)
    print(checkpoint)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()

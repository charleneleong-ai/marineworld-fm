"""Hydra entrypoint and resource-safe orchestration for frozen probes."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from marineworld.data.adapters import DatasetAdapter, build_adapter, build_data_adapter
from marineworld.data.clips import (
    AutoVideoDecoder,
    MaritimeClipDataset,
    SpatialTransform,
    SyntheticVideoDecoder,
)
from marineworld.data.contracts import DatasetManifest, FrameTargets
from marineworld.data.manifest import content_identity, manifest_checksum, validate_manifest
from marineworld.data.splits import SplitUnavailableError, prepare_manifest_splits
from marineworld.eval.encoders import (
    EncoderFeatures,
    FrozenVideoEncoder,
    ResourceUnavailableError,
    load_frozen_encoder,
    validate_encoder_request,
)
from marineworld.eval.probes import (
    DegenerateLabelsError,
    LabelsUnavailableError,
    ProbeResult,
    aggregate_probe_results,
    evaluate_dense_probe_streaming,
    evaluate_probe,
    fit_dense_probe_streaming,
    fit_linear_probe,
    nearest_neighbour_diagnostic,
    sample_labelled_records,
)
from marineworld.train.experiment import RunIdentity, build_run_identity, build_wandb_logger
from marineworld.train.module import VideoMAEPretrainingModule
from marineworld.utils.seed import seed_everything


@dataclass(frozen=True)
class ProbeRun:
    condition: str
    checkpoint: str
    manifest_checksum: str
    dataset: str
    task: str
    fraction: float
    seed: int
    metric: str
    model: str
    device: str
    optional: bool = False
    selected_record_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class _FeatureSet:
    features: torch.Tensor
    labels: torch.Tensor
    record_ids: tuple[str, ...]
    sources: tuple[str, ...]
    datasets: tuple[str, ...]


def run_probe_condition(
    run: ProbeRun,
    *,
    loader: Callable[[], FrozenVideoEncoder],
    evaluator: Callable[[FrozenVideoEncoder], Sequence[ProbeResult]],
) -> tuple[ProbeResult, ...]:
    """Load and evaluate one condition, recording optional resource skips."""
    try:
        encoder = loader()
        return tuple(evaluator(encoder))
    except Exception as error:
        if not run.optional or not _is_resource_error(error):
            raise
        return (probe_result(run, "SKIPPED_RESOURCE"),)


def run_evaluation(cfg: DictConfig) -> tuple[ProbeResult, ...]:
    """Run the configured frozen probe matrix and log one results table."""
    _validate_probe_config(cfg)
    seed_everything(int(cfg.seed))
    adapter = (
        build_data_adapter(cfg.data)
        if cfg.data.get("components")
        else build_adapter(cfg.data.adapter)
    )
    manifest = adapter.build_manifest(Path(cfg.data.root))
    validate_manifest(manifest)
    split_error: SplitUnavailableError | None = None
    try:
        manifest = _prepare_probe_manifest(cfg, manifest)
    except SplitUnavailableError as error:
        split_error = error
    checksum = manifest_checksum(manifest)
    condition = str(cfg.model.condition)
    configured_checkpoint = cfg.eval.checkpoint or cfg.model.get("checkpoint")
    checkpoint_ref = "random-init" if condition == "random" else str(configured_checkpoint)
    revision = cfg.model.get("revision")
    checkpoint = (
        f"{checkpoint_ref}@{revision}" if revision and condition != "random" else checkpoint_ref
    )
    model_config = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    if not isinstance(model_config, dict):
        raise TypeError("model config must resolve to a mapping")

    labelled_records = _labelled_training_records(manifest, int(cfg.model.num_frames))
    runs = tuple(
        ProbeRun(
            condition=condition,
            checkpoint=checkpoint,
            manifest_checksum=checksum,
            dataset=manifest.name,
            task=str(cfg.eval.task),
            fraction=float(fraction),
            seed=int(seed),
            metric="macro_f1",
            model=str(cfg.model.name),
            device=str(cfg.eval.device),
            optional=bool(cfg.eval.optional),
            selected_record_ids=(
                sample_labelled_records(
                    labelled_records,
                    fraction=float(fraction),
                    seed=int(seed),
                )
                if labelled_records
                else ()
            ),
        )
        for fraction in cfg.eval.label_fractions
        for seed in cfg.eval.seeds
    )
    identity = _probe_run_identity(cfg, manifest, checkpoint_ref, revision)
    logger = build_wandb_logger(_tracking_config(cfg), identity)
    diagnostic_path = _prepare_diagnostic_path(Path(cfg.output_dir))
    try:
        selection_path = _write_selection_manifest(cfg, manifest, runs)
        _log_selection_manifest(selection_path, identity.run_id, logger)
        if split_error is not None or not has_labelled_splits(manifest, int(cfg.model.num_frames)):
            results = tuple(probe_result(run, "SKIPPED_UNAVAILABLE_LABELS") for run in runs)
            log_probe_results(results, logger)
            return results
        results = run_probe_condition(
            runs[0],
            loader=lambda: load_frozen_encoder(
                condition,
                checkpoint=None if condition == "random" else checkpoint_ref,
                model_config=model_config,
                device=str(cfg.eval.device),
                allow_download=bool(cfg.eval.allow_download),
            ),
            evaluator=lambda encoder: _evaluate_runs(cfg, manifest, adapter, encoder, runs),
        )
        if len(results) == 1 and results[0].status == "SKIPPED_RESOURCE":
            results = tuple(probe_result(run, "SKIPPED_RESOURCE") for run in runs)
        log_probe_results(results, logger)
        if diagnostic_path.is_file():
            _log_diagnostic_artifact(diagnostic_path, logger)
        return results
    finally:
        if logger is not False:
            logger.experiment.finish()


def log_probe_results(results: Sequence[ProbeResult], logger: Any) -> None:
    """Log every condition/seed/fraction row in one W&B table."""
    if logger is False:
        return
    logger.experiment.log({"probe/results": _wandb_table(aggregate_probe_results(results))})


def _write_selection_manifest(
    cfg: DictConfig,
    manifest: DatasetManifest,
    runs: Sequence[ProbeRun],
) -> Path:
    """Persist exact split membership and label subsets even when W&B is disabled."""
    split_ids = {
        split: sorted(record.id for record in manifest.records if record.split == split)
        for split in ("train", "val", "test")
    }
    payload = {
        "dataset": manifest.name,
        "manifest_checksum": manifest_checksum(manifest),
        "splits": {
            split: {
                "record_ids": record_ids,
                "checksum": _record_ids_checksum(record_ids),
            }
            for split, record_ids in split_ids.items()
        },
        "label_subsets": [
            {
                "fraction": run.fraction,
                "seed": run.seed,
                "record_ids": (
                    list(run.selected_record_ids) if run.selected_record_ids is not None else None
                ),
                "checksum": (
                    _record_ids_checksum(run.selected_record_ids)
                    if run.selected_record_ids is not None
                    else None
                ),
            }
            for run in runs
        ],
    }
    path = Path(cfg.output_dir) / "probe_selection_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _log_selection_manifest(path: Path, run_id: str, logger: Any) -> None:
    if logger is False:
        return
    artifact = _wandb_artifact_type()(name=f"probe-selection-{run_id}", type="dataset")
    artifact.add_file(str(path), name=path.name)
    logger.experiment.log_artifact(artifact)


def _record_ids_checksum(record_ids: Sequence[str]) -> str:
    payload = json.dumps(tuple(sorted(record_ids)), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _tracking_config(cfg: DictConfig) -> DictConfig:
    mode = os.environ.get("WANDB_MODE")
    if mode is None:
        return cfg
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    copied.tracking.mode = mode
    return copied


def _probe_run_identity(
    cfg: DictConfig,
    manifest: DatasetManifest,
    checkpoint: str,
    revision: str | None,
) -> RunIdentity:
    task = str(cfg.eval.task)
    provenance = f"{checkpoint}@{revision}" if revision else checkpoint
    model_config = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    if not isinstance(model_config, dict):
        raise TypeError("model config must resolve to a mapping")
    return build_run_identity(
        str(cfg.model.name),
        (manifest_checksum(manifest),),
        int(cfg.seed),
        None,
        accelerator=str(cfg.eval.device),
        checkpoint_provenance=provenance,
        logical_dimensions={
            "checkpoint_identity": _checkpoint_identity(checkpoint, revision),
            "condition": str(cfg.model.condition),
            "dataset": manifest.name,
            "batch_size": int(cfg.eval.batch_size),
            "dense_epochs": int(cfg.eval.dense_epochs) if task == "dense" else None,
            "dense_lr": float(cfg.eval.dense_lr) if task == "dense" else None,
            "label_fractions": sorted(float(value) for value in cfg.eval.label_fractions),
            "model_config": model_config,
            "transforms": OmegaConf.to_container(cfg.data.transforms, resolve=True),
            "seeds": sorted(int(value) for value in cfg.eval.seeds),
            "task": task,
        },
    )


def validate_probe_matrix(cfg: DictConfig) -> None:
    """Reject an invalid seed / label-fraction matrix."""
    _validate_seed("seed", cfg.seed)
    seeds = _config_values("eval.seeds", cfg.eval.seeds)
    if (
        not seeds
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or any(not 0 <= value < 2**32 for value in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("eval.seeds must be a nonempty list of unique integers in [0, 2**32)")
    fraction_values = _config_values("eval.label_fractions", cfg.eval.label_fractions)
    try:
        fractions = [float(value) for value in fraction_values]
    except (TypeError, ValueError) as error:
        raise ValueError("eval.label_fractions must contain numeric values") from error
    if (
        not fractions
        or any(isinstance(value, bool) for value in fraction_values)
        or len(fractions) != len(set(fractions))
    ):
        raise ValueError("eval.label_fractions must be a nonempty list of unique values")
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in fractions):
        raise ValueError("eval.label_fractions values must be finite and in (0, 1]")
    if bool(cfg.eval.get("report_test", False)) and (len(seeds) != 1 or len(fractions) != 1):
        raise ValueError("eval.report_test requires exactly one selected seed and label fraction")


def validate_probe_hyperparameters(cfg: DictConfig) -> None:
    """Reject invalid probe optimisation and model geometry settings."""
    _require_positive_int("eval.batch_size", cfg.eval.batch_size)
    _require_positive_int("eval.dense_epochs", cfg.eval.dense_epochs)
    try:
        dense_lr = float(cfg.eval.dense_lr)
    except (TypeError, ValueError) as error:
        raise ValueError("eval.dense_lr must be a positive finite number") from error
    if not math.isfinite(dense_lr) or dense_lr <= 0:
        raise ValueError("eval.dense_lr must be a positive finite number")
    _require_positive_int("model.num_frames", cfg.model.num_frames)
    _require_positive_int("model.image_size", cfg.model.image_size)
    if "mask_ratio" in cfg.model:
        mask_ratio = float(cfg.model.mask_ratio)
        if not math.isfinite(mask_ratio) or not 0 < mask_ratio < 1:
            raise ValueError("model.mask_ratio must be finite and in (0, 1)")


def _validate_probe_config(cfg: DictConfig) -> None:
    """Reject invalid probe matrices before adapters, loggers, or models are created."""
    task = str(cfg.eval.task)
    if task not in {"classification", "count", "dense"}:
        raise ValueError(f"eval.task must be one of classification, count, or dense; got {task!r}")
    validate_probe_matrix(cfg)
    validate_probe_hyperparameters(cfg)
    condition = str(cfg.model.condition)
    configured_checkpoint = cfg.eval.checkpoint or cfg.model.get("checkpoint")
    revision = cfg.model.get("revision")
    validate_encoder_request(
        condition,
        checkpoint=str(configured_checkpoint) if configured_checkpoint else None,
        revision=str(revision) if revision is not None else None,
        device=str(cfg.eval.device),
        require_pinned_revision=condition != "random" and configured_checkpoint is not None,
    )


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_seed(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32)")


def _config_values(name: str, value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty list")
    try:
        return list(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty list") from error


def _is_resource_error(error: Exception) -> bool:
    return isinstance(
        error,
        (MemoryError, ResourceUnavailableError, torch.OutOfMemoryError),
    )


def _prepare_probe_manifest(cfg: DictConfig, manifest: DatasetManifest) -> DatasetManifest:
    split = cfg.data.get("split")
    return prepare_manifest_splits(
        manifest,
        val_frac=float(split.val_frac) if split else None,
        test_frac=float(split.test_frac) if split else None,
        seed=int(cfg.seed),
    )


def has_labelled_splits(manifest: DatasetManifest, min_frames: int = 0) -> bool:
    """Report whether train and val each hold a labelled video of usable length."""
    return all(
        any(
            record.split == split
            and record.num_frames >= min_frames
            and (record.dataset == "synthetic" or record.annotation_path is not None)
            for record in manifest.records
        )
        for split in ("train", "val")
    )


def _evaluate_runs(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
    runs: Sequence[ProbeRun],
) -> tuple[ProbeResult, ...]:
    if str(cfg.eval.task) == "dense":
        return _evaluate_dense_runs(cfg, manifest, adapter, encoder, runs)
    try:
        train = _extract_features(cfg, manifest, adapter, encoder, split="train")
        validation = _extract_features(cfg, manifest, adapter, encoder, split="val")
    except LabelsUnavailableError:
        return tuple(probe_result(run, "SKIPPED_UNAVAILABLE_LABELS") for run in runs)
    if validation.labels.unique().numel() < 2:
        return tuple(probe_result(run, "SKIPPED_DEGENERATE_LABELS") for run in runs)
    if len(runs) == 1:
        _write_representation_diagnostics(
            cfg, runs[0], train, validation, manifest, adapter, encoder
        )
    results = []
    for run in runs:
        selected = set(run.selected_record_ids)
        indices = [
            index for index, record_id in enumerate(train.record_ids) if record_id in selected
        ]
        if not indices:
            results.append(probe_result(run, "SKIPPED_UNAVAILABLE_LABELS"))
            continue
        selected_labels = train.labels[indices]
        if selected_labels.unique().numel() < 2:
            results.append(probe_result(run, "SKIPPED_DEGENERATE_LABELS"))
            continue
        probe = fit_linear_probe(
            encoder,
            train.features[indices],
            selected_labels,
            task=run.task,
        )
        metric, value = evaluate_probe(
            probe,
            validation.features,
            validation.labels,
            task=run.task,
        )
        results.append(
            ProbeResult(
                condition=run.condition,
                checkpoint=run.checkpoint,
                manifest_checksum=run.manifest_checksum,
                dataset=run.dataset,
                task=run.task,
                fraction=run.fraction,
                seed=run.seed,
                metric=metric,
                value=value,
                status="COMPLETED",
                model=run.model,
                device=run.device,
                evaluation_split="val",
                **_subset_metadata(run.selected_record_ids),
            )
        )
        test_records = [record for record in manifest.records if record.split == "test"]
        if not bool(cfg.eval.get("report_test", False)) or not test_records:
            continue
        try:
            test = _extract_features(cfg, manifest, adapter, encoder, split="test")
        except LabelsUnavailableError:
            continue
        final_probe = probe
        metric, value = evaluate_probe(final_probe, test.features, test.labels, task=run.task)
        results.append(
            ProbeResult(
                condition=run.condition,
                checkpoint=run.checkpoint,
                manifest_checksum=run.manifest_checksum,
                dataset=run.dataset,
                task=run.task,
                fraction=run.fraction,
                seed=run.seed,
                metric=metric,
                value=value,
                status="COMPLETED",
                model=run.model,
                device=run.device,
                evaluation_split="test",
                **_subset_metadata(run.selected_record_ids),
            )
        )
        results.extend(smd_source_results(run, final_probe, test))
    return tuple(results)


def smd_source_results(run: ProbeRun, probe: Any, test: _FeatureSet) -> list[ProbeResult]:
    """Score each SMD capture source separately so domain shift is visible per subset."""
    rows: list[ProbeResult] = []
    pairs = list(zip(test.datasets, test.sources, strict=True))
    for source in sorted({source for dataset, source in pairs if dataset == "smd"}):
        indices = [
            index
            for index, (dataset, value) in enumerate(pairs)
            if dataset == "smd" and value == source
        ]
        metric, value = evaluate_probe(
            probe, test.features[indices], test.labels[indices], task=run.task
        )
        rows.append(
            replace(
                probe_result(run, "COMPLETED", metric=metric, value=value),
                dataset=f"smd/{source}",
                evaluation_split="test",
            )
        )
    return rows


def _write_representation_diagnostics(
    cfg: DictConfig,
    run: ProbeRun,
    references: _FeatureSet,
    queries: _FeatureSet,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
) -> Path:
    rows = nearest_neighbour_diagnostic(
        queries.features,
        references.features,
        query_ids=queries.record_ids,
        reference_ids=references.record_ids,
        max_references=min(1024, len(references.record_ids)),
        seed=run.seed,
    )
    payload = {
        "checkpoint": run.checkpoint,
        "manifest_checksum": run.manifest_checksum,
        "reference_split": "train",
        "query_split": "val",
        "retrieval": list(rows),
        "query_sources": list(queries.sources),
        "masked_reconstruction": _checkpoint_reconstruction_diagnostic(
            cfg, manifest, adapter, encoder
        ),
    }
    path = Path(cfg.output_dir) / "representation_diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _checkpoint_reconstruction_diagnostic(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
) -> dict[str, str | float | int]:
    if str(cfg.model.condition) != "maritime_videomae":
        return {"status": "unavailable_for_encoder_only_condition"}
    checkpoint = Path(str(cfg.eval.checkpoint or cfg.model.checkpoint))
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_config, dict)
    for key in ("condition", "checkpoint", "revision", "pretrained"):
        model_config.pop(key, None)
    module = VideoMAEPretrainingModule(model_config, seed=int(cfg.seed))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    dataset = _build_probe_dataset(cfg, manifest, adapter, encoder, split="val")
    if not len(dataset):
        raise LabelsUnavailableError("masked reconstruction requires a validation clip")
    pixels = dataset[0]["pixel_values"].unsqueeze(0)
    mask = module.make_mask(1, torch.device("cpu"), step=0)
    with torch.inference_mode():
        loss = module.model(pixel_values=pixels, bool_masked_pos=mask).loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError("masked reconstruction produced a non-finite loss")
    return {
        "status": "completed",
        "masked_loss": float(loss),
        "masked_tokens": int(mask.sum()),
        "clips": 1,
    }


def _log_diagnostic_artifact(path: Path, logger: Any) -> None:
    if logger is False or not path.is_file():
        return
    provenance = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    artifact = _wandb_artifact_type()(
        name=f"representation-diagnostics-{provenance}", type="evaluation"
    )
    artifact.add_file(str(path), name=path.name)
    logger.experiment.log_artifact(artifact)


def _prepare_diagnostic_path(output_dir: Path) -> Path:
    """Reserve this invocation's diagnostic path, removing any stale predecessor."""
    path = output_dir / "representation_diagnostics.json"
    path.unlink(missing_ok=True)
    return path


def _evaluate_dense_runs(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
    runs: Sequence[ProbeRun],
) -> tuple[ProbeResult, ...]:
    train_dataset = _build_probe_dataset(cfg, manifest, adapter, encoder, split="train")
    validation_dataset = _build_probe_dataset(cfg, manifest, adapter, encoder, split="val")
    validation_batches = _dense_batch_factory(
        validation_dataset,
        encoder,
        batch_size=int(cfg.eval.batch_size),
    )
    results = []
    for run in runs:
        selected = set(run.selected_record_ids)
        train_batches = _dense_batch_factory(
            train_dataset,
            encoder,
            batch_size=int(cfg.eval.batch_size),
            record_ids=selected,
        )
        try:
            probe = fit_dense_probe_streaming(
                encoder,
                train_batches,
                num_classes=2,
                epochs=int(cfg.eval.dense_epochs),
                lr=float(cfg.eval.dense_lr),
                seed=run.seed,
            )
        except DegenerateLabelsError:
            results.append(probe_result(run, "SKIPPED_DEGENERATE_LABELS"))
            continue
        except LabelsUnavailableError:
            results.append(probe_result(run, "SKIPPED_UNAVAILABLE_LABELS"))
            continue
        try:
            metric, value = evaluate_dense_probe_streaming(probe, validation_batches)
        except LabelsUnavailableError:
            results.append(probe_result(run, "SKIPPED_UNAVAILABLE_LABELS"))
            continue
        except DegenerateLabelsError:
            results.append(probe_result(run, "SKIPPED_DEGENERATE_LABELS"))
            continue
        results.append(probe_result(run, "COMPLETED", metric=metric, value=value))
        has_test_records = any(record.split == "test" for record in manifest.records)
        if bool(cfg.eval.get("report_test", False)) and has_test_records:
            test_dataset = _build_probe_dataset(cfg, manifest, adapter, encoder, split="test")
            test_batches = _dense_batch_factory(
                test_dataset,
                encoder,
                batch_size=int(cfg.eval.batch_size),
            )
            try:
                test_metric, test_value = evaluate_dense_probe_streaming(probe, test_batches)
            except (LabelsUnavailableError, DegenerateLabelsError):
                results.append(
                    replace(
                        probe_result(run, "SKIPPED_UNAVAILABLE_LABELS"), evaluation_split="test"
                    )
                )
            else:
                results.append(
                    replace(
                        probe_result(run, "COMPLETED", metric=test_metric, value=test_value),
                        evaluation_split="test",
                    )
                )
    return tuple(results)


def probe_result(
    run: ProbeRun,
    status: str,
    *,
    metric: str | None = None,
    value: float | None = None,
) -> ProbeResult:
    """Build one result row, which differs between outcomes only by status and value."""
    return ProbeResult(
        condition=run.condition,
        checkpoint=run.checkpoint,
        manifest_checksum=run.manifest_checksum,
        dataset=run.dataset,
        task=run.task,
        fraction=run.fraction,
        seed=run.seed,
        metric=run.metric if metric is None else metric,
        value=value,
        status=status,
        model=run.model,
        device=run.device,
        **_subset_metadata(run.selected_record_ids),
    )


def _extract_features(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
    *,
    split: str,
) -> _FeatureSet:
    dataset = _build_probe_dataset(cfg, manifest, adapter, encoder, split=split)
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    record_ids: list[str] = []
    sources: list[str] = []
    datasets: list[str] = []
    batch_size = int(cfg.eval.batch_size)
    if batch_size <= 0:
        raise ValueError("probe batch_size must be positive")
    for start in range(0, len(dataset), batch_size):
        decoded = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        samples = [sample for sample in decoded if sample is not None]
        labelled = [
            (sample, label)
            for sample in samples
            if (label := _supervised_scalar_label(sample, str(cfg.eval.task))) is not None
        ]
        if not labelled:
            continue
        samples, scalar_labels = zip(*labelled, strict=True)
        if not samples:
            continue
        encoded = _encode_samples(encoder, samples)
        feature_batches.append(encoded.global_features.cpu())
        label_batches.append(torch.tensor(scalar_labels, dtype=torch.long))
        record_ids.extend(str(sample["record_id"]) for sample in samples)
        sources.extend(str(sample["source"]) for sample in samples)
        datasets.extend(str(sample["dataset"]) for sample in samples)
    if not feature_batches:
        raise LabelsUnavailableError(f"{split} split produced no labelled probe clips")
    return _FeatureSet(
        torch.cat(feature_batches),
        torch.cat(label_batches),
        tuple(record_ids),
        tuple(sources),
        tuple(datasets),
    )


def _build_probe_dataset(
    cfg: DictConfig,
    manifest: DatasetManifest,
    adapter: DatasetAdapter,
    encoder: FrozenVideoEncoder,
    *,
    split: str,
) -> MaritimeClipDataset:
    size = int(cfg.model.image_size)
    processor_owns_geometry = getattr(encoder, "processor", None) is not None
    decoder = (
        SyntheticVideoDecoder(size, size) if manifest.name == "synthetic" else AutoVideoDecoder()
    )
    return MaritimeClipDataset(
        manifest,
        decoder,
        split=split,
        frames=int(cfg.model.num_frames),
        stride=1,
        image_size=None if processor_owns_geometry else size,
        seed=int(cfg.seed),
        target_loader=adapter.load_targets,
        normalization_mean=(
            tuple(cfg.data.transforms.normalization.mean) if not processor_owns_geometry else None
        ),
        normalization_std=(
            tuple(cfg.data.transforms.normalization.std) if not processor_owns_geometry else None
        ),
    )


def _dense_batch_factory(
    dataset: MaritimeClipDataset,
    encoder: FrozenVideoEncoder,
    *,
    batch_size: int,
    record_ids: set[str] | None = None,
) -> Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    if batch_size <= 0:
        raise ValueError("probe batch_size must be positive")

    def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(dataset), batch_size):
            samples = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
                if record_ids is None or dataset.clips[index].record_id in record_ids
            ]
            samples = [sample for sample in samples if bool(sample["is_labelled"])]
            if not samples:
                continue
            with torch.inference_mode():
                spatial = _encode_samples(encoder, samples).spatial_features
            if spatial is None:
                raise ValueError("dense probe requires encoder spatial features")
            spatial = spatial.detach().cpu().clone()
            labels = torch.stack(
                [
                    _dense_token_labels(
                        _sample_with_encoder_geometry(sample, encoder),
                        spatial_shape=spatial.shape[1:4],
                    )
                    for sample in samples
                ]
            )
            yield spatial, labels

    return batches


def _encode_samples(
    encoder: FrozenVideoEncoder,
    samples: Sequence[dict[str, Any]],
) -> EncoderFeatures:
    if getattr(encoder, "processor", None) is None:
        return encoder.encode(torch.stack([sample["pixel_values"] for sample in samples]))
    encoded = [encoder.encode(sample["pixel_values"].unsqueeze(0)) for sample in samples]
    global_features = torch.cat([features.global_features for features in encoded])
    spatial_batches = [features.spatial_features for features in encoded]
    spatial = None
    if all(features is not None for features in spatial_batches):
        spatial = torch.cat([features for features in spatial_batches if features is not None])
    return EncoderFeatures(global_features, spatial)


def _sample_with_encoder_geometry(
    sample: dict[str, Any], encoder: FrozenVideoEncoder
) -> dict[str, Any]:
    transform = sample["spatial_transform"]
    spatial_transform = getattr(encoder, "spatial_transform", None)
    if not callable(spatial_transform) or getattr(encoder, "processor", None) is None:
        return sample
    return {**sample, "spatial_transform": spatial_transform(transform.source_size)}


def _dense_token_labels(
    sample: dict[str, Any],
    *,
    spatial_shape: Sequence[int],
) -> torch.Tensor:
    temporal_tokens, rows, columns = map(int, spatial_shape)
    labels = torch.zeros(temporal_tokens, rows, columns, dtype=torch.long)
    frame_indices = [int(index) for index in sample["frame_indices"]]
    transform: SpatialTransform = sample["spatial_transform"]
    targets: Sequence[FrameTargets] = sample["targets"]
    if not targets and sample["dataset"] == "synthetic":
        source_height, source_width = transform.source_size
        targets = tuple(
            FrameTargets(
                frame_index=frame_index,
                boxes_xyxy=np.array(
                    [[0.0, 0.0, source_width / 2, source_height / 2]],
                    dtype=np.float32,
                ),
                class_ids=np.array([1], dtype=np.int64),
            )
            for position, frame_index in enumerate(frame_indices)
            if position % 2 == 0
        )
    frame_positions = {frame_index: position for position, frame_index in enumerate(frame_indices)}
    output_height, output_width = transform.output_size
    for target in targets:
        if target.frame_index not in frame_positions:
            continue
        temporal = min(
            frame_positions[target.frame_index] * temporal_tokens // len(frame_indices),
            temporal_tokens - 1,
        )
        for x1, y1, x2, y2 in transform.apply_boxes_xyxy(target.boxes_xyxy):
            left = max(0, min(columns, int(np.floor(x1 * columns / output_width))))
            right = max(0, min(columns, int(np.ceil(x2 * columns / output_width))))
            top = max(0, min(rows, int(np.floor(y1 * rows / output_height))))
            bottom = max(0, min(rows, int(np.ceil(y2 * rows / output_height))))
            if left < right and top < bottom:
                labels[temporal, top:bottom, left:right] = 1
    return labels


def _sample_label(sample: dict[str, Any], task: str) -> int | None:
    targets: tuple[FrameTargets, ...] = sample["targets"]
    if not targets and sample["dataset"] == "synthetic":
        start = int(sample["frame_indices"][0])
        return (start // len(sample["frame_indices"])) % 2
    if task == "count":
        return _count_bin(max((len(target.boxes_xyxy) for target in targets), default=0))
    if task != "classification":
        raise ValueError(f"unsupported linear probe task: {task}")
    annotated = [target.class_ids for target in targets if len(target.class_ids)]
    if not annotated:
        return None
    class_ids = np.concatenate(annotated)
    return Counter(class_ids.tolist()).most_common(1)[0][0]


def _supervised_scalar_label(sample: dict[str, Any], task: str) -> int | None:
    if not bool(sample["is_labelled"]):
        return None
    return _sample_label(sample, task)


def _labelled_training_records(manifest: DatasetManifest, min_frames: int) -> tuple[Any, ...]:
    return tuple(
        record
        for record in manifest.records
        if record.split == "train"
        and record.num_frames >= min_frames
        and (record.dataset == "synthetic" or record.annotation_path is not None)
    )


def _subset_metadata(record_ids: Sequence[str] | None) -> dict[str, str | None]:
    if record_ids is None:
        return {"label_subset_checksum": None, "labelled_record_ids": None}
    ids = tuple(sorted(record_ids))
    payload = json.dumps(ids, separators=(",", ":"))
    return {
        "label_subset_checksum": hashlib.sha256(payload.encode()).hexdigest(),
        "labelled_record_ids": payload,
    }


def _count_bin(count: int) -> int:
    """Map vessel counts to stable 0, 1, 2, and 3+ classes."""
    if count < 0:
        raise ValueError("count cannot be negative")
    return min(count, 3)


def _checkpoint_identity(checkpoint: str, revision: str | None = None) -> str:
    """Identify a checkpoint by content, falling back to its reference."""
    path = Path(checkpoint).expanduser()
    if (identity := content_identity(path)) is not None:
        return identity
    return f"{checkpoint}@{revision}" if revision else checkpoint


def _wandb_table(table: Any) -> Any:
    import wandb

    return wandb.Table(dataframe=table)


def _wandb_artifact_type() -> Any:
    import wandb

    return wandb.Artifact


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run probes from Hydra CLI overrides."""
    print(aggregate_probe_results(run_evaluation(cfg)).to_string(index=False))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()

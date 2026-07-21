"""Stable, offline-safe experiment tracking helpers."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from omegaconf import DictConfig, OmegaConf

from marineworld.data.manifest import directory_checksum, file_checksum

if TYPE_CHECKING:
    from pytorch_lightning.loggers import WandbLogger


SECRET_KEY_SUFFIXES = ("apikey", "password", "secret", "token")
SCIENTIFIC_CONFIG_SECTIONS = ("seed", "data", "model", "train")
SCIENTIFIC_RUNTIME_KEYS = (
    "precision",
    "batch_size",
    "accumulate_grad_batches",
    "max_steps",
    "limit_train_batches",
    "limit_val_batches",
)
OPERATIONAL_CONFIG_KEYS = frozenset({"root"})


@dataclass(frozen=True)
class RunIdentity:
    """A unique execution attempt linked to a reproducible scientific condition."""

    run_id: str
    group: str
    tags: tuple[str, ...]
    condition_id: str = ""


def resolved_config(cfg: DictConfig) -> dict[str, Any]:
    """Return allowlisted scientific settings safe for experiment logging."""
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(resolved, dict)
    scientific = {key: resolved[key] for key in SCIENTIFIC_CONFIG_SECTIONS if key in resolved}
    if isinstance(runtime := resolved.get("runtime"), dict):
        scientific["runtime"] = {
            key: runtime[key] for key in SCIENTIFIC_RUNTIME_KEYS if key in runtime
        }
    return _redact_config(scientific)


def build_run_identity(
    model: str,
    manifest_checksums: tuple[str, ...],
    seed: int,
    label_fraction: float | None,
    *,
    git_sha: str | None = None,
    accelerator: str | None = None,
    checkpoint_provenance: str | None = None,
    logical_dimensions: Mapping[str, Any] | None = None,
) -> RunIdentity:
    """Build a portable condition fingerprint and a unique execution attempt ID."""
    condition = {
        "label_fraction": label_fraction,
        "manifest_checksums": sorted(manifest_checksums),
        "model": model,
        "seed": seed,
        "logical_dimensions": dict(logical_dimensions or {}),
    }
    payload = json.dumps(condition, sort_keys=True, separators=(",", ":"))
    condition_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    run_id = secrets.token_hex(8)
    tags = [f"condition:{condition_id}", f"model:{model}", f"seed:{seed}"]
    if label_fraction is not None:
        tags.append(f"label_fraction:{label_fraction:g}")
    for name, value in (
        ("git", git_sha),
        ("accelerator", accelerator),
        ("checkpoint", checkpoint_provenance),
    ):
        if value:
            if name == "checkpoint":
                value = Path(value).name
            tags.append(f"{name}:{value}")
    return RunIdentity(
        run_id=run_id,
        group=condition_id,
        tags=tuple(tags),
        condition_id=condition_id,
    )


def build_wandb_logger(cfg: DictConfig, identity: RunIdentity) -> WandbLogger | Literal[False]:
    """Construct W&B logging without artifacts for offline runs."""
    mode = str(cfg.tracking.mode)
    if mode == "disabled":
        return False
    if mode not in {"online", "offline"}:
        raise ValueError(f"unsupported W&B mode: {mode}")
    return _create_wandb_logger(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        id=identity.run_id,
        resume="never",
        group=identity.group,
        tags=list(identity.tags),
        offline=mode == "offline",
        log_model="all" if mode == "online" else False,
        config=resolved_config(cfg),
    )


def metric_name(stage: str, metric: str, dataset: str | None = None) -> str:
    """Return a slash-separated metric name in the shared namespace."""
    return "/".join(part for part in (stage, dataset, metric) if part)


def _create_wandb_logger(**kwargs: Any) -> WandbLogger:
    """Import Lightning lazily so disabled tracking has no W&B import side effects."""
    from pytorch_lightning.loggers import WandbLogger

    return WandbLogger(**kwargs)


def _redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (_checkpoint_identity(item) if _is_checkpoint_key(key) else _redact_config(item))
            for key, item in value.items()
            if not _is_secret_key(key) and key not in OPERATIONAL_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


def _checkpoint_identity(value: Any) -> Any:
    if not isinstance(value, str):
        return _redact_config(value)
    path = Path(value).expanduser()
    if path.is_file():
        return f"sha256:{file_checksum(path)}"
    if path.is_dir():
        return f"sha256-directory:{directory_checksum(path)}"
    if path.is_absolute() or value.startswith("~"):
        return f"unavailable:{path.name}"
    return value


def _is_checkpoint_key(key: object) -> bool:
    return _normalize_key(key).endswith("checkpoint")


def _is_secret_key(key: object) -> bool:
    normalized = _normalize_key(key)
    return normalized.endswith(SECRET_KEY_SUFFIXES)


def _normalize_key(key: object) -> str:
    """Normalize snake, kebab, and camel keys for credential matching."""
    return "".join(character.lower() for character in str(key) if character.isalnum())

"""Stable, offline-safe experiment tracking helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from pytorch_lightning.loggers import WandbLogger


_SECRET_KEY_SUFFIXES = ("apikey", "password", "secret", "token")
_PRIVATE_CONFIG_PATHS = {
    ("data", "root"),
    ("output_dir",),
    ("runtime", "ckpt_path"),
}


@dataclass(frozen=True)
class RunIdentity:
    """The deterministic W&B identity for one logical experiment."""

    run_id: str
    group: str
    tags: tuple[str, ...]


def resolved_config(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra config into primitive, credential-free values."""
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(resolved, dict)
    return _sanitize_config(resolved)


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
    """Build a run ID independent of source revision and execution hardware."""
    condition = {
        "label_fraction": label_fraction,
        "manifest_checksums": sorted(manifest_checksums),
        "model": model,
        "seed": seed,
    }
    if logical_dimensions:
        condition["logical_dimensions"] = dict(logical_dimensions)
    payload = json.dumps(condition, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    tags = [f"model:{model}", f"seed:{seed}"]
    if label_fraction is not None:
        tags.append(f"label_fraction:{label_fraction:g}")
    for name, value in (
        ("git", git_sha),
        ("accelerator", accelerator),
        ("checkpoint", checkpoint_provenance),
    ):
        if value:
            tags.append(f"{name}:{value}")
    return RunIdentity(run_id=run_id, group=model, tags=tuple(tags))


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


def _sanitize_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_config(item, path + (str(key),))
            for key, item in value.items()
            if not _is_secret_key(key) and not _is_private_config_path(path + (str(key),))
        }
    if isinstance(value, list):
        return [_sanitize_config(item, path) for item in value]
    return value


def _is_private_config_path(path: tuple[str, ...]) -> bool:
    return path in _PRIVATE_CONFIG_PATHS or (
        len(path) == 4 and path[:2] == ("data", "components") and path[-1] == "root"
    )


def _is_secret_key(key: object) -> bool:
    normalized = _normalize_key(key)
    return normalized.endswith(_SECRET_KEY_SUFFIXES)


def _normalize_key(key: object) -> str:
    """Normalize snake, kebab, and camel keys for credential matching."""
    return "".join(character.lower() for character in str(key) if character.isalnum())

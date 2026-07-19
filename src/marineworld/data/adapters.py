"""Common protocol and configuration-driven construction for dataset adapters."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Protocol

from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord


class DatasetAdapter(Protocol):
    """Build canonical records and retrieve their dataset-specific targets."""

    def build_manifest(self, root: Path) -> DatasetManifest: ...

    def load_targets(self, record: VideoRecord) -> tuple[FrameTargets, ...]: ...


def build_adapter(config: Mapping[str, Any]) -> DatasetAdapter:
    """Instantiate an adapter from a Hydra-style ``_target_`` mapping."""
    target = str(config["_target_"])
    adapter_type = _import_string(target)
    kwargs = {key: value for key, value in config.items() if not key.startswith("_")}
    return adapter_type(**kwargs)


def _import_string(target: str) -> type[DatasetAdapter]:
    module_name, separator, attribute = target.rpartition(".")
    if not separator:
        raise ValueError(f"adapter target must be a dotted path: {target}")
    return getattr(import_module(module_name), attribute)

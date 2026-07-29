"""Common protocol and configuration-driven construction for dataset adapters."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Protocol

from marineworld.data.contracts import DatasetManifest, FrameTargets, VideoRecord
from marineworld.data.manifest import manifest_checksum

FRAME_COUNT_CACHE_NAME = ".marineworld_frame_counts.json"
QUARANTINE_NAME = ".marineworld_quarantine.json"

logger = logging.getLogger(__name__)


def write_sidecar_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Best-effort write of a JSON file beside the data; never fail a build on it.

    Frame counts and quarantine records are aids that live next to the corpus and
    are safe to delete; a read-only data directory degrades to not persisting them
    rather than raising.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except OSError:
        pass


def collect_records(
    clips: Iterable[Path],
    record: Callable[[Path], VideoRecord],
    root: Path,
    dataset: str,
) -> tuple[VideoRecord, ...]:
    """Build records from clips, quarantining any that cannot be probed or decoded.

    A clip whose probe/decode fails is set aside -- recorded in a JSON sidecar and
    logged -- rather than failing the whole build: the corpus is otherwise usable,
    and a dead clip that never enters the manifest cannot hang a dataloader later.
    An all-undecodable corpus still raises, since an empty manifest is a broken one.
    """
    records: list[VideoRecord] = []
    quarantine: dict[str, str] = {}
    for clip in clips:
        try:
            records.append(record(clip))
        except (ValueError, RuntimeError) as error:
            # TODO: narrow to a dedicated UndecodableClip sentinel raised at the
            # probe/decode boundary, so a genuine config or record-invariant bug is
            # not masked as a bad clip. Acceptable at v0: uniform per-clip config
            # means a real bug fails every clip and trips the empty guard below.
            quarantine[clip.relative_to(root).as_posix()] = str(error)
    report_quarantine(root, quarantine, dataset)
    if quarantine and not records:
        # A corpus that had clips but decoded none is broken in a specific way worth
        # naming here; an empty corpus (no clips at all) is left to the downstream
        # manifest validation, which owns the generic "no records" invariant.
        raise ValueError(
            f"every {dataset} clip under {root} failed to decode "
            f"(quarantined {len(quarantine)}); see {root / QUARANTINE_NAME}"
        )
    return tuple(records)


def report_quarantine(root: Path, quarantine: Mapping[str, str], dataset: str) -> None:
    """Warn about and persist clips dropped from a manifest as undecodable."""
    path = root / QUARANTINE_NAME
    if not quarantine:
        try:
            path.unlink(missing_ok=True)  # keep a stale audit from misleading a later build
        except OSError:
            pass
        return
    logger.warning(
        "%s quarantined %d undecodable clip(s), excluded from the manifest: %s",
        dataset,
        len(quarantine),
        ", ".join(sorted(quarantine)),
    )
    write_sidecar_json(path, quarantine)


@dataclass
class FrameCountCache:
    """Persist verified frame counts, keyed on file size and mtime, across runs.

    Counting the frames of a damaged video means decoding the whole stream, which
    is paid once per corpus rather than on every manifest build. The cache lives
    beside the data and is safe to delete; a read-only data directory falls back
    to recomputing rather than failing the build.
    """

    path: Path
    entries: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def for_root(cls, root: Path) -> FrameCountCache:
        cache = cls(Path(root) / FRAME_COUNT_CACHE_NAME)
        try:
            loaded = json.loads(cache.path.read_text())
            cache.entries = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            cache.entries = {}
        return cache

    def resolve(self, video: Path, probe: Callable[[Path], int]) -> int:
        # Keyed on (size, mtime) rather than content, unlike file_checksum: a stale
        # count is a self-correcting optimisation (over-counts drop at decode time),
        # not an identity digest, so the same-mtime-rewrite hazard is acceptable here.
        try:
            stat = video.stat()
        except OSError:
            return probe(video)
        key = str(video.resolve())
        cached = self.entries.get(key)
        if cached and cached[:2] == [stat.st_size, stat.st_mtime_ns]:
            return cached[2]
        count = probe(video)
        self.entries[key] = [stat.st_size, stat.st_mtime_ns, count]
        write_sidecar_json(self.path, self.entries)
        return count


def resolve_frame_count(
    configured: int | None,
    video: Path,
    probe: Callable[[Path], int],
    dataset: str,
    cache: FrameCountCache | None = None,
) -> int:
    """Take the configured frame count, or probe the video, and require it positive.

    When a cache is given, the probe result is reused across runs so a damaged
    corpus is decoded to count its frames only when the file changes.
    """
    if configured is not None:
        count = configured
    elif cache is not None:
        count = cache.resolve(video, probe)
    else:
        count = probe(video)
    if count <= 0:
        raise ValueError(f"{dataset} frame count must be positive for {video}, got {count}")
    return count


def resolve_fps(
    configured: float | None, video: Path, probe: Callable[[Path], float], dataset: str
) -> float:
    """Take the configured FPS, or probe the video, and require it finite and positive."""
    fps = configured if configured is not None else probe(video)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{dataset} FPS must be positive for {video}, got {fps}")
    return fps


class DatasetAdapter(Protocol):
    """Build canonical records and retrieve their dataset-specific targets."""

    def build_manifest(self, root: Path) -> DatasetManifest: ...

    def load_targets(self, record: VideoRecord) -> tuple[FrameTargets, ...]: ...


@dataclass(frozen=True)
class CompositeAdapter:
    """Combine arbitrary adapters without erasing component provenance."""

    components: Mapping[str, DatasetAdapter]
    roots: Mapping[str, Path]

    def build_manifest(self, root: Path) -> DatasetManifest:
        del root
        manifests = {
            name: adapter.build_manifest(Path(self.roots[name]))
            for name, adapter in self.components.items()
        }
        if empty := sorted(name for name, manifest in manifests.items() if not manifest.records):
            detail = "; ".join(f"{name} at {Path(self.roots[name])}" for name in empty)
            raise ValueError(
                f"joint corpus component(s) resolved to zero videos: {detail}. "
                "Every component declared in the config must be present, otherwise the "
                "mixture silently becomes a different corpus than the one it is named for."
            )
        records = tuple(
            replace(record, id=f"{name}:{record.id}", dataset=name)
            for name, manifest in manifests.items()
            for record in manifest.records
        )
        return DatasetManifest(
            "maritime_joint",
            "+".join(manifest.version for manifest in manifests.values()),
            "mixed; see component manifests",
            records,
            access="restricted",
            component_checksums={
                name: manifest_checksum(manifest) for name, manifest in manifests.items()
            },
            components={
                name: {
                    "name": manifest.name,
                    "version": manifest.version,
                    "license": manifest.license,
                    "access": manifest.access,
                    "label_mapping": dict(manifest.label_mapping),
                    "native_labels": list(manifest.native_labels),
                }
                for name, manifest in manifests.items()
            },
        )

    def load_targets(self, record: VideoRecord) -> tuple[FrameTargets, ...]:
        name, separator, native_id = record.id.partition(":")
        if not separator or name not in self.components:
            raise ValueError(f"unknown composite record ID: {record.id}")
        return self.components[name].load_targets(replace(record, id=native_id, dataset=name))


def build_adapter(config: Mapping[str, Any]) -> DatasetAdapter:
    """Instantiate an adapter from a Hydra-style ``_target_`` mapping."""
    target = str(config["_target_"])
    adapter_type = _import_string(target)
    kwargs = {key: value for key, value in config.items() if not key.startswith("_")}
    return adapter_type(**kwargs)


def build_data_adapter(config: Mapping[str, Any]) -> DatasetAdapter:
    """Build either one adapter or a config-defined composite uniformly."""
    components = config.get("components")
    if not components:
        return build_adapter(config["adapter"])
    return CompositeAdapter(
        components={
            name: build_adapter(component["adapter"]) for name, component in components.items()
        },
        roots={name: Path(component["root"]) for name, component in components.items()},
    )


def _import_string(target: str) -> type[DatasetAdapter]:
    module_name, separator, attribute = target.rpartition(".")
    if not separator:
        raise ValueError(f"adapter target must be a dotted path: {target}")
    return getattr(import_module(module_name), attribute)

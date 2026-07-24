"""Frozen adapters for the v1 video encoder comparison matrix."""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from httpx import TransportError as HttpxTransportError
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
)
from requests import ConnectionError as RequestsConnectionError
from requests import Timeout as RequestsTimeout
from safetensors import SafetensorError, safe_open

from marineworld.data.clips import SpatialTransform
from marineworld.models.videomae import build_videomae, pair

EncoderCondition = Literal[
    "random",
    "generic_videomae",
    "maritime_videomae",
    "dinov3",
    "vjepa",
]


class ResourceUnavailableError(RuntimeError):
    """A requested accelerator is unavailable before model construction."""


@dataclass(frozen=True)
class EncoderFeatures:
    """Detached global and optional spatiotemporal encoder features."""

    global_features: torch.Tensor
    spatial_features: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.global_features.ndim != 2 or not self.global_features.is_floating_point():
            raise ValueError("global features must be a floating tensor shaped [B, D]")
        if self.spatial_features is not None:
            if self.spatial_features.ndim != 5 or not self.spatial_features.is_floating_point():
                raise ValueError(
                    "spatial features must be a floating tensor shaped [B, T, H, W, D]"
                )
            if self.spatial_features.shape[0] != self.global_features.shape[0]:
                raise ValueError("global and spatial feature batch sizes must match")


@runtime_checkable
class FrozenVideoEncoder(Protocol):
    """One frozen feature-extraction protocol for every encoder condition."""

    name: str

    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures: ...

    def spatial_transform(self, source_size: tuple[int, int]) -> SpatialTransform: ...


class FrozenEncoder(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        name: str,
        device: str | torch.device,
        processor: Any | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.processor = processor

    @property
    def device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def prepare(self, pixel_values: torch.Tensor) -> torch.Tensor:
        values = pixel_values
        if self.processor is None:
            return values.to(self.device)
        if callable(self.processor):
            processed = self.processor(
                [video.cpu() for video in values],
                return_tensors="pt",
                do_rescale=False,
            )
            key = "pixel_values_videos" if "pixel_values_videos" in processed else "pixel_values"
            return processed[key].to(self.device)
        values = values.to(self.device)
        mean = torch.as_tensor(self.processor.image_mean, device=self.device).view(1, 1, -1, 1, 1)
        std = torch.as_tensor(self.processor.image_std, device=self.device).view(1, 1, -1, 1, 1)
        return (values - mean) / std

    def spatial_transform(self, source_size: tuple[int, int]) -> SpatialTransform:
        """Return the processor's resize/center-crop geometry for dense labels."""
        if self.processor is None:
            height, width = source_size
            return SpatialTransform(source_size, source_size, (1.0, 1.0))
        return _processor_spatial_transform(self.processor, source_size)


class VideoMAEFrozenEncoder(FrozenEncoder):
    """Expose frozen VideoMAE tokens as global and dense features."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        name: str,
        device: str | torch.device = "cpu",
        processor: Any | None = None,
    ) -> None:
        backbone = model.videomae if hasattr(model, "videomae") else model
        super().__init__(backbone, name=name, device=device, processor=processor)

    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures:
        with torch.inference_mode():
            tokens = self.model(pixel_values=self.prepare(pixel_values)).last_hidden_state
            spatial = _reshape_tokens(tokens, self.model.config)
            return EncoderFeatures(tokens.mean(dim=1).detach(), spatial.detach())


class DINOv3FrozenEncoder(FrozenEncoder):
    """Apply a frozen image encoder frame-wise and restore temporal layout."""

    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures:
        batch_size, frames, channels, height, width = pixel_values.shape
        with torch.inference_mode():
            images = pixel_values.reshape(batch_size * frames, channels, height, width)
            if callable(self.processor):
                processed = self.processor(
                    [image.cpu() for image in images],
                    return_tensors="pt",
                    do_rescale=False,
                )
                images = processed["pixel_values"].to(self.device)
            else:
                prepared = self.prepare(pixel_values)
                images = prepared.reshape(batch_size * frames, channels, height, width)
            tokens = self.model(pixel_values=images).last_hidden_state
            spatial = _reshape_image_tokens(tokens, self.model.config, frames, batch_size)
            return EncoderFeatures(spatial.mean(dim=(1, 2, 3)).detach(), spatial.detach())


class VJEPAFrozenEncoder(FrozenEncoder):
    """Expose frozen V-JEPA video tokens through the common feature contract."""

    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures:
        with torch.inference_mode():
            output = self.model(
                pixel_values_videos=self.prepare(pixel_values),
                skip_predictor=True,
            )
            tokens = output.last_hidden_state
            spatial = _reshape_tokens(tokens, self.model.config)
            return EncoderFeatures(tokens.mean(dim=1).detach(), spatial.detach())


@dataclass(frozen=True)
class EncoderSpec:
    """How one condition builds an encoder, validates weights, and loads a processor."""

    encoder: type[FrozenEncoder]
    model_type: str | None = None
    model_loader: str = "AutoModel"
    processor: str = "AutoVideoProcessor"


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "random": EncoderSpec(encoder=VideoMAEFrozenEncoder),
    "generic_videomae": EncoderSpec(
        encoder=VideoMAEFrozenEncoder, model_type="videomae", model_loader="VideoMAEModel"
    ),
    "maritime_videomae": EncoderSpec(
        encoder=VideoMAEFrozenEncoder, model_type="videomae", model_loader="VideoMAEModel"
    ),
    "dinov3": EncoderSpec(
        encoder=DINOv3FrozenEncoder, model_type="dinov3_vit", processor="AutoImageProcessor"
    ),
    "vjepa": EncoderSpec(encoder=VJEPAFrozenEncoder, model_type="vjepa2"),
}
MODEL_CONDITIONS: tuple[EncoderCondition, ...] = tuple(ENCODER_SPECS)  # type: ignore[assignment]


def transformers_class(name: str) -> Any:
    """Resolve a Transformers class by name, keeping the heavy import lazy."""
    import transformers

    return getattr(transformers, name)


def load_frozen_encoder(
    condition: EncoderCondition,
    *,
    checkpoint: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    allow_download: bool = False,
) -> FrozenVideoEncoder:
    """Build a local random encoder or opt into a cached/pretrained reference."""
    revision = (
        str(model_config["revision"]) if model_config and model_config.get("revision") else None
    )
    validate_encoder_request(
        condition,
        checkpoint=checkpoint,
        revision=revision,
        device=device,
    )
    if condition == "random":
        if model_config is None:
            raise ValueError("random encoder requires model_config")
        return VideoMAEFrozenEncoder(build_videomae(model_config), name=condition, device=device)
    assert checkpoint is not None

    model = _load_pretrained_model(
        condition,
        checkpoint,
        model_config=model_config,
        device=device,
        allow_download=allow_download,
        revision=revision,
    )
    processor = _load_pretrained_processor(
        condition,
        checkpoint,
        allow_download=allow_download,
        revision=revision,
    )
    return ENCODER_SPECS[condition].encoder(
        model, name=condition, device=device, processor=processor
    )


def validate_encoder_request(
    condition: str,
    *,
    checkpoint: str | None,
    revision: str | None,
    device: str | torch.device,
    require_pinned_revision: bool = False,
) -> None:
    """Validate an encoder request without importing Transformers or loading weights."""
    if condition not in MODEL_CONDITIONS:
        raise ValueError(
            f"model.condition must be one of {', '.join(MODEL_CONDITIONS)}; got {condition!r}"
        )
    _validate_device(device)
    if condition == "random":
        if checkpoint:
            raise ValueError("random encoder does not accept a checkpoint")
        if revision:
            raise ValueError("random encoder does not accept model.revision")
        return
    if not checkpoint:
        raise ValueError(f"{condition} evaluation requires a non-null checkpoint")

    local_checkpoint = _local_checkpoint_path(checkpoint)
    if local_checkpoint is not None:
        validate_local_checkpoint_request(condition, local_checkpoint, revision)
        return
    validate_hub_checkpoint_request(checkpoint, revision, require_pinned_revision)


def validate_local_checkpoint_request(
    condition: str, checkpoint: Path, revision: str | None
) -> None:
    """Require a local checkpoint to exist and to carry no Hub revision."""
    if not checkpoint.exists():
        raise ValueError(f"local checkpoint does not exist: {checkpoint}")
    if revision:
        raise ValueError("model.revision must be null for a local checkpoint")
    _validate_local_checkpoint(condition, checkpoint)


def validate_hub_checkpoint_request(
    checkpoint: str, revision: str | None, require_pinned_revision: bool
) -> None:
    """Require a Hub reference to be owner/repository, pinned when weights are downloaded."""
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", checkpoint):
        raise ValueError(f"Hub checkpoint must use the 'owner/repository' form; got {checkpoint!r}")
    if require_pinned_revision and not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise ValueError("model.revision must be a pinned 40-character commit SHA")


def _validate_local_checkpoint(condition: str, checkpoint: Path) -> None:
    if checkpoint.suffix == ".ckpt":
        if not checkpoint.is_file():
            raise ValueError(f"Lightning checkpoint must be a file: {checkpoint}")
        if condition != "maritime_videomae":
            raise ValueError("Lightning checkpoints require model.condition=maritime_videomae")
        return
    if not checkpoint.is_dir():
        raise ValueError(
            f"local Hugging Face checkpoint must be a save_pretrained directory: {checkpoint}"
        )
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "local Hugging Face checkpoint must contain config.json and model weights: "
            f"{checkpoint}"
        )
    _validate_model_config(condition, config_path)
    safetensors_path = checkpoint / "model.safetensors"
    if safetensors_path.is_file():
        _validate_safetensors(safetensors_path)
        return
    pytorch_path = checkpoint / "pytorch_model.bin"
    if pytorch_path.is_file():
        _validate_pytorch_archive(pytorch_path)
        return
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    indexes = [checkpoint / name for name in index_names if (checkpoint / name).is_file()]
    if not indexes:
        raise ValueError(
            f"local Hugging Face checkpoint has no recognized model weights: {checkpoint}"
        )
    for index in indexes:
        _validate_weight_index(checkpoint, index)


def _validate_model_config(condition: str, path: Path) -> None:
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid config.json in local Hugging Face checkpoint: {path}") from error
    if not isinstance(config, Mapping):
        raise ValueError(f"config.json must contain a JSON mapping: {path}")
    expected = ENCODER_SPECS[condition].model_type
    if config.get("model_type") != expected:
        raise ValueError(f"config.json model_type must be {expected!r} for {condition}: {path}")


def _validate_safetensors(path: Path) -> None:
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            keys = list(weights.keys())
            weights.metadata()
    except (OSError, SafetensorError) as error:
        raise ValueError(f"invalid safetensors weight container: {path}") from error
    if not keys:
        raise ValueError(f"invalid safetensors weight container has no tensors: {path}")


def _validate_pytorch_archive(path: Path) -> None:
    try:
        # weights_only blocks arbitrary pickle globals; mmap + meta validates tensor
        # metadata without copying checkpoint storage into host memory.
        state_dict = torch.load(
            path,
            map_location="meta",
            mmap=True,
            weights_only=True,
        )
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError(f"invalid PyTorch weight archive: {path}") from error
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or not all(isinstance(value, torch.Tensor) for value in state_dict.values())
    ):
        raise ValueError(f"invalid PyTorch weight archive: {path}")


def _validate_weight_index(checkpoint: Path, index: Path) -> None:
    try:
        payload = json.loads(index.read_text())
        weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
        if not isinstance(weight_map, Mapping):
            raise ValueError
        shards = set(weight_map.values())
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Hugging Face weight index: {index}") from error
    if not shards or any(not isinstance(shard, str) or not shard for shard in shards):
        raise ValueError(f"invalid Hugging Face weight index: {index}")
    for shard in sorted(shards):
        relative_path = Path(shard)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid weight shard path {shard!r} in {index}")
        shard_path = checkpoint / relative_path
        if not shard_path.is_file():
            raise ValueError(f"missing referenced weight shard {shard!r} from {index}")
        if index.name == "model.safetensors.index.json":
            if shard_path.suffix != ".safetensors":
                raise ValueError(f"invalid safetensors shard name {shard!r} in {index}")
            _validate_safetensors(shard_path)
        else:
            if shard_path.suffix != ".bin":
                raise ValueError(f"invalid PyTorch shard name {shard!r} in {index}")
            _validate_pytorch_archive(shard_path)


def _load_pretrained_model(
    condition: EncoderCondition,
    checkpoint: str,
    *,
    model_config: Mapping[str, Any] | None,
    device: str | torch.device,
    allow_download: bool,
    revision: str | None,
) -> torch.nn.Module:
    local_only = not allow_download
    pretrained_kwargs: dict[str, Any] = {"local_files_only": local_only}
    if revision:
        pretrained_kwargs["revision"] = revision
    if condition == "maritime_videomae" and checkpoint.endswith(".ckpt"):
        if model_config is None:
            raise ValueError("Lightning maritime checkpoints require model_config")
        from marineworld.train.module import VideoMAEPretrainingModule

        module = VideoMAEPretrainingModule.load_from_checkpoint(
            checkpoint,
            model_config=model_config,
            map_location=device,
        )
        return module.model.videomae

    loader = transformers_class(ENCODER_SPECS[condition].model_loader)
    try:
        return loader.from_pretrained(checkpoint, **pretrained_kwargs)
    except Exception as error:
        _raise_known_hub_failure(error, "checkpoint", checkpoint, local_only)
        raise


def _load_pretrained_processor(
    condition: EncoderCondition,
    checkpoint: str,
    *,
    allow_download: bool,
    revision: str | None,
) -> Any | None:
    if condition == "maritime_videomae" and checkpoint.endswith(".ckpt"):
        return None
    processor_type = transformers_class(ENCODER_SPECS[condition].processor)
    try:
        kwargs: dict[str, Any] = {"local_files_only": not allow_download}
        if revision:
            kwargs["revision"] = revision
        return processor_type.from_pretrained(checkpoint, **kwargs)
    except Exception as error:
        _raise_known_hub_failure(error, "processor", checkpoint, not allow_download)
        raise


def _raise_known_hub_failure(
    error: Exception,
    resource: str,
    checkpoint: str,
    local_only: bool,
) -> None:
    errors = tuple(_exception_chain(error))
    if any(_is_hard_hub_error(item) for item in errors):
        return
    remote_cache_miss = (
        local_only
        and _local_checkpoint_path(checkpoint) is None
        and any(_is_cache_miss_os_error(item) for item in errors)
    )
    known_hub_failure = any(_is_resource_hub_error(item) for item in errors)
    if known_hub_failure or remote_cache_miss:
        suffix = " in the local cache" if remote_cache_miss else ""
        raise ResourceUnavailableError(
            f"{resource} is unavailable{suffix}: {checkpoint}"
        ) from error


def _exception_chain(error: Exception) -> list[Exception]:
    chain = [error]
    seen: set[int] = set()
    current = error
    while _is_neutral_wrapper(current):
        if id(current) in seen:
            break
        seen.add(id(current))
        nested = current.__cause__
        if nested is None and not current.__suppress_context__:
            nested = current.__context__
        if not isinstance(nested, Exception):
            break
        chain.append(nested)
        current = nested
    return chain


def _is_neutral_wrapper(error: Exception) -> bool:
    if type(error) is not OSError or not isinstance(error.__cause__, Exception):
        return False
    message = str(error).lower()
    cause = error.__cause__
    wrappers = (
        (GatedRepoError, "you are trying to access a gated repo.\n"),
        (LocalEntryNotFoundError, "we couldn't connect to "),
        (HfHubHTTPError, "there was a specific connection error when trying to load "),
    )
    return any(isinstance(cause, kind) and message.startswith(prefix) for kind, prefix in wrappers)


def _is_hard_hub_error(error: Exception) -> bool:
    return (
        isinstance(error, HfHubHTTPError)
        and error.response is not None
        and error.response.status_code == 404
    )


def _is_resource_hub_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            GatedRepoError,
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
            HttpxTransportError,
            RequestsConnectionError,
            RequestsTimeout,
        ),
    ):
        return True
    if not isinstance(error, HfHubHTTPError) or error.response is None:
        return False
    status = error.response.status_code
    return status in {401, 403, 408, 429} or status >= 500


def _is_cache_miss_os_error(error: Exception) -> bool:
    if type(error) is not OSError:
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "not cached",
            "local cache",
            "couldn't connect",
            "offline mode",
        )
    )


def _reshape_tokens(tokens: torch.Tensor, config: Any) -> torch.Tensor:
    frames = int(getattr(config, "num_frames", getattr(config, "frames_per_clip", 0)))
    if not frames:
        raise ValueError("encoder config must define num_frames or frames_per_clip")
    temporal = frames // int(getattr(config, "tubelet_size", 1))
    image_size = getattr(config, "image_size", getattr(config, "crop_size", None))
    if image_size is None:
        raise ValueError("encoder config must define image_size or crop_size")
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(config.patch_size)
    rows, columns = image_height // patch_height, image_width // patch_width
    expected = temporal * rows * columns
    if tokens.shape[1] != expected:
        raise ValueError(f"expected {expected} spatial tokens, got {tokens.shape[1]}")
    return tokens.reshape(tokens.shape[0], temporal, rows, columns, tokens.shape[-1])


def _reshape_image_tokens(
    tokens: torch.Tensor,
    config: Any,
    frames: int,
    batch_size: int,
) -> torch.Tensor:
    image_height, image_width = pair(config.image_size)
    patch_height, patch_width = pair(config.patch_size)
    rows, columns = image_height // patch_height, image_width // patch_width
    patch_tokens = rows * columns
    if tokens.shape[1] < patch_tokens:
        raise ValueError(f"expected at least {patch_tokens} image tokens, got {tokens.shape[1]}")
    tokens = tokens[:, -patch_tokens:]
    return tokens.reshape(batch_size, frames, rows, columns, tokens.shape[-1])


def _processor_spatial_transform(
    processor: Any,
    source_size: tuple[int, int],
) -> SpatialTransform:
    source_height, source_width = source_size
    resized_height, resized_width = source_size
    if getattr(processor, "do_resize", True):
        size = processor.size
        height = getattr(size, "height", None)
        width = getattr(size, "width", None)
        shortest = getattr(size, "shortest_edge", None)
        if height is not None and width is not None:
            resized_height, resized_width = int(height), int(width)
        elif shortest is not None:
            scale = float(shortest) / min(source_size)
            resized_height = int(source_height * scale)
            resized_width = int(source_width * scale)
        else:
            raise ValueError("processor resize size must define height/width or shortest_edge")
    output_height, output_width = resized_height, resized_width
    offset_y = offset_x = 0.0
    if getattr(processor, "do_center_crop", False):
        crop = processor.crop_size
        output_height = int(crop.height)
        output_width = int(crop.width)
        offset_y = -float(max(0, (resized_height - output_height) // 2))
        offset_x = -float(max(0, (resized_width - output_width) // 2))
    return SpatialTransform(
        source_size=source_size,
        output_size=(output_height, output_width),
        scale=(resized_height / source_height, resized_width / source_width),
        offset=(offset_y, offset_x),
    )


def _validate_device(device: str | torch.device) -> None:
    try:
        parsed = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"invalid probe device: {device}") from error
    requested = parsed.type
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported probe device: {parsed}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ResourceUnavailableError("CUDA is not available")
    if (
        requested == "cuda"
        and parsed.index is not None
        and parsed.index >= torch.cuda.device_count()
    ):
        raise ResourceUnavailableError(f"CUDA device index is unavailable: {parsed.index}")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ResourceUnavailableError("MPS backend is not available")


def _local_checkpoint_path(checkpoint: str) -> Path | None:
    path = Path(checkpoint).expanduser()
    local_suffixes = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
    if path.is_absolute() or checkpoint.startswith(".") or path.suffix in local_suffixes:
        return path
    if path.exists():
        return path
    return None

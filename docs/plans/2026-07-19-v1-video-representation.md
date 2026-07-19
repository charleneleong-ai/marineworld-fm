# MarineWorld-FM v1 Video Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a locally smoke-testable and single-GPU VideoMAE maritime adaptation pipeline with common dataset adapters, leakage-safe manifests, W&B tracking, frozen reference encoders, and reproducible probes.

**Architecture:** Dataset adapters produce validated immutable manifests and canonical frame targets. Shared clip code decodes every dataset into one tensor contract, which feeds VideoMAE pretraining or frozen-encoder probes. Hydra composes runtime profiles, while a single experiment factory owns W&B identity, configuration, metrics, and checkpoint provenance.

**Tech Stack:** Python 3.11, PyTorch, torchvision, Transformers, Lightning, Hydra/OmegaConf, decord, W&B, scikit-learn, pytest, Ruff.

## Global Constraints

- Local smoke runs must support CPU and Apple MPS without network access or proprietary datasets.
- Scientific training targets one NVIDIA L4 or A100; multi-GPU training is out of scope.
- Train only on SMD and FVessel training videos; never include validation or test video content in SSL.
- Use 16 RGB frames at 224 x 224, tubelet size 2, and 90% tube masking for the primary VideoMAE-S run.
- Never log `WANDB_API_KEY`, raw restricted data, or raw dataset frames as artifacts.
- Split and label-subsample membership must be deterministic and represented by checksummed manifests.
- Follow red-green TDD for every production behaviour.
- Before each commit, run the required clean-elegant-code review on that commit's diff, fold findings into the same commit, and run pre-commit on its changed files.
- Use conventional commits and keep all work on `feat/v1-videomae-foundation`.

---

## File Structure

```text
configs/
  config.yaml                         # compose data/model/runtime/tracking
  data/{fvessel,smd,synthetic}.yaml   # adapter target + roots
  model/{videomae_vit_small,videomae_tiny}.yaml
  runtime/{local_smoke,l4,a100}.yaml
  tracking/wandb.yaml
src/marineworld/
  data/contracts.py                   # canonical immutable records/targets/manifests
  data/manifest.py                    # validation, checksums, split leakage checks
  data/adapters.py                    # adapter protocol + configured construction
  data/clips.py                       # clip indexing, decoding, dataset
  data/fvessel.py                     # FVessel adapter + MOT parsing
  data/smd.py                         # SMD adapter + native target parsing
  data/synthetic.py                   # deterministic local smoke adapter/decoder
  models/videomae.py                  # masks, HF model construction, embeddings
  train/experiment.py                 # W&B/Lightning logger and run identity
  train/module.py                     # pretraining LightningModule
  train/pretrain.py                   # Hydra entrypoint and trainer assembly
  eval/encoders.py                    # frozen encoder protocol + reference loaders
  eval/probes.py                      # deterministic label fractions + probes
  eval/run_probes.py                  # evaluation entrypoint and W&B tables
tests/
  test_data.py                        # contracts, adapters, manifests, clips
  test_training.py                    # masking, model, experiment, checkpoint smoke
  test_evaluation.py                  # frozen encoders, sampling, probes, aggregation
```

### Task 1: Canonical Records and Leakage-Safe Manifests

**Files:**
- Create: `src/marineworld/data/contracts.py`
- Create: `src/marineworld/data/manifest.py`
- Modify: `src/marineworld/data/__init__.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `JSONValue`, `VideoRecord`, `FrameTargets`, `DatasetManifest`, `validate_manifest(manifest)`, and `manifest_checksum(manifest)`.
- `DatasetManifest.records` is a tuple so callers cannot mutate split membership after validation.

- [ ] **Step 1: Write failing manifest tests**

```python
def test_manifest_rejects_video_leakage(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.touch()
    records = (
        _record(video, record_id="train/clip", split="train"),
        _record(video, record_id="test/clip", split="test"),
    )
    with pytest.raises(ValueError, match="video appears in multiple splits"):
        validate_manifest(DatasetManifest("demo", "1", "MIT", records))


def test_manifest_checksum_is_order_independent(tmp_path: Path):
    first = _record(tmp_path / "a.mp4", record_id="a", split="train")
    second = _record(tmp_path / "b.mp4", record_id="b", split="val")
    assert manifest_checksum(_manifest(first, second)) == manifest_checksum(
        _manifest(second, first)
    )
```

- [ ] **Step 2: Verify red**

Run: `uv run --extra dev pytest tests/test_data.py -v`

Expected: collection fails because `marineworld.data.contracts` does not exist.

- [ ] **Step 3: Implement the immutable contracts and validation**

```python
JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class VideoRecord:
    id: str
    dataset: str
    video_path: Path
    split: Literal["train", "val", "test"]
    source: str
    fps: float
    num_frames: int
    annotation_path: Path | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameTargets:
    frame_index: int
    boxes_xyxy: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray | None = None


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    license: str
    records: tuple[VideoRecord, ...]
```

Validation checks non-empty unique IDs, existing video and annotation paths, positive FPS/frame counts, valid splits, and canonicalized video paths appearing in exactly one split. The checksum serializes records sorted by ID and excludes filesystem mtimes.

- [ ] **Step 4: Verify green and full core regression**

Run: `uv run --extra dev pytest tests/test_data.py tests/test_alignment.py tests/test_splits.py -v`

Expected: all tests pass.

- [ ] **Step 5: Review, hook-check, and commit**

Run the clean-elegant-code review, then:

```bash
uvx pre-commit run --files src/marineworld/data/contracts.py src/marineworld/data/manifest.py src/marineworld/data/__init__.py tests/test_data.py
git add src/marineworld/data/contracts.py src/marineworld/data/manifest.py src/marineworld/data/__init__.py tests/test_data.py
git commit -m "feat: add canonical dataset manifests"
```

### Task 2: Dataset Adapter Protocol and Conforming Adapters

**Files:**
- Create: `src/marineworld/data/adapters.py`
- Modify: `src/marineworld/data/fvessel.py`
- Modify: `src/marineworld/data/smd.py`
- Create: `src/marineworld/data/synthetic.py`
- Modify: `tests/test_data.py`
- Create: `configs/data/synthetic.yaml`
- Modify: `configs/data/fvessel.yaml`
- Modify: `configs/data/smd.yaml`

**Interfaces:**
- Consumes: `DatasetManifest`, `VideoRecord`, and `FrameTargets` from Task 1.
- Produces: `DatasetAdapter.build_manifest(root) -> DatasetManifest`, `DatasetAdapter.load_targets(record) -> tuple[FrameTargets, ...]`, `build_adapter(config) -> DatasetAdapter`.

- [ ] **Step 1: Write failing adapter conformance tests**

```python
@pytest.mark.parametrize("adapter_factory", [_synthetic_adapter, _fvessel_adapter, _smd_adapter])
def test_adapter_produces_valid_manifest(adapter_factory, tmp_path: Path):
    adapter, root = adapter_factory(tmp_path)
    manifest = adapter.build_manifest(root)
    validate_manifest(manifest)
    assert manifest.records
    assert all(record.dataset == manifest.name for record in manifest.records)


def test_fvessel_parses_mot_targets(fvessel_root: Path):
    adapter = FVesselAdapter(version="fixture")
    record = adapter.build_manifest(fvessel_root).records[0]
    targets = adapter.load_targets(record)
    assert targets[0].boxes_xyxy.tolist() == [[10.0, 20.0, 40.0, 60.0]]
    assert targets[0].track_ids.tolist() == [7]
```

- [ ] **Step 2: Verify red**

Run: `uv run --extra dev pytest tests/test_data.py -v -k adapter`

Expected: imports for `DatasetAdapter`, `FVesselAdapter`, and `SyntheticAdapter` fail.

- [ ] **Step 3: Implement the protocol and adapters**

```python
class DatasetAdapter(Protocol):
    def build_manifest(self, root: Path) -> DatasetManifest: ...
    def load_targets(self, record: VideoRecord) -> tuple[FrameTargets, ...]: ...


def build_adapter(config: Mapping[str, Any]) -> DatasetAdapter:
    target = str(config["_target_"])
    adapter_type = import_string(target)
    kwargs = {key: value for key, value in config.items() if not key.startswith("_")}
    return adapter_type(**kwargs)
```

`FVesselAdapter` preserves current AIS support and parses MOT rows into XYXY boxes and track IDs. `SMDAdapter` maps native categories only when the fixture proves the mapping; unknown labels remain native IDs in manifest metadata. `SyntheticAdapter` materializes deterministic, empty marker files under its configured ignored data root so ordinary path validation still applies; `SyntheticVideoDecoder` in Task 3 generates tensors without reading those markers.

- [ ] **Step 4: Verify green**

Run: `uv run --extra dev pytest tests/test_data.py -v`

Expected: all adapter and manifest tests pass.

- [ ] **Step 5: Review, hook-check, and commit**

```bash
uvx pre-commit run --files src/marineworld/data/adapters.py src/marineworld/data/fvessel.py src/marineworld/data/smd.py src/marineworld/data/synthetic.py tests/test_data.py configs/data/synthetic.yaml configs/data/fvessel.yaml configs/data/smd.yaml
git add src/marineworld/data tests/test_data.py configs/data
git commit -m "feat: unify maritime dataset adapters"
```

### Task 3: Shared Clip Indexing and Decoding

**Files:**
- Create: `src/marineworld/data/clips.py`
- Modify: `tests/test_data.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: validated manifests and adapter targets.
- Produces: `ClipIndex`, `SpatialTransform`, `VideoDecoder`, `DecordVideoDecoder`, `SyntheticVideoDecoder`, `build_clip_index(...)`, and `MaritimeClipDataset`.
- `MaritimeClipDataset.__getitem__` returns `{"pixel_values": FloatTensor[T,C,H,W], "frame_indices": LongTensor[T], "record_id": str, "dataset": str, "targets": tuple[FrameTargets, ...], "spatial_transform": SpatialTransform}`. The transform is derived from decoded source dimensions before resize and records source/output size, scale, and zero crop offsets.

- [ ] **Step 1: Write failing deterministic-index and tensor-contract tests**

```python
def test_clip_index_is_deterministic_and_split_scoped(synthetic_manifest):
    first = build_clip_index(synthetic_manifest, split="train", frames=4, stride=2, seed=42)
    second = build_clip_index(synthetic_manifest, split="train", frames=4, stride=2, seed=42)
    assert first == second
    assert {clip.record_id for clip in first} == {"train-0"}


def test_clip_dataset_returns_canonical_tensor(synthetic_manifest):
    dataset = MaritimeClipDataset(
        synthetic_manifest,
        SyntheticVideoDecoder(height=32, width=32),
        split="train",
        frames=4,
        stride=1,
        image_size=16,
        seed=42,
    )
    sample = dataset[0]
    assert sample["pixel_values"].shape == (4, 3, 16, 16)
    assert sample["pixel_values"].dtype == torch.float32
    assert sample["frame_indices"].tolist() == [0, 1, 2, 3]
```

- [ ] **Step 2: Verify red**

Run: `uv run --extra train --extra dev pytest tests/test_data.py -v -k clip`

Expected: `marineworld.data.clips` cannot be imported.

- [ ] **Step 3: Implement decoder injection and clip sampling**

```python
class VideoDecoder(Protocol):
    def decode(self, record: VideoRecord, frame_indices: Sequence[int]) -> torch.Tensor: ...


@dataclass(frozen=True)
class ClipIndex:
    record_id: str
    start: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class SpatialTransform:
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    scale: tuple[float, float]
    offset: tuple[float, float] = (0.0, 0.0)

    def apply_boxes_xyxy(self, boxes: np.ndarray) -> np.ndarray: ...


class MaritimeClipDataset(Dataset[dict[str, Any]]):
    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clips[index]
        record = self.records[clip.record_id]
        frames = self.decoder.decode(record, clip.frame_indices)
        pixel_values = self.transform(frames)
        return {
            "pixel_values": pixel_values,
            "frame_indices": torch.tensor(clip.frame_indices),
            "record_id": record.id,
            "dataset": record.dataset,
            "targets": self.targets_for(record, clip.frame_indices),
            "spatial_transform": self.spatial_transform(frames),
        }
```

Use decord only inside `DecordVideoDecoder`; no function-local imports are needed because it is a declared training dependency. Verify non-square source-to-square-output box mapping. Add `scikit-learn>=1.5` for Task 7 and `python-dotenv>=1.0` for the local environment loader.

- [ ] **Step 4: Verify green**

Run: `uv run --extra train --extra dev pytest tests/test_data.py -v`

Expected: manifest, adapter, and clip tests pass.

- [ ] **Step 5: Review, hook-check, and commit**

```bash
uvx pre-commit run --files src/marineworld/data/clips.py tests/test_data.py pyproject.toml
git add src/marineworld/data/clips.py tests/test_data.py pyproject.toml
git commit -m "feat: add shared maritime clip pipeline"
```

### Task 4: W&B Experiment Identity and Offline-Safe Logging

**Files:**
- Create: `src/marineworld/train/experiment.py`
- Create: `configs/tracking/wandb.yaml`
- Modify: `configs/config.yaml`
- Create: `configs/runtime/local_smoke.yaml`
- Create: `configs/runtime/l4.yaml`
- Create: `configs/runtime/a100.yaml`
- Create: `tests/test_training.py`

**Interfaces:**
- Produces: `RunIdentity`, `resolved_config(cfg)`, `build_run_identity(...)`, `build_wandb_logger(...)`, and `metric_name(stage, metric, dataset=None)`.
- The logger factory returns `WandbLogger | False`; disabled mode returns `False`.

- [ ] **Step 1: Write failing identity, secret, and metric tests**

```python
def test_resolved_config_does_not_contain_wandb_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret-value")
    cfg = OmegaConf.create({"tracking": {"mode": "offline", "project": "marineworld-fm"}})
    serialized = json.dumps(resolved_config(cfg))
    assert "secret-value" not in serialized
    assert "WANDB_API_KEY" not in serialized


def test_run_identity_is_stable():
    identity = build_run_identity("videomae", ("sha-a", "sha-b"), seed=42, label_fraction=None)
    assert identity.run_id == build_run_identity(
        "videomae", ("sha-b", "sha-a"), seed=42, label_fraction=None
    ).run_id
    assert metric_name("probe", "macro_f1", "fvessel") == "probe/fvessel/macro_f1"
```

- [ ] **Step 2: Verify red**

Run: `uv run --extra train --extra dev pytest tests/test_training.py -v`

Expected: `marineworld.train.experiment` cannot be imported.

- [ ] **Step 3: Implement the experiment factory**

```python
@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    group: str
    tags: tuple[str, ...]


def resolved_config(cfg: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(resolved, dict)
    return resolved


def build_wandb_logger(cfg: DictConfig, identity: RunIdentity) -> WandbLogger | Literal[False]:
    if cfg.tracking.mode == "disabled":
        return False
    return WandbLogger(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        id=identity.run_id,
        group=identity.group,
        tags=list(identity.tags),
        offline=cfg.tracking.mode == "offline",
        log_model="all" if cfg.tracking.mode == "online" else False,
        config=resolved_config(cfg),
    )
```

Load `.env` only in CLI entrypoints before Hydra composition; library imports never mutate the environment. W&B IDs hash model, sorted manifest checksums, seed, and optional label fraction. Git SHA and accelerator remain tags/metadata so resuming the same logical run after a code fix is explicit rather than accidental.

- [ ] **Step 4: Verify green without network**

Run: `WANDB_MODE=disabled uv run --extra train --extra dev pytest tests/test_training.py -v`

Expected: all experiment tests pass and no W&B network call occurs.

- [ ] **Step 5: Review, hook-check, and commit**

```bash
uvx pre-commit run --files src/marineworld/train/experiment.py tests/test_training.py configs/config.yaml configs/tracking/wandb.yaml configs/runtime/local_smoke.yaml configs/runtime/l4.yaml configs/runtime/a100.yaml
git add src/marineworld/train/experiment.py tests/test_training.py configs
git commit -m "feat: wire reproducible wandb experiments"
```

### Task 5: VideoMAE Masking and Pretraining Module

**Files:**
- Create: `src/marineworld/models/videomae.py`
- Create: `src/marineworld/train/module.py`
- Modify: `src/marineworld/models/__init__.py`
- Modify: `tests/test_training.py`
- Create: `configs/model/videomae_tiny.yaml`
- Modify: `configs/model/videomae_vit_small.yaml`

**Interfaces:**
- Produces: `tube_mask(batch_size, sequence_length, mask_ratio, generator)`, `build_videomae(config)`, `encode_video(model, pixel_values)`, and `VideoMAEPretrainingModule`.

- [ ] **Step 1: Write failing mask and optimization tests**

```python
def test_tube_mask_has_exact_ratio():
    mask = tube_mask(2, sequence_length=80, mask_ratio=0.9, generator=torch.Generator().manual_seed(42))
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 80)
    assert mask.sum(dim=1).tolist() == [72, 72]


def test_tiny_pretraining_step_updates_parameters(tiny_batch):
    module = VideoMAEPretrainingModule(_tiny_model_config(), lr=1e-3, weight_decay=0.0)
    before = next(module.parameters()).detach().clone()
    trainer = Trainer(max_steps=1, accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.fit(module, train_dataloaders=DataLoader([tiny_batch], batch_size=None))
    assert not torch.equal(before, next(module.parameters()).detach())
```

- [ ] **Step 2: Verify red**

Run: `uv run --extra train --extra dev pytest tests/test_training.py -v -k 'mask or pretraining'`

Expected: VideoMAE module imports fail.

- [ ] **Step 3: Implement exact masking and Lightning training**

```python
def tube_mask(
    batch_size: int,
    sequence_length: int,
    mask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    masked = round(sequence_length * mask_ratio)
    noise = torch.rand(batch_size, sequence_length, generator=generator)
    order = noise.argsort(dim=1)
    mask = torch.zeros_like(noise, dtype=torch.bool)
    return mask.scatter(1, order[:, :masked], True)


class VideoMAEPretrainingModule(LightningModule):
    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        mask = self.make_mask(pixel_values.shape[0], pixel_values.device)
        loss = self.model(pixel_values=pixel_values, bool_masked_pos=mask).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batch_idx}")
        self.log("pretrain/loss", loss, on_step=True, on_epoch=True, sync_dist=False)
        return loss
```

Build `VideoMAEConfig` directly for tiny/random models; use `from_pretrained` only for declared reference checkpoints. Seed mask generators from run seed plus global step so resumed runs reproduce the same next mask.

- [ ] **Step 4: Verify green**

Run: `uv run --extra train --extra dev pytest tests/test_training.py -v`

Expected: masking, experiment, and one-step optimization tests pass.

- [ ] **Step 5: Review, hook-check, and commit**

```bash
uvx pre-commit run --files src/marineworld/models/videomae.py src/marineworld/models/__init__.py src/marineworld/train/module.py tests/test_training.py configs/model/videomae_tiny.yaml configs/model/videomae_vit_small.yaml
git add src/marineworld/models src/marineworld/train/module.py tests/test_training.py configs/model
git commit -m "feat: add videomae pretraining module"
```

### Task 6: End-to-End Local Smoke and Cloud Trainer Profiles

**Files:**
- Create: `src/marineworld/train/pretrain.py`
- Modify: `src/marineworld/train/__init__.py`
- Modify: `tests/test_training.py`
- Modify: `mise.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: adapter, clip, experiment, and model factories.
- Produces: `build_trainer(cfg) -> Trainer`, `run_pretraining(cfg) -> Path`, and CLI `python -m marineworld.train.pretrain`.

- [ ] **Step 1: Write failing end-to-end smoke and resume tests**

```python
def test_local_smoke_saves_and_resumes_checkpoint(tmp_path: Path):
    cfg = compose_smoke_config(tmp_path, max_steps=2)
    first_checkpoint = run_pretraining(cfg)
    assert first_checkpoint.exists()

    resumed = OmegaConf.merge(cfg, {"runtime": {"max_steps": 3, "ckpt_path": str(first_checkpoint)}})
    second_checkpoint = run_pretraining(resumed)
    state = torch.load(second_checkpoint, map_location="cpu", weights_only=False)
    assert state["global_step"] == 3
```

- [ ] **Step 2: Verify red**

Run: `WANDB_MODE=disabled uv run --extra train --extra dev pytest tests/test_training.py -v -k smoke`

Expected: `run_pretraining` is missing.

- [ ] **Step 3: Assemble the entrypoint**

```python
def run_pretraining(cfg: DictConfig) -> Path:
    seed_everything(int(cfg.seed))
    adapter = build_adapter(cfg.data.adapter)
    manifest = adapter.build_manifest(Path(cfg.data.root))
    validate_manifest(manifest)
    dataloaders = build_dataloaders(cfg, manifest, adapter)
    identity = build_run_identity(cfg.model.name, (manifest_checksum(manifest),), cfg.seed, None)
    logger = build_wandb_logger(cfg, identity)
    checkpoint = ModelCheckpoint(
        dirpath=Path(cfg.output_dir) / "checkpoints",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        save_on_exception=True,
    )
    trainer = build_trainer(cfg, logger=logger, callbacks=[checkpoint])
    trainer.fit(VideoMAEPretrainingModule.from_config(cfg), **dataloaders, ckpt_path=cfg.runtime.ckpt_path)
    return Path(checkpoint.last_model_path)
```

Add `mise run smoke` using synthetic data, W&B offline, two training steps, and one validation step. Add explicit `train:l4` and `train:a100` tasks that require a real data root and default W&B online.

- [ ] **Step 4: Verify green and CLI behaviour**

Run:

```bash
WANDB_MODE=disabled uv run --extra train --extra dev pytest tests/test_training.py -v
WANDB_MODE=offline uv run --extra train python -m marineworld.train.pretrain data=synthetic model=videomae_tiny runtime=local_smoke
```

Expected: tests pass; CLI exits zero with a local checkpoint and offline W&B run directory.

- [ ] **Step 5: Review, hook-check, and commit**

```bash
uvx pre-commit run --files src/marineworld/train/pretrain.py src/marineworld/train/__init__.py tests/test_training.py mise.toml README.md
git add src/marineworld/train tests/test_training.py mise.toml README.md
git commit -m "feat: add local and single-gpu pretraining runs"
```

### Task 7: Frozen Encoder Matrix and Reproducible Probes

**Files:**
- Create: `src/marineworld/eval/encoders.py`
- Create: `src/marineworld/eval/probes.py`
- Create: `src/marineworld/eval/run_probes.py`
- Modify: `src/marineworld/eval/__init__.py`
- Create: `tests/test_evaluation.py`
- Create: `configs/eval/probes.yaml`
- Modify: `README.md`

**Interfaces:**
- Produces: `FrozenVideoEncoder.encode(pixel_values) -> EncoderFeatures`, `sample_labelled_records(...)`, `fit_linear_probe(...)`, `evaluate_probe(...)`, and `ProbeResult`.
- `EncoderFeatures` contains `global_features: FloatTensor[B,D]` and optional `spatial_features: FloatTensor[B,T,H,W,D]`.

- [ ] **Step 1: Write failing freezing, sampling, and baseline tests**

```python
def test_encoder_remains_frozen_during_probe(fake_encoder, probe_batch):
    before = {name: value.clone() for name, value in fake_encoder.state_dict().items()}
    fit_linear_probe(fake_encoder, probe_batch.features, probe_batch.labels, task="classification")
    assert all(torch.equal(before[name], value) for name, value in fake_encoder.state_dict().items())


@pytest.mark.parametrize("fraction,expected", [(0.01, 1), (0.05, 1), (0.1, 2), (1.0, 20)])
def test_label_subsampling_is_video_level_and_nonempty(records, fraction, expected):
    selected = sample_labelled_records(records, fraction=fraction, seed=42)
    assert len(selected) == expected
    assert selected == sample_labelled_records(records, fraction=fraction, seed=42)


def test_probe_reports_random_and_pretrained_conditions(fake_results):
    table = aggregate_probe_results(fake_results)
    assert set(table["condition"]) == {"random", "generic_videomae", "maritime_videomae"}
    assert set(table["seed"]) == {40, 41, 42}
```

- [ ] **Step 2: Verify red**

Run: `uv run --extra train --extra dev pytest tests/test_evaluation.py -v`

Expected: evaluation interfaces do not exist.

- [ ] **Step 3: Implement encoder adapters and probes**

```python
@dataclass(frozen=True)
class EncoderFeatures:
    global_features: torch.Tensor
    spatial_features: torch.Tensor | None = None


class FrozenVideoEncoder(Protocol):
    name: str
    def encode(self, pixel_values: torch.Tensor) -> EncoderFeatures: ...


def sample_labelled_records(records: Sequence[VideoRecord], fraction: float, seed: int) -> tuple[str, ...]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    record_ids = sorted(record.id for record in records if record.split == "train")
    count = max(1, round(len(record_ids) * fraction))
    rng = np.random.default_rng(seed)
    return tuple(sorted(rng.choice(record_ids, size=count, replace=False).tolist()))
```

Implement random, generic VideoMAE, maritime VideoMAE, DINOv3, and V-JEPA loaders behind the same protocol. Network checkpoint retrieval is opt-in and cached; unit tests use a fake encoder. A resource failure while loading an optional reference emits `SKIPPED_RESOURCE` with the model and device, not a numeric result.

Use scikit-learn logistic regression for class/count probes. The dense probe is a single linear
spatial head trained without encoder gradients. It maps source-pixel boxes through each clip's
`SpatialTransform`, rasterizes binary vessel occupancy onto the encoder token grid, and evaluates
macro F1 through the same result matrix and normal/degenerate statuses. Dense extraction, head
updates, and confusion counts stream bounded clip minibatches rather than accumulating the full
spatiotemporal feature corpus. Log one W&B table with
condition, checkpoint, manifest checksum, dataset, task, label fraction, seed, metric, value, and
status.

- [ ] **Step 4: Verify green**

Run: `WANDB_MODE=disabled uv run --extra train --extra dev pytest tests/test_evaluation.py -v`

Expected: all evaluation tests pass without network or GPU.

- [ ] **Step 5: Review, hook-check, and commit**

```bash
uvx pre-commit run --files src/marineworld/eval/encoders.py src/marineworld/eval/probes.py src/marineworld/eval/run_probes.py src/marineworld/eval/__init__.py tests/test_evaluation.py configs/eval/probes.yaml README.md
git add src/marineworld/eval tests/test_evaluation.py configs/eval README.md
git commit -m "feat: add frozen maritime representation probes"
```

### Task 8: Full Verification and Reproducibility Audit

**Files:**
- Modify only files required to resolve audit findings.

**Interfaces:**
- Verifies every design requirement; introduces no new public interface.

- [ ] **Step 1: Run the complete test and style suite**

```bash
WANDB_MODE=disabled uv run --extra train --extra dev pytest -v tests/
uv run --extra dev ruff check src/ tests/ --select E,W,F,I
uv run --extra dev ruff format --check src/ tests/
uvx pre-commit run --all-files
```

Expected: every command exits zero.

- [ ] **Step 2: Run a clean offline smoke experiment**

```bash
WANDB_MODE=offline uv run --extra train python -m marineworld.train.pretrain data=synthetic model=videomae_tiny runtime=local_smoke
```

Expected: two optimization steps, validation, a local last/best checkpoint, an offline W&B directory, and no network requirement.

- [ ] **Step 3: Audit secrets and split leakage**

```bash
git grep -n 'WANDB_API''_KEY=' -- . ':!.env*'
git check-ignore -v .env
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes data=synthetic model=random eval=probes
```

Expected: the first command has no matches, `.env` is ignored, and the synthetic probe run exits zero with train-only label selection.

- [ ] **Step 4: Review the complete branch diff**

Run the clean-elegant-code review on `origin/main...HEAD`. Fold every accepted finding into the relevant existing commit; do not add a cleanup commit for defects in the unmerged feature.

- [ ] **Step 5: Re-run verification after review changes**

Repeat Steps 1-3 and require the same successful outcomes before claiming completion.

## Acceptance Checklist

- [ ] Synthetic CPU/MPS smoke run completes and resumes.
- [ ] L4 and A100 configurations compose without code changes.
- [ ] SMD, FVessel, and synthetic adapters conform to one protocol.
- [ ] Manifests reject cross-split video leakage and have stable checksums.
- [ ] W&B receives resolved configuration, namespaced metrics, stable identity, and online checkpoint artifacts without secrets.
- [ ] VideoMAE exact mask counts and one-step optimization are tested.
- [ ] Random, generic VideoMAE, maritime VideoMAE, DINOv3, and V-JEPA conditions share one frozen encoder interface.
- [ ] Probe subsets use 1%, 5%, 10%, and 100% of training videos across three seeds.
- [ ] External dataset test labels are never assumed locally.
- [ ] Full tests, Ruff, pre-commit, offline smoke, and secret audit pass.

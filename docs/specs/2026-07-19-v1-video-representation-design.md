# MarineWorld-FM v1 Video Representation Design

**Status:** Approved design, grounded against current releases on 2026-07-19

## Objective

Build a reproducible maritime video self-supervised learning baseline that runs as a tiny local smoke test and trains on one L4 or A100 GPU. The experiment must determine whether maritime-domain adaptation improves frozen video representations over random and general-purpose pretrained encoders.

v1 trains only on video. FVessel AIS remains a typed sidecar for v2 rather than entering the representation objective early.

## Research question and success criterion

The primary question is:

> Does VideoMAE adapted on SMD and FVessel produce better frozen maritime representations than random initialization and general-purpose visual encoders?

Success requires the maritime-adapted encoder to outperform the identical randomly initialized encoder on both SMD and FVessel supervised probes. Results are reported per dataset, label fraction, and seed. Generic VideoMAE, DINOv3, and V-JEPA provide stronger reference points; they are comparisons, not mandatory thresholds for declaring the first baseline valid.

## Model matrix

| Condition | Role | Training in v1 |
| --- | --- | --- |
| Random ViT-S | Negative control | None |
| Generic pretrained VideoMAE | Domain-adaptation control | Frozen evaluation |
| Maritime-adapted VideoMAE-S | Primary model | SMD + FVessel SSL |
| DINOv3 small/base | Dense spatial reference | Frozen evaluation |
| V-JEPA 2/2.1 | Temporal representation reference | Frozen evaluation where single-GPU memory permits |

VideoMAE-S uses 16 RGB frames at 224 x 224, tubelets of two frames, and a 90% tube mask. The local smoke configuration uses the same interfaces with a deliberately tiny encoder and synthetic clips; it is a correctness check, not a scientific run.

VideoMAE remains the trainable baseline because Transformers exposes the masked reconstruction contract directly. V-JEPA 2.1 is retained as the current dense temporal reference, and DINOv3 as the strong dense image reference.

## Dataset tiers

### Core training and in-domain evaluation

- **SMD/SMD-Plus:** visible onshore, visible onboard, and NIR domains.
- **FVessel:** 26 videos with MOT-format targets and asynchronous AIS. Only video and MOT targets are used in v1.

SSL pretraining uses only training videos. Validation and test videos never enter SSL, including without labels, so the reported protocol is inductive rather than transductive.

### Held-out external evaluation

- **MVTD:** maritime visual tracking and domain-transfer evaluation.
- **SeaDronesSee v2:** official detection and tracking export; hidden test labels remain server-evaluated.
- **MODS:** future dense obstacle-transfer evaluation using its official maritime metrics.

The adapters and manifests for external datasets may be registered in v1, but v1 is not blocked on downloading or fully evaluating every external benchmark.

### Deferred

LaRS and MODD2 segmentation, ABOships and SeaShips static-image detection, radar, and multispectral training remain outside v1.

The original SMD licence and distribution terms could not be verified from a current authoritative source during design. Automated SMD download must remain disabled until its access terms are recorded in the manifest.

## Canonical data contract

Dataset-specific code discovers files and parses native annotations. Shared code owns decoding, temporal sampling, augmentation, batching, and masking.

```python
@dataclass(frozen=True)
class VideoRecord:
    id: str
    dataset: str
    video_path: Path
    split: str
    source: str
    fps: float
    num_frames: int
    annotation_path: Path | None
    metadata: Mapping[str, JSONValue]


@dataclass(frozen=True)
class FrameTargets:
    frame_index: int
    boxes_xyxy: Tensor
    class_ids: Tensor
    track_ids: Tensor | None


@dataclass(frozen=True)
class SpatialTransform:
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    scale: tuple[float, float]
    offset: tuple[float, float]

    def apply_boxes_xyxy(self, boxes: ndarray) -> ndarray: ...


class DatasetAdapter(Protocol):
    def build_manifest(self, root: Path) -> DatasetManifest: ...
    def load_targets(self, record: VideoRecord) -> Sequence[FrameTargets]: ...
```

`DatasetManifest` includes dataset name and version, licence/access metadata, canonical label mapping, native labels, records, and a deterministic content checksum. Validation rejects duplicate record IDs, missing media, invalid splits, non-positive frame metadata, targets outside frame bounds, and video overlap across splits.

`MaritimeClipDataset` consumes manifests rather than concrete adapters. Before resizing decoded
frames, it derives a typed `SpatialTransform` from the source height/width to the output canvas.
Each sample returns that transform alongside the fixed-shape video tensor, temporal indices,
record identity, and optional frame targets. Scale and offset are explicit so box geometry remains
correct for non-square sources and can support future crop transforms; v1 resize has zero offsets.
Adapters register through Hydra `_target_` configuration rather than a central conditional.

Future modalities attach as optional typed sidecars:

```text
VideoRecord
|- FrameTargets
|- AIS sidecar          (v2)
|- radar sidecar        (v3)
`- calibration sidecar
```

Sidecars may add data but may not change the core video or target contract.

## Sampling and leakage controls

- Split by whole video before constructing clips.
- Build deterministic clip indices from manifest checksum, clip length, stride, and seed.
- Sample both core datasets explicitly so FVessel cannot vanish beneath the larger SMD corpus.
- Preserve dataset and source identity for stratified metrics.
- Keep augmentation stochasticity separate from deterministic clip membership.
- Record every split and label-subsample selection as a manifest artifact.
- Parse SMD supervision only through the explicit native `smd_objectgt_mat` contract:
  `ObjectGT/<video_stem>_ObjectGT.mat`, one `structXML` element per zero-based frame,
  `BB` rows in `x, y, width, height` form, and matching `Object` class rows. Retain
  vessel classes 1 and 3--7, preserve empty annotated frames, filter invalid class 0,
  buoy class 2, and non-vessel/other classes 8--10, and fail on unmatched filenames
  or malformed/unsupported MAT schemas.

## Evaluation

Frozen encoders are evaluated with lightweight, separately trained probes:

- Dataset-native class prediction where class labels exist.
- Vessel-count-bin prediction from frame targets.
- A single-linear-layer binary vessel-occupancy probe over frozen spatial tokens. Source-pixel
  boxes are mapped through the sample's `SpatialTransform`, then rasterized onto the encoder's
  temporal/spatial token grid before probe fitting. Dense feature extraction, head updates, and
  confusion-count evaluation stream clip minibatches; the full dense feature corpus is never
  retained in memory.
- Nearest-neighbour retrieval and reconstruction diagnostics.

Each supervised probe uses 1%, 5%, 10%, and 100% of labelled training videos with three deterministic subsampling seeds. Metrics follow the dataset's native task and are never averaged across incompatible taxonomies. External encoders use the closest supported spatial and temporal resolution; deviations are logged explicitly.

## Training and runtime profiles

### Local smoke

- Synthetic adapter with generated clips and targets.
- Tiny VideoMAE configuration, two train batches, one validation batch.
- CPU and MPS support.
- W&B offline mode without checkpoint artifact upload.
- Exercises configuration, manifest validation, decoding, masking, forward/backward, checkpoint save/resume, and one probe.

### Single GPU

- VideoMAE-S, 16 x 224 x 224 clips, mixed precision.
- Gradient accumulation configured from effective batch size.
- Explicit best and last checkpoints.
- L4 and A100 profiles differ only in batch, accumulation, worker, and precision settings.
- Resume uses a stable W&B run ID and `Trainer.fit(ckpt_path=...)`.

## W&B experiment contract

All training and probe entrypoints construct logging through one experiment factory.

- Modes are `online`, `offline`, and `disabled`; smoke defaults to `offline`, cloud to `online`.
- Convert the fully resolved Hydra `DictConfig` into primitive containers before logging.
- Run identity includes model condition, manifest checksums, label fraction, seed, git SHA, accelerator, and checkpoint provenance.
- Metric namespaces are `pretrain/*`, `val/*`, `probe/<dataset>/*`, and `system/*`.
- Log manifests, split files, evaluation tables, and online checkpoints as artifacts. Never log raw restricted data or credentials.
- Online checkpoints use `latest` and `best` aliases. Offline smoke runs keep checkpoints local because Lightning does not support offline mode combined with W&B model artifact logging.
- Log reconstruction previews, embedding projections, confusion matrices, and probe tables at bounded intervals.

`WANDB_API_KEY` is read from the process environment or an ignored local `.env`; code, resolved Hydra configuration, artifacts, and logs must never contain its value.

## Failure handling

- Missing or unlicensed datasets fail before model construction with an actionable dataset-specific message.
- Optional reference models that exceed device memory are marked `SKIPPED_RESOURCE`, not failed scientific results.
- Interrupted training preserves the last checkpoint and W&B run ID.
- Non-finite loss terminates the run, records the failing batch metadata without raw frames, and preserves the checkpoint.
- SMD and FVessel classification, count, and dense probes are executable from their
  native annotations; representative SMD fixtures cover boxes, empty frames, class
  filtering, filename pairing, and malformed rows.
- Probe results are emitted only when the encoder checkpoint, manifest checksum, split,
  and label subset are all known. Resource and degeneracy skips retain any computable
  deterministic label subset, and W&B receives the split/subsample JSON as a dataset
  artifact (with the same file retained under `output_dir` when tracking is disabled).

## Test strategy

Tests follow red-green TDD and are grouped by area:

- Manifest and adapter conformance using a module-level fake adapter plus SMD and FVessel fixtures.
- Video-level leakage and deterministic clip-index tests.
- Tensor shape, mask count, augmentation, and target-alignment behaviour.
- Tiny model forward/backward and checkpoint-resume smoke tests.
- Probe freezing, label-subsampling, and metric aggregation tests.
- W&B disabled/offline tests with no network access, covering resolved configuration, namespaced metrics, stable run identity, and resume metadata.

No test may require proprietary data, W&B network access, or a GPU.

## Current references

- [Transformers VideoMAE documentation](https://huggingface.co/docs/transformers/model_doc/videomae)
- [VideoMAE V2](https://arxiv.org/abs/2303.16727)
- [V-JEPA 2.1](https://arxiv.org/abs/2603.14482)
- [DINOv3](https://ai.meta.com/research/dinov3/)
- [FVessel](https://github.com/gy65896/FVessel)
- [MVTD](https://arxiv.org/abs/2506.02866)
- [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee)
- [MODS](https://arxiv.org/abs/2105.02359)
- [MaCVi 2026](https://arxiv.org/abs/2604.13244)
- [W&B Hydra integration](https://docs.wandb.ai/models/integrations/hydra)
- [Lightning W&B logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html)

## Out of scope

- AIS/video fusion and cross-modal losses.
- Radar, NIR-specific objectives, and action conditioning.
- Full detector or tracker fine-tuning.
- Distributed multi-GPU training.
- Automated redistribution of third-party datasets.

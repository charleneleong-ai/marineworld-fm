# MarineWorld-FM

A staged research project building a **self-supervised, multimodal, temporal foundation model** for maritime perception, evaluated on few-shot vessel detection, tracking, and trajectory prediction.

The thesis: start one layer *below* VLAs at **multisensory physical-state representation learning**, and use maritime sensor data (video + AIS + radar) as the wedge into world models.

## Roadmap

| Version | Scope | Status |
| --- | --- | --- |
| v0 | Scaffold, data adapters, AIS temporal alignment, seeding, tests | current |
| v1 | VideoMAE-style video SSL pretraining + few-shot detection/tracking eval | planned |
| v2 | Video + AIS fusion (cross-modal objectives) on FVessel | planned |
| v3 | + radar / NIR modalities | planned |
| v4 | Temporal world model: predict future latent state | planned |
| v5 | Action-conditioned world model `(z_t, a_t) -> z_{t+1}` | planned |

## Datasets

- **FVessel** (`gy65896/FVessel`, MIT): video + time-synchronised AIS + MOT ground truth. Key asset for the video+AIS stages.
- **SMD / SMD-Plus**: onshore/onboard/NIR video with detection + tracking labels.
- **SeaShips**, **MODD/MODD2**, **SeaClips** (license-gated): additional pretraining/eval.

Video and AIS are **not** on a shared clock. FVessel AIS is asynchronously timestamped, so fusing it with video requires explicit temporal alignment — implemented in `src/marineworld/data/alignment.py`.

## Layout

```text
marineworld-fm/
  pyproject.toml          # core + [train] + [dev] dependency groups
  mise.toml               # tool versions + tasks (setup/test/lint)
  .pre-commit-config.yaml # ruff + hygiene + fast tests on push
  configs/                # hydra configs (data / model / train)
  src/marineworld/
    data/                 # alignment.py (AIS<->frame), fvessel.py, smd.py, splits.py
    models/               # video ViT + VideoMAE head (v1)
    train/                # pretrain + finetune entrypoints (v1)
    eval/                 # linear probe, few-shot detection, tracking (v1)
    utils/                # seed.py (seed=42)
  tests/                  # behavioural unit tests
```

## Quick start

```bash
# Install tool versions + deps (uses mise + pip).
mise trust && mise run setup

# Or without mise:
pip install -e '.[train,dev]'

# Run the fast, core-only unit tests (no torch needed).
pytest -v tests/
```

## AIS temporal alignment

`marineworld.data.alignment` maps each video frame timestamp to a per-vessel AIS state:

```python
from marineworld.data.alignment import (
    load_ais_tracks, frame_timestamps_ms, align_tracks_to_frames,
)

tracks = load_ais_tracks("data/raw/fvessel/Video-02/ais")   # per-MMSI tracks
frame_ts = frame_timestamps_ms(start_ms=1652181665000, num_frames=16, fps=25.0)
aligned = align_tracks_to_frames(tracks, frame_ts, method="linear", tolerance_ms=500)
# aligned[i] -> {mmsi: AISRecord} for vessels observable at frame i
```

Behaviour:

- **linear** interpolation for position/speed; **circular** interpolation for course/heading (350 deg -> 10 deg gives 0 deg, not 180 deg).
- AIS heading sentinel `511` ("not available") is kept as `NaN`.
- Dropouts longer than `max_gap_ms` (default `4 * tolerance_ms`) yield no estimate rather than interpolating across a hole.
- Frames with no valid vessel estimate produce an empty dict.

## Reproducibility

All RNGs (torch, cuda, numpy, random) are seeded to **42** via `marineworld.utils.seed.seed_everything`. Splits are made by whole video (never by frame) and are deterministic.

## VideoMAE pretraining

Run the two-step synthetic smoke locally with offline W&B logging:

```bash
mise run smoke
```

The smoke covers manifest construction, synthetic decoding, training, validation, and
best/last checkpoint creation under `outputs/`. To launch the same entrypoint on a single
GPU, provide the real FVessel root explicitly; both cloud profiles use online W&B logging:

```bash
MARINEWORLD_DATA_ROOT=/path/to/fvessel mise run train:l4
MARINEWORLD_DATA_ROOT=/path/to/fvessel mise run train:a100
```

The L4 and A100 profiles share the data, model, optimization schedule, and trainer behavior.
They differ only in precision, batch size, gradient accumulation, and data-loader workers.

## Frozen representation probes

Run the deterministic tiny random smoke matrix without W&B or checkpoint downloads:

```bash
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes \
  data=synthetic model=random eval=probes
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes \
  data=synthetic model=random eval=probes eval.task=dense
```

The shared frozen encoder interface also supports cached generic VideoMAE,
maritime VideoMAE, DINOv3, and V-JEPA checkpoints. Remote checkpoint retrieval is
disabled unless `eval.allow_download=true` is set explicitly. Results retain the
condition, checkpoint, manifest checksum, dataset, task, label fraction, seed,
metric, value, status, and the selected video IDs plus their checksum. Optional
reference models that do not fit available resources are recorded as
`SKIPPED_RESOURCE` without a numeric value. Missing, empty, and single-class label
splits are likewise nonnumeric (`SKIPPED_UNAVAILABLE_LABELS` or
`SKIPPED_DEGENERATE_LABELS`). Count probes use the fixed classes 0, 1, 2, and 3+.
Dense evaluation maps source-pixel boxes through each clip's recorded resize
geometry and trains one binary vessel-occupancy head over frozen spatial tokens.

Use this matrix for scientific comparisons; `random_vit_small` is the matched
384-dimensional, 12-layer ViT-S random control rather than the tiny smoke model:

| Condition | Hydra model | Frames / crop | Checkpoint |
| --- | --- | --- | --- |
| Matched random ViT-S | `random_vit_small` | 16 / 224 | random initialization |
| Generic VideoMAE | `generic_videomae` | 16 / 224 | pinned Hugging Face revision |
| Maritime VideoMAE | `maritime_videomae` | 16 / 224 | required local checkpoint |
| DINOv3 | `dinov3` | 16 / 224 | pinned Hugging Face revision |
| V-JEPA 2 | `vjepa` | 64 / 256 | pinned Hugging Face revision |

```bash
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes \
  data=fvessel model=random_vit_small eval=probes
```

Cached reference conditions use the same CLI and remain network-disabled by
default. The maritime checkpoint must be supplied explicitly:

```bash
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes \
  data=fvessel model=generic_videomae eval=probes eval.optional=true
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes \
  data=fvessel model=dinov3 eval=probes eval.optional=true
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes \
  data=fvessel model=vjepa eval=probes eval.optional=true
WANDB_MODE=disabled uv run --extra train python -m marineworld.eval.run_probes \
  data=fvessel model=maritime_videomae eval=probes \
  model.checkpoint=/path/to/maritime.ckpt
```

Set `eval.allow_download=true` only when remote Hugging Face retrieval is
intentional. Each reference loads its cached checkpoint processor so VideoMAE
and DINOv3/V-JEPA use the processor's declared aspect-preserving resize, crop,
resampling, rescale, and normalization semantics. Hub revisions are pinned in the
model configs and included in result/run provenance. FVessel and SMD adapters probe
each source video's real frame count; a probe is unavailable when either supervised
split has no labelled video long enough for the configured encoder.
SMD supervision uses the explicit `smd_objectgt_mat` adapter format: each annotated
video is paired with `ObjectGT/<video_stem>_ObjectGT.mat`, whose native `structXML` entries
provide zero-based frame boxes and class labels. The parser retains vessel classes
1 and 3--7, preserves annotated empty frames, ignores invalid class 0, buoy class 2,
and non-vessel/other classes 8--10, and rejects unsupported or malformed schemas
rather than guessing.
This convention is grounded in the [SMD benchmark](https://openaccess.thecvf.com/content_CVPRW_2019/html/PBVS/Moosbauer_A_Benchmark_for_Deep_Learning_Based_Object_Detection_in_Maritime_CVPRW_2019_paper.html)
and the public [SMD-Plus ObjectGT reference](https://github.com/kjunhwa/Singapore-Maritime-Dataset-Plus).

Every probe invocation writes `probe_selection_manifest.json` under `output_dir`
with exact split membership, label-subset IDs, and checksums. Enabled W&B runs also
log that file as a `dataset` artifact. Every result row, including resource and
degenerate-label skips, carries its computable label selection. Run identity hashes
the resolved model architecture and optimization-relevant evaluation settings,
including dense batch size, epochs, and learning rate.
Encoder condition, device availability, checkpoint shape/path, and pinned Hub
revision are validated before dataset or logger construction without loading model
weights. Local files and complete `save_pretrained` directory trees are content-hashed,
so in-place checkpoint changes produce a new run identity. A computable empty label
selection is recorded as `[]` with its checksum; `null` is reserved for selections
that cannot be computed.
Local Hugging Face preflight parses `config.json`, safetensors headers, and sharded
indexes. PyTorch archives are loaded with `weights_only=True`, memory mapping, and the
`meta` device to validate state-dict metadata without unsafe pickle globals or copying
checkpoint tensors into host memory. Corrupt or incompatible containers fail before
unavailable-label rows can be emitted.

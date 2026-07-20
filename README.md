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

The developer toolchain defaults to Python 3.13.7; package metadata retains Python
3.11+ compatibility.

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

# Python 3.13 and FVessel Smoke Design

## Goal

Make CPython 3.13.7 the repository's default developer runtime without dropping the
declared Python 3.11 compatibility floor, then prove the real-data path on macOS with a
small, reproducible FVessel Clip-10 training run logged to W&B.

## Grounding

This design was checked on 2026-07-20 against primary sources:

- Python 3.13.14 is the current maintenance release in the 3.13 series:
  <https://www.python.org/downloads/release/python-31314/>.
- PyTorch 2.6 introduced Python 3.13 support, including `torch.compile` support:
  <https://pytorch.org/blog/pytorch2-6/>.
- PyAV publishes binary wheels for macOS, Linux, and Windows and supports Python 3.11+:
  <https://pypi.org/project/av/>.
- The official FVessel repository publishes `Clip-10.zip` at 2.56 GB, compared with
  40.9 GB for FVessel V1 and 23 GB for V2:
  <https://huggingface.co/datasets/gy65896/FVessel/tree/main>.

## Runtime and dependency contract

- Pin `.python-version` and `mise.toml` to the uv-managed CPython 3.13.7 macOS build and set Ruff's target to
  `py313`.
- Retain `project.requires-python = ">=3.11"`; Python 3.13 is the contributor default,
  not a new package minimum.
- Raise the training stack from the Python-3.13-incompatible `torch<2.6` range to a
  resolver-tested PyTorch 2.6+ range. Keep compatible torchvision and Lightning ranges
  explicit and regenerate the lockfile under Python 3.13.
- Add PyAV to the training dependencies. Prefer Decord where it is installed and fall
  back to PyAV otherwise, so Linux/cloud behavior remains stable while macOS can probe
  and decode real videos.
- Import either decoder lazily so core manifest/alignment utilities remain lightweight.

## Decoder boundary

`marineworld.data.video` will own decoder selection and expose backend-neutral frame
metadata and indexed-frame operations. FVessel and clip loading will call that boundary
instead of importing Decord directly. Selection is deterministic: Decord first, PyAV
second, then a single actionable error that names both optional backends without exposing
absolute dataset paths.

PyAV indexed reads will decode the requested frames in one forward pass and return them
in request order. Invalid or unreadable video metadata, missing requested frames, and
non-positive FPS/frame counts fail before trainer construction. Tests will exercise the
selection and output contract with small generated videos or controlled backend fakes;
they will not depend on the downloaded dataset.

## FVessel data flow

The smoke run uses only the official `Clip-10.zip` subset. It will be downloaded into an
ignored local data/cache directory, checked for a valid ZIP before extraction, and
extracted without allowing archive members to escape the destination. The existing
`FVesselAdapter` will discover the extracted MP4 layout; no dataset-specific copy or
renaming step will be introduced unless inspection proves the archive requires one.

The run will use a dedicated real-data smoke configuration with bounded work: one tiny
VideoMAE model, a small clip/sample count, at most two optimization steps, one validation
pass, and local checkpoints under an ignored output directory. It is a correctness and
observability check, not a scientific benchmark.

## W&B and privacy

The run uses the existing `chaleong/marineworld-fm` W&B project in online mode with
`tracking.log_media=true`. It may upload the already-approved bounded FVessel input and
reconstruction previews. Config sanitization must continue removing data roots,
checkpoint paths, and credentials. The verification records the W&B run URL, completion
state, scalar validation loss, and expected four media keys in PR #1; raw data and local
runtime artifacts are never committed.

## Failure handling

- Dependency resolution or import failure stops before dataset download or training.
- Download failure leaves no apparently complete archive; extraction happens only after
  ZIP validation.
- Insufficient disk space or an unexpected archive layout stops with an actionable
  message and does not mutate tracked files.
- A decode or training failure preserves the ignored logs/checkpoints for diagnosis and
  marks the W&B run failed rather than reporting a successful smoke test.

## Verification and acceptance

The change is accepted when:

1. `python --version` reports 3.13.7 through the project toolchain while package metadata
   still declares Python 3.11+.
2. The dependency lock resolves and the full test, Ruff, and pre-commit suites pass under
   Python 3.13.
3. Decoder tests prove Decord preference, PyAV fallback, frame ordering, and actionable
   missing-backend errors.
4. The official FVessel Clip-10 archive is downloaded and inspected outside Git.
5. A bounded real FVessel run finishes online and exposes scalar metrics plus the four
   bounded W&B preview media entries.
6. The repository remains free of dataset files, secrets, absolute paths, and runtime
   outputs, and PR #1 links the exact commands and W&B run.

## Out of scope

Full FVessel V1/V2 training, joint FVessel+SMD training, cloud GPU execution, performance
benchmarking, and AIS fusion remain separate follow-ups.

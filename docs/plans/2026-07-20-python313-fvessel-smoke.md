# Python 3.13 and FVessel Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Python 3.13.14 the local default, add a tested PyAV video fallback, and complete a bounded real FVessel Clip-10 smoke run in W&B.

**Architecture:** Keep Python 3.11 as the package floor while resolving the training environment on Python 3.13. Move all optional video-backend selection into `marineworld.data.video`: Decord remains preferred and PyAV provides metadata and indexed RGB decoding on macOS. A small Typer downloader validates and safely extracts the official FVessel archive, after which the existing adapter and training entrypoint run a dedicated real-data smoke profile.

**Tech Stack:** CPython 3.13.14, uv, PyTorch 2.6+, Decord, PyAV, Typer, Hydra, Lightning, pytest, W&B, Hugging Face FVessel.

## Global Constraints

- `.python-version` and `mise.toml` must default to exactly Python 3.13.14.
- `project.requires-python` must remain exactly `>=3.11`.
- Decord is preferred when importable; PyAV is the fallback.
- Optional video libraries remain lazily imported.
- Download only official `gy65896/FVessel/Clip-10.zip` into ignored local storage.
- The real smoke performs at most two optimization steps and one validation batch.
- `tracking.log_media=true`; W&B configuration must not expose credentials or absolute paths.
- Dataset archives, extracted data, checkpoints, and logs must never be committed.

---

### Task 1: Python 3.13 toolchain and compatible training lock

**Files:**
- Modify: `.python-version`
- Modify: `mise.toml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_training.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `train` and `dev` optional dependency groups.
- Produces: a Python 3.13.14 default environment whose package metadata remains installable on Python 3.11+.

- [ ] **Step 1: Change the tooling contract test first**

Replace `test_project_tooling_targets_python_311` with a test that reads all three configuration files:

```python
def test_project_defaults_to_python_313_without_raising_package_floor() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    mise = tomllib.loads((root / "mise.toml").read_text())

    assert config["project"]["requires-python"] == ">=3.11"
    assert config["tool"]["ruff"]["target-version"] == "py313"
    assert (root / ".python-version").read_text().strip() == "3.13.14"
    assert mise["tools"]["python"] == "3.13.14"
```

- [ ] **Step 2: Verify RED**

Run: `uv run --extra train --extra dev pytest tests/test_training.py::test_project_defaults_to_python_313_without_raising_package_floor -q`

Expected: FAIL because the repository still targets Python 3.11.9/`py311`.

- [ ] **Step 3: Update runtime and dependency declarations**

Set `.python-version` and `mise.toml` to `3.13.14`, Ruff to `py313`, add `av>=18,<19` and `typer>=0.16,<1`, retain the non-Darwin Decord marker, and replace the old PyTorch/Lightning constraints with resolver-tested Python-3.13-compatible ranges. Start with:

```toml
"torch>=2.6,<3",
"torchvision>=0.21,<1",
"pytorch-lightning>=2.5,<3",
"av>=18,<19",
"typer>=0.16,<1",
```

Run `uv lock --python 3.13.14`; narrow only if the resolver or test suite demonstrates an incompatibility. Update README setup text to identify 3.13.14 as the default and 3.11 as the supported floor.

- [ ] **Step 4: Verify GREEN under Python 3.13**

Run:

```bash
uv run --python 3.13.14 --extra train --extra dev python --version
uv run --python 3.13.14 --extra train --extra dev pytest tests/test_training.py::test_project_defaults_to_python_313_without_raising_package_floor -q
uv run --python 3.13.14 --extra train --extra dev python -c 'import av, torch; print(av.__version__, torch.__version__)'
```

Expected: Python 3.13.14, one passing test, and importable PyAV/PyTorch versions.

- [ ] **Step 5: Review, pre-commit, and commit**

Run the required simplify review on the working-tree diff, fold in valid findings, then run:

```bash
uvx pre-commit run --files .python-version mise.toml pyproject.toml uv.lock tests/test_training.py README.md
git diff --check
git add .python-version mise.toml pyproject.toml uv.lock tests/test_training.py README.md
git commit -m "chore: default development to python 3.13"
```

### Task 2: Backend-neutral metadata probes and PyAV decoding

**Files:**
- Modify: `src/marineworld/data/video.py`
- Modify: `src/marineworld/data/clips.py`
- Modify: `src/marineworld/data/fvessel.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Produces: `probe_video(path: Path) -> VideoMetadata`, `probe_video_frame_count(path: Path) -> int`, `probe_video_fps(path: Path) -> float`, and `AutoVideoDecoder.decode(record, frame_indices) -> Tensor`.
- `VideoMetadata` is a frozen dataclass with `frame_count: int` and `fps: float`.
- `FVesselAdapter` consumes the two probe wrapper functions; dataloaders consume `AutoVideoDecoder` through the existing `VideoDecoder` protocol.

- [ ] **Step 1: Write failing backend-selection and contract tests**

Add focused tests to the existing `tests/test_data.py` area, using monkeypatched backend factories rather than importing unavailable libraries:

```python
def _backend_unavailable(*_args: object) -> NoReturn:
    raise VideoBackendUnavailable("unavailable")


def test_auto_decoder_prefers_decord(monkeypatch, synthetic_manifest):
    calls: list[str] = []
    frames = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
    monkeypatch.setattr(video, "_decord_decode", lambda *_: calls.append("decord") or frames)
    monkeypatch.setattr(video, "_pyav_decode", lambda *_: calls.append("pyav") or frames)
    AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))
    assert calls == ["decord"]


def test_auto_decoder_falls_back_to_pyav(monkeypatch, synthetic_manifest):
    frames = torch.tensor([2, 0, 2], dtype=torch.uint8).view(3, 1, 1, 1).expand(-1, 3, 2, 2)
    monkeypatch.setattr(video, "_decord_decode", _backend_unavailable)
    monkeypatch.setattr(video, "_pyav_decode", lambda *_: frames)
    frames = AutoVideoDecoder().decode(synthetic_manifest.records[0], (2, 0, 2))
    assert frames[:, 0, 0, 0].tolist() == [2, 0, 2]


def test_auto_decoder_names_both_missing_backends(monkeypatch, synthetic_manifest):
    monkeypatch.setattr(video, "_decord_decode", _backend_unavailable)
    monkeypatch.setattr(video, "_pyav_decode", _backend_unavailable)
    with pytest.raises(RuntimeError, match="decord.*PyAV"):
        AutoVideoDecoder().decode(synthetic_manifest.records[0], (0,))
```

Add analogous probe tests for Decord preference, PyAV fallback, positive metadata validation, and path-free public errors.

- [ ] **Step 2: Verify RED**

Run: `uv run --python 3.13.14 --extra train --extra dev pytest tests/test_data.py -q`

Expected: FAIL because `VideoMetadata`, `probe_video_fps`, and `AutoVideoDecoder` do not exist.

- [ ] **Step 3: Implement the minimal backend boundary**

In `video.py`, add:

```python
@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float


class VideoBackendUnavailable(RuntimeError):
    pass


def probe_video(video: Path) -> VideoMetadata:
    for probe in (_decord_probe, _pyav_probe):
        try:
            metadata = probe(video)
            return _validate_metadata(metadata)
        except VideoBackendUnavailable:
            continue
    raise RuntimeError("video probing requires optional dependency 'decord' or 'PyAV'")
```

Implement `_pyav_probe` from the first video stream's `frames` and `average_rate`, falling back to a single decode pass when the container reports no frame count. Implement `_pyav_decode` as one forward decode that stores only requested frame indices, then stacks results in the caller's requested order as `[T, C, H, W]` uint8 RGB. Catch import failures as `VideoBackendUnavailable`; sanitize public read/decode failures so they identify the record ID rather than an absolute path.

Move `probe_video_fps` into `video.py`, keep `probe_video_frame_count` as a wrapper, replace FVessel's private Decord probe, and replace training construction of `DecordVideoDecoder` with `AutoVideoDecoder`. Retain `DecordVideoDecoder` as a compatibility wrapper if existing callers/tests require it.

- [ ] **Step 4: Verify GREEN and regression coverage**

Run:

```bash
uv run --python 3.13.14 --extra train --extra dev pytest tests/test_data.py tests/test_training.py -q
uv run --python 3.13.14 --extra dev ruff check src/marineworld/data tests/test_data.py --select E,W,F,I
```

Expected: all selected tests and Ruff checks pass.

- [ ] **Step 5: Review, pre-commit, and commit**

Run the required simplify review, fold in valid findings, then:

```bash
uvx pre-commit run --files src/marineworld/data/video.py src/marineworld/data/clips.py src/marineworld/data/fvessel.py tests/test_data.py
git diff --check
git add src/marineworld/data/video.py src/marineworld/data/clips.py src/marineworld/data/fvessel.py tests/test_data.py
git commit -m "feat: decode maritime video with pyav fallback"
```

### Task 3: Safe FVessel Clip-10 acquisition and bounded smoke profile

**Files:**
- Create: `src/marineworld/data/download.py`
- Create: `configs/runtime/real_smoke.yaml`
- Modify: `mise.toml`
- Modify: `.gitignore`
- Modify: `tests/test_data.py`
- Modify: `tests/test_training.py`
- Modify: `README.md`

**Interfaces:**
- Produces: Typer command `python -m marineworld.data.download fvessel-clip10 --output data/raw/fvessel`.
- Produces: `safe_extract_zip(archive: Path, destination: Path) -> None`.
- The command downloads the fixed official URL to a `.part` file, validates it with `ZipFile.testzip()`, atomically renames it, safely extracts it, and verifies at least one MP4 exists.
- Produces: Hydra runtime `runtime=real_smoke` with at most two train steps and one validation batch.

- [ ] **Step 1: Write failing safe-extraction and config tests**

Add parametrized archive traversal tests and a Hydra composition assertion:

```python
@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "nested/../../escape.txt"])
def test_safe_extract_zip_rejects_members_outside_destination(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, "unsafe")
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "data")


def test_real_smoke_profile_is_bounded() -> None:
    cfg = _compose_config("runtime=real_smoke", "data=fvessel")
    assert cfg.runtime.max_steps == 2
    assert cfg.runtime.limit_train_batches == 2
    assert cfg.runtime.limit_val_batches == 1
    assert cfg.runtime.num_workers == 0
```

Also test successful nested extraction, invalid ZIP rejection, `.part` cleanup on HTTP failure, and missing-MP4 rejection using a local/mock HTTP transport rather than the network.

- [ ] **Step 2: Verify RED**

Run: `uv run --python 3.13.14 --extra train --extra dev pytest tests/test_data.py tests/test_training.py -q`

Expected: FAIL because the downloader and `real_smoke` profile do not exist.

- [ ] **Step 3: Implement the smallest safe downloader and profile**

Use `Path.resolve()` plus `relative_to(destination.resolve())` to reject every unsafe ZIP member before extracting any member. Stream the fixed official URL through `httpx` to `Clip-10.zip.part`, call `ZipFile.testzip()`, atomically rename to `Clip-10.zip`, then extract. The Typer command must refuse to overwrite a non-empty destination unless it already contains a valid extracted MP4 tree; do not add a force/delete option.

Create `real_smoke.yaml` from the local profile with:

```yaml
tracking_mode: online
accelerator: auto
devices: 1
precision: 32-true
batch_size: 1
accumulate_grad_batches: 1
num_workers: 0
max_steps: 2
limit_train_batches: 2
limit_val_batches: 1
ckpt_path: null
```

Replace the existing informational mise download entry with `download:fvessel-clip10` and add `smoke:fvessel` that requires `MARINEWORLD_DATA_ROOT`, uses `data=fvessel model=videomae_tiny runtime=real_smoke`, and explicitly sets `tracking.log_media=true`. Ensure `data/raw/`, `outputs/`, and download `.part` files are ignored. Document the 2.56 GB download, storage warning, commands, and raw-frame W&B upload.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --python 3.13.14 --extra train --extra dev pytest tests/test_data.py tests/test_training.py -q
uv run --python 3.13.14 --extra dev ruff check src/ tests/ --select E,W,F,I
uv run --python 3.13.14 --extra dev ruff format --check src/ tests/
```

Expected: all selected tests, Ruff lint, and formatting checks pass.

- [ ] **Step 5: Review, pre-commit, and commit**

Run the required simplify review, fold in valid findings, then:

```bash
uvx pre-commit run --files src/marineworld/data/download.py configs/runtime/real_smoke.yaml mise.toml .gitignore tests/test_data.py tests/test_training.py README.md
git diff --check
git add src/marineworld/data/download.py configs/runtime/real_smoke.yaml mise.toml .gitignore tests/test_data.py tests/test_training.py README.md
git commit -m "feat: add reproducible FVessel smoke run"
```

### Task 4: Full verification and real online run

**Files:**
- Modify: `README.md` only if observed archive layout or commands differ from the documented contract.
- Modify: PR #1 body through `gh pr edit`; no dataset or W&B artifact file is added to Git.

**Interfaces:**
- Consumes: `mise run download:fvessel-clip10` and `mise run smoke:fvessel`.
- Produces: a completed W&B run URL and reproducible verification evidence in PR #1.

- [ ] **Step 1: Verify the entire repository on Python 3.13.14**

Run:

```bash
WANDB_MODE=disabled uv run --python 3.13.14 --extra train --extra dev pytest -q tests/
uv run --python 3.13.14 --extra dev ruff check src/ tests/ --select E,W,F,I
uv run --python 3.13.14 --extra dev ruff format --check src/ tests/
uvx pre-commit run --all-files
git diff --check
```

Expected: all tests and checks pass with no uncommitted hook rewrites.

- [ ] **Step 2: Check local capacity, download, and inspect FVessel**

Run `df -h .` and require at least 8 GB free before continuing. Then run:

```bash
mise run download:fvessel-clip10
find data/raw/fvessel -type f -name '*.mp4' | head
git status --short --ignored data outputs
```

Expected: the official 2.56 GB archive validates, one or more MP4s are extracted, and all data/runtime paths are ignored.

- [ ] **Step 3: Run the bounded real FVessel smoke online**

Run:

```bash
MARINEWORLD_DATA_ROOT="$PWD/data/raw/fvessel" \
WANDB_ENTITY=chaleong WANDB_PROJECT=marineworld-fm WANDB_MODE=online \
mise run smoke:fvessel
```

Expected: no more than two optimization steps, one validation batch, a final checkpoint, and a finished W&B run. If execution exceeds five minutes, relaunch it with the repository's detached-daemon convention and verify PPID 1.

- [ ] **Step 4: Verify W&B and privacy**

Query the emitted run ID with the W&B API and assert: state `finished`, finite validation loss, media keys `validation_inputs`, `validation_reconstruction`, `best_inputs`, and `best_reconstruction`, and no absolute data/checkpoint path or API key in the uploaded config.

- [ ] **Step 5: Update documentation if observation requires it**

If the actual official archive root differs from the README example, make the smallest documentation/config correction, run pre-commit, and commit it as:

```bash
git commit -m "docs: record FVessel smoke verification"
```

Do not create an empty verification commit.

- [ ] **Step 6: Final review, push, and render-check PR #1**

Run the verification-before-completion checklist and whole-branch review. Push `feat/v1-videomae-foundation`, then update PR #1 with linked source files, grouped commit tables, exact test outcomes, download/smoke commands, and the W&B URL. Render-check with:

```bash
gh pr view 1 --json body --jq '.body' | head -40
gh pr checks 1
git status --short --branch
```

Expected: correctly rendered links/code, checks reported accurately, and a clean branch synchronized with origin.

# W&B VideoMAE Media Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log bounded input grids and masked-reconstruction panels during validation and once from the best checkpoint for every W&B-backed pretraining run.

**Architecture:** A tracking-independent preview builder converts normalized VideoMAE inputs, masks, and decoder logits into display-ready image arrays. A rank-zero Lightning callback owns scheduling, W&B conversion, validation-sample retention, and best-checkpoint loading, keeping W&B calls out of the training module.

**Tech Stack:** Python 3.11, PyTorch 2.5+, Transformers VideoMAE, PyTorch Lightning, Hydra/OmegaConf, W&B, pytest.

## Global Constraints

- `tracking.log_media` defaults to `true` for synthetic, SMD, and FVessel runs.
- Online real-data runs may upload sampled raw frames and derived reconstructions to W&B; `tracking.log_media=false` is the explicit opt-out.
- Log at most one video, four frames, and one event per configured epoch interval.
- Media logging runs only on global rank zero and must never make disabled/non-W&B tracking invalid.
- Periodic and best-checkpoint previews use distinct metric keys.
- Preview construction is independent of W&B and uses the exact training normalization, tubelet size, patch size, and deterministic validation mask.
- Best-checkpoint inspection must not mutate the live trained module.
- Do not log credentials, filesystem paths, unrestricted batches, or additional raw media artifacts.

---

## File Map

- Create `src/marineworld/train/media.py`: preview data structures, denormalization, mask expansion, decoder-logit placement, image-grid assembly, and the Lightning W&B callback.
- Modify `src/marineworld/train/pretrain.py`: validate media configuration and register the callback beside checkpoint callbacks.
- Modify `configs/tracking/wandb.yaml`: default-enabled media controls.
- Modify `tests/test_training.py`: unit, callback, checkpoint, and smoke regressions in the existing training test area.
- Modify `README.md`: preview keys, cadence, opt-out, and real-data upload warning.
- Modify `docs/specs/2026-07-20-wandb-media-logging-design.md`: only if implementation review exposes a genuine design correction.

---

### Task 1: Tracking-independent VideoMAE preview builder

**Files:**
- Create: `src/marineworld/train/media.py`
- Modify: `tests/test_training.py`

**Interfaces:**
- Consumes: normalized `pixel_values: FloatTensor[B,T,C,H,W]`, `bool_masked_pos: BoolTensor[B,N]`, decoder `logits: FloatTensor[B,M,P]`, normalization mean/std, patch size, and tubelet size.
- Produces: `MediaPreview(input_grid: np.ndarray, reconstruction_panel: np.ndarray, caption: str)` and `build_media_preview(...) -> MediaPreview`.

- [ ] **Step 1: Write failing denormalization, shape, and caption tests**

Add tests using a two-frame, one-channel-equivalent RGB tensor whose normalized values are analytically reversible:

```python
def test_build_media_preview_denormalizes_and_bounds_output() -> None:
    pixels = torch.linspace(-2, 2, steps=24).reshape(1, 2, 3, 2, 2)
    mask = torch.tensor([[True, False, True, False, True, False, True, False]])
    logits = torch.zeros(1, 4, 3)

    preview = build_media_preview(
        pixel_values=pixels,
        bool_masked_pos=mask,
        logits=logits,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        patch_size=(1, 1),
        tubelet_size=1,
        max_frames=2,
        dataset="synthetic",
        source="generated",
    )

    assert preview.input_grid.shape == (2, 4, 3)
    assert preview.reconstruction_panel.shape == (4, 8, 3)
    assert preview.input_grid.min() >= 0
    assert preview.input_grid.max() <= 1
    assert preview.caption == "dataset=synthetic source=generated"
```

Add parametrized failures for non-positive `max_frames`, mismatched mask length, mismatched decoder patch width, and empty frame tensors.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
WANDB_MODE=disabled uv run --extra train --extra dev pytest \
  tests/test_training.py -k 'media_preview' -v
```

Expected: collection fails because `marineworld.train.media` and `build_media_preview` do not exist.

- [ ] **Step 3: Implement immutable preview types and basic image helpers**

Create:

```python
@dataclass(frozen=True)
class MediaPreview:
    input_grid: np.ndarray
    reconstruction_panel: np.ndarray
    caption: str


def denormalize_video(
    pixel_values: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    mean_tensor = pixel_values.new_tensor(mean).view(1, 1, -1, 1, 1)
    std_tensor = pixel_values.new_tensor(std).view(1, 1, -1, 1, 1)
    return (pixel_values * std_tensor + mean_tensor).clamp(0, 1)
```

Add private `_frame_indices(total, maximum)`, `_horizontal_grid(frames)`, and `_to_hwc(array)` helpers. Select evenly spaced frames with integer `torch.linspace`; never select more than `max_frames`.

- [ ] **Step 4: Implement tube-mask expansion and decoder patch placement**

Implement `_complete_reconstruction(...) -> tuple[Tensor, Tensor]`:

1. Validate `N == (T // tubelet) * (H // patch_h) * (W // patch_w)`.
2. Validate decoder patch width `P == tubelet * patch_h * patch_w * C`.
3. Create a full `[B,N,P]` patch tensor from patchified input.
4. Replace only masked patch rows with decoder logits in mask order.
5. Unpatchify to `[B,T,C,H,W]`.
6. Expand token masks over tubelets and spatial patches to construct a gray masked input.
7. Compute channel-mean absolute error and map it to a three-channel heatmap in `[0,1]`.

The public builder creates one four-column row per selected frame—original, masked, completed reconstruction, error—stacks those rows vertically, and converts both outputs to CPU NumPy HWC arrays.

- [ ] **Step 5: Run preview tests and verify GREEN**

Run the Step 2 command.

Expected: all preview-builder tests pass.

- [ ] **Step 6: Run the broader training tests**

Run:

```bash
WANDB_MODE=disabled uv run --extra train --extra dev pytest tests/test_training.py -v
```

Expected: all training tests pass.

- [ ] **Step 7: Simplify-review and commit Task 1**

Run the mandatory clean/elegant review on the working diff. Fold accepted findings, rerun Steps 5–6, then:

```bash
uvx pre-commit run --files src/marineworld/train/media.py tests/test_training.py
git add src/marineworld/train/media.py tests/test_training.py
git commit -m "feat: build videomae media previews"
```

---

### Task 2: Periodic validation and best-checkpoint W&B callback

**Files:**
- Modify: `src/marineworld/train/media.py`
- Modify: `src/marineworld/train/pretrain.py`
- Modify: `configs/tracking/wandb.yaml`
- Modify: `tests/test_training.py`

**Interfaces:**
- Consumes: `MediaPreview`, a `VideoMAEPretrainingModule`, validation batches, the active Lightning logger, and Lightning's `ModelCheckpoint.best_model_path`.
- Produces: `WandbMediaCallback(mean, std, enabled, every_n_epochs, max_frames, checkpoint_callback)` and four W&B keys: `media/validation_inputs`, `media/validation_reconstruction`, `media/best_inputs`, and `media/best_reconstruction`.

- [ ] **Step 1: Write failing callback-gating and fake-W&B tests**

Use a fake experiment whose `log(payload, step)` retains calls and a fake W&B logger recognized through a small `_wandb_experiment(logger)` adapter. Cover:

```python
@pytest.mark.parametrize(
    "enabled,rank,epoch,expected_calls",
    [(False, 0, 0, 0), (True, 1, 0, 0), (True, 0, 1, 0), (True, 0, 2, 1)],
)
def test_media_callback_gates_validation_logging(enabled, rank, epoch, expected_calls): ...
```

Assert one permitted event logs exactly the two validation keys, uses one video, includes dataset/source captions, and records the current global step. Assert disabled and ordinary CSV/custom loggers neither import W&B nor run model inference.

- [ ] **Step 2: Write failing best-checkpoint tests**

Create a tiny module, save two checkpoints with different deterministic decoder weights, and set `checkpoint_callback.best_model_path` to the first. Assert `on_fit_end`:

- logs exactly `media/best_inputs` and `media/best_reconstruction`;
- uses the best checkpoint rather than the last/current weights;
- leaves every live-module parameter byte-identical; and
- warns and skips when `best_model_path` is empty.

- [ ] **Step 3: Verify callback tests are RED**

Run:

```bash
WANDB_MODE=disabled uv run --extra train --extra dev pytest \
  tests/test_training.py -k 'media_callback or best_checkpoint_preview' -v
```

Expected: tests fail because `WandbMediaCallback` and the tracking configuration do not exist.

- [ ] **Step 4: Add and validate tracking configuration**

Extend `configs/tracking/wandb.yaml`:

```yaml
log_media: true
media_log_every_n_epochs: 1
media_max_frames: 4
```

Add `_validate_media_config(cfg)` in `pretrain.py` and call it before adapter/logger construction:

```python
def _validate_media_config(cfg: DictConfig) -> None:
    for key in ("media_log_every_n_epochs", "media_max_frames"):
        if int(cfg.tracking[key]) <= 0:
            raise ValueError(f"tracking.{key} must be positive")
```

Parametrize tests for zero, negative, and string/non-integral values, asserting failure before W&B construction.

- [ ] **Step 5: Implement periodic callback logging**

In `on_validation_batch_end`, return before inference unless all gates pass: enabled, global rank zero, first validation batch, epoch divisible by interval, and active W&B experiment. Retain a detached CPU copy of only the first sample for the final preview. Use `pl_module.make_mask(..., step=trainer.global_step, microbatch=batch_idx)` and one inference-mode model call, then log:

```python
experiment.log(
    {
        "media/validation_inputs": wandb.Image(preview.input_grid, caption=preview.caption),
        "media/validation_reconstruction": wandb.Image(
            preview.reconstruction_panel,
            caption=preview.caption,
        ),
    },
    step=trainer.global_step,
)
```

Import W&B lazily inside the conversion function only after logger gating.

- [ ] **Step 6: Implement isolated best-checkpoint logging**

At `on_fit_end`, locate the retained bounded sample and `best_model_path`. Instantiate a fresh `VideoMAEPretrainingModule` from the live module's immutable constructor values, load the checkpoint's `state_dict` on CPU, switch to eval mode, and build the preview under inference mode. Never call `load_state_dict` on the live module. Log the two best keys at `trainer.global_step` and release retained sample/model references.

Catch only expected missing-file/state incompatibility errors, warn, and skip; do not broadly swallow programming errors.

- [ ] **Step 7: Register the callback in the real trainer path**

In `_run_with_identity`, construct `WandbMediaCallback` beside the model checkpoint callback using:

```python
media = WandbMediaCallback(
    mean=tuple(cfg.data.transforms.normalization.mean),
    std=tuple(cfg.data.transforms.normalization.std),
    enabled=bool(cfg.tracking.log_media),
    every_n_epochs=int(cfg.tracking.media_log_every_n_epochs),
    max_frames=int(cfg.tracking.media_max_frames),
    checkpoint_callback=checkpoint,
)
```

Append it for every run; its own gates make disabled/non-W&B tracking a no-op.

- [ ] **Step 8: Run callback and training tests**

Run:

```bash
WANDB_MODE=disabled uv run --extra train --extra dev pytest \
  tests/test_training.py -v
```

Expected: all tests pass, including fake-W&B periodic/final logs and existing checkpoint/resume cases.

- [ ] **Step 9: Run an offline smoke with media enabled**

Run outputs under a temporary directory:

```bash
WANDB_MODE=offline uv run --extra train python -m marineworld.train.pretrain \
  data=joint_synthetic model=videomae_tiny runtime=local_smoke \
  tracking.log_media=true output_dir=/tmp/marineworld-media-smoke/output \
  hydra.run.dir=/tmp/marineworld-media-smoke/hydra
```

Expected: two optimization steps, validation, best/last checkpoints, and an offline W&B run whose history contains all four media keys.

- [ ] **Step 10: Simplify-review and commit Task 2**

Run the mandatory clean/elegant review, fold findings, repeat Steps 8–9, then:

```bash
uvx pre-commit run --files src/marineworld/train/media.py src/marineworld/train/pretrain.py configs/tracking/wandb.yaml tests/test_training.py
git add src/marineworld/train/media.py src/marineworld/train/pretrain.py configs/tracking/wandb.yaml tests/test_training.py
git commit -m "feat: log wandb reconstruction previews"
```

---

### Task 3: Documentation, online verification, and PR update

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-07-20-wandb-media-logging.md`
- Modify only files required by review findings.

**Interfaces:**
- Verifies the complete media-logging feature; introduces no new public Python interface.

- [x] **Step 1: Document W&B media behaviour and privacy**

Add a concise README section showing:

```yaml
tracking:
  log_media: true
  media_log_every_n_epochs: 1
  media_max_frames: 4
```

Document the four media keys, the best-checkpoint semantics, bounded cadence, and the explicit warning that online SMD/FVessel runs upload sampled raw frames and derived previews. Include `tracking.log_media=false` as the opt-out command.

- [x] **Step 2: Run complete local verification**

Run:

```bash
WANDB_MODE=disabled uv run --extra train --extra dev pytest -q tests/
uv run --extra dev ruff check src/ tests/ --select E,W,F,I
uv run --extra dev ruff format --check src/ tests/
uvx pre-commit run --all-files
git diff --check
```

Expected: every command exits zero; the full test count is at least 255.

- [x] **Step 3: Verify the online synthetic W&B run**

With the user's existing explicit approval for synthetic run metadata and image upload, run the two-step `joint_synthetic` smoke in online mode with isolated checkpoint output. Query the resulting run through `wandb.Api()` and assert:

- state is `finished`;
- scalar validation/training losses exist;
- W&B history contains the periodic and best media keys;
- exactly bounded synthetic previews are present;
- the training-manifest and model artifacts still exist; and
- no raw SMD/FVessel data was involved in this verification.

Verified in [W&B run `0f854e141fbacc0a`](https://wandb.ai/chaleong/marineworld-fm/runs/0f854e141fbacc0a): state `finished`, validation loss `0.99274`, exactly four synthetic media files under the four configured keys, one dataset-manifest artifact, and two model artifact versions.

- [x] **Step 4: Final whole-diff review**

Review `origin/main...HEAD` with emphasis on raw-data upload clarity, W&B logger lifecycle, rank/interval gates, checkpoint isolation, tensor-to-image correctness, memory bounds, and regression coverage. Fix every accepted Critical/Important finding before continuing.

- [x] **Step 5: Commit documentation and push the reviewed branch**

Pure documentation may skip simplify; hook-check it before commit:

```bash
uvx pre-commit run --files README.md docs/plans/2026-07-20-wandb-media-logging.md
git add README.md docs/plans/2026-07-20-wandb-media-logging.md
git commit -m "docs: document wandb media previews"
git push
```

Update PR #1 with the new design/plan/source links and refreshed test/W&B evidence, then render-check the body with:

```bash
gh pr view 1 --json body --jq '.body' | head -40
```

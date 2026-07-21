# W&B VideoMAE Media Logging Design

## Goal

Add bounded visual monitoring to every W&B-backed pretraining run. Each logged preview shows both the input video and the masked-reconstruction behaviour so training failures are visible without downloading checkpoints.

## User-facing behaviour

`tracking.log_media` defaults to `true`. When W&B tracking is enabled, the first validation batch at the configured interval logs:

- an input grid containing up to four evenly spaced frames from one video; and
- a reconstruction panel containing the original frame, masked input, reconstructed frame, and absolute-error heatmap.

After training, the callback restores the best checkpoint and generates one final preview from the deterministic validation sample. Periodic previews use `media/validation_inputs` and `media/validation_reconstruction`; the best-checkpoint preview uses `media/best_inputs` and `media/best_reconstruction` so W&B keeps the final result distinct from the training timeline.

Captions include the dataset and source identities already carried by the canonical batch contract. Logging is rank-zero-only and bounded to one video per event, four frames per grid, and one event per configured epoch interval.

The default applies to synthetic, SMD, and FVessel data. Online runs may therefore upload sampled raw dataset frames and derived reconstructions to W&B. This behaviour is intentional and user-approved. `tracking.log_media=false` remains available for deployments whose dataset terms prohibit media upload.

## Architecture

Implement a Lightning callback responsible for preview scheduling, best-checkpoint preview generation, and W&B interaction. The callback receives a validation batch and model outputs through a small, tracking-independent preview builder. It retains only the bounded CPU validation sample needed for the final preview, then loads the checkpoint selected by Lightning's `ModelCheckpoint` callback after fitting. The live training module is not mutated when the best checkpoint is inspected.

The preview builder:

1. denormalizes model inputs using the configured training mean and standard deviation;
2. recreates the deterministic validation mask;
3. maps masked tube tokens back to the frame/patch canvas;
4. reconstructs masked patches from VideoMAE logits;
5. produces clamped image tensors for the input grid and four-column reconstruction panel; and
6. returns captions and image arrays without importing or calling W&B.

The callback converts those arrays to `wandb.Image` only when the active logger is a W&B logger. Disabled or non-W&B loggers remain valid and perform no media work. If no best checkpoint exists, final preview generation is skipped with a concise warning while periodic validation logging remains valid.

## Data and reconstruction semantics

Preview values are converted from normalized model space back to display-space RGB in `[0, 1]`. The original and reconstructed images use the same normalization contract as training. Mask visualization operates at the configured VideoMAE tubelet and patch sizes rather than applying an unrelated pixel mask.

Only masked patches are taken from model predictions. Visible patches in the reconstruction panel retain the original input, making the panel a direct view of the masked reconstruction objective. The error heatmap is the channel-mean absolute error between the original and completed reconstruction.

## Configuration

Extend the W&B tracking configuration with:

```yaml
log_media: true
media_log_every_n_epochs: 1
media_max_frames: 4
```

All values are included in the resolved configuration. `media_log_every_n_epochs` and `media_max_frames` must be positive integers.

## Failure handling

Media logging must never fail training. Unsupported logger types, absent validation batches, or missing reconstruction logits skip the preview with a concise warning. Invalid configuration fails before training starts. Programming errors in the preview builder remain test failures rather than being swallowed broadly.

## Testing

Tests cover:

- denormalization and `[0, 1]` clamping;
- exact input-grid and reconstruction-panel shapes;
- tubelet/patch mask expansion;
- reconstruction placement and error-map values;
- dataset/source captions;
- default-enabled configuration and explicit opt-out;
- epoch interval, rank-zero, disabled logger, and non-W&B logger gating;
- bounded frame/video selection; and
- a fake W&B experiment integration asserting periodic and final named image logs without network access;
- best-checkpoint selection rather than last-checkpoint selection; and
- final preview generation without mutating the trained in-memory module.

The existing offline smoke must continue to finish two optimization steps, validation, and checkpoint creation with media logging enabled.

## Documentation

Update the README training and W&B sections with the default upload behaviour, configuration controls, and the privacy implication for real SMD/FVessel frames. The PR test plan will include the media callback tests and refreshed offline/online smoke evidence.

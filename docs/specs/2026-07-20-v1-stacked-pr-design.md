# V1 Stacked Pull Request Design

## Goal

Replace the 11,000-line v1 pull request with a reviewable stack while preserving PR #1
as the foundation scaffold and retaining all existing work in recoverable Git history.

## Stack

1. `chore/v1-foundation-scaffold` targets `main`. It contains the v1 design and plan,
   Python 3.13.7 developer default, the `>=3.11` package floor, compatible dependency
   metadata and lockfile, Ruff `py313`, and Hydra/tooling skeletons.
2. `feat/maritime-data-pipeline` targets `chore/v1-foundation-scaffold`. It contains
   canonical manifests, adapters, splitting, clip sampling, video decoding, the PyAV
   fallback, and safe FVessel Clip-10 acquisition.
3. `feat/videomae-pretraining` targets `feat/maritime-data-pipeline`. It contains the
   VideoMAE model/module, experiment identity, core W&B logging, and local/L4/A100
   training entrypoints, including the bounded real FVessel smoke profile and observed
   A100/W&B verification evidence.
4. `feat/representation-evaluation` targets `feat/videomae-pretraining`. It contains
   shared frozen encoders, probes, and bounded diagnostics.
5. `feat/joint-maritime-pretraining` targets `feat/representation-evaluation`. It
   contains balanced SMD/FVessel sampling, deterministic joint manifests, and resume
   behavior.
6. `feat/wandb-media` targets `feat/joint-maritime-pretraining`. It contains validation
   and best-checkpoint reconstruction previews plus the approved media documentation.

The originally planned `test/fvessel-smoke` layer was folded into PR #3 so reviewers can
verify the real data path alongside the pretraining entrypoint it exercises.

## History strategy

Create a remote backup branch at the current PR #1 head before changing any branch.
Reconstruct each layer from `main` using the existing commits where their boundaries are
clean and focused replacement commits where files currently mix layers. Preserve authored
content and commit attribution; do not copy runtime data, secrets, checkpoints, or W&B
artifacts.

PR #1 will be force-updated with `--force-with-lease` to point at
`chore/v1-foundation-scaffold`, retain its number, and receive a new scaffold-focused
title/body. Each subsequent branch receives its own stacked PR with an explicit dependency
notice and links to adjacent PRs.

## Review and verification

Each branch must pass the smallest relevant test suite plus Ruff, pre-commit, and
`git diff --check` before push. Each PR body follows the repository template, links every
named file/symbol on its branch, groups commits in tables, and is render-checked after
creation. The final stack tip must reproduce the current complete test outcome before the
real FVessel work begins.

## Safety

- Never modify `main`.
- Create and verify the backup branch before rewriting PR #1.
- Use `--force-with-lease`, never an unconditional force push.
- Do not delete the original feature branch or backup until the whole stack is verified.
- Stop if GitHub reports an unexpected base/head or if any reconstructed layer loses
  files required by its own tests.

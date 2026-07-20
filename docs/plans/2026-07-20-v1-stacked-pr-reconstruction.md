# V1 Stacked Pull Request Reconstruction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve all current v1 work while replacing PR #1 with a Python-3.13 foundation scaffold and opening six reviewable dependent PRs.

**Architecture:** Snapshot the current tip locally and remotely, reconstruct branches from `main` at functional boundaries, and use GitHub's branch-rename endpoint so PR #1 keeps its number while its head becomes `chore/v1-foundation-scaffold`. Each later branch is based on the previous branch and contains only its layer's files/commits.

**Tech Stack:** Git worktrees, GitHub CLI/API, uv, Python 3.13.7, pytest, Ruff, pre-commit.

## Global Constraints

- Never modify or push `main`.
- Preserve the current tip as `backup/v1-videomae-foundation-20260720` locally and remotely before rewriting anything.
- Use only `--force-with-lease` for rewritten remote refs.
- PR #1 must remain open and retain number 1 after the remote branch rename.
- Branch 1 is exactly `chore/v1-foundation-scaffold` and includes Python 3.13.7.
- Every later PR targets the immediately preceding stack branch.
- No dataset, checkpoint, secret, W&B media file, or runtime output may enter Git.

---

### Task 1: Snapshot and inventory the current pull request

**Files:**
- Create: remote/local branch `backup/v1-videomae-foundation-20260720`
- Modify: none

**Interfaces:**
- Produces: immutable recovery ref at the current full-stack SHA and a recorded PR #1 head/base/state.

- [ ] Record `git rev-parse HEAD`, `git status --short`, `git log --reverse --oneline main..HEAD`, and `gh pr view 1 --json number,state,baseRefName,headRefName,url`.
- [ ] Create `backup/v1-videomae-foundation-20260720` at that exact SHA and push it without force.
- [ ] Verify local and remote backup SHAs equal the recorded original SHA using `git rev-parse` and `git ls-remote`.
- [ ] Stop immediately if the worktree is dirty or either backup SHA differs.

### Task 2: Reconstruct and verify the foundation scaffold

**Files:**
- Include: `.python-version`, `.gitignore`, `mise.toml`, `pyproject.toml`, `uv.lock`
- Include: base and local/cloud Hydra configuration skeletons under `configs/`
- Include: v1, Python 3.13, and stack design/plan documents under `docs/`
- Include: focused tooling/config tests only
- Exclude: substantive `src/marineworld/data`, `models`, `train`, and `eval` implementations

**Interfaces:**
- Produces: `chore/v1-foundation-scaffold` based directly on `main`.

- [ ] Create a separate reconstruction worktree from `main` so the backup/full branch remains untouched.
- [ ] Restore only the approved scaffold files from the backup ref, then remove configs/tests that require later implementations.
- [ ] Change the developer default to Python 3.13.7, keep `requires-python = ">=3.11"`, set Ruff `py313`, and resolve Python-3.13-compatible training constraints plus `uv.lock`.
- [ ] Add or retain one focused test asserting `.python-version`, mise, Ruff, and package-floor contracts; verify it fails before the version edits and passes afterward.
- [ ] Run scaffold tests, Ruff, pre-commit, and `git diff --check` under Python 3.13.7.
- [ ] Run the required simplify review, fold valid findings into the same commit, and commit with conventional subjects.
- [ ] Rename the existing remote PR head `feat/v1-videomae-foundation` to `chore/v1-foundation-scaffold` through `POST /repos/charleneleong-ai/marineworld-fm/branches/feat%2Fv1-videomae-foundation/rename` with `new_name=chore/v1-foundation-scaffold`; do not pre-create the destination ref.
- [ ] Verify PR #1 is still open, numbered 1, based on `main`, and headed by `chore/v1-foundation-scaffold`; stop if any assertion fails.
- [ ] Push the reconstructed local branch to the renamed remote ref with `--force-with-lease=refs/heads/chore/v1-foundation-scaffold:<recorded-original-sha>`, then verify the remote SHA equals the local scaffold SHA.
- [ ] Rewrite PR #1 title/body around the scaffold only and render-check the result.

### Task 3: Reconstruct the six dependent branches

**Files:**
- `feat/maritime-data-pipeline`: data contracts, manifests, adapters, splits, clips, video backend, PyAV fallback/downloader, data tests/config/docs.
- `feat/videomae-pretraining`: VideoMAE model/module, experiment tracking, pretraining entrypoint, runtime configs, training tests/docs.
- `feat/representation-evaluation`: encoders, probes, diagnostic runner, evaluation configs/tests/docs.
- `feat/joint-maritime-pretraining`: balanced joint configs, sampling/resume logic, focused training tests/docs.
- `feat/wandb-media`: media callback/rendering, tracking config, media tests/design/plan/docs.
- `test/fvessel-smoke`: real smoke profile, real-data command/docs, and W&B verification evidence only.

**Interfaces:**
- Each branch consumes the complete preceding branch and produces one independently reviewable layer.

- [ ] For each branch in order, branch from the verified predecessor and apply the matching existing commits from the backup ref; use focused replacement commits only where an original commit crosses boundaries.
- [ ] Resolve conflicts by preserving the predecessor's Python 3.13/tooling state and adding only the current layer.
- [ ] Run that layer's focused tests first, then Ruff, pre-commit, and `git diff --check`.
- [ ] Run the required simplify review before each non-trivial commit and fold valid findings into that commit.
- [ ] Push each branch normally, verify its remote SHA, and do not delete any predecessor.
- [ ] At the final stack tip, run the complete test suite and confirm it reproduces or improves the current 314-test result before real-data execution.

### Task 4: Open and cross-link the stacked pull requests

**Files:**
- Modify: GitHub PR metadata only

**Interfaces:**
- Produces: PRs 2-7, each based on the preceding stack branch, with PR #1 as the root.

- [ ] Open each PR with the repository-required Summary, Test plan, grouped Commits tables, and applicable Out-of-scope follow-ups.
- [ ] Put `Stack dependency: #N` at the top of every dependent PR and link both its predecessor and successor after all numbers are known.
- [ ] Move W&B synthetic/media links to the PR that introduces their functionality; keep real FVessel evidence only in the final smoke PR.
- [ ] Render-check every body with `gh pr view <N> --json body --jq '.body' | head -40` and verify base/head pairs with `gh pr view <N> --json baseRefName,headRefName,state,url`.
- [ ] Report the ordered stack, URLs, test evidence, backup ref, and any intentionally deferred real-data execution.

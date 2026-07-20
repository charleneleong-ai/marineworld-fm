# Architecture Diagrams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one canonical Mermaid architecture document and consistent contribution-highlight diagrams to stacked PRs #3–#6.

**Architecture:** The canonical diagram shows dataset adapters and manifests feeding clip decoding, optional balanced sampling, VideoMAE pretraining, frozen representation evaluation, and bounded W&B outputs. Each PR description reuses this topology and highlights only the component introduced by that PR.

**Tech Stack:** Markdown, GitHub Mermaid, GitHub CLI

## Global Constraints

- Keep the diagram topology and labels consistent across PRs #3–#6.
- Represent joint sampling as an input to pretraining, not a later pipeline stage.
- Put the canonical project view at `docs/architecture.md`.
- Preserve the existing PR description structure and place `## Visual aid` before `## Commits`.

---

### Task 1: Canonical architecture document

**Files:**
- Create: `docs/architecture.md`

**Interfaces:**
- Consumes: dataset, training, evaluation, and W&B module boundaries already present in the stack.
- Produces: the canonical Mermaid topology and PR-to-component map reused in review descriptions.

- [ ] **Step 1: Write the architecture page**

Add a Mermaid flowchart with dataset, shared data contract, balanced sampling, pretraining, evaluation, and observability groups. Follow it with a compact table mapping PRs #3–#6 to their architectural contribution.

- [ ] **Step 2: Validate the Markdown and Mermaid structure**

Run: `rg -n '^```mermaid|^## Pull request map|PR #3|PR #4|PR #5|PR #6' docs/architecture.md`

Expected: one Mermaid fence, one PR map heading, and all four PR rows.

### Task 2: Stacked PR visual aids

**Files:**
- Modify externally: GitHub PR descriptions #3, #4, #5, and #6

**Interfaces:**
- Consumes: the topology in `docs/architecture.md`.
- Produces: one `## Visual aid` Mermaid diagram per PR with its contribution highlighted.

- [ ] **Step 1: Add consistent diagrams**

Insert `## Visual aid` before `## Commits` in every description. Use blue for the current contribution, green for merged foundations, and grey for adjacent stack components.

- [ ] **Step 2: Render-check descriptions**

Run: `gh pr view <number> --json body --jq '.body' | head -80` for PRs #3–#6.

Expected: unescaped Mermaid fences, consistent topology, and exactly one highlighted contribution per PR.

### Task 3: Verify and publish

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/plans/2026-07-20-architecture-diagrams.md`

**Interfaces:**
- Consumes: completed documentation and PR metadata.
- Produces: a committed, pushed architecture document and merge-ready PR #3.

- [ ] **Step 1: Run documentation checks**

Run: `uvx --python 3.13.7 pre-commit run --files docs/architecture.md docs/plans/2026-07-20-architecture-diagrams.md`

Expected: all configured hooks pass.

- [ ] **Step 2: Commit and push**

Run:

```bash
git add docs/architecture.md docs/plans/2026-07-20-architecture-diagrams.md
git commit -m "docs: map the foundation architecture"
git push origin feat/videomae-pretraining
```

Expected: PR #3 head advances and remains mergeable.

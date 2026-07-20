# Foundation architecture

The v1 stack keeps dataset-specific discovery separate from the shared
[DatasetManifest](../src/marineworld/data/contracts.py#L36) and
[VideoRecord](../src/marineworld/data/contracts.py#L15) contract. The
[MaritimeClipDataset](../src/marineworld/data/clips.py#L122) consumes that
contract. Balanced SMD/FVessel sampling is an input to pretraining; evaluation
loads the resulting encoder from a checkpoint, and observability receives only
bounded experiment outputs.

```mermaid
flowchart LR
    subgraph datasets["Dataset adapters"]
        FV["FVessel adapter"]
        SMD["SMD adapter"]
    end

    subgraph contract["Shared data contract"]
        MANIFEST["DatasetManifest + VideoRecord"]
        CLIPS["MaritimeClipDataset<br/>clip indexing, decoding, tensors"]
    end

    subgraph sampling["Balanced sampling"]
        JOINT["Deterministic joint manifests<br/>SMD + FVessel"]
        BALANCED["Balanced sampler<br/>resume-deterministic"]
    end

    subgraph pretraining["Pretraining"]
        ENTRY["Hydra + Lightning entrypoint"]
        VIDEOMAE["VideoMAE masked reconstruction"]
        CHECKPOINTS["Best and last checkpoints"]
    end

    subgraph evaluation["Evaluation"]
        ENCODER["Frozen encoder"]
        PROBES["Probes + bounded diagnostics"]
    end

    subgraph observability["Observability"]
        WANDB["W&B run identity, config, and metrics"]
        MEDIA["Bounded reconstruction previews"]
    end

    FV --> MANIFEST
    SMD --> MANIFEST
    MANIFEST --> CLIPS
    MANIFEST --> JOINT
    CLIPS --> ENTRY
    JOINT --> BALANCED
    BALANCED --> ENTRY
    ENTRY --> VIDEOMAE --> CHECKPOINTS
    CHECKPOINTS --> ENCODER --> PROBES
    ENTRY --> WANDB
    PROBES --> WANDB
    CHECKPOINTS --> MEDIA --> WANDB
```

## Pull request map

| Pull request | Architectural contribution |
| --- | --- |
| [PR #3 — VideoMAE pretraining](https://github.com/charleneleong-ai/marineworld-fm/pull/3) | VideoMAE model and Lightning pretraining entrypoint, with core W&B run identity and metrics. |
| [PR #4 — Representation evaluation](https://github.com/charleneleong-ai/marineworld-fm/pull/4) | Frozen encoder loading, lightweight probes, and bounded evaluation diagnostics. |
| [PR #5 — Joint maritime pretraining](https://github.com/charleneleong-ai/marineworld-fm/pull/5) | Deterministic joint manifests plus balanced SMD/FVessel sampling and resume behavior feeding pretraining. |
| [PR #6 — W&B media](https://github.com/charleneleong-ai/marineworld-fm/pull/6) | Validation and best-checkpoint reconstruction previews sent to W&B at bounded intervals. |

"""Reproducible frozen-feature probes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from marineworld.data.contracts import VideoRecord

ProbeTask = Literal["classification", "count"]
ProbeStatus = Literal[
    "COMPLETED",
    "SKIPPED_RESOURCE",
    "SKIPPED_DEGENERATE_LABELS",
    "SKIPPED_UNAVAILABLE_LABELS",
]


class ParameterizedEncoder(Protocol):
    def parameters(self) -> Sequence[torch.nn.Parameter]: ...


DenseBatchFactory = Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]


class DegenerateLabelsError(ValueError):
    """A labelled subset cannot support a two-class probe."""


class LabelsUnavailableError(ValueError):
    """A probe split contains no supervised examples."""


@dataclass(frozen=True)
class ProbeResult:
    condition: str
    checkpoint: str
    manifest_checksum: str
    dataset: str
    task: str
    fraction: float
    seed: int
    metric: str
    value: float | None
    status: ProbeStatus
    evaluation_split: str = "val"
    model: str | None = None
    device: str | None = None
    label_subset_checksum: str | None = None
    labelled_record_ids: str | None = None


def sample_labelled_records(
    records: Sequence[VideoRecord], fraction: float, seed: int
) -> tuple[str, ...]:
    """Select a deterministic, nonempty fraction of training videos."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    record_ids = sorted(record.id for record in records if record.split == "train")
    if not record_ids:
        raise ValueError("cannot sample labels without training records")
    count = max(1, round(len(record_ids) * fraction))
    rng = np.random.default_rng(seed)
    return tuple(sorted(rng.choice(record_ids, size=count, replace=False).tolist()))


def fit_linear_probe(
    encoder: ParameterizedEncoder,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    task: ProbeTask,
) -> LogisticRegression:
    """Fit a class/count probe without constructing an encoder autograd graph."""
    if task not in ("classification", "count"):
        raise ValueError(f"unsupported linear probe task: {task}")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    probe = LogisticRegression(random_state=0)
    return probe.fit(features.detach().cpu().numpy(), labels.detach().cpu().numpy())


def evaluate_probe(
    probe: LogisticRegression,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    task: ProbeTask,
) -> tuple[str, float]:
    """Evaluate class/count probes with macro F1 over their discrete labels."""
    if task not in ("classification", "count"):
        raise ValueError(f"unsupported linear probe task: {task}")
    predictions = probe.predict(features.detach().cpu().numpy())
    value = f1_score(labels.detach().cpu().numpy(), predictions, average="macro")
    return "macro_f1", float(value)


def fit_dense_probe(
    encoder: ParameterizedEncoder,
    spatial_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    epochs: int = 20,
    lr: float = 0.1,
    seed: int = 42,
) -> torch.nn.Linear:
    """Train one linear spatial head over detached encoder tokens."""
    if spatial_features.ndim != 5:
        raise ValueError("spatial features must be shaped [B, T, H, W, D]")
    _freeze_encoder(encoder)
    features = spatial_features.detach().reshape(-1, spatial_features.shape[-1])
    targets = labels.detach().reshape(-1).to(dtype=torch.long, device=features.device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = torch.nn.Linear(features.shape[-1], num_classes, device=features.device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(head(features), targets).backward()
        optimizer.step()
    return head.eval()


def evaluate_dense_probe(
    head: torch.nn.Linear,
    spatial_features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[str, float]:
    """Evaluate foreground/background token predictions with macro F1."""
    features = spatial_features.detach().reshape(-1, spatial_features.shape[-1])
    predictions = head(features).argmax(dim=1).cpu().numpy()
    value = f1_score(labels.detach().reshape(-1).cpu().numpy(), predictions, average="macro")
    return "macro_f1", float(value)


def fit_dense_probe_streaming(
    encoder: ParameterizedEncoder,
    batches: DenseBatchFactory,
    *,
    num_classes: int,
    epochs: int = 5,
    lr: float = 0.1,
    seed: int = 42,
) -> torch.nn.Linear:
    """Train one dense head while retaining only one feature batch at a time."""
    _freeze_encoder(encoder)
    head: torch.nn.Linear | None = None
    optimizer: torch.optim.Optimizer | None = None
    observed_labels: set[int] = set()
    for _ in range(epochs):
        for spatial_features, labels in batches():
            features = spatial_features.detach().reshape(-1, spatial_features.shape[-1])
            targets = labels.detach().reshape(-1).to(dtype=torch.long, device=features.device)
            observed_labels.update(int(value) for value in targets.unique())
            if head is None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    head = torch.nn.Linear(
                        features.shape[-1],
                        num_classes,
                        device=features.device,
                    )
                optimizer = torch.optim.SGD(head.parameters(), lr=lr)
            assert optimizer is not None
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(head(features), targets).backward()
            optimizer.step()
    if head is None:
        raise LabelsUnavailableError("dense label subset produced no training clips")
    if len(observed_labels) < 2:
        raise DegenerateLabelsError("dense label subset contains fewer than two classes")
    return head.eval()


def evaluate_dense_probe_streaming(
    head: torch.nn.Linear,
    batches: DenseBatchFactory,
    *,
    num_classes: int = 2,
) -> tuple[str, float]:
    """Aggregate dense macro F1 from bounded-batch confusion counts."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    observations = 0
    observed_labels: set[int] = set()
    with torch.inference_mode():
        for spatial_features, labels in batches():
            features = spatial_features.reshape(-1, spatial_features.shape[-1])
            targets = labels.reshape(-1).to(dtype=torch.long, device=features.device)
            predictions = head(features).argmax(dim=1)
            observations += targets.numel()
            observed_labels.update(int(value) for value in targets.unique())
            counts = torch.bincount(
                targets * num_classes + predictions,
                minlength=num_classes**2,
            ).reshape(num_classes, num_classes)
            confusion += counts.cpu()
    if observations == 0:
        raise LabelsUnavailableError("validation split produced no labelled dense tokens")
    if len(observed_labels) < 2:
        raise DegenerateLabelsError("validation split contains fewer than two classes")
    scores = []
    for label in range(num_classes):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum()) - true_positive
        false_negative = int(confusion[label, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return "macro_f1", float(np.mean(scores))


def aggregate_probe_results(results: Sequence[ProbeResult]) -> pd.DataFrame:
    """Return a stable tabular representation suitable for one W&B table."""
    table = pd.DataFrame(asdict(result) for result in results)
    if "value" in table:
        table["value"] = pd.Series(
            [result.value for result in results],
            dtype=object,
        )
    return table


def nearest_neighbour_diagnostic(
    queries: torch.Tensor,
    references: torch.Tensor,
    *,
    query_ids: Sequence[str],
    reference_ids: Sequence[str],
    max_references: int = 1_024,
    neighbours: int = 1,
    seed: int = 42,
) -> tuple[dict[str, str | int], ...]:
    """Return bounded cosine neighbours without materializing a corpus-scale index."""
    if queries.ndim != 2 or references.ndim != 2 or queries.shape[1] != references.shape[1]:
        raise ValueError("query and reference features must be [N, D] with matching D")
    if len(query_ids) != len(queries) or len(reference_ids) != len(references):
        raise ValueError("feature and identifier lengths must align")
    if not len(references):
        raise ValueError("nearest-neighbour diagnostic requires a nonempty reference corpus")
    if max_references <= 0 or neighbours <= 0:
        raise ValueError("diagnostic bounds must be positive")
    bounded = min(max_references, len(references))
    selected = torch.randperm(len(references), generator=torch.Generator().manual_seed(seed))[
        :bounded
    ]
    normalized_queries = torch.nn.functional.normalize(queries.detach().cpu(), dim=1)
    normalized_references = torch.nn.functional.normalize(
        references[selected].detach().cpu(), dim=1
    )
    ranks = (
        (normalized_queries @ normalized_references.T).topk(min(neighbours, bounded), dim=1).indices
    )
    return tuple(
        {
            "query_id": query_id,
            "neighbour_id": reference_ids[int(selected[int(reference_index)])],
            "rank": rank + 1,
        }
        for query_id, row in zip(query_ids, ranks, strict=True)
        for rank, reference_index in enumerate(row)
    )


def masked_reconstruction_diagnostic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    """Summarize reconstruction error over masked tokens only."""
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target, and mask shapes must match")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("mask must select at least one token")
    error = (prediction.detach() - target.detach()).square()[mask]
    return {"masked_mse": float(error.mean()), "masked_tokens": int(error.numel())}


def _freeze_encoder(encoder: ParameterizedEncoder) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

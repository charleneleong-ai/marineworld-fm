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


def _freeze_encoder(encoder: ParameterizedEncoder) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

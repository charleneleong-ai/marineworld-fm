"""Single authority for W&B metric and media key names."""

from __future__ import annotations

__all__ = ["metric_name"]


def metric_name(stage: str, metric: str, dataset: str | None = None) -> str:
    """Return a slash-separated metric name in the shared namespace."""
    return "/".join(part for part in (stage, dataset, metric) if part)

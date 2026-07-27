"""Frozen representation evaluation interfaces."""

from marineworld.eval.encoders import EncoderFeatures, FrozenVideoEncoder
from marineworld.eval.probes import (
    ProbeResult,
    evaluate_probe,
    fit_linear_probe,
)

__all__ = [
    "EncoderFeatures",
    "FrozenVideoEncoder",
    "ProbeResult",
    "evaluate_probe",
    "fit_linear_probe",
]

"""
Duration-threshold classifier — a minimal worked example.

Assigns each detected event one of two call-type labels purely from its
duration, using a single configurable threshold. It is intentionally
simple: a template to copy when writing a real classifier, not a
validated call-type model.

To write your own classifier: copy this file, change id/display_name/
description/Params, and implement classify(). It is auto-discovered and
registered — nothing else needs editing (see classifiers/__init__.py).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from squeak_peek.classifiers.base import AbstractClassifier

if TYPE_CHECKING:
    from squeak_peek.labels.model import Label


class DurationClassifierParams(BaseModel):
    """Parameters for the duration-threshold classifier."""

    threshold: float = Field(
        0.02, ge=0.0, le=10.0,
        description="Duration cutoff separating the two call-type buckets.",
        json_schema_extra={"unit": "s", "decimals": 4},
    )
    short_label: str = Field(
        "sk", description="Call type assigned to events shorter than the threshold.",
    )
    long_label: str = Field(
        "d", description="Call type assigned to events at or above the threshold.",
    )

    model_config = {"populate_by_name": True}


class DurationClassifier(AbstractClassifier):
    """Buckets events into two call types by duration alone."""

    id = "DURATION"
    display_name = "Duration threshold"
    description = (
        "Baseline example classifier — assigns a call type purely from event "
        "duration. Not a validated call-type model; use it as a template."
    )
    Params = DurationClassifierParams

    def classify(self, labels: list[Label], signal: np.ndarray, fs: int) -> list[Label]:
        p = self.params
        return [
            replace(lbl, label=p.short_label if lbl.duration < p.threshold else p.long_label)
            for lbl in labels
        ]

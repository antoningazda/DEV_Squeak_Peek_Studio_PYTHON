"""
Trained USV call-type model (CNN USV/NOISE + Random Forest call types).

Runs a model trained in the Classification tab (or by the standalone
USV_Klasifikace tool) on a detector's events, so detection and
classification happen in one pass from the Detection tab. See
squeak_peek.usv_classifier for the pipeline itself.

Each event's label becomes its predicted call type, or NOISE (the CNN
rejected it as a USV), UNCERTAIN (the Random Forest was not confident —
optionally shown as the best guess with a "?", e.g. "5t?") or
PROCESSING_ERROR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from squeak_peek.classifiers.base import AbstractClassifier

if TYPE_CHECKING:
    from squeak_peek.labels.model import Label


class USVModelParams(BaseModel):
    """Parameters for the trained USV call-type model."""

    modelPath: str = Field(
        "",
        description="Trained model: the manifest.json inside a model folder "
                    "(Classification tab → Train model writes one to <output>/run/model).",
        json_schema_extra={"widget": "file", "file_filter": "Model manifest (manifest.json);;All files (*)"},
    )
    device: str = Field(
        "cpu",
        description="Where the CNN runs. 'auto' uses CUDA when available; 'mps' is Apple-GPU (experimental).",
        json_schema_extra={"choices": ["cpu", "auto", "mps"]},
    )
    dropNoise: bool = Field(
        False, description="Remove events the CNN classifies as NOISE instead of labeling them 'NOISE'.",
    )
    guessUncertain: bool = Field(
        True, description="Label UNCERTAIN events with the Random Forest's best guess plus '?' (e.g. '5t?').",
    )

    model_config = {"populate_by_name": True}


class USVModelClassifier(AbstractClassifier):
    """CNN + Random Forest call-type model trained on expert labels."""

    id = "USV_MODEL"
    display_name = "USV model"
    description = (
        "Trained two-stage model: a CNN separates real USVs from noise, then a Random "
        "Forest assigns the call type, flagging low-confidence calls as UNCERTAIN. "
        "Train or pick a model in the Classification tab."
    )
    Params = USVModelParams

    def classify(self, labels: list[Label], signal: np.ndarray, fs: int) -> list[Label]:
        from squeak_peek.usv_classifier import api

        p = self.params
        if not p.modelPath:
            raise RuntimeError(
                "No USV model selected. Choose one in the Classification tab or in "
                "Settings → USV model classifier."
            )
        predictions = api.classify_signal(labels, signal, fs, p.modelPath, device=p.device)
        if predictions.empty:
            return list(labels)
        return api.predictions_to_labels(
            predictions, labels, drop_noise=p.dropNoise, guess_uncertain=p.guessUncertain,
        )

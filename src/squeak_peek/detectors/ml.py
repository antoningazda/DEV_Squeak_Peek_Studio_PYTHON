"""
ML (Random Forest) detector: inference over a model trained by
squeak_peek.ml.train.

Port of MLDetector.m. Loads a model_dict (see ml/train.py's
train_model()/save_model()) and applies it via a sliding window of
acoustic features to find USV event boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import medfilt

from squeak_peek.config import MLParams
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.features.extract import extract_frame_features
from squeak_peek.labels.model import Label
from squeak_peek.ml.train import _nfft_for, _preprocess, load_model


class MLDetector(AbstractDetector):
    """
    Random Forest sliding-window USV detector.

    The model file (params.modelPath) is self-describing: its frame_params
    (frame length/hop, bandpass cutoffs) and feature_cols travel with it, so
    inference always matches how the model was trained.
    """

    def __init__(self, params: MLParams):
        self.params = params
        self._model_dict: dict[str, Any] | None = None

    def _load_model(self) -> dict[str, Any]:
        if self._model_dict is None:
            if not self.params.modelPath:
                raise ValueError(
                    "MLDetector: params.modelPath is not set. Train a model "
                    "(squeak_peek.ml.train.train_model) and point MLParams.modelPath "
                    "at the saved .joblib file."
                )
            self._model_dict = load_model(self.params.modelPath)
        return self._model_dict

    def detect(self, signal: np.ndarray, fs: int) -> list[Label]:
        model_dict = self._load_model()
        model = model_dict["model"]
        feature_cols = model_dict["feature_cols"]
        fp = model_dict["frame_params"]

        x = _preprocess(signal, fs, fp["fcutMin"], fp["fcutMax"])

        frame_len = round(fp["frame_len_s"] * fs)
        hop_len = round(fp["hop_len_s"] * fs)
        nfft = _nfft_for(frame_len)

        X, mid_times = extract_frame_features(x, fs, frame_len, hop_len, nfft, fp["fcutMin"], fp["fcutMax"])
        if X.shape[0] == 0:
            return []

        if X.shape[1] != len(feature_cols):
            raise ValueError(
                f"MLDetector: model expects {len(feature_cols)} features "
                f"({feature_cols}) but the current extractor produced {X.shape[1]}. "
                "Retrain the model against the current feature extractor."
            )

        classes = list(model.classes_)
        usv_idx = classes.index(1)
        probs = model.predict_proba(X)[:, usv_idx]

        binary = probs > self.params.sensitivity
        if len(binary) >= 3:
            # zero-padded 3-tap median filter, same edge behavior as MATLAB's
            # medfilt1(double(predNum), 3) default.
            binary = medfilt(binary.astype(np.float64), 3) > 0.5

        edges = np.diff(np.concatenate([[0], binary.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1

        labels: list[Label] = []
        for s, e in zip(starts, ends):
            t_start = float(mid_times[s])
            t_end = float(mid_times[e])
            if (t_end - t_start) < self.params.minEventDuration:
                continue
            labels.append(
                Label(
                    start_time=t_start,
                    end_time=t_end,
                    label="d",
                    start_frequency=0.0,
                    end_frequency=0.0,
                    start_index=round(t_start * fs),
                    stop_index=round(t_end * fs),
                )
            )

        return labels

    @property
    def name(self) -> str:
        return "ML"

"""
ML detector training: sliding-window Random Forest classifier.

Ports MLDetectorTrain.m. MATLAB's TreeBagger -> sklearn's
RandomForestClassifier (a new model format, not a .mat loader — see
model_dict's shape below, agreed with detectors/ml.py which loads it).

Improvements over the direct MATLAB port:
  - Frame ground-truth labeling (`_label_frames`) is vectorized instead of
    MATLAB's O(n_frames * n_labels) nested loop.
  - `train_from_features()` is split out from `train_model()` so
    ml/optimize.py's noise-ratio sweep can reuse one feature extraction
    pass across multiple ratios instead of re-extracting per ratio
    (MLDetectorOptimize.m's OptimizeNoiseRatio path re-runs
    MLDetectorTrain, and therefore full feature extraction, per ratio).
  - RandomForestClassifier uses n_jobs=-1 (parallel tree building).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from squeak_peek.audio.filters import bandpass_filter_filtfilt
from squeak_peek.audio.io import load_wav
from squeak_peek.features.extract import extract_frame_features
from squeak_peek.labels.io import import_labels
from squeak_peek.labels.model import Label

FEATURE_COLS = [
    "BandPower",
    "SpecCentroid",
    "SpecSpread",
    "SpecFlatness",
    "SpecEntropy",
    "ZCR",
    "SNR_est",
    "SpecFlux",
    "DomFreq",
    "Delta_BandPower",
    "Delta_Centroid",
    "Delta_Entropy",
]


def _preprocess(signal: np.ndarray, fs: int, fcut_min: float, fcut_max: float) -> np.ndarray:
    """DC-removal, normalization, order-8 zero-phase bandpass (MLDetector.m/
    MLDetectorTrain.m both use FilterOrder=8, unlike PSD/BSCD's order 12)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - np.mean(x)
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x = x / max_val
    return bandpass_filter_filtfilt(x, fs, fcut_min, fcut_max, order=8)


def _nfft_for(frame_len: int) -> int:
    """MATLAB: 2^nextpow2(frameLen)."""
    return 1 << int(np.ceil(np.log2(max(frame_len, 1))))


def _label_frames(mid_times: np.ndarray, gt_labels: list[Label]) -> np.ndarray:
    """
    Binary USV(1)/NOISE(0) label per frame: a frame is USV if its mid-time
    falls within [StartTime, EndTime] of any ground-truth label.

    Vectorized equivalent of MLDetectorTrain.m's per-frame, per-label loop.
    """
    y = np.zeros(len(mid_times), dtype=np.int64)
    for lbl in gt_labels:
        mask = (mid_times >= lbl.start_time) & (mid_times <= lbl.end_time)
        y[mask] = 1
    return y


def extract_labeled_features(
    wav_path: str | Path,
    label_path: str | Path,
    *,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    frame_len_s: float = 0.02,
    hop_len_s: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one WAV + label file, return (X, y) for that file."""
    signal, fs = load_wav(wav_path)
    x = _preprocess(signal, fs, fcut_min, fcut_max)

    frame_len = round(frame_len_s * fs)
    hop_len = round(hop_len_s * fs)
    nfft = _nfft_for(frame_len)

    X, mid_times = extract_frame_features(x, fs, frame_len, hop_len, nfft, fcut_min, fcut_max)
    if X.shape[0] == 0:
        return X, np.zeros(0, dtype=np.int64)

    gt_labels = import_labels(label_path, fs)
    y = _label_frames(mid_times, gt_labels)

    # Defensive NaN/Inf removal (extract_frame_features already sanitises X,
    # but keep y in sync in case any rows were ever dropped upstream).
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        X = X[~bad]
        y = y[~bad]

    return X, y


def collect_training_data(
    wav_label_pairs: list[tuple[str | Path, str | Path]],
    *,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    frame_len_s: float = 0.02,
    hop_len_s: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and concatenate (X, y) across multiple (wav, label) file pairs."""
    if not wav_label_pairs:
        raise ValueError("collect_training_data: no (wav, label) pairs given.")

    xs, ys = [], []
    for wav_path, label_path in wav_label_pairs:
        X, y = extract_labeled_features(
            wav_path, label_path,
            fcut_min=fcut_min, fcut_max=fcut_max,
            frame_len_s=frame_len_s, hop_len_s=hop_len_s,
        )
        if X.shape[0] > 0:
            xs.append(X)
            ys.append(y)

    if not xs:
        raise ValueError("collect_training_data: no features extracted. Check WAV/label paths.")

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _balance_classes(
    X: np.ndarray, y: np.ndarray, noise_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample the noise (0) class to noise_ratio * n_usv frames."""
    rng = np.random.default_rng(seed)
    idx_usv = np.where(y == 1)[0]
    idx_noise = np.where(y == 0)[0]

    n_usv = len(idx_usv)
    n_noise_desired = round(n_usv * noise_ratio)
    n_noise_actual = min(len(idx_noise), n_noise_desired)

    idx_noise_sample = rng.choice(idx_noise, size=n_noise_actual, replace=False)
    idx_all = np.concatenate([idx_usv, idx_noise_sample])
    rng.shuffle(idx_all)

    return X[idx_all], y[idx_all]


def train_from_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    frame_len_s: float = 0.02,
    hop_len_s: float = 0.005,
    n_trees: int = 200,
    min_leaf_size: int = 3,
    noise_ratio: float = 3.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a RandomForestClassifier from an already-extracted (X, y) pair.

    Split out from train_model() so callers with pre-extracted features
    (e.g. ml/optimize.py sweeping NoiseRatio) can retrain without redoing
    feature extraction each time.

    Returns a model_dict: {"model", "feature_cols", "frame_params", "training_info"}.
    """
    n_usv = int(np.sum(y == 1))
    n_noise = int(np.sum(y == 0))
    if n_usv < 5:
        raise ValueError(f"train_from_features: too few USV frames ({n_usv}). Check label files match WAVs.")

    X_bal, y_bal = _balance_classes(X, y, noise_ratio, seed)

    model = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=min_leaf_size,
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_bal, y_bal)

    return {
        "model": model,
        "feature_cols": list(FEATURE_COLS),
        "frame_params": {
            "frame_len_s": frame_len_s,
            "hop_len_s": hop_len_s,
            "fcutMin": fcut_min,
            "fcutMax": fcut_max,
        },
        "training_info": {
            "n_trees": n_trees,
            "min_leaf_size": min_leaf_size,
            "noise_ratio": noise_ratio,
            "seed": seed,
            "n_frames_total": int(len(y)),
            "n_usv_total": n_usv,
            "n_noise_total": n_noise,
            "n_frames_trained": int(len(y_bal)),
            "oob_accuracy": float(model.oob_score_),
        },
    }


def train_model(
    wav_label_pairs: list[tuple[str | Path, str | Path]],
    *,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    frame_len_s: float = 0.02,
    hop_len_s: float = 0.005,
    n_trees: int = 200,
    min_leaf_size: int = 3,
    noise_ratio: float = 3.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a Random Forest USV-frame classifier from (wav, label) file pairs.

    Port of MLDetectorTrain.m. Returns the model_dict consumed by
    detectors.ml.MLDetector and save_model()/load_model().
    """
    X, y = collect_training_data(
        wav_label_pairs,
        fcut_min=fcut_min, fcut_max=fcut_max,
        frame_len_s=frame_len_s, hop_len_s=hop_len_s,
    )
    return train_from_features(
        X, y,
        fcut_min=fcut_min, fcut_max=fcut_max,
        frame_len_s=frame_len_s, hop_len_s=hop_len_s,
        n_trees=n_trees, min_leaf_size=min_leaf_size,
        noise_ratio=noise_ratio, seed=seed,
    )


def save_model(model_dict: dict[str, Any], path: str | Path) -> None:
    """Save a model_dict (from train_model()) as a joblib file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_dict, path)


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a model_dict saved by save_model()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ML model not found: {path}")
    return joblib.load(path)

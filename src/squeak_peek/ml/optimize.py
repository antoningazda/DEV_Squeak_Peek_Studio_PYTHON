"""
ML detector calibration: find the best Sensitivity (and optionally
NoiseRatio) for a trained model.

Ports MLDetectorOptimize.m's numeric core (no plotting — this returns
structured results only; a GUI progress display is a separate concern).
Uses labels.metrics.compare_labels for F1/Precision/Recall rather than
reimplementing MATLAB's local calculateStats(), which is an exact
duplicate of compareLabels.m's midpoint-matching logic.

Improvement over the direct MATLAB port: sweep_noise_ratio() extracts the
*training* features once and reuses them for every ratio tried, instead
of MLDetectorOptimize.m's OptimizeNoiseRatio path, which calls
MLDetectorTrain (full feature re-extraction) once per ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import medfilt

from squeak_peek.audio.io import load_wav
from squeak_peek.features.extract import extract_frame_features
from squeak_peek.labels.io import import_labels
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.model import Label
from squeak_peek.ml.train import (
    ProgressFn,
    _nfft_for,
    _preprocess,
    collect_training_data,
    train_from_features,
)

DEFAULT_SENSITIVITY_RANGE = tuple(np.round(np.arange(0.1, 0.951, 0.02), 4))
DEFAULT_NOISE_RATIO_RANGE = (1, 3, 5, 10, 15)


def _segments_from_probabilities(
    probs: np.ndarray,
    mid_times: np.ndarray,
    fs: int,
    sensitivity: float,
    min_event_duration: float,
) -> list[Label]:
    binary = probs > sensitivity
    if len(binary) >= 3:
        binary = medfilt(binary.astype(np.float64), 3) > 0.5

    edges = np.diff(np.concatenate([[0], binary.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    labels = []
    for s, e in zip(starts, ends):
        t_start = float(mid_times[s])
        t_end = float(mid_times[e])
        if (t_end - t_start) < min_event_duration:
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


def _extract_validation_features(
    frame_params: dict[str, float], val_wav_path: str | Path, val_label_path: str | Path
) -> tuple[np.ndarray, np.ndarray, list[Label], int]:
    fp = frame_params
    signal, fs = load_wav(val_wav_path)
    gt_labels = [lbl for lbl in import_labels(val_label_path, fs) if lbl.detection_state != "Rejected"]

    x = _preprocess(signal, fs, fp["fcutMin"], fp["fcutMax"])
    frame_len = round(fp["frame_len_s"] * fs)
    hop_len = round(fp["hop_len_s"] * fs)
    nfft = _nfft_for(frame_len)

    X, mid_times = extract_frame_features(x, fs, frame_len, hop_len, nfft, fp["fcutMin"], fp["fcutMax"])
    return X, mid_times, gt_labels, fs


def _score_sensitivities(
    model: Any,
    X: np.ndarray,
    mid_times: np.ndarray,
    gt_labels: list[Label],
    fs: int,
    sensitivity_range: tuple[float, ...],
    min_event_duration: float,
) -> dict[str, Any]:
    """Sweep Sensitivity for one (model, already-extracted-features) pair."""
    if X.shape[0] == 0:
        empty_stats = compare_labels([], gt_labels)
        return {"best_sensitivity": sensitivity_range[0], "best_stats": empty_stats, "sweep": []}

    classes = list(model.classes_)
    usv_idx = classes.index(1)
    probs = model.predict_proba(X)[:, usv_idx]

    sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for sens in sensitivity_range:
        detected = _segments_from_probabilities(probs, mid_times, fs, sens, min_event_duration)
        stats = compare_labels(detected, gt_labels)
        entry = {"sensitivity": float(sens), "stats": stats}
        sweep.append(entry)
        if best is None or stats.f1_score > best["stats"].f1_score:
            best = entry

    return {"best_sensitivity": best["sensitivity"], "best_stats": best["stats"], "sweep": sweep}


def sweep_sensitivity(
    model_dict: dict[str, Any],
    val_wav_path: str | Path,
    val_label_path: str | Path,
    *,
    sensitivity_range: tuple[float, ...] = DEFAULT_SENSITIVITY_RANGE,
    min_event_duration: float = 0.003,
) -> dict[str, Any]:
    """
    Sweep Sensitivity against one validation (wav, label) pair, scoring each
    threshold with labels.metrics.compare_labels. Validation features are
    extracted once and reused across the whole sweep.

    Returns {"best_sensitivity", "best_stats", "sweep": [{"sensitivity", "stats"}]}.
    """
    X, mid_times, gt_labels, fs = _extract_validation_features(
        model_dict["frame_params"], val_wav_path, val_label_path
    )
    return _score_sensitivities(
        model_dict["model"], X, mid_times, gt_labels, fs, sensitivity_range, min_event_duration
    )


def sweep_noise_ratio(
    train_pairs: list[tuple[str | Path, str | Path]],
    val_wav_path: str | Path,
    val_label_path: str | Path,
    *,
    noise_ratio_range: tuple[float, ...] = DEFAULT_NOISE_RATIO_RANGE,
    sensitivity_range: tuple[float, ...] = DEFAULT_SENSITIVITY_RANGE,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    frame_len_s: float = 0.02,
    hop_len_s: float = 0.005,
    n_trees: int = 200,
    min_leaf_size: int = 3,
    min_event_duration: float = 0.003,
    seed: int = 42,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Retrain across several NoiseRatio values and, for each, sweep
    Sensitivity, picking the (ratio, sensitivity) pair with the best F1.

    Both training and validation features are extracted exactly once, up
    front, and reused for every ratio (unlike MLDetectorOptimize.m's
    OptimizeNoiseRatio path, which reruns full training AND validation
    feature extraction per ratio via MLDetectorTrain/predict); only
    class-balancing + refitting + threshold scoring repeats per ratio.

    Returns {"best_noise_ratio", "best_sensitivity", "best_stats",
             "best_model", "sweep": [{"noise_ratio", "best_sensitivity",
             "best_stats", "sweep", "model"}]}.
    """
    X, y = collect_training_data(
        train_pairs, fcut_min=fcut_min, fcut_max=fcut_max, frame_len_s=frame_len_s, hop_len_s=hop_len_s,
        progress=progress,
    )
    frame_params = {
        "frame_len_s": frame_len_s, "hop_len_s": hop_len_s,
        "fcutMin": fcut_min, "fcutMax": fcut_max,
    }
    if progress is not None:
        progress(None, f"Extracting validation features: {Path(val_wav_path).name}")
    X_val, mid_times_val, gt_labels, fs = _extract_validation_features(
        frame_params, val_wav_path, val_label_path
    )

    sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for i, ratio in enumerate(noise_ratio_range):
        if progress is not None:
            progress(i / len(noise_ratio_range),
                     f"Training with noise ratio {ratio} ({i + 1}/{len(noise_ratio_range)})")
        model_dict = train_from_features(
            X, y,
            fcut_min=fcut_min, fcut_max=fcut_max,
            frame_len_s=frame_len_s, hop_len_s=hop_len_s,
            n_trees=n_trees, min_leaf_size=min_leaf_size,
            noise_ratio=ratio, seed=seed,
        )
        sens_result = _score_sensitivities(
            model_dict["model"], X_val, mid_times_val, gt_labels, fs,
            sensitivity_range, min_event_duration,
        )
        entry = {
            "noise_ratio": ratio,
            "best_sensitivity": sens_result["best_sensitivity"],
            "best_stats": sens_result["best_stats"],
            "sweep": sens_result["sweep"],
            "model": model_dict,
        }
        sweep.append(entry)
        if best is None or entry["best_stats"].f1_score > best["best_stats"].f1_score:
            best = entry

    return {
        "best_noise_ratio": best["noise_ratio"],
        "best_sensitivity": best["best_sensitivity"],
        "best_stats": best["best_stats"],
        "best_model": best["model"],
        "sweep": sweep,
    }

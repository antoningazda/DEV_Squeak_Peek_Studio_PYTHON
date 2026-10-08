"""
Detector parameter tuning: search a detector's numeric parameters for the
best F1 against ground-truth labels.

Python successor to the MATLAB app's runBayesianOptPSD.m, generalized to
every registered detector (PSD, BSCD, RBD, and the ML/CNN detectors'
sensitivity / min-event settings). The search is a dependency-free
"random start, then local refinement" loop:

  1. trial 0 is the current parameters, so the result is never worse;
  2. about a third of the budget samples the search box uniformly
     (log-uniformly for ranges spanning more than a decade);
  3. the rest perturbs the best point found so far with a shrinking step.

Every recording is loaded once; F1 is pooled over all recordings (TP/FP/FN
summed), scored with labels.metrics.compare_labels — the same midpoint
matching the Metrics tab uses.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from squeak_peek.audio.io import load_wav
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.io import import_labels
from squeak_peek.labels.metrics import ComparisonStats, compare_labels
from squeak_peek.labels.model import Label

ProgressFn = Callable[[float | None, str], None]
PostProcessFn = Callable[[list[Label], np.ndarray, int], list[Label]]
PreProcessFn = Callable[[np.ndarray, int], np.ndarray]

# Fields left out of the default search: the analysis band and ROI are
# experiment choices, not knobs to fit; model orders change cost a lot.
_NOT_TUNED_BY_DEFAULT = {
    "fcutMin", "fcutMax", "ROIstart", "ROIlength",
    "AR_order_left", "AR_order_right", "Bayesian_Evidence_order",
}


@dataclass
class ParamRange:
    name: str
    lo: float
    hi: float
    integer: bool = False
    enabled: bool = True

    @property
    def log(self) -> bool:
        """Search log-uniformly when the range spans more than a decade."""
        return self.lo > 0 and self.hi / self.lo > 10


@dataclass
class Trial:
    params: dict[str, Any]
    stats: ComparisonStats | None     # None = this combination failed
    error: str = ""

    @property
    def f1(self) -> float:
        return self.stats.f1_score if self.stats is not None else -1.0


@dataclass
class TuningResult:
    detector_id: str
    best_params: BaseModel
    best: Trial
    baseline: Trial
    trials: list[Trial] = field(default_factory=list)
    tuned: list[str] = field(default_factory=list)


def _bounds(info) -> tuple[float | None, float | None]:
    lo = hi = None
    for c in info.metadata:
        lo = getattr(c, "ge", None) if getattr(c, "ge", None) is not None else lo
        lo = getattr(c, "gt", None) if getattr(c, "gt", None) is not None else lo
        hi = getattr(c, "le", None) if getattr(c, "le", None) is not None else hi
        hi = getattr(c, "lt", None) if getattr(c, "lt", None) is not None else hi
    return lo, hi


def default_ranges(params: BaseModel) -> list[ParamRange]:
    """A search box around the current value of every numeric field.

    Fractions (bounded to [0, 1]) get ±0.25; everything else ÷4 … ×4,
    clipped to the field's own validation bounds.
    """
    ranges: list[ParamRange] = []
    for name, info in type(params).model_fields.items():
        if info.annotation not in (int, float):
            continue
        cur = float(getattr(params, name))
        lo_b, hi_b = _bounds(info)
        lo_b = -math.inf if lo_b is None else float(lo_b)
        hi_b = math.inf if hi_b is None else float(hi_b)
        if hi_b <= 1.0 and lo_b >= 0.0 and cur >= 0.05:
            lo, hi = cur - 0.25, cur + 0.25
        elif cur > 0:
            lo, hi = cur / 4, cur * 4
        else:
            lo, hi = lo_b if math.isfinite(lo_b) else 0.0, hi_b if math.isfinite(hi_b) else 1.0
        lo, hi = max(lo, lo_b), min(hi, hi_b)
        integer = info.annotation is int
        if integer:
            lo, hi = math.floor(lo), math.ceil(hi)
        if hi <= lo:
            continue
        ranges.append(ParamRange(name, lo, hi, integer, enabled=name not in _NOT_TUNED_BY_DEFAULT))
    return ranges


def _sample(rng: np.random.Generator, r: ParamRange) -> float:
    if r.log:
        return float(math.exp(rng.uniform(math.log(r.lo), math.log(r.hi))))
    return float(rng.uniform(r.lo, r.hi))


def _perturb(rng: np.random.Generator, r: ParamRange, value: float, scale: float) -> float:
    if r.log:
        a, b, v = math.log(r.lo), math.log(r.hi), math.log(max(value, r.lo))
        return float(math.exp(np.clip(v + rng.normal(0, scale * (b - a)), a, b)))
    return float(np.clip(value + rng.normal(0, scale * (r.hi - r.lo)), r.lo, r.hi))


def _cast(r: ParamRange, value: float) -> float | int:
    # 4 significant digits: finer than any detector is sensitive to, and
    # keeps Settings readable.
    return int(round(value)) if r.integer else float(f"{value:.4g}")


def _pooled(stats: Sequence[ComparisonStats]) -> ComparisonStats:
    tp = sum(s.true_positives for s in stats)
    fp = sum(s.false_positives for s in stats)
    fn = sum(s.false_negatives for s in stats)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ComparisonStats(
        total_provided_labels=sum(s.total_provided_labels for s in stats),
        total_detected_labels=sum(s.total_detected_labels for s in stats),
        true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=precision, recall=recall, f1_score=f1,
    )


def _prepare_params(detector_id: str, base: BaseModel) -> BaseModel:
    """The tuner slices the signal itself, so detectors with their own
    region-of-interest switch (PSD) must analyse whatever they are given."""
    if "runWholeSignal" in type(base).model_fields:
        return base.model_copy(update={"runWholeSignal": True})
    return base


def tune_detector(
    detector_id: str,
    base_params: BaseModel,
    wav_label_pairs: Sequence[tuple[str | Path, str | Path]],
    ranges: Sequence[ParamRange],
    *,
    n_trials: int = 40,
    max_seconds: float | None = 30.0,
    post_process: PostProcessFn | None = None,
    pre_process: PreProcessFn | None = None,
    seed: int = 0,
    progress: ProgressFn | None = None,
) -> TuningResult:
    """Search ``ranges`` (only the enabled ones) for the parameters with the
    best pooled F1 over ``wav_label_pairs``.

    ``max_seconds`` limits each recording to its first N seconds (labels
    outside are dropped) to keep each trial fast; ``None`` uses everything.
    ``pre_process(samples, fs)`` (e.g. denoising) runs once per recording
    before every trial; ``post_process(labels, samples, fs)`` mirrors Run
    detectors' steps and receives the original samples.
    ``progress`` may raise to cancel.
    """
    active = [r for r in ranges if r.enabled]
    if not active:
        raise ValueError("Select at least one parameter to tune.")
    if not wav_label_pairs:
        raise ValueError("No (wav, label) pairs given.")
    det_cls = AbstractDetector.get(detector_id)
    rng = np.random.default_rng(seed)

    data: list[tuple[np.ndarray, int, list[Label]]] = []
    for i, (wav, lbl_path) in enumerate(wav_label_pairs):
        if progress is not None:
            progress(None, f"Loading {Path(wav).name} ({i + 1}/{len(wav_label_pairs)})…")
        samples, fs = load_wav(wav)
        labels = [lbl for lbl in import_labels(lbl_path, fs) if lbl.detection_state != "Rejected"]
        if max_seconds:
            samples = samples[: int(round(max_seconds * fs))]
            labels = [lbl for lbl in labels if lbl.end_time <= max_seconds]
        data.append((samples, pre_process(samples, fs) if pre_process else samples, fs, labels))
    if sum(len(lbl) for _s, _d, _f, lbl in data) == 0:
        raise ValueError("The label files contain no calls in the analysed part of the recordings.")

    base = _prepare_params(detector_id, base_params)
    base_values = base.model_dump()

    def evaluate(values: dict[str, Any]) -> Trial:
        try:
            params = type(base).model_validate({**base_values, **values})
            detector = det_cls(params)
            stats = []
            for samples, det_input, fs, ref in data:
                detected = detector.detect(det_input, fs)
                if post_process is not None:
                    detected = post_process(detected, samples, fs)
                stats.append(compare_labels(detected, ref))
            return Trial(values, _pooled(stats))
        except Exception as exc:  # noqa: BLE001 — a bad combination, not a bad run
            return Trial(values, None, str(exc) or type(exc).__name__)

    trials: list[Trial] = []
    n_trials = max(n_trials, 1)
    n_random = max(1, (n_trials - 1) // 3)
    best: Trial | None = None
    for i in range(n_trials):
        if i == 0:
            values = {r.name: getattr(base, r.name) for r in active}
        elif i <= n_random or best is None or best.stats is None:
            values = {r.name: _cast(r, _sample(rng, r)) for r in active}
        else:
            frac = (i - n_random) / max(n_trials - n_random, 1)
            scale = 0.2 * (1 - frac) + 0.03 * frac
            values = {r.name: _cast(r, _perturb(rng, r, best.params[r.name], scale)) for r in active}
        if progress is not None:
            best_txt = f" · best F1 so far {best.f1:.3f}" if best is not None and best.stats else ""
            progress(i / n_trials, f"Trial {i + 1}/{n_trials}{best_txt}")
        trial = evaluate(values)
        trials.append(trial)
        if best is None or trial.f1 > best.f1:
            best = trial

    if best is None or best.stats is None:
        raise RuntimeError(f"Every trial failed; first error: {trials[0].error}")
    best_params = type(base_params).model_validate({**base_params.model_dump(), **best.params})
    return TuningResult(
        detector_id=detector_id, best_params=best_params, best=best, baseline=trials[0],
        trials=trials, tuned=[r.name for r in active],
    )

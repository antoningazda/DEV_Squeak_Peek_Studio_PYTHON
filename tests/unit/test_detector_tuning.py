"""Tests for squeak_peek.detectors.tuning (detector parameter search)."""

from __future__ import annotations

import pytest

import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek.detectors.psd import PSDParams
from squeak_peek.detectors.tuning import ParamRange, default_ranges, tune_detector


def test_default_ranges_respect_bounds_and_defaults():
    ranges = {r.name: r for r in default_ranges(PSDParams())}
    # Band / ROI are not tuned by default; string/bool fields are skipped.
    assert not ranges["fcutMin"].enabled and not ranges["ROIstart"].enabled
    assert "runWholeSignal" not in ranges
    # Fraction field: ±0.25 around the current value, inside its bounds.
    ov = ranges["overlapFactor"]
    assert ov.lo == pytest.approx(0.34) and ov.hi == pytest.approx(0.84)
    # Integer field: ÷4 … ×4, still integer.
    seg = ranges["segmentLength"]
    assert seg.integer and seg.lo == 2048 and seg.hi == 32768
    for r in ranges.values():
        assert r.lo < r.hi


def test_tune_never_worse_than_current(example_wav_path, example_ref_labels_path):
    params = PSDParams(runWholeSignal=True)
    ranges = [ParamRange("k", 0.005, 0.1)]
    result = tune_detector(
        "PSD", params, [(example_wav_path, example_ref_labels_path)], ranges,
        n_trials=3, max_seconds=30.0, seed=1,
    )
    assert len(result.trials) == 3
    assert result.baseline.params == {"k": params.k}
    assert result.best.f1 >= result.baseline.f1
    assert result.best_params.k == result.best.params["k"]
    assert result.tuned == ["k"]


def test_tune_requires_an_enabled_parameter(example_wav_path, example_ref_labels_path):
    with pytest.raises(ValueError):
        tune_detector("PSD", PSDParams(), [(example_wav_path, example_ref_labels_path)],
                      [ParamRange("k", 0.01, 0.1, enabled=False)])


def test_progress_callback_can_cancel(example_wav_path, example_ref_labels_path):
    class Stop(Exception):
        pass

    def progress(_fraction, message):
        if message.startswith("Trial 2"):
            raise Stop

    with pytest.raises(Stop):
        tune_detector("PSD", PSDParams(runWholeSignal=True), [(example_wav_path, example_ref_labels_path)],
                      [ParamRange("k", 0.01, 0.1)], n_trials=5, max_seconds=30.0, progress=progress)

"""
Unit tests for squeak_peek.features.tonality and the broadband label filter.
"""

from __future__ import annotations

import numpy as np

from squeak_peek.features.tonality import tonality_score
from squeak_peek.labels.model import Label
from squeak_peek.labels.postprocess import filter_broadband_labels

FS = 250_000


def _tone(freq_hz: float, duration_s: float = 0.03) -> np.ndarray:
    t = np.arange(round(duration_s * FS)) / FS
    return np.sin(2 * np.pi * freq_hz * t)


def _sweep(f0: float, rate: float, duration_s: float = 0.03) -> np.ndarray:
    t = np.arange(round(duration_s * FS)) / FS
    return np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t**2))


def _noise(duration_s: float = 0.03, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(round(duration_s * FS))


class TestTonalityScore:
    def test_pure_tone_scores_high(self):
        assert tonality_score(_tone(70_000), FS) > 0.9

    def test_fm_sweep_scores_high(self):
        """Real USVs are frequency-modulated, so a sweep must not be penalised."""
        assert tonality_score(_sweep(60_000, 400_000), FS) > 0.9

    def test_broadband_noise_scores_low(self):
        assert tonality_score(_noise(), FS) < 0.4

    def test_tone_above_noise_still_scores_high(self):
        assert tonality_score(_tone(70_000) + 0.3 * _noise(), FS) > 0.7

    def test_segment_shorter_than_window_is_nan(self):
        assert np.isnan(tonality_score(np.zeros(10), FS))

    def test_silent_segment_is_nan(self):
        assert np.isnan(tonality_score(np.zeros(4096), FS))


class TestFilterBroadbandLabels:
    def _signal_with(self, chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks).astype(np.float32)

    def test_keeps_tonal_drops_broadband(self):
        dur = 0.03
        usv = self._signal_with([_tone(70_000, dur), _noise(dur)])
        labels = [
            Label(start_time=0.0, end_time=dur, label="tone"),
            Label(start_time=dur, end_time=2 * dur, label="noise"),
        ]
        kept = filter_broadband_labels(labels, usv, FS, min_tonality=0.5)
        assert [lbl.label for lbl in kept] == ["tone"]

    def test_unscorable_segments_are_kept(self):
        usv = self._signal_with([_tone(70_000, 0.03)])
        labels = [Label(start_time=0.0, end_time=0.0001, label="tiny")]
        assert len(filter_broadband_labels(labels, usv, FS, min_tonality=0.9)) == 1

    def test_empty_inputs_pass_through(self):
        assert filter_broadband_labels([], np.zeros(10), FS) == []
        labels = [Label(start_time=0.0, end_time=0.01)]
        assert filter_broadband_labels(labels, np.zeros(0), FS) == labels

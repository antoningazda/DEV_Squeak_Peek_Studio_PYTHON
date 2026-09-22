"""
Unit tests for squeak_peek.cnn.spectrogram_image — no torch dependency.
"""

from __future__ import annotations

import numpy as np

from squeak_peek.cnn.spectrogram_image import freq_to_row, signal_to_image, time_to_col

FS = 250_000
FCUT_MIN, FCUT_MAX = 40_000, 120_000


def _tone(freq_hz: float, duration_s: float, amplitude: float) -> np.ndarray:
    t = np.arange(round(duration_s * FS)) / FS
    rng = np.random.default_rng(0)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)
            + 0.01 * amplitude * rng.standard_normal(len(t))).astype(np.float32)


class TestSignalToImage:
    def test_axes_match_image_dimensions(self):
        image, freqs, times = signal_to_image(_tone(70_000, 0.2, 0.5), FS, FCUT_MIN, FCUT_MAX)
        assert image.shape == (len(freqs), len(times))
        assert freqs[0] >= FCUT_MIN and freqs[-1] <= FCUT_MAX
        assert np.all(np.diff(freqs) > 0)

    def test_quiet_recording_is_not_blank(self):
        """
        Real USV recordings sit far below 0 dBFS. An absolute dB window
        flattened them to an all-zero image; levels must be referenced to
        the tile's own noise floor instead.
        """
        image, _, _ = signal_to_image(_tone(70_000, 0.2, amplitude=1e-3), FS, FCUT_MIN, FCUT_MAX)
        assert image.max() > 0.5
        assert (image == 0).mean() < 0.95

    def test_gain_invariant(self):
        """Same signal at 100x the recording gain must yield the same image."""
        quiet, _, _ = signal_to_image(_tone(70_000, 0.2, amplitude=1e-3), FS, FCUT_MIN, FCUT_MAX)
        loud, _, _ = signal_to_image(_tone(70_000, 0.2, amplitude=1e-1), FS, FCUT_MIN, FCUT_MAX)
        assert np.abs(quiet - loud).max() < 1e-4

    def test_tone_lands_in_expected_frequency_row(self):
        image, freqs, _ = signal_to_image(_tone(70_000, 0.2, 0.5), FS, FCUT_MIN, FCUT_MAX)
        brightest_row = int(np.argmax(image.mean(axis=1)))
        assert abs(freqs[brightest_row] - 70_000) < 2_000


class TestAxisMapping:
    def test_freq_and_time_lookups_are_clamped(self):
        freqs = np.array([40_000.0, 50_000.0, 60_000.0])
        times = np.array([0.0, 0.5, 1.0])
        assert freq_to_row(freqs, 10_000) == 0
        assert freq_to_row(freqs, 999_000) == len(freqs) - 1
        assert time_to_col(times, -5.0) == 0
        assert time_to_col(times, 99.0) == len(times) - 1

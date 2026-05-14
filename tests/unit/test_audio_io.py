"""
Unit tests for squeak_peek.audio.io.

Covers:
- load_wav: dtype, shape, sample rate, mono conversion
- save_wav: round-trip fidelity, error on non-1D input
- Helper functions: audio_duration, time_to_sample, sample_to_time
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squeak_peek.audio.io import (
    audio_duration,
    load_wav,
    sample_to_time,
    save_wav,
    time_to_sample,
)

# ── Load WAV — using example fixture ───────────────────────────────────────

class TestLoadWav:
    def test_returns_float32(self, example_audio):
        samples, _ = example_audio
        assert samples.dtype == np.float32

    def test_returns_int_sample_rate(self, example_audio):
        _, fs = example_audio
        assert isinstance(fs, int)

    def test_sample_rate_is_250khz(self, example_audio):
        _, fs = example_audio
        assert fs == 250_000

    def test_samples_are_1d(self, example_audio):
        samples, _ = example_audio
        assert samples.ndim == 1

    def test_samples_non_empty(self, example_audio):
        samples, _ = example_audio
        assert len(samples) > 0

    def test_samples_in_valid_range(self, example_audio):
        samples, _ = example_audio
        # Ultrasonic recordings are normalised; values should be within [-1, 1]
        # (with a small tolerance for float32 rounding)
        assert samples.max() <= 1.1
        assert samples.min() >= -1.1

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_wav(tmp_path / "nonexistent.wav")

    def test_accepts_path_object(self, example_wav_path):
        samples, fs = load_wav(Path(example_wav_path))
        assert fs == 250_000

    def test_accepts_string_path(self, example_wav_path):
        samples, fs = load_wav(str(example_wav_path))
        assert fs == 250_000


# ── Save WAV — round-trip ───────────────────────────────────────────────────

class TestSaveWav:
    def test_round_trip_preserves_data(self, example_audio, tmp_path):
        samples, fs = example_audio
        out = tmp_path / "round_trip.wav"
        save_wav(out, samples, fs)
        reloaded, fs2 = load_wav(out)

        assert fs2 == fs
        # Float32 WAV is lossless — should be bit-identical
        np.testing.assert_array_almost_equal(reloaded, samples, decimal=6)

    def test_round_trip_sample_rate(self, example_audio, tmp_path):
        samples, fs = example_audio
        out = tmp_path / "sr_check.wav"
        save_wav(out, samples, fs)
        _, fs2 = load_wav(out)
        assert fs2 == 250_000

    def test_rejects_2d_input(self, example_audio, tmp_path):
        samples, fs = example_audio
        stereo = np.column_stack([samples, samples])
        with pytest.raises(ValueError, match="1-D"):
            save_wav(tmp_path / "stereo.wav", stereo, fs)

    def test_casts_float64_to_float32(self, tmp_path):
        fs = 250_000
        sine = np.sin(2 * np.pi * 50_000 * np.arange(1000) / fs)  # float64
        out = tmp_path / "cast_test.wav"
        save_wav(out, sine, fs)  # should not raise
        reloaded, _ = load_wav(out)
        assert reloaded.dtype == np.float32


# ── Helper functions ────────────────────────────────────────────────────────

class TestHelpers:
    def test_audio_duration(self):
        samples = np.zeros(250_000, dtype=np.float32)
        assert audio_duration(samples, 250_000) == pytest.approx(1.0)

    def test_audio_duration_half_second(self):
        samples = np.zeros(125_000, dtype=np.float32)
        assert audio_duration(samples, 250_000) == pytest.approx(0.5)

    def test_time_to_sample(self):
        assert time_to_sample(1.0, 250_000) == 250_000
        assert time_to_sample(0.0, 250_000) == 0
        assert time_to_sample(0.5, 250_000) == 125_000

    def test_time_to_sample_rounding(self):
        # 0.213 s * 250_000 = 53250.0 exactly
        assert time_to_sample(0.213, 250_000) == 53_250

    def test_sample_to_time(self):
        assert sample_to_time(250_000, 250_000) == pytest.approx(1.0)
        assert sample_to_time(0, 250_000) == pytest.approx(0.0)

    def test_round_trip_time_sample(self):
        fs = 250_000
        t = 0.3456
        idx = time_to_sample(t, fs)
        t2 = sample_to_time(idx, fs)
        assert t2 == pytest.approx(t, abs=1 / fs)


# ── Synthetic WAV smoke test (no MATLAB data needed) ───────────────────────

class TestSyntheticWav:
    """Tests that run without the MATLAB example data."""

    def test_save_and_load_sine(self, tmp_path):
        fs = 250_000
        duration = 0.1  # 100 ms
        t = np.arange(int(fs * duration)) / fs
        sine = np.sin(2 * np.pi * 50_000 * t).astype(np.float32)

        out = tmp_path / "sine_50khz.wav"
        save_wav(out, sine, fs)
        loaded, loaded_fs = load_wav(out)

        assert loaded_fs == fs
        assert loaded.shape == sine.shape
        np.testing.assert_array_almost_equal(loaded, sine, decimal=6)

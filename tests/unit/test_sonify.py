"""
Unit tests for sonify_segment phase-vocoder implementation.

Tests focus on shape/finiteness/normalization invariants rather than
exact numerical match against MATLAB, since perfect equivalence requires
a MATLAB runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from squeak_peek.audio.sonify import sonify_segment


class TestSonifySegmentBasics:
    """Test basic properties and edge cases of sonify_segment."""

    def test_output_finite(self):
        """Output contains no NaN or inf values."""
        fs = 250000
        duration = 0.1  # 100 ms
        n_samples = int(fs * duration)

        # Synthetic 60 kHz tone
        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=4
        )

        assert np.all(np.isfinite(result)), "Output contains NaN or inf"

    def test_output_normalized(self):
        """Output normalized to max absolute value <= 1.0 (plus epsilon)."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=4
        )

        max_abs = np.max(np.abs(result))
        # Allow small epsilon due to floating-point precision
        assert max_abs <= 1.0 + 1e-6, f"Output exceeds max (max={max_abs})"

    def test_output_non_empty(self):
        """Output is non-empty."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=4
        )

        assert len(result) > 0, "Output is empty"

    def test_identity_case_no_pitch_shift_no_stretch(self):
        """With semitones=0 and slowdown=1, output should closely resemble input.

        Due to windowed overlap-add not being perfectly lossless at edges,
        we check correlation rather than exact equality.
        """
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=1
        )

        # Extract the segment that was processed
        start_idx = round(0.01 * fs)
        end_idx = round(0.09 * fs)
        segment = audio[start_idx:end_idx]

        # Normalize for comparison
        segment_norm = segment / (np.max(np.abs(segment)) + np.finfo(float).eps)

        # Check correlation (should be reasonably high)
        # Need to handle length mismatch due to overlap-add buffer
        min_len = min(len(segment_norm), len(result))
        correlation = np.corrcoef(
            segment_norm[:min_len], result[:min_len]
        )[0, 1]

        # Correlation should be > 0.5 for this identity-like case
        # (not perfect due to windowing, but clearly recognizable)
        assert correlation > 0.5, (
            f"Correlation with no-op settings too low: {correlation}"
        )

    def test_slowdown_increases_duration(self):
        """With slowdown > 1, output should be roughly slowdown times longer.

        Duration check must account for STFT frame-boundary rounding.
        """
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        slowdown = 4.0
        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=slowdown
        )

        segment = audio[round(0.01 * fs) : round(0.09 * fs)]
        expected_samples = len(segment) * slowdown

        # Allow ±10% tolerance for frame-boundary rounding
        lower_bound = expected_samples * 0.9
        upper_bound = expected_samples * 1.1

        assert lower_bound <= len(result) <= upper_bound, (
            f"Output duration {len(result)} not within 10% of expected {expected_samples}"
        )

    def test_pitch_shift_changes_content(self):
        """Pitch-shifted output should differ from unshifted input."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # No pitch shift, no stretch
        result_no_shift = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=1
        )

        # With pitch shift
        result_shift = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=1
        )

        # Should not be identical (though may have some overlap)
        min_len = min(len(result_no_shift), len(result_shift))
        mse = np.mean((result_no_shift[:min_len] - result_shift[:min_len]) ** 2)

        assert mse > 0.01, "Pitch-shifted output too similar to unshifted"


class TestSonifySegmentRobustness:
    """Test robustness to edge cases and parameter variations."""

    def test_short_segment(self):
        """Short segment with moderate window size."""
        fs = 250000
        # 0.01 s = 2500 samples, use smaller window
        duration = 0.01
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # Use smaller window for shorter audio
        result = sonify_segment(
            audio,
            0.001,
            0.009,
            fs,
            semitones=-10,
            slowdown=2,
            win_len=256,
            win_hop=64,
        )

        # Should produce valid output
        assert np.all(np.isfinite(result))
        assert len(result) > 0

    def test_full_segment(self):
        """Segment spanning nearly the entire audio."""
        fs = 250000
        duration = 0.2
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result = sonify_segment(
            audio, 0.0, 0.2, fs, semitones=-20, slowdown=2
        )

        assert np.all(np.isfinite(result))
        assert len(result) > 0

    def test_custom_window_parameters(self):
        """Test with custom window length and hop."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result = sonify_segment(
            audio,
            0.01,
            0.09,
            fs,
            semitones=-20,
            slowdown=2,
            win_len=512,
            win_hop=128,
        )

        assert np.all(np.isfinite(result))
        max_abs = np.max(np.abs(result))
        assert max_abs <= 1.0 + 1e-6

    def test_zero_semitones(self):
        """Pitch shift of 0 semitones should have minimal phase effect."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # R_pitch = 2^(0/12) = 1.0
        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=1
        )

        assert np.all(np.isfinite(result))
        # Should be reasonably close to original (not exact due to windowing)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_negative_pitch_shift(self):
        """Negative pitch shift (lower pitch)."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # R_pitch = 2^(-35/12) ≈ 0.293
        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=1
        )

        assert np.all(np.isfinite(result))
        assert len(result) > 0
        max_abs = np.max(np.abs(result))
        assert max_abs <= 1.0 + 1e-6

    def test_positive_pitch_shift(self):
        """Positive pitch shift (higher pitch)."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # R_pitch = 2^(12/12) = 2.0
        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=12, slowdown=1
        )

        assert np.all(np.isfinite(result))
        assert len(result) > 0
        max_abs = np.max(np.abs(result))
        assert max_abs <= 1.0 + 1e-6

    def test_default_win_hop(self):
        """Test that default win_hop (None) is computed correctly."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # Call without specifying win_hop
        result = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-20, slowdown=2
        )

        # Should use win_len // 4 by default
        assert np.all(np.isfinite(result))
        assert len(result) > 0


class TestSonifySegmentInputValidation:
    """Test handling of various input types and ranges."""

    def test_different_sample_rates(self):
        """Test with different sample rates (must be > 240 kHz for 40-120 kHz filter)."""
        for fs in [250000, 256000, 300000]:
            duration = 0.1
            n_samples = int(fs * duration)

            t = np.arange(n_samples) / fs
            audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

            result = sonify_segment(
                audio, 0.01, 0.09, fs, semitones=-20, slowdown=2
            )

            assert np.all(np.isfinite(result))
            max_abs = np.max(np.abs(result))
            assert max_abs <= 1.0 + 1e-6

    def test_different_audio_dtypes(self):
        """Test with different audio data types."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs

        for dtype in [np.float32, np.float64]:
            audio = np.sin(2 * np.pi * 60000 * t).astype(dtype)

            result = sonify_segment(
                audio, 0.01, 0.09, fs, semitones=-20, slowdown=2
            )

            assert np.all(np.isfinite(result))
            max_abs = np.max(np.abs(result))
            assert max_abs <= 1.0 + 1e-6

    def test_time_boundaries_beyond_audio(self):
        """Test with time boundaries that exceed audio length."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # Request segment beyond the audio duration
        result = sonify_segment(
            audio, 0.05, 0.3, fs, semitones=-20, slowdown=2
        )

        # Should gracefully handle by clamping to audio bounds
        assert np.all(np.isfinite(result))

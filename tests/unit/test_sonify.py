"""
Unit tests for sonify_segment (pitch-shift + time-stretch + resample).

sonify_segment must return audio whose real-world duration and pitch
actually match its semitones/slowdown parameters (see sonify.py's module
docstring for why an earlier hand-rolled implementation didn't).
"""

from __future__ import annotations

import numpy as np

from squeak_peek.audio.sonify import sonify_full_track, sonify_segment


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

        result, target_fs = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=4
        )

        assert target_fs == 44_100
        assert np.all(np.isfinite(result)), "Output contains NaN or inf"

    def test_output_normalized(self):
        """Output normalized to max absolute value <= 1.0 (plus epsilon)."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result, _ = sonify_segment(
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

        result, _ = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=4
        )

        assert len(result) > 0, "Output is empty"

    def test_identity_case_no_pitch_shift_no_stretch(self):
        """With semitones=0 and slowdown=1, output duration (at target_fs)
        should closely match the input segment's real-world duration —
        pitch_shift/time_stretch are each a no-op at these settings, and
        the only remaining operation is a true resample."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result, target_fs = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=1
        )

        expected_duration = 0.09 - 0.01
        actual_duration = len(result) / target_fs
        assert abs(actual_duration - expected_duration) < 0.01

    def test_slowdown_increases_duration(self):
        """With slowdown > 1, real-world output duration (at target_fs)
        should be ~slowdown times the input segment's duration."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        slowdown = 4.0
        result, target_fs = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=slowdown
        )

        expected_duration = (0.09 - 0.01) * slowdown
        actual_duration = len(result) / target_fs

        # Allow ±15% tolerance for STFT frame-boundary rounding
        assert expected_duration * 0.85 <= actual_duration <= expected_duration * 1.15, (
            f"Output duration {actual_duration} not within 15% of expected {expected_duration}"
        )

    def test_pitch_shift_preserves_duration(self):
        """Pitch shift alone (slowdown=1) should not change real-world
        duration — that's the whole point of using a duration-preserving
        pitch-shift primitive instead of a naive playback-rate trick."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result_no_shift, fs_no_shift = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=1
        )
        result_shift, fs_shift = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=1
        )

        dur_no_shift = len(result_no_shift) / fs_no_shift
        dur_shift = len(result_shift) / fs_shift
        assert abs(dur_no_shift - dur_shift) < 0.01

    def test_pitch_shift_changes_content(self):
        """Pitch-shifted output should differ from unshifted input."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        # No pitch shift, no stretch
        result_no_shift, _ = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=1
        )

        # With pitch shift
        result_shift, _ = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-35, slowdown=1
        )

        # Should not be identical (though may have some overlap)
        min_len = min(len(result_no_shift), len(result_shift))
        mse = np.mean((result_no_shift[:min_len] - result_shift[:min_len]) ** 2)

        assert mse > 0.01, "Pitch-shifted output too similar to unshifted"


class TestSonifySegmentRobustness:
    """Test robustness to edge cases and parameter variations."""

    def test_short_segment_raises(self):
        """A segment shorter than the minimum analysis window raises a
        clear error rather than an opaque one from librosa/scipy."""
        fs = 250000
        duration = 0.005  # far shorter than the ~2048-sample minimum
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        try:
            sonify_segment(audio, 0.0, duration, fs, semitones=-10, slowdown=2)
            raised = False
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for a too-short segment"

    def test_full_segment(self):
        """Segment spanning nearly the entire audio."""
        fs = 250000
        duration = 0.2
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result, _ = sonify_segment(
            audio, 0.0, 0.2, fs, semitones=-20, slowdown=2
        )

        assert np.all(np.isfinite(result))
        assert len(result) > 0

    def test_zero_semitones(self):
        """Pitch shift of 0 semitones should have minimal effect."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result, _ = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=0, slowdown=1
        )

        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_negative_pitch_shift(self):
        """Negative pitch shift (lower pitch)."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result, _ = sonify_segment(
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

        result, _ = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=12, slowdown=1
        )

        assert np.all(np.isfinite(result))
        assert len(result) > 0
        max_abs = np.max(np.abs(result))
        assert max_abs <= 1.0 + 1e-6

    def test_custom_target_fs(self):
        """Test with a non-default target sample rate."""
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)

        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        result, target_fs = sonify_segment(
            audio, 0.01, 0.09, fs, semitones=-20, slowdown=2, target_fs=48_000
        )

        assert target_fs == 48_000
        assert np.all(np.isfinite(result))
        max_abs = np.max(np.abs(result))
        assert max_abs <= 1.0 + 1e-6

    def test_invalid_slowdown_raises(self):
        fs = 250000
        duration = 0.1
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

        try:
            sonify_segment(audio, 0.01, 0.09, fs, semitones=0, slowdown=0)
            raised = False
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for non-positive slowdown"


class TestSonifySegmentInputValidation:
    """Test handling of various input types and ranges."""

    def test_different_sample_rates(self):
        """Test with different sample rates (must be > 240 kHz for 40-120 kHz filter)."""
        for fs in [250000, 256000, 300000]:
            duration = 0.1
            n_samples = int(fs * duration)

            t = np.arange(n_samples) / fs
            audio = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

            result, _ = sonify_segment(
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

            result, _ = sonify_segment(
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
        result, _ = sonify_segment(
            audio, 0.05, 0.3, fs, semitones=-20, slowdown=2
        )

        # Should gracefully handle by clamping to audio bounds
        assert np.all(np.isfinite(result))


class TestSonifyFullTrack:
    """sonify_full_track must preserve real-world duration (no time-stretch),
    unlike sonify_segment — it stays locked to a real-time clock (video)."""

    def test_duration_preserved(self):
        fs = 250000
        duration = 0.5
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        audio = (0.3 * np.sin(2 * np.pi * 60000 * t)).astype(np.float32)

        target_fs = 48000
        out, out_fs = sonify_full_track(audio, fs, semitones=-35, target_fs=target_fs)

        assert out_fs == target_fs
        out_duration = len(out) / out_fs
        in_duration = len(audio) / fs
        assert abs(out_duration - in_duration) < 0.005  # within 5ms over 0.5s

    def test_output_finite_and_normalized(self):
        fs = 250000
        n_samples = int(fs * 0.3)
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)

        out, out_fs = sonify_full_track(audio, fs, semitones=-35)

        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

"""
Unit tests for feature extraction.

Tests include:
  - Basic shape and structure validation
  - End-to-end test on real audio slice (sanity checks)
  - Synthetic pure tone (DomFreq, SpecCentroid sanity)
  - Delta features computation
  - Edge cases (no frames, empty signal, etc.)
"""

from __future__ import annotations

import numpy as np

from squeak_peek.features.extract import extract_frame_features


class TestExtractFrameFeaturesBasic:
    """Test basic shape, type, and structure of feature extraction."""

    def test_output_shape_matches_nframes_by_12(self):
        """Test that output has shape (n_frames, 12)."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        # Create simple test signal (DC removed, normalized)
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 80_000 * t).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # Check shape
        assert X.ndim == 2
        assert X.shape[1] == 12

        # Compute expected number of frames
        n_expected = len(np.arange(0, n_samples - frame_len + 1, hop_len))
        assert X.shape[0] == n_expected

        # mid_times should match n_frames
        assert len(mid_times) == X.shape[0]

    def test_output_dtype_is_float(self):
        """Test that output arrays are float type."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        n_samples = int(0.02 * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        assert X.dtype in [np.float32, np.float64]
        assert mid_times.dtype in [np.float32, np.float64]

    def test_empty_signal_returns_empty_frames(self):
        """Test that signal shorter than frame_len returns no frames."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192

        # Signal shorter than frame_len
        signal = np.zeros(frame_len - 1, dtype=np.float32)

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        assert X.shape == (0, 12)
        assert len(mid_times) == 0

    def test_no_nan_or_inf_in_output(self):
        """Test that output contains no NaN or Inf (sanitized)."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        assert np.all(np.isfinite(X)), "Found NaN or Inf in feature matrix X"
        assert np.all(np.isfinite(mid_times)), "Found NaN or Inf in mid_times"

    def test_mid_times_in_valid_range(self):
        """Test that mid_times are within signal duration."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.1
        n_samples = int(duration_sec * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # All mid_times should be within [0, duration_sec]
        assert np.all(mid_times >= 0)
        assert np.all(mid_times <= duration_sec)
        # mid_times should be monotonically increasing
        assert np.all(np.diff(mid_times) > 0)


class TestExtractFrameFeaturesSyntheticPureTone:
    """Test feature extraction on synthetic pure tones."""

    def test_pure_tone_dominant_frequency_near_true_frequency(self):
        """Test that DomFreq is near the true tone frequency for a pure tone."""
        fs = 250_000
        tone_freq = 80_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        # Pure tone
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # DomFreq is column 8
        dom_freq = X[:, 8]

        # For a pure tone, DomFreq should be close to tone_freq (within FFT resolution)
        fft_resolution = fs / nfft
        assert np.all(np.abs(dom_freq - tone_freq) < 5 * fft_resolution), (
            f"DomFreq not near true tone frequency. "
            f"Mean DomFreq: {np.mean(dom_freq)}, Expected: {tone_freq}"
        )

    def test_pure_tone_spectral_centroid_near_frequency(self):
        """Test that SpecCentroid is close to the pure tone frequency."""
        fs = 250_000
        tone_freq = 80_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # SpecCentroid is column 1
        spec_centroid = X[:, 1]

        # For a pure tone, SpecCentroid should be close to tone_freq
        fft_resolution = fs / nfft
        assert np.all(np.abs(spec_centroid - tone_freq) < 10 * fft_resolution), (
            f"SpecCentroid not near true tone frequency. "
            f"Mean: {np.mean(spec_centroid)}, Expected: {tone_freq}"
        )

    def test_pure_tone_low_entropy_and_flatness(self):
        """Test that a pure tone has low entropy and low flatness (tone-like)."""
        fs = 250_000
        tone_freq = 80_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # SpecEntropy (col 4) and SpecFlatness (col 3)
        spec_entropy = X[:, 4]
        spec_flatness = X[:, 3]

        # Pure tone should have low entropy (concentrated energy)
        assert np.mean(spec_entropy) < 0.5, (
            f"Pure tone should have low entropy. Got mean: {np.mean(spec_entropy)}"
        )

        # Pure tone should have low flatness (tone-like, not noise)
        assert np.mean(spec_flatness) < 1.0, (
            f"Pure tone should have flatness < 1.0. Got mean: {np.mean(spec_flatness)}"
        )

    def test_pure_tone_positive_band_power(self):
        """Test that BandPower is positive for a signal in the band."""
        fs = 250_000
        tone_freq = 80_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # BandPower is column 0
        band_power = X[:, 0]

        assert np.all(band_power > 0), "BandPower should be positive for in-band signal"


class TestExtractFrameFeaturesRealAudio:
    """Test feature extraction on real audio."""

    def test_real_audio_slice_produces_valid_features(self, example_audio):
        """Test that feature extraction runs on real audio and produces valid output."""
        samples, fs = example_audio

        # Use first 2 seconds for fast test
        duration_sec = 2.0
        n_samples = int(duration_sec * fs)
        signal = samples[:n_samples]

        # Parameters matching default PSD detector
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        fmin, fmax = 40_000, 120_000

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, fmin, fmax)

        # Should produce non-empty output
        assert X.shape[0] > 0, "No frames extracted from real audio"
        assert X.shape[1] == 12

        # All features should be finite
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(mid_times))

        # All times should be within bounds
        assert np.all(mid_times >= 0)
        assert np.all(mid_times <= duration_sec)

        # Sanity checks on feature ranges
        # BandPower (col 0) should be positive
        assert np.all(X[:, 0] >= 0)

        # SpecCentroid (col 1) should be within frequency band
        assert np.all(X[:, 1] >= fmin)
        assert np.all(X[:, 1] <= fmax)

        # SpecSpread (col 2) should be non-negative
        assert np.all(X[:, 2] >= 0)

        # SpecFlatness (col 3) should be non-negative
        assert np.all(X[:, 3] >= 0)

        # SpecEntropy (col 4) should be in [0, 1] (normalized)
        assert np.all((X[:, 4] >= 0) & (X[:, 4] <= 1.001))  # Small tolerance

        # ZCR (col 5) should be in [0, 1]
        assert np.all((X[:, 5] >= 0) & (X[:, 5] <= 1.001))

        # DomFreq (col 8) should be within frequency band
        assert np.all(X[:, 8] >= fmin)
        assert np.all(X[:, 8] <= fmax)


class TestExtractFrameFeaturesDeltaFeatures:
    """Test delta feature computation."""

    def test_delta_features_have_correct_shape(self):
        """Test that delta features have same number of frames as base features."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # Delta features should exist and have same length
        assert X.shape[1] == 12
        delta_power = X[:, 9]
        delta_centroid = X[:, 10]
        delta_entropy = X[:, 11]

        assert len(delta_power) == X.shape[0]
        assert len(delta_centroid) == X.shape[0]
        assert len(delta_entropy) == X.shape[0]

    def test_delta_features_first_frame_is_zero(self):
        """Test that first delta feature is 0 (no previous frame)."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.1
        n_samples = int(duration_sec * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # First frame deltas should be 0
        assert X[0, 9] == 0, "Delta_BandPower[0] should be 0"
        assert X[0, 10] == 0, "Delta_Centroid[0] should be 0"
        assert X[0, 11] == 0, "Delta_Entropy[0] should be 0"

    def test_delta_features_are_differences(self):
        """Test that delta features are correct differences between consecutive frames."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.1
        n_samples = int(duration_sec * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # Check that deltas are correct differences
        # Delta_BandPower should equal diff of BandPower
        expected_delta_power = np.concatenate([[0], np.diff(X[:, 0])])
        np.testing.assert_array_almost_equal(X[:, 9], expected_delta_power)

        # Delta_Centroid should equal diff of SpecCentroid
        expected_delta_centroid = np.concatenate([[0], np.diff(X[:, 1])])
        np.testing.assert_array_almost_equal(X[:, 10], expected_delta_centroid)

        # Delta_Entropy should equal diff of SpecEntropy
        expected_delta_entropy = np.concatenate([[0], np.diff(X[:, 4])])
        np.testing.assert_array_almost_equal(X[:, 11], expected_delta_entropy)


class TestExtractFrameFeaturesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_frame_signal(self):
        """Test signal that produces exactly one frame."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192

        # Signal exactly frame_len long
        n_samples = frame_len
        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        assert X.shape == (1, 12)
        assert len(mid_times) == 1
        assert np.all(np.isfinite(X))

    def test_two_frame_signal(self):
        """Test signal that produces exactly two frames."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192

        # Signal long enough for 2 frames: frame_len + hop_len
        n_samples = frame_len + hop_len
        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, mid_times = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        assert X.shape[0] == 2
        assert X.shape[1] == 12
        assert len(mid_times) == 2

    def test_different_frequency_bands(self):
        """Test with different frequency band settings."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.05
        n_samples = int(duration_sec * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        # Test with different band
        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 60_000, 100_000)

        # Should still produce valid output
        assert X.shape[1] == 12
        assert np.all(np.isfinite(X))

        # SpecCentroid and DomFreq should be within the band
        assert np.all((X[:, 1] >= 60_000) & (X[:, 1] <= 100_000))
        assert np.all((X[:, 8] >= 60_000) & (X[:, 8] <= 100_000))


class TestExtractFrameFeaturesColumnOrder:
    """Test that column order exactly matches MATLAB specification."""

    def test_column_count_is_12(self):
        """Test that exactly 12 columns are produced."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        n_samples = int(0.05 * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        assert X.shape[1] == 12

    def test_columns_0_8_are_instantaneous_features(self):
        """Test that columns 0-8 are instantaneous (not delta) features."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        duration_sec = 0.1
        n_samples = int(duration_sec * fs)

        # Use a chirp signal to ensure feature variations
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * (70_000 + 20_000 * t / duration_sec) * t).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # Instantaneous features should vary frame-to-frame (at least some)
        # For a chirp, BandPower, SpecCentroid, etc. should change
        # Check that columns 0-8 are not all constant
        assert np.std(X[:, 0]) > 0, "BandPower (col 0) should vary"
        assert np.std(X[:, 1]) > 0, "SpecCentroid (col 1) should vary"
        assert np.std(X[:, 8]) > 0, "DomFreq (col 8) should vary"

    def test_columns_9_11_are_delta_features(self):
        """Test that columns 9-11 are delta features (first frame = 0)."""
        fs = 250_000
        frame_len = 4096
        hop_len = 2048
        nfft = 8192
        n_samples = int(0.1 * fs)

        signal = np.sin(2 * np.pi * 80_000 * np.arange(n_samples) / fs).astype(np.float32)
        signal = signal / np.max(np.abs(signal))

        X, _ = extract_frame_features(signal, fs, frame_len, hop_len, nfft, 40_000, 120_000)

        # First row of delta features should be 0
        assert X[0, 9] == 0  # Delta_BandPower
        assert X[0, 10] == 0  # Delta_Centroid
        assert X[0, 11] == 0  # Delta_Entropy

        # Subsequent deltas can be non-zero
        if X.shape[0] > 1:
            # At least check they exist and are finite
            assert np.all(np.isfinite(X[1:, 9]))
            assert np.all(np.isfinite(X[1:, 10]))
            assert np.all(np.isfinite(X[1:, 11]))

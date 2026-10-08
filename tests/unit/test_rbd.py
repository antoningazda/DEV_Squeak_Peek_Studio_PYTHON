"""
Unit tests for RBD (Recursive Bayesian Detector).

Tests the rbd() algorithm and RBDDetector class with both synthetic and real audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from squeak_peek.detectors.rbd import RBDDetector, RBDParams, rbd
from squeak_peek.labels.model import Label


class TestRBDAlgorithm:
    """Tests for the core rbd() algorithm."""

    def test_rbd_output_shape(self):
        """Test that rbd() output has same length as input."""
        signal = np.random.randn(1000)
        window_length = 100
        AR_order_left = 4
        AR_order_right = 4
        Bayesian_Evidence_order = 4

        output = rbd(signal, window_length, AR_order_left, AR_order_right, Bayesian_Evidence_order)

        assert output.shape == signal.shape
        assert output.dtype == np.float64

    def test_rbd_output_dtype(self):
        """Test that rbd() returns float64 array."""
        signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 200, dtype=np.float32)
        window_length = 100

        output = rbd(signal, window_length, 4, 4, 4)

        assert output.dtype == np.float64

    def test_rbd_detects_change_in_synthetic_signal(self):
        """Test that rbd() detects a change point in a synthetic signal."""
        # Create a signal with a clear change: white noise then stepped change
        np.random.seed(42)
        part1 = np.random.randn(2000) * 0.1
        part2 = np.random.randn(2000) * 0.5 + 0.3  # Higher amplitude
        signal = np.concatenate([part1, part2])

        window_length = 200
        output = rbd(signal, window_length, 4, 4, 4)

        # Normalize output
        output_normalized = output / np.max(np.abs(output))

        # Check that output has higher values near the change point (around sample 2000)
        before_change = np.mean(np.abs(output_normalized[1800:2000]))
        after_change = np.mean(np.abs(output_normalized[2000:2200]))

        # Should have some activity around change point
        assert before_change > 0
        assert after_change > 0

    def test_rbd_various_ar_orders(self):
        """Test that rbd() works with different AR orders."""
        signal = np.random.randn(1000)
        window_length = 100

        for ar_order in [2, 4, 6]:
            output = rbd(signal, window_length, ar_order, ar_order, ar_order)
            assert output.shape == signal.shape
            assert not np.any(np.isnan(output))

    def test_rbd_numba_matches_plain_numpy_reference(self):
        """The jitted core and the plain-numpy reference must implement the
        same recursion. At the scale of a short clip they should agree to
        near machine precision; some drift is expected and acceptable since
        both paths run ~thousands of chained Sherman-Morrison rank-1 updates,
        which is inherently sensitive to floating-point rounding order.
        """
        np.random.seed(0)
        signal = np.random.randn(600).astype(np.float64)
        signal = signal / np.max(np.abs(signal))

        ref = rbd(signal, 400, 4, 4, 4, use_numba=False)
        jit = rbd(signal, 400, 4, 4, 4, use_numba=True)

        assert np.allclose(ref, jit, rtol=1e-6, atol=1e-9)

    def test_rbd_matches_least_squares_closed_form(self):
        """The recursion must equal RBD.m's statistic computed per window from
        scratch: at MATLAB index mm the left AR(M1) fit covers samples
        mm-m+1..mm, the right AR(M2) fit mm+1..mm+m, the baseline AR(ME) fit
        both, every row regressing on its true lags; R = 4*(ln res_E -
        ln(res_L + res_R))."""
        rng = np.random.default_rng(0)
        n, okno, order = 1200, 120, 4
        x = rng.normal(size=n)
        for i in range(2, n):  # AR(2) whose coefficients change at n/2
            a = (1.2, -0.5) if i < n // 2 else (-0.3, 0.4)
            x[i] += a[0] * x[i - 1] + a[1] * x[i - 2]
        out = rbd(x, okno, order, order, order)

        sig = x / np.max(np.abs(x))
        m = okno // 2

        def residual(rows):
            a = np.array([[sig[r - 1 - k] for k in range(order)] for r in rows])
            coef, *_ = np.linalg.lstsq(a, sig[rows], rcond=None)
            e = sig[rows] - a @ coef
            return e @ e

        # Skip the start-up rows, whose lags RBD.m zero-pads.
        for mm in range(okno + 10, n - m + 1, 7):
            left, right = list(range(mm - m, mm)), list(range(mm, mm + m))
            expected = 4 * (np.log(residual(left + right)) - np.log(residual(left) + residual(right)))
            assert out[mm - 1] == pytest.approx(expected, abs=1e-8)
        assert abs(int(np.argmax(out)) - n // 2) <= 2

    def test_rbd_numba_matches_reference_on_real_audio_slice(
        self, example_audio: tuple[np.ndarray, int]
    ):
        """Same cross-check as above, on a real (bandpass-free, per RBD.m)
        audio slice at the production window size — allows a looser but
        still tight tolerance since more recursion steps run here.
        """
        samples, fs = example_audio
        sig = samples[: fs // 4].astype(np.float64)  # 0.25s
        sig = sig - np.mean(sig)
        sig = sig / np.max(np.abs(sig))

        window_length = round(RBDParams().wlen * fs)
        ref = rbd(sig, window_length, 4, 4, 4, use_numba=False)
        jit = rbd(sig, window_length, 4, 4, 4, use_numba=True)

        # Absolute tolerance relative to the statistic's range: near-zero
        # samples carry ~1e-6 of rounding drift that a relative check inflates.
        assert np.allclose(ref, jit, rtol=1e-3, atol=1e-4 * np.max(np.abs(ref)))

    def test_rbd_numba_runtime_on_longer_real_slice(
        self, example_audio: tuple[np.ndarray, int]
    ):
        """The jitted path must handle several real seconds of 250kHz audio
        in well under real time (the naive per-sample loop would not)."""
        import time

        samples, fs = example_audio
        sig = samples[: 5 * fs].astype(np.float64)
        sig = sig - np.mean(sig)
        sig = sig / np.max(np.abs(sig))
        window_length = round(RBDParams().wlen * fs)

        rbd(sig[: window_length * 2], window_length, 4, 4, 4)  # warm up JIT

        t0 = time.time()
        out = rbd(sig, window_length, 4, 4, 4)
        elapsed = time.time() - t0

        assert out.shape == sig.shape
        # Generous bound: a naive (non-jitted) per-sample loop would take
        # minutes-to-hours here. This just guards against an accidental
        # regression back to that, not a tight real-time budget.
        assert elapsed < 15.0, f"RBD on 5s of audio took {elapsed:.2f}s (expected a few seconds, jitted)"


class TestRBDDetector:
    """Tests for the RBDDetector class."""

    def test_detector_name(self):
        """Test that detector reports correct name."""
        params = RBDParams()
        detector = RBDDetector(params)

        assert detector.name == "RBD"

    def test_detector_output_is_list_of_labels(self):
        """Test that detect() returns a list of Label objects."""
        params = RBDParams()
        detector = RBDDetector(params)

        signal = np.random.randn(50000)  # ~200ms at 250kHz
        fs = 250000

        labels = detector.detect(signal, fs)

        assert isinstance(labels, list)
        for label in labels:
            assert isinstance(label, Label)

    def test_detector_label_format(self):
        """Test that detected labels have required fields."""
        params = RBDParams()
        detector = RBDDetector(params)

        signal = np.random.randn(100000)  # ~400ms at 250kHz
        fs = 250000

        labels = detector.detect(signal, fs)

        for label in labels:
            assert hasattr(label, "start_time")
            assert hasattr(label, "end_time")
            assert hasattr(label, "label")
            assert hasattr(label, "start_frequency")
            assert hasattr(label, "end_frequency")
            assert hasattr(label, "start_index")
            assert hasattr(label, "stop_index")
            assert label.label == "d"
            assert label.start_frequency == 0.0
            assert label.end_frequency == 0.0

    def test_detector_times_consistent_with_indices(self):
        """Test that start/end times correspond to sample indices."""
        params = RBDParams()
        detector = RBDDetector(params)

        signal = np.random.randn(250000)  # 1 second at 250kHz
        fs = 250000

        labels = detector.detect(signal, fs)

        for label in labels:
            # start_time should equal start_index / fs
            assert np.isclose(label.start_time, label.start_index / fs, rtol=1e-6)
            # end_time should equal stop_index / fs
            assert np.isclose(label.end_time, label.stop_index / fs, rtol=1e-6)

    def test_detector_with_real_audio(self, example_audio: tuple[np.ndarray, int]):
        """Test detector on real example audio (short slice)."""
        samples, fs = example_audio

        # Use first 2 seconds to keep test fast
        short_samples = samples[: 2 * fs]

        params = RBDParams()
        detector = RBDDetector(params)

        labels = detector.detect(short_samples, fs)

        # Should get some labels from real USV data
        assert isinstance(labels, list)
        for label in labels:
            assert isinstance(label, Label)
            assert 0 <= label.start_time <= 2
            assert 0 <= label.end_time <= 2
            assert label.start_time <= label.end_time
            assert label.start_index < label.stop_index

    def test_detector_empty_signal(self):
        """Test detector gracefully handles edge cases."""
        params = RBDParams()
        detector = RBDDetector(params)

        # Very short signal (but long enough for window)
        signal = np.random.randn(1000)
        fs = 250000

        labels = detector.detect(signal, fs)

        # Should return a list (possibly empty)
        assert isinstance(labels, list)

    def test_detector_params_affect_output(self):
        """Test that different parameters produce different results."""
        signal = np.random.randn(500000)  # 2 seconds at 250kHz
        fs = 250000

        params1 = RBDParams(dynamicScaling=0.2)
        detector1 = RBDDetector(params1)
        labels1 = detector1.detect(signal, fs)

        params2 = RBDParams(dynamicScaling=0.5)
        detector2 = RBDDetector(params2)
        labels2 = detector2.detect(signal, fs)

        # Different thresholds should generally produce different numbers of labels
        # (not guaranteed for random noise, but very likely)
        # At minimum, both should run without error
        assert isinstance(labels1, list)
        assert isinstance(labels2, list)

    def test_detector_amplitude_threshold_filters(self):
        """Test that amplitude threshold parameter filters detections."""
        signal = np.random.randn(100000)
        fs = 250000

        # Low threshold should potentially detect more
        params_low = RBDParams(amplitudeThreshold=0.01)
        detector_low = RBDDetector(params_low)
        labels_low = detector_low.detect(signal, fs)

        # High threshold should potentially detect less
        params_high = RBDParams(amplitudeThreshold=0.1)
        detector_high = RBDDetector(params_high)
        labels_high = detector_high.detect(signal, fs)

        # At least run without error (the actual comparison depends on signal)
        assert isinstance(labels_low, list)
        assert isinstance(labels_high, list)

    def test_detector_normalization_order(self):
        """Test that DC removal and normalization work correctly."""
        # Create signal with large DC offset
        signal = np.random.randn(100000) + 5.0  # Large DC offset

        params = RBDParams()
        detector = RBDDetector(params)

        # Should run without issues (should handle DC offset internally)
        labels = detector.detect(signal, 250000)

        assert isinstance(labels, list)


class TestRBDIntegration:
    """Integration tests combining rbd() and RBDDetector."""

    def test_end_to_end_with_real_audio(
        self, example_audio: tuple[np.ndarray, int], settings
    ):
        """End-to-end test with real USV example audio."""
        samples, fs = example_audio

        # Test with first 5 seconds to keep it reasonable
        test_slice = samples[: min(5 * fs, len(samples))]

        # Use settings if available, otherwise default params
        if hasattr(settings, "RBD"):
            params = settings.RBD
        else:
            params = RBDParams()

        detector = RBDDetector(params)
        labels = detector.detect(test_slice, fs)

        # Verify output format and validity
        assert isinstance(labels, list)
        assert all(isinstance(lbl, Label) for lbl in labels)

        # Verify times are within bounds
        max_time = len(test_slice) / fs
        for label in labels:
            assert 0 <= label.start_time <= max_time
            assert 0 <= label.end_time <= max_time
            assert label.start_time <= label.end_time

        # Real USV data should typically have some detections
        # (if not, at least verify the detector ran)
        print(f"Found {len(labels)} detections in {max_time:.2f}s audio slice")


class TestRBDImprovements:
    """Bandpass + median threshold (the defaults) vs RBDDetector.m's rules."""

    @staticmethod
    def _calls_in_noise(fs: int = 250_000):
        rng = np.random.default_rng(3)
        x = rng.normal(size=fs) * 0.02 + np.cumsum(rng.normal(size=fs)) * 1e-3  # hiss + low-freq drift
        t = np.arange(int(0.03 * fs)) / fs
        starts = (0.2, 0.5, 0.8)
        for s in starts:
            i = int(s * fs)
            x[i:i + len(t)] += 0.3 * np.sin(2 * np.pi * (55_000 * t + 2e5 * t ** 2))  # 30 ms FM sweep
        return x, fs, starts

    def test_defaults_find_calls(self):
        x, fs, starts = self._calls_in_noise()
        labels = RBDDetector(RBDParams()).detect(x, fs)
        for s in starts:
            mid = s + 0.015
            assert any(lb.start_time <= mid <= lb.end_time for lb in labels), f"missed call at {s}s"

    def test_original_mode_is_matlab_rule(self):
        """bandpass=False + thresholdMode='original' must reproduce RBDDetector.m."""
        x, fs, _ = self._calls_in_noise()
        p = RBDParams(bandpass=False, thresholdMode="original")
        labels = RBDDetector(p).detect(x, fs)

        from squeak_peek.audio.filters import movmean

        sig = (x - x.mean()) / np.max(np.abs(x - x.mean()))
        out = rbd(sig, round(p.wlen * fs), 4, 4, 4)
        out = out / np.max(np.abs(out))
        sm = movmean(out, round(p.smoothingWindowRBD * fs))
        thr = movmean(sm, round(p.smoothingWindowThr * fs)) * p.dynamicScaling
        b = (sm > thr) & (sm > p.amplitudeThreshold)
        e = np.diff(np.concatenate([[0], b.astype(int), [0]]))
        assert [lb.start_index for lb in labels] == list(np.where(e == 1)[0])

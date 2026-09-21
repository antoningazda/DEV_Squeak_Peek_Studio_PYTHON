"""
Power Spectral Density (PSD) detector for ultrasonic vocalization events.

Ported from MATLAB's PSDDetector.m, which applies bandpass filtering,
spectrogram-based thresholding, and noise-adaptive event detection to
identify USV segments in high-frequency audio.

The algorithm:
  1. Normalizes the signal (DC-removal, amplitude normalization)
  2. Applies a zero-phase bandpass filter (order 12 IIR with filtfilt)
  3. Computes a Hamming-window STFT
  4. Calculates power envelope within the frequency band
  5. Estimates noise floor via moving minimum
  6. Applies adaptive threshold based on local SNR statistics
  7. Detects events via binary thresholding and edge detection
  8. Filters by minimum effective power
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from squeak_peek.audio.filters import bandpass_filter_filtfilt, compute_stft
from squeak_peek.detectors.base import AbstractDetector

if TYPE_CHECKING:
    from squeak_peek.config import PSDParams
    from squeak_peek.labels.model import Label


class PSDDetector(AbstractDetector):
    """
    Detects USV events using Power Spectral Density analysis.

    This detector applies a series of signal-processing steps to extract
    events from the power envelope of a bandpass-filtered, high-frequency signal.
    It is highly tunable via PSDParams.
    """

    def __init__(self, params: PSDParams):
        """
        Initialize the PSD detector with parameters.

        Parameters
        ----------
        params : PSDParams
            Configuration object containing all detector hyperparameters.
        """
        self.params = params

    def detect(
        self,
        signal: np.ndarray,
        fs: int,
    ) -> list[Label]:
        """
        Detect USV events in the input signal.

        Performs DC-removal, normalization, bandpass filtering, spectrogram
        analysis, and adaptive thresholding to identify USV segments.

        Parameters
        ----------
        signal : 1-D float32 array
            Raw or preprocessed audio signal
        fs     : int
            Sample rate in Hz

        Returns
        -------
        list[Label]
            Detected USV events as Label objects, each with start_time, end_time,
            start_index, and stop_index (frame indices in the STFT).
        """
        from squeak_peek.labels.model import Label

        p = self.params

        # ── Normalization (DC-removal and amplitude normalization) ──────────
        x = signal.astype(np.float64)  # Ensure float for numerical stability
        x = x - np.mean(x)
        max_val = np.max(np.abs(x))
        if max_val > 0:
            x = x / max_val
        # else: if signal is all zeros, leave it as is

        # ── ROI (Region of Interest) slicing ──────────────────────────────
        if p.runWholeSignal:
            x_roi = x
            roi_start_time = 0.0
        else:
            idx1 = int(np.round(p.ROIstart * fs))
            idx2 = int(np.round((p.ROIstart + p.ROIlength) * fs))
            x_roi = x[idx1:idx2]
            roi_start_time = p.ROIstart

        # ── Bandpass filter (zero-phase, order 12) ───────────────────────
        x_roi = bandpass_filter_filtfilt(
            x_roi, fs, p.fcutMin, p.fcutMax, order=12
        )

        # ── STFT with Hamming window ─────────────────────────────────────
        frequencies, times, Sxx = compute_stft(
            x_roi,
            fs,
            p.segmentLength,
            p.overlapFactor,
            window="hamming",  # MATLAB uses hamming() for PSD
        )

        # ── Power envelope within the frequency band ─────────────────────
        freq_mask = (frequencies >= p.fcutMin) & (frequencies <= p.fcutMax)
        # Sum squared magnitudes across frequency (power), then normalize
        power_envelope = np.sum(np.power(10.0, Sxx[freq_mask, :] / 10.0), axis=0)
        power_envelope = power_envelope / np.max(power_envelope)

        # ── Smoothing (moving average) ───────────────────────────────────
        power_envelope = self._movmean(power_envelope, p.maWindow)

        # ── Noise floor estimation (moving minimum) ──────────────────────
        noise_floor = self._movmin(power_envelope, p.noiseWindow)

        # ── Effective envelope ───────────────────────────────────────────
        effective_envelope = np.maximum(power_envelope - noise_floor, 0)

        # ── Local SNR calculation ────────────────────────────────────────
        local_snr = np.minimum(
            effective_envelope / (noise_floor + np.finfo(float).eps), 10.0
        )

        # ── Local statistics for adaptive threshold ──────────────────────
        local_mean = self._movmean(effective_envelope, p.localWindow)
        local_std = self._movstd(effective_envelope, p.localWindow)

        # ── Adaptive threshold ───────────────────────────────────────────
        threshold = (local_mean + p.k * local_std) / (1.0 + p.w * local_snr)

        # ── Binary thresholding ──────────────────────────────────────────
        binary = effective_envelope > threshold

        # ── Edge detection (find event boundaries) ──────────────────────
        edges = np.diff(np.concatenate(([0], binary.astype(int), [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1

        # ── Create labels for each event ─────────────────────────────────
        labels: list[Label] = []
        for start_idx, end_idx in zip(starts, ends):
            # Skip zero-duration events (start_idx == end_idx)
            if start_idx >= end_idx:
                continue

            # Check minimum effective power criterion
            mean_power = np.mean(effective_envelope[start_idx : end_idx + 1])
            if mean_power >= p.minEffectivePower:
                label = Label(
                    start_time=times[start_idx] + roi_start_time,
                    end_time=times[end_idx] + roi_start_time,
                    label="d",  # PSD detector uses "d" as default label
                    start_frequency=0.0,
                    end_frequency=0.0,
                    start_index=int(start_idx),
                    stop_index=int(end_idx),
                )
                labels.append(label)

        return labels

    @property
    def name(self) -> str:
        """Return the detector name."""
        return "PSD"

    # ── Utility functions matching MATLAB behavior ─────────────────────

    @staticmethod
    def _movwin(window: int, n: int) -> tuple[int, int]:
        """
        MATLAB moving-window centering: for a k-element window, the window
        spans k//2 elements before the current one, the current one, and
        k-1-(k//2) after. For odd k this is symmetric; for even k there is
        one more element before than after (matches MATLAB's movmean/
        movstd/movmin/smoothdata('movmean',...) convention exactly, which
        differs from a naive symmetric-clip approach for even window sizes).
        """
        before = window // 2
        after = window - 1 - before
        return before, after

    @classmethod
    def _movmean(cls, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving mean (matching MATLAB's movmean)."""
        if window <= 1:
            return data.copy()

        before, after = cls._movwin(window, len(data))
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - before)
            end = min(len(data), i + after + 1)
            result[i] = np.mean(data[start:end])

        return result

    @classmethod
    def _movstd(cls, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving standard deviation (matching MATLAB's movstd)."""
        if window <= 1:
            return np.zeros_like(data)

        before, after = cls._movwin(window, len(data))
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - before)
            end = min(len(data), i + after + 1)
            result[i] = np.std(data[start:end], ddof=1) if end - start > 1 else 0.0

        return result

    @classmethod
    def _movmin(cls, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving minimum (matching MATLAB's movmin)."""
        if window <= 1:
            return data.copy()

        before, after = cls._movwin(window, len(data))
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - before)
            end = min(len(data), i + after + 1)
            result[i] = np.min(data[start:end])

        return result

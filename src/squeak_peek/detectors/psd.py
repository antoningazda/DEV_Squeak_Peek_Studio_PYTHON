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
from pydantic import BaseModel, Field

from squeak_peek.audio.filters import bandpass_filter_filtfilt
from squeak_peek.detectors.base import AbstractDetector

if TYPE_CHECKING:
    from squeak_peek.labels.model import Label


class PSDParams(BaseModel):
    """Parameters for the Power Spectral Density detector."""

    fcutMin: float = Field(
        40_000, ge=0, le=250_000,
        description="Lower bound of the frequency band the PSD detector analyses.",
        json_schema_extra={"unit": "Hz", "group": "freq_band"},
    )
    fcutMax: float = Field(
        120_000, ge=0, le=250_000,
        description="Upper bound of the frequency band the PSD detector analyses.",
        json_schema_extra={
            "unit": "Hz", "group": "freq_band",
            "caption": "Frequencies outside [fcutMin, fcutMax] are ignored when computing signal power.",
        },
    )
    denoise: bool = Field(
        True,
        description="Suppress stationary background noise before this detector runs.",
        json_schema_extra={
            "caption": "Recommended: in-band noise otherwise inflates the power envelope and costs most of PSD's precision. Noise-suppression settings: Settings → Pre-processing. "
                       "Export, review and classification always keep the original audio.",
        },
    )
    ROIstart: float = Field(
        60, ge=0, le=100_000,
        description="Start of the Region of Interest (used only when runWholeSignal is off).",
        json_schema_extra={"unit": "s"},
    )
    ROIlength: float = Field(
        10, ge=0, le=100_000,
        description="Length of the Region of Interest (used only when runWholeSignal is off).",
        json_schema_extra={"unit": "s"},
    )
    runWholeSignal: bool = Field(
        True, description="Ignore the ROI and process the full signal.",
    )
    segmentLength: int = Field(
        8192, ge=256, le=65_536,
        description="STFT window length used to compute the power spectrum.",
        json_schema_extra={
            "unit": "samples",
            "caption": "Larger windows give finer frequency resolution but coarser time resolution.",
        },
    )
    overlapFactor: float = Field(
        0.59, ge=0.0, le=0.99,
        description="Fraction of each analysis window that overlaps with the next.",
        json_schema_extra={
            "decimals": 3,
            "caption": "Higher overlap gives a smoother power estimate but increases processing time.",
        },
    )
    maWindow: int = Field(
        3, ge=1, le=1_000_000,
        description="Moving-average window used to smooth the power envelope (frames).",
        json_schema_extra={"unit": "frames"},
    )
    noiseWindow: int = Field(
        240, ge=1, le=1_000_000,
        description="Moving-minimum window used to estimate the noise floor (frames).",
        json_schema_extra={"unit": "frames"},
    )
    localWindow: int = Field(
        194, ge=1, le=1_000_000,
        description="Window used to compute local mean/std for the adaptive threshold (frames).",
        json_schema_extra={"unit": "frames"},
    )
    k: float = Field(
        0.023, ge=0.0, le=10.0,
        description="Scales the local-statistics term of the adaptive threshold.",
        json_schema_extra={
            "decimals": 4,
            "caption": "Higher k -> stricter threshold -> fewer, more confident detections.",
        },
    )
    w: float = Field(
        0.994, ge=0.0, le=20.0,
        description="Weight given to the local SNR term when computing the adaptive threshold.",
        json_schema_extra={"decimals": 4},
    )
    minEffectivePower: float = Field(
        8.5e-5, ge=0.0, le=1.0,
        description="Minimum mean effective power required to accept a candidate event.",
        json_schema_extra={"decimals": 8},
    )

    model_config = {"populate_by_name": True}


class PSDDetector(AbstractDetector):
    """
    Detects USV events using Power Spectral Density analysis.

    This detector applies a series of signal-processing steps to extract
    events from the power envelope of a bandpass-filtered, high-frequency signal.
    It is highly tunable via PSDParams.
    """

    id = "PSD"
    display_name = "PSD"
    description = (
        "Power Spectral Density — thresholds band power against an adaptive "
        "noise floor. Fast and robust on clear, high-SNR recordings."
    )
    Params = PSDParams

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

        # ── STFT with Hamming window (MATLAB: spectrogram(x, hamming(L),
        #    round(L*overlap), L, fs)) ─────────────────────────────────────
        from scipy.signal import spectrogram as _sp_spectrogram

        frequencies, times, Sxx = _sp_spectrogram(
            x_roi,
            fs=fs,
            window="hamming",
            nperseg=p.segmentLength,
            noverlap=int(round(p.segmentLength * p.overlapFactor)),
            nfft=p.segmentLength,
            scaling="spectrum",
            mode="psd",
            detrend=False,
        )

        # ── Power envelope within the frequency band ─────────────────────
        # MATLAB: sum(abs(S(freqMask,:)).^2, 1), normalised to its maximum.
        # Summed directly from the power spectrum — no round trip through dB,
        # whose +eps floor would bias the envelope (and the noise floor, and
        # so the SNR term of the threshold) on quiet recordings.
        freq_mask = (frequencies >= p.fcutMin) & (frequencies <= p.fcutMax)
        power_envelope = np.sum(Sxx[freq_mask, :], axis=0)
        max_power = np.max(power_envelope) if power_envelope.size else 0.0
        if max_power <= 0:
            return []
        power_envelope = power_envelope / max_power

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
            # Single-frame events (start_idx == end_idx) are kept, as in
            # PSDDetector.m; Remove Short Labels drops them if enabled.

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

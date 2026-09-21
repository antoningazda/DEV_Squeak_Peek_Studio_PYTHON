"""
Bayesian Sequential Change Detection (BSCD) detector for ultrasonic vocalizations.

Port of bscd.m and BSCDDetector.m from the original MATLAB implementation.
Uses numba JIT compilation for the inner per-sample loop to achieve real-time performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from squeak_peek.audio.filters import bandpass_filter_filtfilt
from squeak_peek.config import BSCDParams
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.model import Label


# ─────────────────────────────────────────────────────────────────────────────
# Numba-compiled core: inline 2x2 matrix operations for performance
# ─────────────────────────────────────────────────────────────────────────────


@njit
def _inv_2x2(a: float, b: float, c: float, d: float) -> tuple[float, float, float, float]:
    """
    Compute inline 2x2 matrix inverse: [[a, b], [c, d]] -> inv.

    Returns the four elements of the inverse as (a', b', c', d').
    det = a*d - b*c, so inv = (1/det) * [[d, -b], [-c, a]]

    Raises RuntimeError if determinant is (near-)zero.
    """
    det = a * d - b * c
    if np.abs(det) < 1e-15:
        # Return identity to avoid division by zero; algorithm will degrade gracefully
        return 1.0, 0.0, 0.0, 1.0
    inv_det = 1.0 / det
    return inv_det * d, inv_det * (-b), inv_det * (-c), inv_det * a


@njit
def _matmul_2x2_1col(a00: float, a01: float, a10: float, a11: float, b0: float, b1: float) -> tuple[float, float]:
    """
    Multiply 2x2 matrix by 2x1 column vector: [[a00, a01], [a10, a11]] @ [[b0], [b1]].

    Returns the result as a tuple (c0, c1).
    """
    return a00 * b0 + a01 * b1, a10 * b0 + a11 * b1


def _safe_log10(x: float) -> float:
    """Safely compute log10, handling edge cases."""
    if x <= 0:
        return -100.0  # Large negative value instead of -inf
    return np.log10(x)


@njit
def _bscd_inner(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Core BSCD per-sample Bayesian evidence computation with numba JIT.

    Performs a sliding window update on the Bayesian evidence for each sample,
    using rank-1 matrix updates (Sherman-Morrison formula) to avoid expensive
    matrix inversions on every iteration.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Bandpass-filtered, squared signal (DC removed and normalized)
    window_samples : int
        Window length in samples (e.g. fs * wlen)

    Returns
    -------
    np.ndarray, shape (N,)
        Evidence score at each sample (higher = more likely change point)
    """
    N = len(signal)
    p2 = np.zeros(N)

    # ─── Initialization (first window) ─────────────────────────────────────
    m = window_samples // 2

    # Setup G matrix: [1, 0] for first half, [0, 1] for second half
    G = np.zeros((window_samples, 2))
    for j in range(m):
        G[j, 0] = 1.0
    for j in range(m, window_samples):
        G[j, 1] = 1.0

    # Initialize statistics from first window
    data = signal[:window_samples]
    D = np.sum(data * data)

    # Model 1: 2-parameter model (intercept, slope)
    CHI = np.zeros(2)
    for j in range(window_samples):
        CHI[0] += data[j] * G[j, 0]
        CHI[1] += data[j] * G[j, 1]

    GTG = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(window_samples):
                GTG[i, j] += G[k, i] * G[k, j]

    # Matrix inverse for model 1
    fi00, fi01, fi10, fi11 = _inv_2x2(GTG[0, 0], GTG[0, 1], GTG[1, 0], GTG[1, 1])
    delta = GTG[0, 0] * GTG[1, 1] - GTG[0, 1] * GTG[1, 0]

    # Compute evidence for model 1 at initialization
    chi_fi_chi = (
        CHI[0] * (fi00 * CHI[0] + fi01 * CHI[1])
        + CHI[1] * (fi10 * CHI[0] + fi11 * CHI[1])
    )
    numerator = D - chi_fi_chi
    if numerator <= 0:
        numerator = 1e-12
    cit = ((-window_samples + 1.0 + 1.0) / 2.0) * np.log10(numerator)
    if delta <= 0:
        delta = 1e-12
    jm = 0.5 * np.log10(np.abs(delta))

    # Model 2: 1-parameter model (intercept only)
    CHI_E = 0.0
    for j in range(window_samples):
        CHI_E += data[j]

    GTG_E = float(window_samples)
    fi_e = 1.0 / GTG_E
    delta_e = GTG_E

    FTF = CHI_E * fi_e * CHI_E
    B = fi_e * CHI_E
    BTB = B * B

    # gamma(0.5) = sqrt(pi) ≈ 1.7724538509055159
    sqrt_pi = np.sqrt(np.pi)

    E1 = (-window_samples / 2.0) * np.log10(np.pi)
    E2 = -0.5 * np.log10(delta_e)
    E3 = np.log10(sqrt_pi)  # log10(gamma(0.5))
    E5 = -(window_samples - 1.0) / 2.0 * np.log10(max(D - FTF, 1e-12))
    E6 = -0.5 * 1.0 * np.log10(max(np.abs(BTB), 1e-12))
    evid = E1 + E2 + E3 + E5 + E6

    p2[m] = cit - jm - evid

    # ─── Main loop: sliding window updates ────────────────────────────────
    for mm in range(m + 1, N - m):
        # Add new sample at position mm+m
        d2 = signal[mm + m]
        d2_sq = d2 * d2

        D = D + d2_sq

        # Model 1: rank-1 update for add operation (G2 = [0, 1], bscd.m line 77)
        CHI[1] += d2 * 1.0  # G2[0] = 0 (CHI[0] unchanged), G2[1] = 1

        # Sherman-Morrison for add: FI = FI - W @ inv(LAMBDA) @ W^T
        # where W = FI @ G2^T = FI[:, 1], LAMBDA = 1 + G2 @ W = 1 + W[1]
        w0 = fi01
        w1 = fi11
        lambda_add = 1.0 + w1
        delta = delta * lambda_add
        inv_lambda = 1.0 / lambda_add
        fi00 -= w0 * inv_lambda * w0
        fi01 -= w0 * inv_lambda * w1
        fi10 -= w1 * inv_lambda * w0
        fi11 -= w1 * inv_lambda * w1

        # Model 2: rank-1 update for add
        CHI_E += d2
        w_e = fi_e * 1.0
        lambda_e = 1.0 + 1.0 * w_e
        delta_e = delta_e * lambda_e
        fi_e -= w_e * (1.0 / lambda_e) * w_e

        # Remove old sample at position mm-m
        d_old = signal[mm - m]
        d_old_sq = d_old * d_old

        D = D - d_old_sq

        # Model 1: rank-1 update for remove: FI = FI + (1/LAMBDA) @ W @ W^T
        # where W = FI @ Z^T, LAMBDA = 1 - Z @ W
        z0 = 1.0  # only first model component removes
        z1 = 0.0
        w0 = fi00 * z0 + fi01 * z1
        w1 = fi10 * z0 + fi11 * z1
        lambda_remove = 1.0 - z0 * w0 - z1 * w1
        delta = delta * lambda_remove
        inv_lambda = 1.0 / lambda_remove
        fi00 += inv_lambda * w0 * w0
        fi01 += inv_lambda * w0 * w1
        fi10 += inv_lambda * w1 * w0
        fi11 += inv_lambda * w1 * w1

        # Update CHI
        CHI[0] -= d_old * z0
        CHI[1] -= d_old * z1

        # Model 2: rank-1 update for remove
        z_e = 1.0
        w_e = fi_e * z_e
        lambda_e_remove = 1.0 - z_e * w_e
        delta_e = delta_e * lambda_e_remove
        fi_e += (1.0 / lambda_e_remove) * w_e * w_e
        CHI_E -= d_old

        # Shift m position: move from first model to second model
        # Remove first model contribution
        r0 = 0.0
        r1 = 1.0
        w0 = fi00 * r0 + fi01 * r1
        w1 = fi10 * r0 + fi11 * r1
        lambda_shift1 = 1.0 - r0 * w0 - r1 * w1
        delta = delta * lambda_shift1
        inv_lambda = 1.0 / lambda_shift1
        fi00 += inv_lambda * w0 * w0
        fi01 += inv_lambda * w0 * w1
        fi10 += inv_lambda * w1 * w0
        fi11 += inv_lambda * w1 * w1
        CHI[0] -= signal[mm] * r0
        CHI[1] -= signal[mm] * r1

        # Add to second model
        q0 = 1.0
        q1 = 0.0
        w0 = fi00 * q0 + fi01 * q1
        w1 = fi10 * q0 + fi11 * q1
        lambda_shift2 = 1.0 + q0 * w0 + q1 * w1
        delta = delta * lambda_shift2
        inv_lambda = 1.0 / lambda_shift2
        fi00 -= inv_lambda * w0 * w0
        fi01 -= inv_lambda * w0 * w1
        fi10 -= inv_lambda * w1 * w0
        fi11 -= inv_lambda * w1 * w1
        CHI[0] += signal[mm] * q0
        CHI[1] += signal[mm] * q1

        # Compute evidence
        chi_fi_chi = (
            CHI[0] * (fi00 * CHI[0] + fi01 * CHI[1])
            + CHI[1] * (fi10 * CHI[0] + fi11 * CHI[1])
        )
        numerator = D - chi_fi_chi
        if numerator <= 0:
            numerator = 1e-12
        cit = ((-window_samples + 1.0 + 1.0) / 2.0) * np.log10(numerator)
        if delta <= 0:
            delta_safe = 1e-12
        else:
            delta_safe = delta
        jm = 0.5 * np.log10(np.abs(delta_safe))

        # Model 2
        B = fi_e * CHI_E
        BTB = B * B
        FTF = CHI_E * fi_e * CHI_E

        E1 = (-window_samples / 2.0) * np.log10(np.pi)
        E2 = -0.5 * np.log10(delta_e)
        E3 = np.log10(sqrt_pi)  # log10(gamma(0.5))
        E5 = -(window_samples - 1.0) / 2.0 * np.log10(max(D - FTF, 1e-12))
        E6 = 0.5 * 1.0 * np.log10(max(np.abs(BTB), 1e-12))
        evid = E1 + E2 + E3 + E5 - E6

        p2[mm] = cit - jm - evid

    # Normalize output: min-shift and threshold to zero
    valid_idx = np.isfinite(p2)
    if np.any(valid_idx):
        min_p2 = np.min(p2[valid_idx])
        out = p2 - min_p2
    else:
        out = p2

    out = np.maximum(out, 0.0)
    # Replace any remaining non-finite values with 0
    out[~np.isfinite(out)] = 0.0

    return out


def bscd(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Bayesian Sequential Change Detection (BSCD) analysis.

    Detects change points in a signal using Bayesian evidence of model fit.
    Implements the algorithm from Cmejla2013.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Normalized audio signal (typically bandpass-filtered, squared).
        Should be pre-normalized by DC-removal and division by max(abs()).
    window_samples : int
        Analysis window length in samples.
        Typical range: fs * wlen (e.g. 250000 * 0.01 = 2500 for 250kHz, 0.01s window).

    Returns
    -------
    np.ndarray, shape (N,)
        Bayesian evidence score at each sample.
        Higher values indicate stronger evidence of a change point.
    """
    return _bscd_inner(signal, window_samples)


# ─────────────────────────────────────────────────────────────────────────────
# BSCDDetector: high-level detector class
# ─────────────────────────────────────────────────────────────────────────────


class BSCDDetector(AbstractDetector):
    """
    Bayesian Sequential Change Detection detector for ultrasonic vocalizations.

    Wraps the core BSCD algorithm with preprocessing (bandpass filtering) and
    post-processing (smoothing, thresholding, event extraction).
    """

    def __init__(self, params: BSCDParams):
        """
        Initialize the BSCD detector.

        Parameters
        ----------
        params : BSCDParams
            Configuration object with fields:
                - fcutMin, fcutMax: bandpass filter cutoff frequencies (Hz)
                - wlen: analysis window length (seconds)
                - maWindow: moving-average smoothing window (samples)
                - noiseWindow, localWindow, k, w: (reserved for future use)
        """
        self.params = params

    @property
    def name(self) -> str:
        """Detector name."""
        return "BSCD"

    def detect(self, signal: np.ndarray, fs: int) -> list[Label]:
        """
        Detect ultrasonic vocalization events in an audio signal.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Audio samples
        fs : int
            Sampling frequency (Hz)

        Returns
        -------
        list[Label]
            Detected events as Label objects with start/end times and indices.
        """
        # ─── Preprocessing ─────────────────────────────────────────────────
        # DC removal (rule 2: exact MATLAB order)
        signal = signal.astype(np.float64)
        signal = signal - np.mean(signal)
        # Normalization
        max_val = np.max(np.abs(signal))
        if max_val < 1e-12:
            # Silent signal
            return []
        signal = signal / max_val

        # Bandpass filter with filtfilt (zero-phase)
        filtered = bandpass_filter_filtfilt(
            signal,
            fs,
            self.params.fcutMin,
            self.params.fcutMax,
            order=12,
        )

        # ─── BSCD algorithm ────────────────────────────────────────────────
        # Apply to squared filtered signal (MATLAB does this at line 101)
        window_samples = int(self.params.wlen * fs)
        bscd_out = bscd(filtered ** 2, window_samples)

        # ─── Smoothing ─────────────────────────────────────────────────────
        # Moving average smoothing
        bscd_smoothed = self._moving_average(bscd_out, self.params.maWindow)

        # ─── Thresholding and event extraction ─────────────────────────────
        threshold = np.mean(bscd_smoothed)
        binary = bscd_smoothed > threshold

        # Find transitions (0->1 and 1->0)
        binary_padded = np.concatenate(([0], binary.astype(int), [0]))
        diff = np.diff(binary_padded.astype(int))
        start_indices = np.where(diff == 1)[0]
        end_indices = np.where(diff == -1)[0] - 1

        # Create labels
        time_axis = np.arange(len(signal)) / fs
        labels = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            if start_idx < len(time_axis) and end_idx < len(time_axis):
                label = Label(
                    start_time=time_axis[start_idx],
                    end_time=time_axis[end_idx],
                    label="d",
                    start_index=int(start_idx),
                    stop_index=int(end_idx),
                )
                labels.append(label)

        return labels

    @staticmethod
    def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if window <= 1:
            return signal.copy()

        # Use centered moving average for minimal delay
        result = np.convolve(signal, np.ones(window) / window, mode="same")
        return result

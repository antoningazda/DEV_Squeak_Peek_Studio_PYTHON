"""
Bayesian Sequential Change Detection (BSCD) detector for ultrasonic vocalizations.

Port of bscd.m and BSCDDetector.m from the original MATLAB implementation.
Uses numba JIT compilation for the inner per-sample loop to achieve real-time performance.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from pydantic import BaseModel, Field
from scipy.ndimage import minimum_filter1d

from squeak_peek.audio.filters import bandpass_filter_filtfilt, movmean
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.model import Label


class BSCDParams(BaseModel):
    """Parameters for the Bayesian Sequential Change Detection detector."""

    fcutMin: float = Field(
        40_000, ge=0, le=250_000,
        description="Lower bound of the frequency band the BSCD detector analyses.",
        json_schema_extra={"unit": "Hz", "group": "freq_band"},
    )
    fcutMax: float = Field(
        120_000, ge=0, le=250_000,
        description="Upper bound of the frequency band the BSCD detector analyses.",
        json_schema_extra={"unit": "Hz", "group": "freq_band"},
    )
    denoise: bool = Field(
        False,
        description="Suppress stationary background noise before this detector runs.",
        json_schema_extra={
            "caption": "Off by default: the mean threshold is tuned on raw audio and denoising lowers precision. Noise-suppression settings: Settings → Pre-processing. "
                       "Export, review and classification always keep the original audio.",
        },
    )
    wlen: float = Field(
        0.01, ge=0.0, le=1.0,
        description="Length of the sliding analysis window used to detect abrupt changes in signal statistics.",
        json_schema_extra={"unit": "s", "decimals": 4},
    )
    maWindow: int = Field(
        5_000, ge=1, le=2_000_000,
        description="Number of frames averaged to smooth the change-detection statistic.",
    )
    thresholdMode: str = Field(
        "mean",
        description="How the smoothed BSCD statistic is thresholded.",
        json_schema_extra={
            "choices": ["mean", "adaptive"],
            "caption": "'mean' = one global threshold at the statistic's mean (the original MATLAB "
                       "detector); 'adaptive' = local noise floor + SNR-weighted threshold driven by "
                       "noiseWindow/localWindow/k/w, which are ignored in 'mean' mode.",
        },
    )
    noiseWindow: int = Field(
        256, ge=1, le=2_000_000,
        description="Number of frames used to estimate the background noise level.",
    )
    localWindow: int = Field(
        256, ge=1, le=2_000_000,
        description="Number of frames used to estimate the local (short-term) signal level.",
    )
    k: float = Field(
        0.023, ge=0.0, le=100.0,
        description="Scales the local-statistics term of the adaptive threshold.",
        json_schema_extra={
            "decimals": 4,
            "caption": "Higher k -> stricter threshold -> fewer, more confident detections.",
        },
    )
    w: float = Field(
        0.994, ge=0.0, le=1000.0,
        description="Weight given to the local SNR term when computing the adaptive threshold.",
        json_schema_extra={"decimals": 4},
    )

    model_config = {"populate_by_name": True}


# ─────────────────────────────────────────────────────────────────────────────
# Numba-compiled core: literal port of bscd.m (Cmejla2013)
# ─────────────────────────────────────────────────────────────────────────────

_TINY = 1e-300


@njit(cache=True)
def _log10_abs(x: float) -> float:
    """MATLAB's real(log10(x)): log10|x|, guarded against an exact zero."""
    return np.log10(np.abs(x) + _TINY)


@njit(cache=True)
def _bscd_inner(sig: np.ndarray, okno: int) -> np.ndarray:
    """
    Line-for-line port of bscd.m's sliding-window recursion.

    Two linear models are compared over an ``okno``-sample window centred on
    each sample: a two-level mean (step at the window centre, the "cit"/"jm"
    terms) and a single mean (the Bayesian-evidence baseline, "evid"). Both
    inverses are updated by rank-1 (Sherman-Morrison) steps as the window
    slides, so the cost is linear in len(sig).

    ``sig`` must already be normalised to max(|sig|) == 1 (bscd.m does this
    itself; see :func:`bscd`). Indices in the comments are MATLAB's 1-based
    ones; positions the MATLAB loop never writes stay 0, as there.
    """
    L = sig.shape[0]
    N = okno
    m = okno // 2
    p2 = np.zeros(L)
    if L < okno or L - m < m + 1:
        return p2

    # ---- initialisation over data = sig(1:okno) ----
    # G(j,:) = [1 0] for j <= m, [0 1] for j > m  ->  GTG = diag(m, N-m)
    D = 0.0
    chi0 = 0.0
    chi1 = 0.0
    for j in range(okno):
        D += sig[j] * sig[j]
        if j < m:
            chi0 += sig[j]
        else:
            chi1 += sig[j]
    g00 = float(m)
    g11 = float(N - m)
    fi00 = 1.0 / g00
    fi01 = 0.0
    fi10 = 0.0
    fi11 = 1.0 / g11
    DELTA = g00 * g11

    cit = ((-N + 2.0) / 2.0) * _log10_abs(D - (chi0 * (fi00 * chi0 + fi01 * chi1) + chi1 * (fi10 * chi0 + fi11 * chi1)))
    jm = 0.5 * _log10_abs(DELTA)

    chi_e = chi0 + chi1
    fi_e = 1.0 / N
    DELTA_E = float(N)
    B = fi_e * chi_e
    BTB = B * B
    FTF = chi_e * fi_e * chi_e
    E1 = (-N / 2.0) * np.log10(np.pi)
    E2 = -0.5 * _log10_abs(DELTA_E)
    E3 = np.log10(np.sqrt(np.pi))  # log10(gamma(1/2))
    E5 = -((N - 1.0) / 2.0) * _log10_abs(D - FTF)
    E6 = -0.5 * _log10_abs(BTB)
    evid = E1 + E2 + E3 + E5 + E6
    p2[m - 1] = cit - jm - evid  # p2(m)

    # ---- main loop: mm = m+1 .. L-m (MATLAB, inclusive) ----
    for mm in range(m + 1, L - m + 1):
        # pridani novych dat: d2 = sig(mm+m), G2 = [0 1]
        d2 = sig[mm + m - 1]
        D += d2 * d2
        chi1 += d2
        w0 = fi01
        w1 = fi11
        lam = 1.0 + w1
        DELTA *= lam
        fi00 -= w0 * w0 / lam
        fi01 -= w0 * w1 / lam
        fi10 -= w1 * w0 / lam
        fi11 -= w1 * w1 / lam
        chi_e += d2
        lam_e = 1.0 + fi_e
        DELTA_E *= lam_e
        fi_e -= fi_e * fi_e / lam_e

        # vlozeni nul: drop sig(mm-m), Z = [1 0]
        old = sig[mm - m - 1]
        D -= old * old
        chi0 -= old
        w0 = fi00
        w1 = fi10
        lam = 1.0 - w0
        DELTA *= lam
        fi00 += w0 * w0 / lam
        fi01 += w0 * w1 / lam
        fi10 += w1 * w0 / lam
        fi11 += w1 * w1 / lam
        chi_e -= old
        lam_e = 1.0 - fi_e
        DELTA_E *= lam_e
        fi_e += fi_e * fi_e / lam_e

        B = fi_e * chi_e
        BTB = B * B
        FTF = chi_e * fi_e * chi_e
        E2 = -0.5 * _log10_abs(DELTA_E)
        E5 = -((N - 1.0) / 2.0) * _log10_abs(D - FTF)
        E6 = 0.5 * _log10_abs(BTB)
        evid = E1 + E2 + E3 + E5 - E6

        # posunuti pozice m+1: sig(mm) moves from the right to the left mean
        cur = sig[mm - 1]
        chi1 -= cur  # R = [0 1]
        w0 = fi01
        w1 = fi11
        lam = 1.0 - w1
        DELTA *= lam
        fi00 += w0 * w0 / lam
        fi01 += w0 * w1 / lam
        fi10 += w1 * w0 / lam
        fi11 += w1 * w1 / lam
        chi0 += cur  # Q = [1 0]
        w0 = fi00
        w1 = fi10
        lam = 1.0 + w0
        DELTA *= lam
        fi00 -= w0 * w0 / lam
        fi01 -= w0 * w1 / lam
        fi10 -= w1 * w0 / lam
        fi11 -= w1 * w1 / lam

        cit = ((-N + 2.0) / 2.0) * _log10_abs(
            D - (chi0 * (fi00 * chi0 + fi01 * chi1) + chi1 * (fi10 * chi0 + fi11 * chi1))
        )
        jm = 0.5 * _log10_abs(DELTA)
        p2[mm - 1] = cit - jm - evid

    # x = find(p2); minp = min(p2(x)); out = p2 - minp; out = out .* (out > 0)
    found = False
    minp = 0.0
    for i in range(L):
        if p2[i] != 0.0 and np.isfinite(p2[i]):
            if not found or p2[i] < minp:
                minp = p2[i]
                found = True
    out = np.zeros(L)
    for i in range(L):
        v = p2[i] - minp
        if np.isfinite(v) and v > 0.0:
            out[i] = v
    return out


def bscd(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Bayesian Sequential Change Detection (BSCD) statistic — port of bscd.m.

    Implements the changepoint evidence of Cmejla et al. (2013): for every
    sample, log10 Bayesian evidence of a step in the mean at the centre of a
    ``window_samples``-long window versus a constant mean, shifted so its
    minimum over the computed samples is 0 and clipped at 0.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Typically the squared, bandpass-filtered audio. Normalised to
        max(|signal|) == 1 here, exactly as bscd.m does.
    window_samples : int
        Analysis window length in samples (e.g. fs * wlen).

    Returns
    -------
    np.ndarray, shape (N,)
        Non-negative evidence score per sample (0 where the window does not fit).
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    max_val = np.max(np.abs(sig)) if sig.size else 0.0
    if max_val > 0:
        sig = sig / max_val
    return _bscd_inner(sig, int(window_samples))


# ─────────────────────────────────────────────────────────────────────────────
# BSCDDetector: high-level detector class
# ─────────────────────────────────────────────────────────────────────────────


class BSCDDetector(AbstractDetector):
    """
    Bayesian Sequential Change Detection detector for ultrasonic vocalizations.

    Wraps the core BSCD algorithm with preprocessing (bandpass filtering) and
    post-processing (smoothing, thresholding, event extraction).
    """

    id = "BSCD"
    display_name = "BSCD"
    description = (
        "Bayesian Sequential Change Detection — flags points where the signal's "
        "statistics shift abruptly. Good for call onsets/offsets in noisier audio."
    )
    Params = BSCDParams

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

        # ─── Thresholding ──────────────────────────────────────────────────
        if self.params.thresholdMode == "adaptive":
            binary = self._adaptive_binary(bscd_smoothed)
        else:
            # BSCDDetector.m: optimalThreshold = mean(powerEnvelope)
            binary = bscd_smoothed > np.mean(bscd_smoothed)

        # Find transitions (0->1 and 1->0)
        binary_padded = np.concatenate(([0], binary.astype(int), [0]))
        diff = np.diff(binary_padded.astype(int))
        start_indices = np.where(diff == 1)[0]
        end_indices = np.where(diff == -1)[0] - 1

        # Create labels
        time_axis = np.arange(len(signal)) / fs
        labels = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            if start_idx <= end_idx < len(time_axis):
                label = Label(
                    start_time=time_axis[start_idx],
                    end_time=time_axis[end_idx],
                    label="d",
                    start_index=int(start_idx),
                    stop_index=int(end_idx),
                )
                labels.append(label)

        return labels

    def _adaptive_binary(self, smoothed: np.ndarray) -> np.ndarray:
        """Local noise floor + SNR-weighted threshold, the same scheme
        PSDDetector uses, driven by noiseWindow/localWindow/k/w. Not in the
        MATLAB original, which declared these parameters but thresholded at
        the global mean."""
        noise_window = max(1, int(self.params.noiseWindow))
        local_window = max(1, int(self.params.localWindow))

        noise_floor = minimum_filter1d(smoothed, size=noise_window, mode="nearest")
        effective = np.maximum(smoothed - noise_floor, 0.0)
        local_snr = np.minimum(effective / (noise_floor + np.finfo(float).eps), 10.0)

        local_mean = self._moving_average(effective, local_window)
        local_mean_sq = self._moving_average(effective**2, local_window)
        local_std = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 0.0))

        threshold = (local_mean + self.params.k * local_std) / (1.0 + self.params.w * local_snr)
        return effective > threshold

    @staticmethod
    def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
        """Centered moving average, MATLAB smoothdata('movmean') convention
        (edges shrink rather than pad). O(N), which matters here: BSCD's
        traces have one point per audio sample (tens of millions of points)."""
        return movmean(signal, window)

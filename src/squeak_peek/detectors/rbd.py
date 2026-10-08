"""
Recursive Bayesian Detector (RBD) for ultrasonic vocalization detection.

Ports the MATLAB RBD.m (sliding-window piecewise-AR changepoint statistic
vs. a single-AR Bayesian-evidence baseline) and RBDDetector.m.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from pydantic import BaseModel, Field

from squeak_peek.audio.filters import bandpass_filter_filtfilt, movmean
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.model import Label

_EPS = 1e-300


class RBDParams(BaseModel):
    """Parameters for the Relative Bayesian Difference detector."""

    fcutMin: float = Field(
        40_000, ge=0, le=250_000,
        description="Lower bound of the frequency band the RBD detector analyses.",
        json_schema_extra={"unit": "Hz", "group": "freq_band"},
    )
    fcutMax: float = Field(
        120_000, ge=0, le=250_000,
        description="Upper bound of the frequency band the RBD detector analyses.",
        json_schema_extra={"unit": "Hz", "group": "freq_band"},
    )
    denoise: bool = Field(
        False,
        description="Suppress stationary background noise before this detector runs.",
        json_schema_extra={
            "caption": "Off by default: with the bandpass on it did not improve RBD on the reference recordings. Noise-suppression settings: Settings → Pre-processing. "
                       "Export, review and classification always keep the original audio.",
        },
    )
    bandpass: bool = Field(
        True,
        description="Bandpass-filter the signal to [fcutMin, fcutMax] before fitting the AR models.",
        json_schema_extra={
            "caption": "Off reproduces RBDDetector.m, which fits the AR models to the full-band signal — "
                       "there they mostly describe noise below the USV band.",
        },
    )
    thresholdMode: str = Field(
        "median",
        description="How the smoothed RBD statistic is thresholded.",
        json_schema_extra={
            "choices": ["median", "original"],
            "caption": "'median' = above medianFactor × the recording's median (robust to a single loud "
                       "transient); 'original' = RBDDetector.m's dynamicScaling / amplitudeThreshold rule, "
                       "relative to the statistic's global maximum.",
        },
    )
    medianFactor: float = Field(
        4.0, ge=1.0, le=100.0,
        description="Threshold in 'median' mode, as a multiple of the smoothed statistic's median.",
        json_schema_extra={
            "decimals": 2,
            "caption": "Higher -> fewer, more confident detections.",
        },
    )
    wlen: float = Field(
        0.02, ge=0.0, le=1.0,
        description="Length of the sliding window compared on either side of each candidate boundary.",
        json_schema_extra={"unit": "s", "decimals": 4},
    )
    AR_order_left: int = Field(
        4, ge=0, le=50,
        description="Order of the autoregressive model fitted to the segment left of a candidate boundary.",
        json_schema_extra={"caption": "Higher orders capture more complex spectral shape but need more data and computation."},
    )
    AR_order_right: int = Field(
        4, ge=0, le=50,
        description="Order of the autoregressive model fitted to the segment right of a candidate boundary.",
        json_schema_extra={"caption": "Higher orders capture more complex spectral shape but need more data and computation."},
    )
    Bayesian_Evidence_order: int = Field(
        4, ge=0, le=50,
        description="Order of the autoregressive model used when computing the Bayesian evidence ratio.",
    )
    dynamicScaling: float = Field(
        0.3, ge=0.0, le=10.0,
        description="Scales the adaptive detection threshold relative to the signal's local statistics ('original' threshold mode only).",
        json_schema_extra={"decimals": 4},
    )
    smoothingWindowRBD: float = Field(
        0.03, ge=0.0, le=1.0,
        description="Smoothing window applied to the RBD detection statistic before thresholding.",
        json_schema_extra={"unit": "s", "decimals": 4},
    )
    smoothingWindowThr: float = Field(
        0.02, ge=0.0, le=1.0,
        description="Smoothing window applied to the adaptive threshold itself ('original' threshold mode only).",
        json_schema_extra={"unit": "s", "decimals": 4},
    )
    amplitudeThreshold: float = Field(
        0.02, ge=0.0, le=1.0,
        description="Minimum normalized signal amplitude required for a candidate event to be kept ('original' threshold mode only).",
        json_schema_extra={"decimals": 4},
    )

    model_config = {"populate_by_name": True}


def _matlab_round(x: float) -> int:
    """MATLAB round(): half away from zero (unlike Python/NumPy's round-half-to-even)."""
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


def _rbd_impl(sig: np.ndarray, okno: int, M1: int, M2: int, ME: int) -> np.ndarray:
    """
    Shared body for the plain-numpy reference and numba-jitted RBD core.

    Literal, line-for-line port of RBD.m's sliding-window recursion. Maintains
    two running least-squares systems that share a single energy accumulator D
    over an `okno`-sample sliding window centred `m = okno // 2` samples behind
    the newest sample:
      - (CHI, FI) over an (M1+M2)-column design matrix whose first M1 columns
        regress on the "left" half of the window and last M2 columns regress
        on the "right" half (a single-changepoint AR model) -> feeds `cit`;
      - (CHI_E, FI_E) over an ME-column design matrix spanning the whole
        window (a no-changepoint AR(ME) baseline) -> feeds `E5`.
    RBD.m's returned `out` is built only from `E5 - cit` (`CSF`); everything
    else it computes (evid/jm/p2, E1-E4/E6, DELTA/DELTA_E) does not feed the
    output, so it is intentionally not ported.

    Returns (cit, E5) as float64 arrays of length len(sig). Positions before
    the initial window centre and after the main loop's last index are left
    at 0, exactly as MATLAB's pre-zeroed arrays leave them untouched.
    """
    Ntot = sig.shape[0]
    m = okno // 2
    D2 = M1 + M2

    cit = np.zeros(Ntot)
    E5 = np.zeros(Ntot)

    # ---- initialization over data = sig[0:okno] (MATLAB data = sig(1:okno)) ----
    G = np.zeros((okno, D2))
    G_E = np.zeros((okno, ME))
    for j in range(2, okno + 1):
        row = j - 1
        if j <= M1:
            if j <= m:
                for k in range(j - 1):
                    G[row, k] = sig[j - 2 - k]
            else:
                for k in range(j - 1):
                    G[row, M1 + k] = sig[j - 2 - k]
        else:
            if j <= m:
                for k in range(M1):
                    G[row, k] = sig[j - 2 - k]
            else:
                for k in range(M2):
                    G[row, M1 + k] = sig[j - 2 - k]
        if j <= ME:
            for k in range(j - 1):
                G_E[row, k] = sig[j - 2 - k]
        else:
            for k in range(ME):
                G_E[row, k] = sig[j - 2 - k]

    data = sig[:okno]
    D = data @ data
    CHI = data @ G
    GTG = G.T @ G
    FI = np.linalg.inv(GTG)

    CHI_E = data @ G_E
    GTG_E = G_E.T @ G_E
    FI_E = np.linalg.inv(GTG_E)

    cit_val = D - CHI @ FI @ CHI
    cit[m - 1] = np.log(np.abs(cit_val) + _EPS)
    FTF0 = CHI_E @ FI_E @ CHI_E
    E5[m - 1] = np.log(np.abs(D - FTF0) + _EPS)

    # Scratch buffers reused across every iteration of the main loop below.
    # Numba can't stack-allocate arrays whose size is a runtime value (D2/ME
    # here), so allocating G2/W/etc. fresh inside a 10s-of-millions-of-
    # iterations loop makes heap allocation (not FLOPs) the bottleneck.
    # Preallocating once and mutating in place (with explicit index loops in
    # place of np.outer/@, which would themselves allocate) removes that cost.
    G2 = np.zeros(D2)
    G2_E = np.zeros(ME)
    Z = np.zeros(D2)
    Z_E = np.zeros(ME)
    R = np.zeros(D2)
    Q = np.zeros(D2)
    W = np.zeros(D2)
    W_E = np.zeros(ME)
    tmp_D2 = np.zeros(D2)
    tmp_ME = np.zeros(ME)

    # ---- main loop: mm = m+1 .. Ntot-m (MATLAB 1-based, inclusive) ----
    for mm in range(m + 1, Ntot - m + 1):
        # --- pridani novych dat (add newest sample, position mm+m) ---
        d2 = sig[mm + m - 1]  # sig(mm+m)

        for k in range(M2):
            G2[M1 + k] = sig[mm + m - 2 - k]
        D += d2 * d2
        for i in range(D2):
            CHI[i] += d2 * G2[i]
        for i in range(D2):
            acc = 0.0
            for j in range(D2):
                acc += FI[i, j] * G2[j]
            W[i] = acc
        LAMBDA = 1.0
        for i in range(D2):
            LAMBDA += G2[i] * W[i]
        for i in range(D2):
            for j in range(D2):
                FI[i, j] -= W[i] * W[j] / LAMBDA

        for k in range(ME):
            G2_E[k] = sig[mm + m - 2 - k]
        for i in range(ME):
            CHI_E[i] += d2 * G2_E[i]
        for i in range(ME):
            acc = 0.0
            for j in range(ME):
                acc += FI_E[i, j] * G2_E[j]
            W_E[i] = acc
        LAMBDA_E = 1.0
        for i in range(ME):
            LAMBDA_E += G2_E[i] * W_E[i]
        for i in range(ME):
            for j in range(ME):
                FI_E[i, j] -= W_E[i] * W_E[j] / LAMBDA_E

        # --- vlozeni nul (drop oldest sample, position mm-m) ---
        old = sig[mm - m - 1]  # sig(mm-m)
        D -= old * old
        if mm - m > M1:
            for k in range(M1):
                Z[k] = sig[mm - m - 2 - k]
        else:
            avail = mm - 1 - m
            for k in range(avail):
                Z[k] = sig[mm - m - 2 - k]
        for i in range(D2):
            CHI[i] -= old * Z[i]
        for i in range(D2):
            acc = 0.0
            for j in range(D2):
                acc += FI[i, j] * Z[j]
            W[i] = acc
        LAMBDA = 1.0
        for i in range(D2):
            LAMBDA -= Z[i] * W[i]
        for i in range(D2):
            for j in range(D2):
                FI[i, j] += W[i] * W[j] / LAMBDA

        if mm - m > ME:
            for k in range(ME):
                Z_E[k] = sig[mm - m - 2 - k]
        else:
            avail_e = mm - 1 - m
            for k in range(avail_e):
                Z_E[k] = sig[mm - m - 2 - k]
        for i in range(ME):
            CHI_E[i] -= old * Z_E[i]
        for i in range(ME):
            acc = 0.0
            for j in range(ME):
                acc += FI_E[i, j] * Z_E[j]
            W_E[i] = acc
        LAMBDA_E = 1.0
        for i in range(ME):
            LAMBDA_E -= Z_E[i] * W_E[i]
        for i in range(ME):
            for j in range(ME):
                FI_E[i, j] += W_E[i] * W_E[j] / LAMBDA_E

        for i in range(ME):
            acc = 0.0
            for j in range(ME):
                acc += FI_E[i, j] * CHI_E[j]
            tmp_ME[i] = acc
        FTF = 0.0
        for i in range(ME):
            FTF += CHI_E[i] * tmp_ME[i]
        E5[mm - 1] = np.log(np.abs(D - FTF) + _EPS)

        # --- posunuti pozice m+1 (shift the left/right split boundary) ---
        for k in range(M2):
            R[M1 + k] = sig[mm - 2 - k]
        cur = sig[mm - 1]  # sig(mm)
        for i in range(D2):
            CHI[i] -= cur * R[i]
        for i in range(D2):
            acc = 0.0
            for j in range(D2):
                acc += FI[i, j] * R[j]
            W[i] = acc
        LAMBDA = 1.0
        for i in range(D2):
            LAMBDA -= R[i] * W[i]
        for i in range(D2):
            for j in range(D2):
                FI[i, j] += W[i] * W[j] / LAMBDA

        for k in range(M1):
            Q[k] = sig[mm - 2 - k]
        for i in range(D2):
            CHI[i] += cur * Q[i]
        for i in range(D2):
            acc = 0.0
            for j in range(D2):
                acc += FI[i, j] * Q[j]
            W[i] = acc
        LAMBDA = 1.0
        for i in range(D2):
            LAMBDA += Q[i] * W[i]
        for i in range(D2):
            for j in range(D2):
                FI[i, j] -= W[i] * W[j] / LAMBDA

        for i in range(D2):
            acc = 0.0
            for j in range(D2):
                acc += FI[i, j] * CHI[j]
            tmp_D2[i] = acc
        cit_val = D
        for i in range(D2):
            cit_val -= CHI[i] * tmp_D2[i]
        cit[mm - 1] = np.log(np.abs(cit_val) + _EPS)

    return cit, E5


def _rbd_reference(sig: np.ndarray, okno: int, M1: int, M2: int, ME: int) -> tuple[np.ndarray, np.ndarray]:
    """Plain-numpy (non-jitted) reference implementation, for correctness validation only."""
    return _rbd_impl(sig, okno, M1, M2, ME)


_rbd_numba_impl = njit(cache=True)(_rbd_impl)


def rbd(
    signal: np.ndarray,
    window_length: int,
    AR_order_left: int,
    AR_order_right: int,
    Bayesian_Evidence_order: int,
    use_numba: bool = True,
) -> np.ndarray:
    """
    Recursive Bayesian (Autoregressive Changepoint) Detector.

    Port of RBD.m. Computes a changepoint detection statistic (4 * (E5 - cit))
    from a sliding-window piecewise-AR vs. single-AR(``Bayesian_Evidence_order``)
    Bayesian-evidence comparison.

    Parameters
    ----------
    signal : np.ndarray
        1-D audio signal (should already be normalized, matching MATLAB's
        RBD.m which also re-normalizes by max(abs()) internally).
    window_length : int
        Length of the sliding window in samples.
    AR_order_left, AR_order_right : int
        AR model order for the left/right half of the window.
    Bayesian_Evidence_order : int
        AR order for the single-model Bayesian-evidence baseline.
    use_numba : bool
        Use the numba-jitted core (default). False forces the plain-numpy
        reference path (slow; for validation on short clips only).

    Returns
    -------
    np.ndarray
        Detection output (real-valued), shape (len(signal),).
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val

    okno = int(window_length)
    m = okno // 2
    Ntot = sig.shape[0]
    if Ntot < okno or Ntot - 2 * m < 1:
        return np.zeros(Ntot)

    impl = _rbd_numba_impl if use_numba else _rbd_reference
    cit, E5 = impl(sig, okno, int(AR_order_left), int(AR_order_right), int(Bayesian_Evidence_order))

    CSF = E5 - cit
    start = _matlab_round(okno / 2.0)
    out_tail = 4.0 * CSF[start:Ntot]
    out = np.concatenate([np.zeros(m), out_tail])

    if out.shape[0] < Ntot:
        out = np.pad(out, (0, Ntot - out.shape[0]))
    elif out.shape[0] > Ntot:
        out = out[:Ntot]
    return out


class RBDDetector(AbstractDetector):
    """
    RBD (Recursive Bayesian Detector) for ultrasonic vocalization detection.

    Based on RBDDetector.m: normalize -> [bandpass] -> rbd() -> smooth ->
    threshold -> binary edges -> Label events.

    Two departures from the MATLAB detector, each switchable back:
    ``bandpass`` (RBDDetector.m fits the AR models to the full-band signal,
    where they mostly model noise below the USV band) and
    ``thresholdMode="median"`` (RBDDetector.m thresholds relative to the
    statistic's global maximum, so one loud transient moves the threshold
    for the whole recording). On the five reference recordings the two
    together took RBD from F1 ~0.25-0.4 to ~0.6-0.8.
    """

    id = "RBD"
    display_name = "RBD"
    description = (
        "Relative Bayesian Difference — compares autoregressive models on either "
        "side of a candidate boundary. More precise boundaries, more compute per call."
    )
    Params = RBDParams

    def detect(self, signal: np.ndarray, fs: int) -> list[Label]:
        signal = np.asarray(signal, dtype=np.float64).ravel()

        window_samples = round(self.params.wlen * fs)
        if len(signal) < window_samples * 2:
            return []

        # DC removal and normalization (MATLAB RBDDetector.m lines 64-65)
        signal = signal - np.mean(signal)
        max_val = np.max(np.abs(signal))
        if max_val == 0:
            return []
        signal = signal / max_val

        if self.params.bandpass:
            signal = bandpass_filter_filtfilt(signal, fs, self.params.fcutMin, self.params.fcutMax, order=12)

        rbd_out = rbd(
            signal,
            window_samples,
            self.params.AR_order_left,
            self.params.AR_order_right,
            self.params.Bayesian_Evidence_order,
        )

        rbd_max = np.max(np.abs(rbd_out))
        if rbd_max == 0:
            return []
        rbd_out = rbd_out / rbd_max

        # Smoothing (MATLAB RBDDetector.m line 73: movmean, edges shrink).
        smooth_window_rbd = max(1, round(self.params.smoothingWindowRBD * fs))
        smooth_rbd = movmean(rbd_out, smooth_window_rbd)

        if self.params.thresholdMode == "original":
            # Dynamic threshold (MATLAB RBDDetector.m lines 74, 76)
            smooth_window_thr = max(1, round(self.params.smoothingWindowThr * fs))
            dynamic_threshold = movmean(smooth_rbd, smooth_window_thr) * self.params.dynamicScaling
            binary = (smooth_rbd > dynamic_threshold) & (smooth_rbd > self.params.amplitudeThreshold)
        else:
            # Relative to the typical (median) level of the statistic, which the
            # sparse calls barely move — unlike its maximum.
            binary = smooth_rbd > self.params.medianFactor * np.median(smooth_rbd)

        # Find edges (MATLAB RBDDetector.m lines 77-79)
        edges = np.diff(np.concatenate([[0], binary.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1

        labels = []
        for start_idx, end_idx in zip(starts, ends):
            labels.append(
                Label(
                    start_time=start_idx / fs,
                    end_time=end_idx / fs,
                    label="d",
                    start_frequency=0.0,
                    end_frequency=0.0,
                    start_index=int(start_idx),
                    stop_index=int(end_idx),
                )
            )

        return labels

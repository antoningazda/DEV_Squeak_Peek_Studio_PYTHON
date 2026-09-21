"""
Recursive Bayesian Detector (RBD) for ultrasonic vocalization detection.

Ports the MATLAB RBD.m (sliding-window piecewise-AR changepoint statistic
vs. a single-AR Bayesian-evidence baseline) and RBDDetector.m.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from squeak_peek.config import RBDParams
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.model import Label

_EPS = 1e-300


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

    # ---- main loop: mm = m+1 .. Ntot-m (MATLAB 1-based, inclusive) ----
    for mm in range(m + 1, Ntot - m + 1):
        # --- pridani novych dat (add newest sample, position mm+m) ---
        d2 = sig[mm + m - 1]  # sig(mm+m)

        G2 = np.zeros(D2)
        for k in range(M2):
            G2[M1 + k] = sig[mm + m - 2 - k]
        D += d2 * d2
        CHI = CHI + d2 * G2
        W = FI @ G2
        LAMBDA = 1.0 + G2 @ W
        FI = FI - np.outer(W, W) / LAMBDA

        G2_E = np.zeros(ME)
        for k in range(ME):
            G2_E[k] = sig[mm + m - 2 - k]
        CHI_E = CHI_E + d2 * G2_E
        W_E = FI_E @ G2_E
        LAMBDA_E = 1.0 + G2_E @ W_E
        FI_E = FI_E - np.outer(W_E, W_E) / LAMBDA_E

        # --- vlozeni nul (drop oldest sample, position mm-m) ---
        old = sig[mm - m - 1]  # sig(mm-m)
        D -= old * old
        Z = np.zeros(D2)
        if mm - m > M1:
            for k in range(M1):
                Z[k] = sig[mm - m - 2 - k]
        else:
            avail = mm - 1 - m
            for k in range(avail):
                Z[k] = sig[mm - m - 2 - k]
        CHI = CHI - old * Z
        W = FI @ Z
        LAMBDA = 1.0 - Z @ W
        FI = FI + np.outer(W, W) / LAMBDA

        Z_E = np.zeros(ME)
        if mm - m > ME:
            for k in range(ME):
                Z_E[k] = sig[mm - m - 2 - k]
        else:
            avail_e = mm - 1 - m
            for k in range(avail_e):
                Z_E[k] = sig[mm - m - 2 - k]
        CHI_E = CHI_E - old * Z_E
        W_E = FI_E @ Z_E
        LAMBDA_E = 1.0 - Z_E @ W_E
        FI_E = FI_E + np.outer(W_E, W_E) / LAMBDA_E

        FTF = CHI_E @ FI_E @ CHI_E
        E5[mm - 1] = np.log(np.abs(D - FTF) + _EPS)

        # --- posunuti pozice m+1 (shift the left/right split boundary) ---
        R = np.zeros(D2)
        for k in range(M2):
            R[M1 + k] = sig[mm - 2 - k]
        cur = sig[mm - 1]  # sig(mm)
        CHI = CHI - cur * R
        W = FI @ R
        LAMBDA = 1.0 - R @ W
        FI = FI + np.outer(W, W) / LAMBDA

        Q = np.zeros(D2)
        for k in range(M1):
            Q[k] = sig[mm - 2 - k]
        CHI = CHI + cur * Q
        W = FI @ Q
        LAMBDA = 1.0 + Q @ W
        FI = FI - np.outer(W, W) / LAMBDA

        cit_val = D - CHI @ FI @ CHI
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

    Port of RBDDetector.m: normalize -> rbd() -> normalize -> smooth -> dynamic
    threshold -> binary edges -> Label events. Unlike PSD/BSCD, RBDDetector.m
    does NOT bandpass-filter the input; it runs directly on the DC-removed,
    normalized signal.
    """

    def __init__(self, params: RBDParams):
        self.params = params

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

        # Smoothing (MATLAB RBDDetector.m line 73: movmean)
        smooth_window_rbd = max(1, round(self.params.smoothingWindowRBD * fs))
        smooth_rbd = np.convolve(
            rbd_out, np.ones(smooth_window_rbd) / smooth_window_rbd, mode="same"
        )

        # Dynamic threshold (MATLAB RBDDetector.m line 74)
        smooth_window_thr = max(1, round(self.params.smoothingWindowThr * fs))
        dynamic_threshold = (
            np.convolve(smooth_rbd, np.ones(smooth_window_thr) / smooth_window_thr, mode="same")
            * self.params.dynamicScaling
        )

        # Thresholding (MATLAB RBDDetector.m line 76)
        binary = (smooth_rbd > dynamic_threshold) & (smooth_rbd > self.params.amplitudeThreshold)

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

    @property
    def name(self) -> str:
        return "RBD"

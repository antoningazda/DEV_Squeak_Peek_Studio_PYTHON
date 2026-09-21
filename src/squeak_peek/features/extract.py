"""
Feature extraction for USV detection.

Computes 12-dimensional acoustic features per frame (frame-based analysis).
This module ports extractUSVFrameFeatures.m from the MATLAB Squeak Peek Studio.

MATLAB equivalent
-----------------
    [X, midTimes] = extractUSVFrameFeatures(x, fs, frameLen, hopLen, nfft, fmin, fmax)

Author
------
    Antonín Gazda
    Master's Thesis — Software for Visualization, Segmentation,
    and Sonification of Ultrasonic Vocalizations of Laboratory Rats
    Czech Technical University in Prague, 2025
"""

from __future__ import annotations

import numpy as np


def extract_frame_features(
    x: np.ndarray,
    fs: int,
    frame_len: int,
    hop_len: int,
    nfft: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 12-dimensional acoustic features from USV audio (frame-wise).

    Computes a 12-feature vector per sliding-window frame. Used by ML-based
    detectors to guarantee a consistent feature set across training, optimization,
    and inference.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Bandpass-filtered, DC-removed, normalized audio vector.
    fs : int
        Sampling frequency (Hz).
    frame_len : int
        Frame length in samples.
    hop_len : int
        Hop length (stride) in samples.
    nfft : int
        FFT length (recommended: 2^ceil(log2(frame_len))).
    fmin : float
        Lower frequency bound (Hz).
    fmax : float
        Upper frequency bound (Hz).

    Returns
    -------
    X : np.ndarray, shape (n_frames, 12)
        Feature matrix. Columns are:
            0   BandPower        - Total power in [fmin, fmax]
            1   SpecCentroid     - Spectral centroid frequency (Hz)
            2   SpecSpread       - Spectral spread (std-dev of freq. distribution)
            3   SpecFlatness     - Spectral flatness (tone vs. noise measure)
            4   SpecEntropy      - Normalized spectral entropy
            5   ZCR              - Zero crossing rate
            6   SNR_est          - Per-frame SNR estimate (dB) vs. local noise floor
            7   SpecFlux         - Spectral flux (frame-to-frame spectral change)
            8   DomFreq          - Dominant (peak-power) frequency (Hz)
            9   Delta_BandPower  - First-order difference of BandPower across frames
            10  Delta_Centroid   - First-order difference of SpecCentroid across frames
            11  Delta_Entropy    - First-order difference of SpecEntropy across frames
        NaN and Inf values are replaced with 0.
    mid_times : np.ndarray, shape (n_frames,)
        Mid-point time of each frame in seconds.

    Notes
    -----
    - Features 9-11 (delta features) capture temporal dynamics, critical for
      detecting USV onsets and offsets with frequency modulation.
    - All NaN and Inf values are replaced with 0 (sanitization).
    """
    N = len(x)

    # Compute frame start indices (0-based Python indexing)
    # MATLAB: starts = (1 : hopLen : (N - frameLen + 1))'
    # In Python: starts = range(0, N - frameLen + 1, hopLen)
    starts = np.arange(0, N - frame_len + 1, hop_len)
    n_frames = len(starts)

    # Handle empty case
    if n_frames == 0:
        return np.zeros((0, 12)), np.zeros(0)

    # --- Noise floor (300 ms moving minimum of smoothed power) ---
    noise_win = round(0.3 * fs)
    power_sig = _movmean(x**2, noise_win)
    noise_floor = _movmin(power_sig, noise_win) + np.finfo(float).eps

    # --- Frequency band mask ---
    freq_axis = np.arange(nfft // 2 + 1) * (fs / nfft)
    band_idx = (freq_axis >= fmin) & (freq_axis <= fmax)
    freq_band = freq_axis[band_idx]
    n_band = np.sum(band_idx)

    # Periodic Hann window (MATLAB: hann(frameLen, 'periodic')).
    # np.hanning() is the *symmetric* variant and is NOT equivalent here.
    from scipy.signal.windows import hann as _hann_window

    win = _hann_window(frame_len, sym=False)

    # --- Pre-allocate (9 instantaneous features per frame) ---
    X_base = np.zeros((n_frames, 9))
    mid_times = np.zeros(n_frames)
    Fb_prev = np.zeros(n_band)  # For spectral flux computation

    for i in range(n_frames):
        i1 = starts[i]
        i2 = i1 + frame_len
        frame = x[i1:i2] * win

        # FFT magnitude in-band
        F = np.abs(np.fft.fft(frame, nfft))
        F = F[: nfft // 2 + 1]
        Fb = F[band_idx]

        P = Fb**2 + np.finfo(float).eps
        P_total = np.sum(P)
        P_norm = P / P_total

        # --- Core spectral features ---
        band_power = P_total
        spec_centroid = np.sum(freq_band * P_norm)
        spec_spread = np.sqrt(np.sum(((freq_band - spec_centroid) ** 2) * P_norm))
        spec_flatness = np.exp(np.mean(np.log(P))) / np.mean(P)
        spec_entropy = -np.sum(P_norm * np.log(P_norm + np.finfo(float).eps)) / np.log(
            max(len(P_norm), 2)
        )

        # --- Zero crossing rate ---
        sgn = np.sign(x[i1:i2])
        sgn[sgn == 0] = 1
        zcr = np.sum(np.abs(np.diff(sgn))) / (2 * frame_len)

        # --- Per-frame SNR estimate ---
        # MATLAB: midSamp = round((matlab_i1 + matlab_i2)/2) with 1-based
        # matlab_i1=i1+1, matlab_i2=i2 (i2 here is the 0-based exclusive end,
        # equal to the 1-based inclusive last-sample index) -> 0-based:
        # round((i1+i2+1)/2) - 1, using MATLAB's round-half-away-from-zero,
        # which for positive integers is (i1+i2+2)//2 - 1.
        mid_samp = max(0, min(N - 1, (i1 + i2 + 2) // 2 - 1))
        snr_est = 10 * np.log10(
            (np.mean(x[i1:i2] ** 2) + np.finfo(float).eps) / noise_floor[mid_samp]
        )

        # --- Spectral flux (L2 distance from previous frame in band) ---
        spec_flux = np.sum((Fb - Fb_prev) ** 2) / max(n_band, 1)
        Fb_prev = Fb.copy()

        # --- Dominant frequency (peak power bin in band) ---
        pk_idx = np.argmax(P)
        dom_freq = freq_band[pk_idx]

        X_base[i, :] = [
            band_power,
            spec_centroid,
            spec_spread,
            spec_flatness,
            spec_entropy,
            zcr,
            snr_est,
            spec_flux,
            dom_freq,
        ]
        mid_times[i] = (mid_samp) / fs

    # --- Delta features (first-order temporal differences, cols 9-11) ---
    delta_power = np.concatenate([[0], np.diff(X_base[:, 0])])  # Delta BandPower
    delta_centroid = np.concatenate([[0], np.diff(X_base[:, 1])])  # Delta SpecCentroid
    delta_entropy = np.concatenate([[0], np.diff(X_base[:, 4])])  # Delta SpecEntropy

    X = np.column_stack([X_base, delta_power, delta_centroid, delta_entropy])

    # --- Sanitise (replace NaN/Inf with 0) ---
    X[~np.isfinite(X)] = 0

    return X, mid_times


def _movmean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute moving mean (centered window, matching MATLAB's movmean).

    MATLAB's movmean(x, k) window spans k//2 elements before the current one,
    the current one, and k-1-(k//2) after (symmetric for odd k, one more
    element before than after for even k), and *shrinks* (not pads) at the
    edges -- averaging over only the in-range elements.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    window : int
        Window length in samples.

    Returns
    -------
    np.ndarray
        Moving mean, same shape as input.
    """
    if window <= 1:
        return x.astype(float).copy()

    n = len(x)
    before = window // 2
    after = window - 1 - before
    idx = np.arange(n)
    start = np.clip(idx - before, 0, n)
    end = np.clip(idx + after + 1, 0, n)

    csum = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    return (csum[end] - csum[start]) / (end - start)


def _movmin(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute moving minimum (centered window, shrink at edges).

    Mimics MATLAB's movmin(x, window) with centered window.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    window : int
        Window length in samples.

    Returns
    -------
    np.ndarray
        Moving minimum, same shape as input.
    """
    from scipy.ndimage import minimum_filter

    if window <= 0:
        return x.copy()

    # For centered window, use size and mode='nearest'
    return minimum_filter(x, size=window, mode="nearest")

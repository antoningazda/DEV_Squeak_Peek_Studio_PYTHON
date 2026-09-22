"""
Signal processing utilities — bandpass filtering and STFT spectrogram.

MATLAB equivalents
------------------
    bandpass_filter  ↔  bandpass(y, [fcutMin, fcutMax], fs)
    compute_stft     ↔  [s, f, t] = spectrogram(y, window, noverlap, nfft, fs)

Phase 1 status: stubs only — full implementation is Phase 2.
"""

from __future__ import annotations

import numpy as np


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    f_low: float,
    f_high: float,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Matches MATLAB's ``bandpass(y, [f_low, f_high], fs)`` with the
    default filter order (5).  Uses second-order sections (SOS) for
    numerical stability at high sample rates (250 kHz).

    Parameters
    ----------
    signal  : 1-D float32 audio array
    fs      : sample rate in Hz
    f_low   : lower cut-off frequency in Hz
    f_high  : upper cut-off frequency in Hz
    order   : Butterworth filter order (default 5)

    Returns
    -------
    np.ndarray  filtered signal, same shape and dtype as *signal*

    Notes
    -----
    Full numerical validation against MATLAB output is part of Phase 2.
    """
    from scipy.signal import butter, sosfilt  # lazy — scipy not always installed

    nyq = fs / 2.0
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    filtered = sosfilt(sos, signal)
    return filtered.astype(signal.dtype)


def bandpass_filter_filtfilt(
    signal: np.ndarray,
    fs: int,
    f_low: float,
    f_high: float,
    order: int = 12,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter using filtfilt.

    Matches MATLAB's ``designfilt('bandpassiir', FilterOrder, ...)`` followed by
    ``filtfilt()``, which applies the filter forward and backward for zero-phase
    distortion. Equivalent to MATLAB's filtfilt with order-12 IIR bandpass.

    Parameters
    ----------
    signal  : 1-D float32 audio array
    fs      : sample rate in Hz
    f_low   : lower cut-off frequency in Hz
    f_high  : upper cut-off frequency in Hz
    order   : total Butterworth filter order (default 12, matching MATLAB
              designfilt('bandpassiir', 'FilterOrder', 12, ...) — this is the
              TOTAL order of the resulting bandpass filter, as MATLAB's
              FilterOrder means)

    Returns
    -------
    np.ndarray  filtered signal, same shape and dtype as *signal*

    Notes
    -----
    Uses scipy.signal.sosfiltfilt for zero-phase filtering (forward-backward
    pass). scipy.signal.butter(N, ..., btype="bandpass") returns a filter of
    total order 2*N (it composites a lowpass and a highpass prototype), so we
    pass order // 2 to butter() to get a resulting filter whose TOTAL order
    matches MATLAB's FilterOrder — passing `order` straight through would
    silently produce a filter twice as steep as MATLAB's, with a materially
    different roll-off near the band edges.
    """
    from scipy.signal import butter, sosfiltfilt  # lazy — scipy not always installed

    nyq = fs / 2.0
    sos = butter(order // 2, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, signal)
    return filtered.astype(signal.dtype)


def compute_stft(
    signal: np.ndarray,
    fs: int,
    segment_length: int,
    overlap_factor: float,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Short-Time Fourier Transform (STFT) spectrogram in dB.

    Matches MATLAB's ``spectrogram(y, window, noverlap, nfft, fs)`` with a
    Hann window (default) and 'density' scaling.

    Parameters
    ----------
    signal         : 1-D audio array
    fs             : sample rate in Hz
    segment_length : STFT window length in samples (e.g. 8192)
    overlap_factor : overlap fraction (0–1), e.g. 0.59
    window         : window function name (default "hann"), e.g. "hamming", "hann"

    Returns
    -------
    frequencies : np.ndarray, shape (F,)   — Hz
    times       : np.ndarray, shape (T,)   — seconds
    Sxx_dB      : np.ndarray, shape (F, T) — power in dB

    Notes
    -----
    Full numerical validation against MATLAB output is part of Phase 2.
    Backward compatible: omitting window= uses "hann" as before.
    """
    from scipy.signal import spectrogram as _sp_spectrogram  # lazy

    noverlap = int(segment_length * overlap_factor)
    f, t, Sxx = _sp_spectrogram(
        signal,
        fs=fs,
        window=window,
        nperseg=segment_length,
        noverlap=noverlap,
        scaling="density",
    )
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_dB


def band_restrict(
    frequencies: np.ndarray,
    Sxx: np.ndarray,
    f_low: float,
    f_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a spectrogram to a frequency band of interest.

    Parameters
    ----------
    frequencies : 1-D array of frequency bins (Hz)
    Sxx         : 2-D spectrogram array, shape (F, T)
    f_low       : lower frequency bound (Hz)
    f_high      : upper frequency bound (Hz)

    Returns
    -------
    f_band  : 1-D array of frequency bins within the band
    Sxx_band: 2-D spectrogram sliced to the band, shape (F_band, T)
    """
    mask = (frequencies >= f_low) & (frequencies <= f_high)
    return frequencies[mask], Sxx[mask, :]

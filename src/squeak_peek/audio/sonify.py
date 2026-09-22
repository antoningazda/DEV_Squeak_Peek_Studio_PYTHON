"""
Phase-vocoder pitch-shift and time-stretch for audio sonification.

MATLAB equivalent:
    sonifySegment (SqueakPeekStudio_exported.m:858-952)

Algorithm
---------
Two-pass phase vocoder:
1. Extract segment and bandpass filter (40-120 kHz)
2. STFT analysis with Hamming window
3. PITCH SHIFT: resample frame timeline by 2^(semitones/12), interpolate magnitude
   and phase differences, reconstruct complex spectrum, overlap-add synthesize
4. TIME STRETCH: repeat STFT/resample/synthesize on pitch-shifted output
5. Normalize output
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def sonify_segment(
    audio: np.ndarray,
    start_time: float,
    end_time: float,
    fs: int,
    semitones: float,
    slowdown: float,
    win_len: int = 1024,
    win_hop: int | None = None,
) -> np.ndarray:
    """
    Phase-vocoder pitch-shift and time-stretch a segment of audio.

    Port of MATLAB's sonifySegment (SqueakPeekStudio_exported.m:858-952).
    Applies a bandpass filter (40-120 kHz, order 12), then a two-pass
    phase vocoder (pitch shift via frame-timeline resampling, then time
    stretch on the pitch-shifted output), using manual STFT + overlap-add
    synthesis (not equivalent to librosa.effects.pitch_shift/time_stretch).

    Parameters
    ----------
    audio : np.ndarray
        Raw audio samples (1-D float array)
    start_time : float
        Segment start time in seconds
    end_time : float
        Segment end time in seconds
    fs : int
        Sample rate in Hz
    semitones : float
        Pitch shift in semitones (e.g., -35 for -35 semitones)
    slowdown : float
        Time stretch factor (e.g., 4 for 4x slower)
    win_len : int
        STFT window length in samples (default 1024)
    win_hop : int or None
        STFT hop length in samples (default win_len // 4)

    Returns
    -------
    np.ndarray
        Sonified signal, normalized to max absolute value <= 1.0,
        at the original sample rate fs
    """
    if win_hop is None:
        win_hop = win_len // 4

    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})"
        )

    # === Extract segment ===
    start_idx = max(0, round(start_time * fs))
    end_idx = min(len(audio), round(end_time * fs))
    x = audio[start_idx:end_idx]

    # A zero-phase filtfilt-style filter needs more samples than its padding
    # length, and a usable STFT needs at least one full analysis window —
    # whichever is larger sets the real minimum segment length. Fail with a
    # clear message rather than letting scipy or a negative array dimension
    # raise an opaque error further down.
    min_samples = max(win_len, 3 * (12 + 1))  # matches scipy sosfiltfilt's default padlen for a 12th-order SOS
    if len(x) < min_samples:
        raise ValueError(
            f"Segment too short to sonify: {len(x)} samples "
            f"(need at least {min_samples} for filtering and STFT analysis). "
            "Choose a longer time range."
        )

    # === Bandpass filter (40–120 kHz, order 12, zero-phase) ===
    from squeak_peek.audio.filters import bandpass_filter_filtfilt

    x = bandpass_filter_filtfilt(x, fs, 40000, 120000, order=12)

    # === STFT analysis with Hamming window ===
    N = len(x)
    n_frames = (N - win_len) // win_hop + 1
    win = np.hamming(win_len)

    X = np.zeros((win_len, n_frames), dtype=complex)
    for k in range(n_frames):
        idx = slice(k * win_hop, k * win_hop + win_len)
        frame = x[idx] * win
        X[:, k] = np.fft.fft(frame)

    # === Pitch shift ===
    R_pitch = 2 ** (semitones / 12)
    t_orig = np.arange(n_frames, dtype=float)
    t_new = np.arange(0, n_frames, 1 / R_pitch, dtype=float)

    # Interpolate magnitude (absolute value of complex spectrum)
    Xmag = np.abs(X)  # shape (win_len, n_frames)
    f_mag = interp1d(
        t_orig, Xmag, axis=1, kind="linear", fill_value="extrapolate"
    )
    Xmag_pitch = f_mag(t_new)  # shape (win_len, len(t_new))

    # Interpolate phase difference (frame-to-frame phase differences)
    phase = np.angle(X)  # shape (win_len, n_frames)
    dphase = np.diff(phase, axis=1)  # shape (win_len, n_frames-1)

    f_dphase = interp1d(
        t_orig[1:], dphase, axis=1, kind="linear", fill_value="extrapolate"
    )
    t_new_mid = t_new[1:]  # Skip first frame (t=0)
    dphase_interp = f_dphase(t_new_mid)  # shape (win_len, len(t_new)-1)

    # Reconstruct phase by cumulative sum of interpolated differences
    # phaseN = [phase(:,1), repmat(phase(:,1),1,size(dPint,2)) + cumsum(dPint,2)]
    phase_new = np.zeros((win_len, len(t_new)), dtype=float)
    phase_new[:, 0] = phase[:, 0]
    phase_new[:, 1:] = phase[:, 0:1] + np.cumsum(dphase_interp, axis=1)

    # Reconstruct complex spectrum and inverse-FFT
    Yp = Xmag_pitch * np.exp(1j * phase_new)

    # === Overlap-add synthesis after pitch shift ===
    n_fp = Yp.shape[1]
    y_pitch_len = (n_fp - 1) * win_hop + win_len
    y_pitch = np.zeros(y_pitch_len, dtype=float)

    for k in range(n_fp):
        idx = slice(k * win_hop, k * win_hop + win_len)
        seg = np.real(np.fft.ifft(Yp[:, k])) * win
        y_pitch[idx] += seg

    # === Time stretch (repeat STFT/resample/synthesize on y_pitch) ===
    N2 = len(y_pitch)
    n_fr2 = (N2 - win_len) // win_hop + 1

    X2 = np.zeros((win_len, n_fr2), dtype=complex)
    for k in range(n_fr2):
        idx = slice(k * win_hop, k * win_hop + win_len)
        X2[:, k] = np.fft.fft(y_pitch[idx] * win)

    # Resample frame timeline by slowdown factor
    t_orig2 = np.arange(n_fr2, dtype=float)
    t_new2 = np.arange(0, n_fr2, 1 / slowdown, dtype=float)

    # Interpolate magnitude
    Xmag2 = np.abs(X2)
    f_mag2 = interp1d(
        t_orig2, Xmag2, axis=1, kind="linear", fill_value="extrapolate"
    )
    Xmag2_stretch = f_mag2(t_new2)

    # Interpolate phase difference
    phase2 = np.angle(X2)
    dphase2 = np.diff(phase2, axis=1)

    f_dphase2 = interp1d(
        t_orig2[1:],
        dphase2,
        axis=1,
        kind="linear",
        fill_value="extrapolate",
    )
    t_new2_mid = t_new2[1:]
    dphase2_interp = f_dphase2(t_new2_mid)

    # Reconstruct phase
    phase_new2 = np.zeros((win_len, len(t_new2)), dtype=float)
    phase_new2[:, 0] = phase2[:, 0]
    phase_new2[:, 1:] = phase2[:, 0:1] + np.cumsum(dphase2_interp, axis=1)

    # Reconstruct complex spectrum
    Y2 = Xmag2_stretch * np.exp(1j * phase_new2)

    # === Overlap-add synthesis after time stretch ===
    n_fn = Y2.shape[1]
    y_out_len = (n_fn - 1) * win_hop + win_len
    y_out = np.zeros(y_out_len, dtype=float)

    for k in range(n_fn):
        idx = slice(k * win_hop, k * win_hop + win_len)
        seg = np.real(np.fft.ifft(Y2[:, k])) * win
        y_out[idx] += seg

    # === Normalize output ===
    signal_out = y_out / (np.max(np.abs(y_out)) + np.finfo(float).eps)

    return signal_out

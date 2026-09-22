"""
Spectrogram-to-image conversion shared by CNN training and inference.

Faster R-CNN operates on images, not raw spectrogram-dB arrays, so this
module turns an audio window into a normalized [0, 1] float32 image plus
the frequency/time axes needed to map pixel-space boxes back to
seconds/Hz.
"""

from __future__ import annotations

import numpy as np

from squeak_peek.audio.filters import band_restrict, compute_stft


def signal_to_image(
    signal: np.ndarray,
    fs: int,
    fcut_min: float,
    fcut_max: float,
    *,
    segment_length: int = 1024,
    overlap_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an audio window into a normalized spectrogram image.

    Returns
    -------
    image : np.ndarray, shape (F, T), float32 in [0, 1] — row 0 is the
            lowest frequency bin (fcut_min), consistent with `freqs`.
    freqs : np.ndarray, shape (F,) — Hz, ascending, matches image rows.
    times : np.ndarray, shape (T,) — seconds (window-relative), matches
            image columns.
    """
    f, t, Sxx_dB = compute_stft(signal, fs, segment_length, overlap_factor)
    f_band, Sxx_band = band_restrict(f, Sxx_dB, fcut_min, fcut_max)

    if Sxx_band.size == 0:
        return np.zeros((1, len(t)), dtype=np.float32), f_band, t

    # Fixed dB range rather than per-tile min/max: per-tile normalization
    # would make the same call look different in a quiet vs. loud window,
    # which the network should not have to learn around.
    lo, hi = -80.0, 0.0
    image = np.clip((Sxx_band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return image, f_band, t


def freq_to_row(freqs: np.ndarray, freq_hz: float) -> int:
    """Map a frequency (Hz) to the nearest row index in an image from signal_to_image."""
    return int(np.clip(np.searchsorted(freqs, freq_hz), 0, len(freqs) - 1))


def time_to_col(times: np.ndarray, time_s: float) -> int:
    """Map a time (s, window-relative) to the nearest column index."""
    return int(np.clip(np.searchsorted(times, time_s), 0, len(times) - 1))

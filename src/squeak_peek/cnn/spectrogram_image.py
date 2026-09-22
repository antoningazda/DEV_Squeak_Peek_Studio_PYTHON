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
    dynamic_range_db: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an audio window into a normalized spectrogram image.

    Levels are expressed as dB above the tile's own noise floor (its median
    in-band level), then scaled over `dynamic_range_db`. An absolute dB
    window cannot work here: recording gain varies between rigs and
    datasets, so the same call sits at a different absolute level in each
    (real USVSEG recordings land near -120..-60 dB, a lab with hotter gain
    would land elsewhere). Referencing the median rather than the peak keeps
    the mapping stable whether or not a loud call is in the tile.

    Returns
    -------
    image : np.ndarray, shape (F, T), float32 in [0, 1] — row 0 is the
            lowest frequency bin (fcut_min), consistent with `freqs`.
    freqs : np.ndarray, shape (F,) — Hz, ascending, matches image rows.
    times : np.ndarray, shape (T,) — seconds (window-relative), matches
            image columns.
    """
    # Scale to unit RMS first. compute_stft floors its log at +1e-12, which
    # on raw USV recordings (in-band levels near -120 dB) would clamp the
    # noise floor away; unit RMS keeps the power spectrum clear of that
    # epsilon. The median subtraction below cancels this scaling, so it does
    # not change the resulting image, only its numerical conditioning.
    x = np.asarray(signal, dtype=np.float64).ravel()
    rms = float(np.sqrt(np.mean(x * x)))
    if rms > 0:
        x = x / rms

    f, t, Sxx_dB = compute_stft(x, fs, segment_length, overlap_factor)
    f_band, Sxx_band = band_restrict(f, Sxx_dB, fcut_min, fcut_max)

    if Sxx_band.size == 0:
        return np.zeros((1, len(t)), dtype=np.float32), f_band, t

    noise_floor = float(np.median(Sxx_band))
    image = np.clip((Sxx_band - noise_floor) / dynamic_range_db, 0.0, 1.0).astype(np.float32)
    return image, f_band, t


def freq_to_row(freqs: np.ndarray, freq_hz: float) -> int:
    """Map a frequency (Hz) to the nearest row index in an image from signal_to_image."""
    return int(np.clip(np.searchsorted(freqs, freq_hz), 0, len(freqs) - 1))


def time_to_col(times: np.ndarray, time_s: float) -> int:
    """Map a time (s, window-relative) to the nearest column index."""
    return int(np.clip(np.searchsorted(times, time_s), 0, len(times) - 1))

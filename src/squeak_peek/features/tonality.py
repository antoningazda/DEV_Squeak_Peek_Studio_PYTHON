"""
Tonality scoring for candidate USV detections.

A rodent USV is a narrowband, frequency-modulated whistle: within any one
short frame, nearly all of its energy sits close to a single peak
frequency. The false positives that dominate the classical detectors'
output — cage knocks, bedding rustle, scratching — are broadband, with
energy smeared across the band. Scoring that difference lets a detector
keep its recall while dropping much of its noise.
"""

from __future__ import annotations

import numpy as np

from squeak_peek.audio.filters import band_restrict


def tonality_score(
    segment: np.ndarray,
    fs: int,
    *,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    segment_length: int = 256,
    overlap_factor: float = 0.5,
    peak_bandwidth_hz: float = 5_000.0,
) -> float:
    """
    Energy concentration around each frame's peak frequency, averaged over
    frames and weighted by frame energy.

    Returns a value in [0, 1] — near 1 for a pure tone, near the
    bandwidth's share of the band for broadband noise — or NaN when the
    segment is too short or carries no in-band energy.
    """
    from scipy.signal import spectrogram as _spectrogram

    x = np.asarray(segment, dtype=np.float64).ravel()
    if len(x) < segment_length or fs <= 0:
        return float("nan")

    x = x - np.mean(x)
    noverlap = int(segment_length * overlap_factor)
    freqs, _t, power = _spectrogram(
        x, fs=fs, window="hann", nperseg=segment_length, noverlap=noverlap, scaling="density"
    )
    freqs, power = band_restrict(freqs, power, fcut_min, fcut_max)
    if power.size == 0:
        return float("nan")

    frame_energy = power.sum(axis=0)
    total = frame_energy.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")

    peak_rows = np.argmax(power, axis=0)
    peak_freqs = freqs[peak_rows]
    near_peak = np.abs(freqs[:, None] - peak_freqs[None, :]) <= peak_bandwidth_hz

    concentration = np.where(frame_energy > 0, (power * near_peak).sum(axis=0) / np.maximum(frame_energy, 1e-30), 0.0)
    return float(np.sum(concentration * frame_energy) / total)

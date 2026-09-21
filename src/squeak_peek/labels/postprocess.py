"""
Post-processing operations on label lists.

Ported from MATLAB functions:
  - mergeCloseLabels.m → merge_close_labels()
  - removeShortLabels.m → remove_short_labels()
  - padLabels.m → pad_labels()
  - filterLowCentroidLabels.m → filter_low_centroid_labels()
  - adaptiveStrongestFilter.m → adaptive_strongest_filter()
"""

from __future__ import annotations

import numpy as np

from squeak_peek.labels.model import Label


def merge_close_labels(labels: list[Label], max_gap: float) -> list[Label]:
    """
    Merge consecutive labels separated by a gap smaller than max_gap.

    Useful for cleaning up label outputs with small temporal interruptions.

    Args:
        labels: List of Label objects (should be sorted by time)
        max_gap: Maximum time gap to merge (seconds)

    Returns:
        List with merged labels
    """
    if not labels:
        return []

    merged: list[Label] = [labels[0]]

    for i in range(1, len(labels)):
        gap = labels[i].start_time - merged[-1].end_time
        if gap < max_gap:
            # Merge: extend the end time of the last label
            last = merged[-1]
            merged[-1] = type(last)(
                start_time=last.start_time,
                end_time=labels[i].end_time,
                label=last.label,
                start_frequency=last.start_frequency,
                end_frequency=labels[i].end_frequency,
                start_index=last.start_index,
                stop_index=labels[i].stop_index,
            )
        else:
            # Keep separate
            merged.append(labels[i])

    return merged


def remove_short_labels(labels: list[Label], min_duration: float) -> list[Label]:
    """
    Remove labels shorter than min_duration.

    Args:
        labels: List of Label objects
        min_duration: Minimum allowed duration (seconds)

    Returns:
        Filtered list containing only labels with duration >= min_duration
    """
    return [lbl for lbl in labels if lbl.duration >= min_duration]


def pad_labels(
    labels: list[Label],
    padding_sec: float,
    total_duration_sec: float | None = None,
) -> list[Label]:
    """
    Add a fixed left/right margin to each label, clamped to [0, total_duration_sec].

    Args:
        labels: List of Label objects
        padding_sec: Padding amount in seconds (must be >= 0)
        total_duration_sec: Total signal duration for clamping. If None, no upper clamp.

    Returns:
        List of padded labels
    """
    if not labels or padding_sec <= 0:
        return labels

    if total_duration_sec is None:
        total_duration_sec = float("inf")

    padded = []
    for lbl in labels:
        new_start = max(0.0, lbl.start_time - padding_sec)
        new_end = min(total_duration_sec, lbl.end_time + padding_sec)

        # Recalculate indices if they exist
        start_idx = lbl.start_index
        stop_idx = lbl.stop_index
        if lbl.start_index > 0 and lbl.stop_index > 0:
            # Rough scaling (assumes original indices were based on original times)
            # Note: this is approximate since we don't have original fs here
            fs_approx = (lbl.stop_index - lbl.start_index) / (
                lbl.end_time - lbl.start_time
            )
            start_idx = round(new_start * fs_approx)
            stop_idx = round(new_end * fs_approx)

        padded.append(
            type(lbl)(
                start_time=new_start,
                end_time=new_end,
                label=lbl.label,
                start_frequency=lbl.start_frequency,
                end_frequency=lbl.end_frequency,
                start_index=start_idx,
                stop_index=stop_idx,
            )
        )

    return padded


def filter_low_centroid_labels(
    labels: list[Label],
    usv: np.ndarray,
    fs: int,
    min_centroid_hz: float = 30000.0,
) -> list[Label]:
    """
    Reject detections whose energy centroid (in the 5–fs/2 kHz band) is too low.

    Designed to drop broadband low-frequency artefacts (bedding, cage knocks)
    that have energy mass below min_centroid_hz, while keeping real rat USVs
    (typically above ~30 kHz).

    Edge cases (empty/zero-length/NaN centroid) are KEPT.

    Args:
        labels: List of Label objects
        usv: Audio signal (mono, 1D array)
        fs: Sampling rate (Hz)
        min_centroid_hz: Minimum centroid frequency threshold (Hz, default 30 kHz)

    Returns:
        Filtered list
    """
    if not labels or len(usv) == 0 or fs <= 0:
        return labels

    keep = []
    nsamples = len(usv)

    for lbl in labels:
        # Extract segment
        i0 = max(0, round(lbl.start_time * fs))
        i1 = min(nsamples, round(lbl.end_time * fs))

        if i1 <= i0:
            # Keep zero-length segments (edge case)
            keep.append(lbl)
            continue

        seg = usv[i0:i1].astype(np.float64)
        seg = seg - np.mean(seg)

        # FFT
        nfft = max(256, 2 ** int(np.ceil(np.log2(len(seg)))))
        Y = np.fft.fft(seg, nfft)
        mag = np.abs(Y[: nfft // 2 + 1])
        freqs = np.arange(nfft // 2 + 1) * (fs / nfft)

        # Restrict to ultrasonic band (5 kHz – fs/2) to avoid DC/hum dominance
        band_mask = (freqs >= 5000) & (freqs <= fs / 2)
        m = mag[band_mask]
        ff = freqs[band_mask]

        denom = np.sum(m)
        if denom <= 0 or not np.isfinite(denom):
            # Keep edge case
            keep.append(lbl)
            continue

        centroid = np.sum(ff * m) / denom

        if np.isfinite(centroid) and centroid < min_centroid_hz:
            # Reject: centroid too low
            continue

        keep.append(lbl)

    return keep


def adaptive_strongest_filter(
    labels: list[Label],
    usv: np.ndarray,
    fs: int,
    relative_threshold: float = 0.20,
) -> list[Label]:
    """
    Drop detections weaker than relative_threshold * strongest detection RMS.

    Anchors threshold to the strongest event in the clip for robustness
    across recordings with varying SNR.

    Edge cases (empty slices, non-finite values) are KEPT.

    Args:
        labels: List of Label objects
        usv: Audio signal (mono, 1D array)
        fs: Sampling rate (Hz)
        relative_threshold: Threshold factor in (0, 1], default 0.20

    Returns:
        Filtered list
    """
    if len(labels) < 2 or len(usv) == 0 or fs <= 0:
        return labels

    # Compute RMS for each label
    nsamples = len(usv)
    power = np.full(len(labels), np.nan)

    for i, lbl in enumerate(labels):
        i0 = max(0, round(lbl.start_time * fs))
        i1 = min(nsamples, round(lbl.end_time * fs))

        if i1 <= i0:
            continue

        seg = usv[i0:i1].astype(np.float64)
        seg = seg - np.mean(seg)
        power[i] = np.sqrt(np.mean(seg * seg))  # RMS

    ref_power = np.nanmax(power)
    if not np.isfinite(ref_power) or ref_power <= 0:
        return labels

    threshold = relative_threshold * ref_power

    keep = []
    for i, lbl in enumerate(labels):
        if np.isfinite(power[i]) and power[i] < threshold:
            # Reject: too weak
            continue
        keep.append(lbl)

    return keep

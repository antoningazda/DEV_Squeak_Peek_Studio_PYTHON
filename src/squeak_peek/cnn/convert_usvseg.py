"""
Convert the USVSEG public dataset (Zenodo 3428024) into this repo's
label-file format so it can feed squeak_peek.cnn.train like any other
(wav, label) pair.

USVSEG ships each recording as `<name>.wav` + `<name>.csv`, where the CSV
is `start_time,end_time` per call (seconds, no header, no frequency
column — confirmed by inspecting the real files, not assumed). The
frequency box each call needs for Faster R-CNN training is derived here
from the actual signal in [start_time, end_time], not invented: we take
the frequency band around the window's peak energy that stays within
`dynamic_range_db` of that peak.

No torch dependency — this module only needs numpy/scipy, so USVSEG data
can be converted and inspected without installing the 'cnn' extra.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.audio.io import load_wav
from squeak_peek.labels.io import export_labels
from squeak_peek.labels.model import Label


def estimate_freq_band(
    signal_segment: np.ndarray,
    fs: int,
    fcut_min: float,
    fcut_max: float,
    *,
    segment_length: int = 512,
    overlap_factor: float = 0.75,
    dynamic_range_db: float = 15.0,
    min_band_hz: float = 2_000.0,
) -> tuple[float, float]:
    """
    Estimate a call's frequency band from real signal energy, not a fixed
    default: the contiguous band around the segment's peak that stays
    within `dynamic_range_db` of that peak.
    """
    if len(signal_segment) < segment_length:
        return fcut_min, fcut_max

    f, _t, Sxx_dB = compute_stft(signal_segment, fs, segment_length, overlap_factor)
    f_band, Sxx_band = band_restrict(f, Sxx_dB, fcut_min, fcut_max)
    if Sxx_band.size == 0:
        return fcut_min, fcut_max

    profile = Sxx_band.max(axis=1)  # peak dB per frequency bin over the whole segment
    peak_idx = int(np.argmax(profile))
    threshold = profile[peak_idx] - dynamic_range_db
    above = profile >= threshold

    lo = hi = peak_idx
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1

    f_lo, f_hi = float(f_band[lo]), float(f_band[hi])
    if f_hi - f_lo < min_band_hz:
        center = (f_hi + f_lo) / 2.0
        f_lo = max(fcut_min, center - min_band_hz / 2.0)
        f_hi = min(fcut_max, center + min_band_hz / 2.0)
    return f_lo, f_hi


def convert_usvseg_csv(
    csv_path: str | Path,
    wav_path: str | Path,
    *,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    pad_s: float = 0.002,
) -> list[Label]:
    """Read a USVSEG (start_time,end_time) CSV and return boxed Labels."""
    signal, fs = load_wav(wav_path)

    labels: list[Label] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                start_time, end_time = float(row[0]), float(row[1])
            except ValueError:
                continue
            if end_time <= start_time:
                continue

            seg_start = max(0, round((start_time - pad_s) * fs))
            seg_end = min(len(signal), round((end_time + pad_s) * fs))
            if seg_end - seg_start < 8:
                continue

            f_lo, f_hi = estimate_freq_band(signal[seg_start:seg_end], fs, fcut_min, fcut_max)
            labels.append(
                Label(
                    start_time=start_time,
                    end_time=end_time,
                    label="USV",
                    start_frequency=f_lo,
                    end_frequency=f_hi,
                    start_index=round(start_time * fs),
                    stop_index=round(end_time * fs),
                )
            )
    return labels


def convert_usvseg_dir(
    src_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
) -> list[tuple[Path, Path]]:
    """
    Convert every matching `<name>.wav` + `<name>.csv` pair in src_dir
    (USVSEG's per-species zip layout, extracted) into a label file next
    to the WAV (or in out_dir if given).

    Returns the (wav_path, label_path) pairs, ready for cnn.train.train_cnn.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir) if out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for wav_path in sorted(src_dir.rglob("*.wav")):
        csv_path = wav_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        labels = convert_usvseg_csv(csv_path, wav_path, fcut_min=fcut_min, fcut_max=fcut_max)

        label_path = (out_dir / f"{wav_path.stem}_labels.txt") if out_dir else wav_path.with_name(f"{wav_path.stem}_labels.txt")
        export_labels(label_path, labels)
        pairs.append((wav_path, label_path))

    return pairs

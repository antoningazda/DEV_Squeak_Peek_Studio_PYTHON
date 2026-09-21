"""
Label I/O: import and export of USV event annotations.

Implements the canonical 2-line-per-label text format:
  Line 1: StartTime    EndTime     Label
  Line 2: \            StartFreq   EndFreq

Ported from MATLAB:
  - importLabels.m → import_labels()
  - exportLabels.m → export_labels()
  - exportLabelsDetector.m → export_labels_detector()
"""

from __future__ import annotations

from pathlib import Path

from squeak_peek.labels.model import Label


def import_labels(path: str | Path, fs: int | None = None) -> list[Label]:
    """
    Load labels from a 2-line-per-label text file.

    Each label occupies two lines:
      Line 1: StartTime    EndTime     Label
      Line 2: \\           StartFreq   EndFreq

    Args:
        path: Path to the label file
        fs: Sampling rate (Hz). If provided, converts times to sample indices.

    Returns:
        List of Label objects
    """
    labels: list[Label] = []

    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line1 = lines[i].strip()
        i += 1

        # Skip empty lines and comments
        if not line1 or line1.startswith("#"):
            continue

        # Parse line 1: StartTime EndTime Label
        parts1 = line1.split("\t")
        if len(parts1) < 3:
            continue

        try:
            start_time = float(parts1[0])
            end_time = float(parts1[1])
            label_name = parts1[2]

            # Parse line 2: \ StartFreq EndFreq (if present)
            start_freq = 0.0
            end_freq = 0.0
            if i < len(lines):
                line2 = lines[i].strip()
                if line2.startswith("\\") or line2.startswith("backslash"):
                    parts2 = line2.split("\t")
                    if len(parts2) >= 3:
                        try:
                            start_freq = float(parts2[1])
                            end_freq = float(parts2[2])
                        except ValueError:
                            pass
                    i += 1

            # Calculate sample indices
            start_index = 0
            stop_index = 0
            if fs is not None and fs > 0:
                start_index = round(start_time * fs)
                stop_index = round(end_time * fs)

            labels.append(
                Label(
                    start_time=start_time,
                    end_time=end_time,
                    label=label_name,
                    start_frequency=start_freq,
                    end_frequency=end_freq,
                    start_index=start_index,
                    stop_index=stop_index,
                )
            )
        except (ValueError, IndexError):
            continue

    return labels


def export_labels(path: str | Path, labels: list[Label]) -> None:
    """
    Export labels to the canonical 2-line-per-label text format.

    Args:
        path: Output file path
        labels: List of Label objects to export
    """
    with open(path, "w", encoding="utf-8") as fh:
        for lbl in labels:
            # Line 1: StartTime EndTime Label
            fh.write(f"{lbl.start_time:.6f}\t{lbl.end_time:.6f}\t{lbl.label}\n")
            # Line 2: \ StartFreq EndFreq
            fh.write(f"\\\t{lbl.start_frequency:.6f}\t{lbl.end_frequency:.6f}\n")


def export_labels_detector(path: str | Path, labels: list[Label]) -> None:
    """
    Export detected labels in simplified format (detector output).

    The label type is fixed as 'd' (detected), and frequency bounds are zeroed.

    Args:
        path: Output file path
        labels: List of Label objects to export
    """
    with open(path, "w", encoding="utf-8") as fh:
        for lbl in labels:
            # Line 1: StartTime EndTime d
            fh.write(f"{lbl.start_time:.6f}\t{lbl.end_time:.6f}\td\n")
            # Line 2: \ 0.0 0.0
            fh.write("\\\t0.000000\t0.000000\n")

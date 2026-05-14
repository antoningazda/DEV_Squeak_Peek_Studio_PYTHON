from __future__ import annotations

from pathlib import Path

from squeak_peek.labels.model import Label


def load_labels(path: Path | str) -> list[Label]:
    """Load labels from a tab-separated text file (onset  offset  [label]).

    Handles both 2-column (start, end) and 3-column (start, end, label)
    formats produced by MATLAB exportLabels and Audacity.
    """
    labels: list[Label] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()  # fall back to whitespace split
            if len(parts) < 2:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                tag = parts[2].strip() if len(parts) > 2 else ""
                labels.append(
                    Label(
                        start_time=start,
                        end_time=end,
                        label=tag,
                    )
                )
            except (ValueError, IndexError):
                continue
    return labels


def save_labels(path: Path | str, labels: list[Label]) -> None:
    """Write labels to a tab-separated text file (onset  offset  label)."""
    with open(path, "w", encoding="utf-8") as fh:
        for lbl in labels:
            fh.write(f"{lbl.start_time:.6f}\t{lbl.end_time:.6f}\t{lbl.label}\n")

"""
Label I/O wrapper for GUI.

Delegates to the canonical label I/O implementation in squeak_peek.labels.io.
This module maintains the existing function names for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

from squeak_peek.labels.io import export_labels, import_labels
from squeak_peek.labels.model import Label


def load_labels(path: Path | str) -> list[Label]:
    """Load labels from a text file.

    Delegates to the canonical import_labels function.
    """
    return import_labels(path, fs=None)


def save_labels(path: Path | str, labels: list[Label]) -> None:
    """Write labels to a text file.

    Delegates to the canonical export_labels function.
    """
    export_labels(path, labels)

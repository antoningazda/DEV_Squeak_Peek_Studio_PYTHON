from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from squeak_peek.audio.io import load_wav
from squeak_peek.config import AppSettings
from squeak_peek.labels.model import Label


class AppState(QObject):
    wav_loaded = pyqtSignal()
    labels_changed = pyqtSignal()
    settings_changed = pyqtSignal()
    segment_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.samples: np.ndarray | None = None
        self.fs: int = 250_000
        self.wav_path: Path | None = None
        self.detected_labels: list[Label] = []
        self.reference_labels: list[Label] = []
        self.settings: AppSettings = AppSettings.defaults()
        self.segment_start: float = 0.0

    @property
    def duration(self) -> float:
        if self.samples is None:
            return 0.0
        return float(len(self.samples)) / self.fs

    @property
    def segment_length(self) -> float:
        return self.settings.visualization.segment_length_seconds

    @property
    def segment_end(self) -> float:
        return min(self.segment_start + self.segment_length, self.duration)

    def load_wav(self, path: Path | str) -> None:
        self.samples, self.fs = load_wav(path)
        self.wav_path = Path(path)
        self.segment_start = 0.0
        self.wav_loaded.emit()
        self.segment_changed.emit()

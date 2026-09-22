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
    video_loaded = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.samples: np.ndarray | None = None
        self.fs: int = 250_000
        self.wav_path: Path | None = None
        self.detected_labels: list[Label] = []
        self.reference_labels: list[Label] = []
        self.settings: AppSettings = AppSettings.defaults()
        self.segment_start: float = 0.0

        # Video import / sync (see squeak_peek.video)
        self.video_path: Path | None = None
        self.video_sync_offset: float = 0.0   # seconds to ADD to a WAV time to get video time
        self.video_sync_method: str = ""
        self.sonified_track: tuple[np.ndarray, int] | None = None  # (samples, fs)

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
        # A previously imported video/sync/sonification belonged to the old
        # WAV — stale data here would silently mis-sync the Video tab.
        self.video_path = None
        self.video_sync_offset = 0.0
        self.video_sync_method = ""
        self.sonified_track = None
        self.wav_loaded.emit()
        self.segment_changed.emit()

    def load_video(self, path: Path | str) -> None:
        """Import a behavior video, sync it to the loaded WAV's timeline,
        and cache a real-time-locked sonified version of the whole WAV.

        Raises whatever ``extract_audio_track``/``compute_sync_offset``
        raise (e.g. no audio track in the video, no WAV loaded yet) —
        callers show these to the user rather than silently swallowing them.
        """
        if self.samples is None:
            raise RuntimeError("Load a WAV recording before importing a video.")

        from squeak_peek.audio.sonify import sonify_full_track
        from squeak_peek.video.sync import compute_sync_offset, extract_audio_track

        video_samples, video_fs = extract_audio_track(path)
        sync_result = compute_sync_offset(self, video_samples, video_fs)

        self.video_path = Path(path)
        self.video_sync_offset = sync_result.offset
        self.video_sync_method = sync_result.method
        self.sonified_track = sonify_full_track(
            self.samples,
            self.fs,
            semitones=self.settings.visualization.sonification_st,
            denoise=self.settings.visualization.sonification_denoise,
        )
        self.video_loaded.emit()

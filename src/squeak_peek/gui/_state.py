from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from squeak_peek.audio.io import load_wav
from squeak_peek.config import AppSettings
from squeak_peek.labels.model import Label

# Cap on how many detected-label snapshots the undo stack keeps. A snapshot
# is a deep copy of the whole label list, so this bounds memory rather than
# letting a long Label Edit session grow it unboundedly.
_MAX_UNDO_DEPTH = 50


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
        # Where detected_labels last came from, so Data Input can show which
        # file is in memory after Detection or Classification replaces them.
        self.detected_labels_path: Path | None = None
        self.reference_labels: list[Label] = []
        self.settings: AppSettings = AppSettings.defaults()
        self.segment_start: float = 0.0

        # Video import / sync (see squeak_peek.video)
        self.video_path: Path | None = None
        self.video_sync_offset: float = 0.0   # seconds to ADD to a WAV time to get video time
        self.video_sync_method: str = ""
        self.sonified_track: tuple[np.ndarray, int] | None = None  # (samples, fs)

        # Undo/redo for detected_labels edits (accept/reject, drag-resize,
        # manual create/delete). Callers that mutate detected_labels call
        # snapshot_labels() first, which is cheap (a deep copy of a list of
        # small dataclasses) compared to re-running a detector.
        self._undo_stack: list[list[Label]] = []
        self._redo_stack: list[list[Label]] = []

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
        self.segment_start = self.settings.visualization.segment_start_seconds
        # A previously imported video/sync/sonification belonged to the old
        # WAV — stale data here would silently mis-sync the Video tab.
        self.video_path = None
        self.video_sync_offset = 0.0
        self.video_sync_method = ""
        self.sonified_track = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.wav_loaded.emit()
        self.segment_changed.emit()

    # ── Label edit undo/redo ────────────────────────────────────────────

    def snapshot_labels(self) -> None:
        """Push the current detected_labels onto the undo stack. Call this
        right before any in-place edit (accept/reject, drag-resize, manual
        create/delete) so the edit can be undone."""
        self._undo_stack.append(copy.deepcopy(self.detected_labels))
        if len(self._undo_stack) > _MAX_UNDO_DEPTH:
            del self._undo_stack[0]
        self._redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(copy.deepcopy(self.detected_labels))
        self.detected_labels = self._undo_stack.pop()
        self.labels_changed.emit()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(copy.deepcopy(self.detected_labels))
        self.detected_labels = self._redo_stack.pop()
        self.labels_changed.emit()

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

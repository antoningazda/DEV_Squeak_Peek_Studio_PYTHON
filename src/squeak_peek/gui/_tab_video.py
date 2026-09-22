from __future__ import annotations

from pathlib import Path

import sounddevice as sd
from PyQt6.QtCore import QThread, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from squeak_peek.video.export import VideoExportWorker

from . import _theme as t
from ._spectrogram_widget import _get_color_for_name
from ._state import AppState


def _format_ms(ms: int) -> str:
    total_s = max(0, ms) // 1000
    return f"{total_s // 60:02d}:{total_s % 60:02d}"


class _ImportWorker(QThread):
    """Runs ``AppState.load_video`` (ffmpeg extraction + sync + full-track
    sonification — a few seconds of work) off the UI thread."""

    succeeded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, state: AppState, path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._path = path

    def run(self) -> None:
        try:
            self._state.load_video(self._path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.succeeded.emit()


class _LabelStrip(QWidget):
    """Compact horizontal timeline: label spans plus a moving playhead,
    drawn in the VIDEO's time domain (labels are offset-shifted from
    WAV time)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(28)
        self.setMaximumHeight(28)
        self._duration = 0.0
        self._spans: list[tuple[float, float, QColor]] = []
        self._position = 0.0

    def set_data(self, duration_s: float, spans: list[tuple[float, float, QColor]]) -> None:
        self._duration = max(duration_s, 1e-6)
        self._spans = spans
        self.update()

    def set_position(self, position_s: float) -> None:
        self._position = position_s
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        painter = QPainter(self)
        try:
            w, h = self.width(), self.height()
            painter.fillRect(0, 0, w, h, QColor(t.SURFACE_MUTED))

            for start, end, color in self._spans:
                if end < 0 or start > self._duration:
                    continue
                x0 = max(0.0, start) / self._duration * w
                x1 = min(self._duration, end) / self._duration * w
                painter.fillRect(int(x0), 4, max(1, int(x1 - x0)), h - 8, color)

            px = int(self._position / self._duration * w)
            painter.setPen(QColor("red"))
            painter.drawLine(px, 0, px, h)
        finally:
            painter.end()


class VideoTab(QWidget):
    """Import a behavior video, auto-sync it to the loaded WAV, play it back
    with a continuous real-time-locked sonified soundtrack and a label
    timeline, and export a copy with that soundtrack muxed in."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        self._player = QMediaPlayer(self)
        self._video_widget = QVideoWidget()
        self._player.setVideoOutput(self._video_widget)
        # Deliberately no QAudioOutput attached: the video's own audio track
        # never plays back — the sonified WAV track plays instead, via
        # sounddevice, kept in lockstep with the player's position.

        self._import_worker: _ImportWorker | None = None
        self._export_worker: VideoExportWorker | None = None
        self._sd_playing = False
        self._scrub_was_playing = False
        self._spans_cache: list[tuple[float, float, QColor]] = []

        self._setup_ui()

        state.video_loaded.connect(self._on_video_loaded)
        state.wav_loaded.connect(self._on_wav_loaded)
        state.labels_changed.connect(self._on_labels_changed)
        t.signal.changed.connect(self._apply_theme)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)

    # ── UI ───────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        outer.setSpacing(t.SP_3)

        import_row = QHBoxLayout()
        self._import_btn = QPushButton("Import video…")
        self._import_btn.setToolTip(
            "Import a behavior video and auto-sync it to the loaded WAV's timeline "
            "(via an 'sk' label, or an auto-detected finger-snap transient)."
        )
        self._import_btn.clicked.connect(self._browse_video)
        import_row.addWidget(self._import_btn)
        self._video_path_label = QLabel("No video loaded")
        self._video_path_label.setWordWrap(True)
        import_row.addWidget(self._video_path_label, stretch=1)
        outer.addLayout(import_row)

        sync_row = QHBoxLayout()
        self._sync_label = QLabel("")
        self._sync_label.setWordWrap(True)
        sync_row.addWidget(self._sync_label, stretch=1)
        self._resync_btn = QPushButton("Re-sync")
        self._resync_btn.setToolTip("Re-run sync detection for the current video.")
        self._resync_btn.setEnabled(False)
        self._resync_btn.clicked.connect(self._resync)
        sync_row.addWidget(self._resync_btn)
        outer.addLayout(sync_row)

        self._video_widget.setMinimumHeight(360)
        outer.addWidget(self._video_widget, stretch=1)

        self._label_strip = _LabelStrip()
        outer.addWidget(self._label_strip)

        controls = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self._play_btn)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, 0)
        self._position_slider.sliderPressed.connect(self._on_slider_pressed)
        self._position_slider.sliderMoved.connect(self._on_slider_moved)
        self._position_slider.sliderReleased.connect(self._on_slider_released)
        controls.addWidget(self._position_slider, stretch=1)

        self._time_label = QLabel("00:00 / 00:00")
        controls.addWidget(self._time_label)
        outer.addLayout(controls)

        export_row = QHBoxLayout()
        export_row.addStretch()
        self._export_btn = QPushButton("Export video with sonified audio…")
        self._export_btn.setObjectName("primaryBtn")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        export_row.addWidget(self._export_btn)
        outer.addLayout(export_row)

        self._apply_theme()

    def _apply_theme(self) -> None:
        self._video_path_label.setStyleSheet(f"color: {t.TEXT_SECONDARY};")
        self._sync_label.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")
        self._label_strip.update()

    # ── Import / sync ────────────────────────────────────────────────────

    def _browse_video(self) -> None:
        if self._state.samples is None:
            QMessageBox.warning(
                self, "No WAV loaded",
                "Load a WAV recording (Data Input tab) before importing a video.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open behavior video", "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)",
        )
        if not path:
            return
        self._start_import(path)

    def _resync(self) -> None:
        if self._state.video_path is None:
            return
        self._start_import(str(self._state.video_path))

    def _start_import(self, path: str) -> None:
        self._import_btn.setEnabled(False)
        self._resync_btn.setEnabled(False)
        self._sync_label.setText("Syncing and sonifying — this can take a few seconds…")

        self._import_worker = _ImportWorker(self._state, path, self)
        self._import_worker.succeeded.connect(self._on_import_succeeded)
        self._import_worker.failed.connect(self._on_import_failed)
        self._import_worker.start()

    def _on_import_succeeded(self) -> None:
        self._import_btn.setEnabled(True)
        self._resync_btn.setEnabled(True)

    def _on_import_failed(self, message: str) -> None:
        self._import_btn.setEnabled(True)
        self._resync_btn.setEnabled(self._state.video_path is not None)
        self._sync_label.setText("")
        QMessageBox.critical(self, "Video import failed", message)

    def _on_video_loaded(self) -> None:
        s = self._state
        self._video_path_label.setText(str(s.video_path))
        self._sync_label.setText(
            f"Sync offset: {s.video_sync_offset:+.3f} s  ({s.video_sync_method})"
        )
        self._export_btn.setEnabled(True)
        self._play_btn.setEnabled(True)
        self._spans_cache = self._compute_spans()
        # Refresh immediately with whatever duration is already known — for a
        # re-sync (same source), setSource() below is a no-op and won't fire
        # durationChanged, so this is the only place the new offset reaches
        # the strip. For a genuinely new video, durationChanged corrects
        # this again once the media's metadata has loaded.
        self._label_strip.set_data(self._player.duration() / 1000.0, self._spans_cache)
        self._player.setSource(QUrl.fromLocalFile(str(s.video_path)))

    def _on_wav_loaded(self) -> None:
        self._stop_sonified_playback()
        self._player.stop()
        self._player.setSource(QUrl())
        self._video_path_label.setText("No video loaded")
        self._sync_label.setText("")
        self._resync_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._play_btn.setEnabled(False)
        self._position_slider.setRange(0, 0)
        self._time_label.setText("00:00 / 00:00")
        self._spans_cache = []
        self._label_strip.set_data(0.0, [])

    def _on_labels_changed(self) -> None:
        if self._state.video_path is None:
            return
        self._spans_cache = self._compute_spans()
        self._label_strip.set_data(self._player.duration() / 1000.0, self._spans_cache)

    def _compute_spans(self) -> list[tuple[float, float, QColor]]:
        s = self._state
        vis = s.settings.visualization
        offset = s.video_sync_offset
        det_rgba = _get_color_for_name(vis.label_color) or (0.0, 1.0, 1.0, 1.0)
        ref_rgba = _get_color_for_name(vis.reference_label_color) or (1.0, 1.0, 1.0, 1.0)
        det_color = QColor.fromRgbF(*det_rgba)
        ref_color = QColor.fromRgbF(*ref_rgba)

        spans: list[tuple[float, float, QColor]] = []
        for lbl in s.reference_labels:
            spans.append((lbl.start_time + offset, lbl.end_time + offset, ref_color))
        for lbl in s.detected_labels:
            spans.append((lbl.start_time + offset, lbl.end_time + offset, det_color))
        return spans

    # ── Playback ─────────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("⏸ Pause")
            self._start_sonified_playback()
        else:
            self._play_btn.setText("▶ Play")
            self._stop_sonified_playback()

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._stop_sonified_playback()
            self._play_btn.setText("▶ Play")

    def _start_sonified_playback(self) -> None:
        s = self._state
        if s.sonified_track is None:
            return
        sonified, sfs = s.sonified_track
        video_t = self._player.position() / 1000.0
        wav_t = video_t - s.video_sync_offset
        start_sample = max(0, int(round(wav_t * sfs)))
        if start_sample >= len(sonified):
            return
        sd.play(sonified[start_sample:], samplerate=sfs)
        self._sd_playing = True

    def _stop_sonified_playback(self) -> None:
        if self._sd_playing:
            sd.stop()
            self._sd_playing = False

    def _on_position_changed(self, position_ms: int) -> None:
        if not self._position_slider.isSliderDown():
            self._position_slider.setValue(position_ms)
        self._label_strip.set_position(position_ms / 1000.0)
        self._time_label.setText(f"{_format_ms(position_ms)} / {_format_ms(self._player.duration())}")

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._position_slider.setRange(0, duration_ms)
        self._label_strip.set_data(duration_ms / 1000.0, self._spans_cache)
        self._time_label.setText(f"{_format_ms(self._player.position())} / {_format_ms(duration_ms)}")

    def _on_slider_pressed(self) -> None:
        self._scrub_was_playing = self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self._stop_sonified_playback()

    def _on_slider_moved(self, value: int) -> None:
        self._player.setPosition(value)  # scrub video live; sonified audio stays stopped

    def _on_slider_released(self) -> None:
        self._player.setPosition(self._position_slider.value())
        if self._scrub_was_playing:
            self._start_sonified_playback()

    # ── Export ───────────────────────────────────────────────────────────

    def _export(self) -> None:
        s = self._state
        if s.video_path is None or s.sonified_track is None:
            return
        default_name = str(Path(s.video_path).with_name(Path(s.video_path).stem + "_sonified.mp4"))
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export synced video", default_name, "MP4 video (*.mp4)"
        )
        if not out_path:
            return

        sonified, sfs = s.sonified_track
        self._export_btn.setEnabled(False)
        self._export_btn.setText("Exporting…")

        self._export_worker = VideoExportWorker(
            s.video_path, sonified, sfs, s.video_sync_offset, out_path, self
        )
        self._export_worker.succeeded.connect(self._on_export_succeeded)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.start()

    def _on_export_succeeded(self) -> None:
        self._export_btn.setEnabled(True)
        self._export_btn.setText("Export video with sonified audio…")
        QMessageBox.information(self, "Export complete", "Video exported successfully.")

    def _on_export_failed(self, message: str) -> None:
        self._export_btn.setEnabled(True)
        self._export_btn.setText("Export video with sonified audio…")
        QMessageBox.critical(self, "Export failed", message)

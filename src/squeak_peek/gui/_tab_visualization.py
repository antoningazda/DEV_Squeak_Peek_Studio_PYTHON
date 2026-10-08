from __future__ import annotations

import time
from typing import Any

import sounddevice as sd
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from squeak_peek.audio.sonify import sonify_segment
from squeak_peek.labels.model import Label

from . import _shortcuts as shortcuts
from . import _theme as t
from ._spectrogram_widget import SpectrogramWidget
from ._state import AppState


class VisualizationTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()

        state.wav_loaded.connect(self._refresh)
        state.labels_changed.connect(self._refresh)
        state.segment_changed.connect(self._refresh)
        state.settings_changed.connect(self._on_settings_changed)
        t.signal.changed.connect(self._on_theme_changed)

        # Sonification playback state
        self._is_playing: bool = False
        self._playback_timer: QTimer | None = None
        self._playback_start_time: float = 0.0
        self._playback_duration: float = 0.0
        self._playback_start_cursor: float = 0.0  # original segment start time
        self._playback_ratio: float = 1.0  # ratio of original duration to stretched duration
        self._playback_line: Any = None  # reference to the cursor line on spec plot
        self._playback_wave_line: Any = None  # reference to the cursor line on wave plot

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(t.SP_2, t.SP_2, t.SP_2, t.SP_2)
        layout.setSpacing(t.SP_2)

        self._spec = SpectrogramWidget()
        layout.addWidget(self._spec, stretch=1)

        # ── Navigation bar ────────────────────────────────────────────────
        nav = QGroupBox()
        nav_row = QHBoxLayout(nav)
        nav_row.setSpacing(t.SP_3)

        start_lbl = QLabel("Start (s):")
        start_tip = "Start time of the segment currently displayed."
        start_lbl.setToolTip(start_tip)
        nav_row.addWidget(start_lbl)
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 99_999.0)
        self._start_spin.setDecimals(3)
        self._start_spin.setSingleStep(0.1)
        self._start_spin.setMinimumWidth(90)
        self._start_spin.setToolTip(start_tip)
        self._start_spin.valueChanged.connect(self._on_start_changed)
        nav_row.addWidget(self._start_spin)

        len_lbl = QLabel("Length (s):")
        len_tip = "Length of the segment currently displayed. Also set in Settings → Visualization."
        len_lbl.setToolTip(len_tip)
        nav_row.addWidget(len_lbl)
        self._len_spin = QDoubleSpinBox()
        self._len_spin.setRange(0.01, 60.0)
        self._len_spin.setDecimals(3)
        self._len_spin.setSingleStep(0.1)
        self._len_spin.setValue(self._state.segment_length)
        self._len_spin.setMinimumWidth(80)
        self._len_spin.setToolTip(len_tip)
        self._len_spin.valueChanged.connect(self._on_length_changed)
        nav_row.addWidget(self._len_spin)

        prev_btn = QPushButton("◀  Prev")
        prev_btn.setToolTip(
            f"Move one segment length backward "
            f"(shortcut: {shortcuts.get_shortcut('prev_segment').toString()})."
        )
        prev_btn.clicked.connect(self._prev_segment)
        nav_row.addWidget(prev_btn)

        next_btn = QPushButton("Next  ▶")
        next_btn.setToolTip(
            f"Move one segment length forward "
            f"(shortcut: {shortcuts.get_shortcut('next_segment').toString()})."
        )
        next_btn.clicked.connect(self._next_segment)
        nav_row.addWidget(next_btn)

        self._sonify_btn = QPushButton("🔊 Sonify")
        self._sonify_btn.setToolTip(
            "Play the current segment pitch-shifted and slowed down so ultrasonic calls "
            "become audible. Adjust the amount in Settings → Visualization."
        )
        self._sonify_btn.clicked.connect(self._on_sonify_clicked)
        nav_row.addWidget(self._sonify_btn)

        nav_row.addStretch()

        self._det_cb = QCheckBox("Detected labels")
        self._det_cb.setToolTip("Show/hide detected-label boxes on the spectrogram.")
        self._det_cb.setChecked(self._state.settings.visualization.show_labels)
        self._det_cb.stateChanged.connect(self._refresh)
        nav_row.addWidget(self._det_cb)

        self._ref_cb = QCheckBox("Reference labels")
        self._ref_cb.setToolTip("Show/hide reference-label boxes on the spectrogram.")
        self._ref_cb.setChecked(self._state.settings.visualization.show_reference_labels)
        self._ref_cb.stateChanged.connect(self._refresh)
        nav_row.addWidget(self._ref_cb)

        self._pitch_cb = QCheckBox("Pitch trace")
        self._pitch_cb.setToolTip("Show/hide the estimated pitch (fundamental frequency) overlay.")
        self._pitch_cb.setChecked(True)
        self._pitch_cb.stateChanged.connect(self._refresh)
        nav_row.addWidget(self._pitch_cb)

        layout.addWidget(nav)

        # Connect right-click on spectrogram to manual label creation
        self._spec.spectrogram_right_clicked.connect(self._on_spectrogram_right_clicked)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_start_changed(self, value: float) -> None:
        self._state.segment_start = value
        self._state.segment_changed.emit()

    def _on_length_changed(self, value: float) -> None:
        self._state.settings.visualization.segment_length_seconds = value
        self._state.segment_changed.emit()

    def _prev_segment(self) -> None:
        new_start = max(0.0, self._state.segment_start - self._state.segment_length)
        self._start_spin.setValue(new_start)

    def _next_segment(self) -> None:
        new_start = self._state.segment_start + self._state.segment_length
        if new_start < self._state.duration:
            self._start_spin.setValue(new_start)

    def _on_settings_changed(self) -> None:
        vis = self._state.settings.visualization
        self._det_cb.blockSignals(True)
        self._det_cb.setChecked(vis.show_labels)
        self._det_cb.blockSignals(False)

        self._ref_cb.blockSignals(True)
        self._ref_cb.setChecked(vis.show_reference_labels)
        self._ref_cb.blockSignals(False)

        self._refresh()

    def _on_theme_changed(self) -> None:
        self._spec.refresh_theme()
        self._refresh()

    def _on_sonify_clicked(self) -> None:
        """Handle Sonify button click: playback with moving cursor."""
        try:
            s = self._state
            if s.samples is None or s.fs is None:
                return
            vis = s.settings.visualization

            # Gather parameters (matching MATLAB SonifyButtonPushed)
            start_time = s.segment_start
            end_time = s.segment_end
            segment_length = end_time - start_time

            # Call sonify_segment — it returns audio already correctly
            # pitch-shifted/time-stretched/resampled, so it's played back
            # at exactly the sample rate it reports.
            sonified, target_fs = sonify_segment(
                audio=s.samples,
                start_time=start_time,
                end_time=end_time,
                fs=s.fs,
                semitones=vis.sonification_st,
                # None = slow down by exactly the pitch ratio, which needs no
                # phase vocoder and is therefore artefact-free.
                slowdown=(
                    None if vis.sonification_natural_speed
                    else vis.sonification_slowdown
                ),
                denoise=vis.sonification_denoise,
            )

            # Compute mapping ratio (original duration / stretched output duration)
            dur_out = len(sonified) / target_fs
            ratio = segment_length / dur_out

            # Disable sonify button during playback
            self._sonify_btn.setEnabled(False)

            # Start playback with sounddevice. sd.play() is fire-and-forget —
            # it returns None, not a stream handle — so "is playback active"
            # is tracked separately via self._is_playing rather than by
            # checking its return value.
            sd.play(sonified, samplerate=target_fs)
            self._is_playing = True
            self._playback_start_time = time.monotonic()
            self._playback_duration = dur_out
            self._playback_start_cursor = start_time
            self._playback_ratio = ratio

            # Start timer to update cursor line (every 30ms)
            if self._playback_timer is None:
                self._playback_timer = QTimer(self)
                self._playback_timer.timeout.connect(self._on_playback_tick)
            self._playback_timer.start(30)

        except Exception:
            import traceback
            traceback.print_exc()
            self._sonify_btn.setEnabled(True)

    def _on_playback_tick(self) -> None:
        """Update moving cursor line during playback."""
        try:
            if not self._is_playing or self._playback_timer is None:
                return

            elapsed = time.monotonic() - self._playback_start_time

            # Check if playback finished
            if elapsed >= self._playback_duration:
                self._stop_playback()
                return

            # Map elapsed time back to original segment time
            current_time = self._playback_start_cursor + elapsed * self._playback_ratio

            # Draw vertical line on both plots
            import pyqtgraph as pg
            pen = pg.mkPen("red", width=2)

            # Remove old lines
            if self._playback_line is not None:
                self._spec._spec_plot.removeItem(self._playback_line)
            if self._playback_wave_line is not None:
                self._spec._wave_plot.removeItem(self._playback_wave_line)

            # Create new lines at current position
            self._playback_line = pg.InfiniteLine(
                pos=current_time, angle=90, pen=pen, movable=False
            )
            self._spec._spec_plot.addItem(self._playback_line)

            self._playback_wave_line = pg.InfiniteLine(
                pos=current_time, angle=90, pen=pen, movable=False
            )
            self._spec._wave_plot.addItem(self._playback_wave_line)

        except Exception:
            import traceback
            traceback.print_exc()
            self._stop_playback()

    def _stop_playback(self) -> None:
        """Stop playback and clean up."""
        if self._playback_timer is not None:
            self._playback_timer.stop()
        if self._playback_line is not None:
            self._spec._spec_plot.removeItem(self._playback_line)
            self._playback_line = None
        if self._playback_wave_line is not None:
            self._spec._wave_plot.removeItem(self._playback_wave_line)
            self._playback_wave_line = None
        sd.stop()
        self._is_playing = False
        self._sonify_btn.setEnabled(True)

    def _on_spectrogram_right_clicked(self, clicked_time: float) -> None:
        """Handle right-click on spectrogram: create a manual label."""
        try:
            s = self._state
            vis = s.settings.visualization

            # Create label centered on clicked time
            label_duration = vis.manual_label_length
            start_time = clicked_time - label_duration / 2
            end_time = clicked_time + label_duration / 2

            # Create new label
            new_label = Label(
                start_time=start_time,
                end_time=end_time,
                label="md",
                detection_state="Accepted",
            )

            # Insert sorted by start_time
            s.detected_labels.append(new_label)
            s.detected_labels.sort(key=lambda lbl: lbl.start_time)

            # Emit signal to refresh
            s.labels_changed.emit()

        except Exception:
            import traceback
            traceback.print_exc()

    def _refresh(self) -> None:
        try:
            self._refresh_inner()
        except Exception:
            import traceback
            traceback.print_exc()

    def _refresh_inner(self) -> None:
        s = self._state
        if s.samples is None:
            return
        vis = s.settings.visualization

        # Sync spinboxes without triggering another segment_changed
        self._start_spin.blockSignals(True)
        self._start_spin.setMaximum(max(0.0, s.duration - 0.001))
        self._start_spin.setValue(s.segment_start)
        self._start_spin.blockSignals(False)

        self._len_spin.blockSignals(True)
        self._len_spin.setValue(s.segment_length)
        self._len_spin.blockSignals(False)

        self._spec.display(
            samples=s.samples,
            fs=s.fs,
            t_start=s.segment_start,
            t_end=s.segment_end,
            fmin_hz=vis.spectrogram_min_freq_hz,
            fmax_hz=vis.spectrogram_max_freq_hz,
            nperseg=vis.spectrogram_window,
            noverlap=vis.spectrogram_overlap,
            detected_labels=s.detected_labels,
            reference_labels=s.reference_labels,
            show_detected=self._det_cb.isChecked(),
            show_reference=self._ref_cb.isChecked(),
            show_pitch=self._pitch_cb.isChecked(),
            colormap_name=vis.colormap,
            detected_color_name=vis.label_color,
            reference_color_name=vis.reference_label_color,
        )

    # ── Keyboard shortcuts ────────────────────────────────────────────────

    def move_to_previous_segment(self) -> None:
        """Move to previous segment (keyboard shortcut handler)."""
        self._prev_segment()

    def move_to_next_segment(self) -> None:
        """Move to next segment (keyboard shortcut handler)."""
        self._next_segment()

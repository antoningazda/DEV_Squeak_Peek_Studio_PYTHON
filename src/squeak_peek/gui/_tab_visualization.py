from __future__ import annotations

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
        state.settings_changed.connect(self._refresh)
        t.signal.changed.connect(self._on_theme_changed)

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

        nav_row.addWidget(QLabel("Start (s):"))
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 99_999.0)
        self._start_spin.setDecimals(3)
        self._start_spin.setSingleStep(0.1)
        self._start_spin.setMinimumWidth(90)
        self._start_spin.valueChanged.connect(self._on_start_changed)
        nav_row.addWidget(self._start_spin)

        nav_row.addWidget(QLabel("Length (s):"))
        self._len_spin = QDoubleSpinBox()
        self._len_spin.setRange(0.01, 60.0)
        self._len_spin.setDecimals(3)
        self._len_spin.setSingleStep(0.1)
        self._len_spin.setValue(self._state.segment_length)
        self._len_spin.setMinimumWidth(80)
        self._len_spin.valueChanged.connect(self._on_length_changed)
        nav_row.addWidget(self._len_spin)

        prev_btn = QPushButton("◀  Prev")
        prev_btn.clicked.connect(self._prev_segment)
        nav_row.addWidget(prev_btn)

        next_btn = QPushButton("Next  ▶")
        next_btn.clicked.connect(self._next_segment)
        nav_row.addWidget(next_btn)

        nav_row.addStretch()

        self._det_cb = QCheckBox("Detected labels")
        self._det_cb.setChecked(True)
        self._det_cb.stateChanged.connect(self._refresh)
        nav_row.addWidget(self._det_cb)

        self._ref_cb = QCheckBox("Reference labels")
        self._ref_cb.setChecked(True)
        self._ref_cb.stateChanged.connect(self._refresh)
        nav_row.addWidget(self._ref_cb)

        layout.addWidget(nav)

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

    def _on_theme_changed(self) -> None:
        self._spec.refresh_theme()
        self._refresh()

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
        )

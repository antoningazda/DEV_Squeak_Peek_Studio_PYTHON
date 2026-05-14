from __future__ import annotations

from dataclasses import replace

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._label_io import save_labels
from ._spectrogram_widget import SpectrogramWidget
from ._state import AppState


class LabelEditTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._idx: int = 0
        self._setup_ui()

        state.wav_loaded.connect(self._on_labels_changed)
        state.labels_changed.connect(self._on_labels_changed)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._spec = SpectrogramWidget()
        layout.addWidget(self._spec, stretch=1)

        # ── Label info row ────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self._info_label = QLabel("No detected labels loaded")
        info_row.addWidget(self._info_label, stretch=1)
        self._counter = QLabel("— / —")
        info_row.addWidget(self._counter)
        layout.addLayout(info_row)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = QGroupBox()
        ctrl_row = QHBoxLayout(ctrl)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(40)
        self._prev_btn.clicked.connect(self._prev)
        ctrl_row.addWidget(self._prev_btn)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(40)
        self._next_btn.clicked.connect(self._next)
        ctrl_row.addWidget(self._next_btn)

        ctrl_row.addStretch()

        ctrl_row.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.setMinimumWidth(80)
        self._reload_classes()
        ctrl_row.addWidget(self._class_combo)

        self._accept_btn = QPushButton("Accept ✓")
        self._accept_btn.setStyleSheet(
            "QPushButton { background: #4DCDD2; color: #111; font-weight: bold; }"
            "QPushButton:hover { background: #6DDDE2; }"
        )
        self._accept_btn.clicked.connect(self._accept)
        ctrl_row.addWidget(self._accept_btn)

        self._reject_btn = QPushButton("Reject ✗")
        self._reject_btn.setStyleSheet(
            "QPushButton { background: #F8C06F; color: #111; font-weight: bold; }"
            "QPushButton:hover { background: #FAD08F; }"
        )
        self._reject_btn.clicked.connect(self._reject)
        ctrl_row.addWidget(self._reject_btn)

        layout.addWidget(ctrl)

        # ── Export ────────────────────────────────────────────────────────
        export_row = QHBoxLayout()
        export_row.addStretch()
        export_btn = QPushButton("Export Labels…")
        export_btn.clicked.connect(self._export)
        export_row.addWidget(export_btn)
        layout.addLayout(export_row)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_labels_changed(self) -> None:
        self._reload_classes()
        labels = self._state.detected_labels
        if not labels:
            self._info_label.setText("No detected labels loaded")
            self._counter.setText("— / —")
            self._spec.clear()
            return
        self._idx = 0
        self._show_current()

    def _reload_classes(self) -> None:
        classes = self._state.settings.label_edit.classification_list
        self._class_combo.blockSignals(True)
        current = self._class_combo.currentText()
        self._class_combo.clear()
        self._class_combo.addItems(classes)
        idx = self._class_combo.findText(current)
        if idx >= 0:
            self._class_combo.setCurrentIndex(idx)
        self._class_combo.blockSignals(False)

    def _show_current(self) -> None:
        s = self._state
        labels = s.detected_labels
        if not labels or s.samples is None:
            return

        n = len(labels)
        lbl = labels[self._idx]
        self._counter.setText(f"{self._idx + 1} / {n}")
        self._info_label.setText(
            f"[{lbl.start_time:.4f} – {lbl.end_time:.4f} s]  "
            f"dur = {lbl.duration * 1000:.1f} ms  "
            f"label = '{lbl.label}'"
        )

        # Sync class combo
        ci = self._class_combo.findText(lbl.label)
        if ci >= 0:
            self._class_combo.blockSignals(True)
            self._class_combo.setCurrentIndex(ci)
            self._class_combo.blockSignals(False)

        # Show label with 50 % padding on each side, minimum 10 ms
        pad = max(lbl.duration * 0.5, 0.01)
        t0 = max(0.0, lbl.start_time - pad)
        t1 = min(s.duration, lbl.end_time + pad)

        le = s.settings.label_edit
        vis = s.settings.visualization
        self._spec.display(
            samples=s.samples,
            fs=s.fs,
            t_start=t0,
            t_end=t1,
            fmin_hz=le.spectrogram_min_freq_khz * 1_000.0,
            fmax_hz=le.spectrogram_max_freq_khz * 1_000.0,
            nperseg=le.spectrogram_window,
            noverlap=le.spectrogram_overlap,
            detected_labels=[lbl],
            reference_labels=None,
            show_detected=True,
            show_reference=False,
        )

    def _prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._show_current()

    def _next(self) -> None:
        if self._idx < len(self._state.detected_labels) - 1:
            self._idx += 1
            self._show_current()

    def _accept(self) -> None:
        labels = self._state.detected_labels
        if not labels:
            return
        new_class = self._class_combo.currentText()
        labels[self._idx] = replace(labels[self._idx], label=new_class)
        # Advance to next if available
        if self._idx < len(labels) - 1:
            self._idx += 1
        self._show_current()

    def _reject(self) -> None:
        labels = self._state.detected_labels
        if not labels:
            return
        labels.pop(self._idx)
        self._state.labels_changed.emit()
        if labels:
            self._idx = min(self._idx, len(labels) - 1)
            self._show_current()
        else:
            self._info_label.setText("No detected labels")
            self._counter.setText("— / —")
            self._spec.clear()

    def _export(self) -> None:
        labels = self._state.detected_labels
        if not labels:
            QMessageBox.warning(self, "Nothing to export", "No detected labels to export.")
            return
        default = (
            str(self._state.wav_path.with_suffix(".txt"))
            if self._state.wav_path
            else ""
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export labels", default, "Text files (*.txt);;All files (*)"
        )
        if path:
            save_labels(path, labels)
            QMessageBox.information(self, "Exported", f"Saved {len(labels)} labels to:\n{path}")

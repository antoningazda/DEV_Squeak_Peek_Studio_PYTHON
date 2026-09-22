from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from squeak_peek.labels.io import export_labels
from squeak_peek.labels.model import Label

from . import _theme as t
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
        t.signal.changed.connect(self._on_theme_changed)

        # Connect spectrogram signals
        self._spec.boundary_dragged.connect(self._on_boundary_dragged)
        self._spec.spectrogram_right_clicked.connect(self._on_spectrogram_right_clicked)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(t.SP_2, t.SP_2, t.SP_2, t.SP_2)
        layout.setSpacing(t.SP_2)

        self._spec = SpectrogramWidget()
        layout.addWidget(self._spec, stretch=1)

        # ── Label info row ────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self._info_label = QLabel("No detected labels loaded")
        info_row.addWidget(self._info_label, stretch=1)
        self._counter = QLabel("— / —")
        info_row.addWidget(self._counter)
        layout.addLayout(info_row)

        # ── State counters ────────────────────────────────────────────────
        counters_row = QHBoxLayout()
        counters_row.addStretch()

        counters_row.addWidget(QLabel("Detections:"))
        self._counter_det_accepted = QLabel("0")
        counters_row.addWidget(self._counter_det_accepted)
        counters_row.addWidget(QLabel("accepted /"))

        self._counter_det_rejected = QLabel("0")
        counters_row.addWidget(self._counter_det_rejected)
        counters_row.addWidget(QLabel("rejected"))

        counters_row.addSpacing(20)
        counters_row.addWidget(QLabel("Classifications:"))

        self._counter_cls_accepted = QLabel("0")
        counters_row.addWidget(self._counter_cls_accepted)
        counters_row.addWidget(QLabel("accepted /"))

        self._counter_cls_rejected = QLabel("0")
        counters_row.addWidget(self._counter_cls_rejected)
        counters_row.addWidget(QLabel("rejected"))

        layout.addLayout(counters_row)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = QGroupBox()
        ctrl_layout = QVBoxLayout(ctrl)

        # Navigation row
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.clicked.connect(self._prev)
        nav_row.addWidget(self._prev_btn)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(36)
        self._next_btn.clicked.connect(self._next)
        nav_row.addWidget(self._next_btn)
        nav_row.addStretch()

        # Class combo
        nav_row.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.setMinimumWidth(90)
        self._class_combo.currentTextChanged.connect(self._on_class_changed)
        self._reload_classes()
        nav_row.addWidget(self._class_combo)

        ctrl_layout.addLayout(nav_row)

        # Detection state row
        det_row = QHBoxLayout()
        det_row.addWidget(QLabel("Detection State:"))
        self._det_state_combo = QComboBox()
        self._det_state_combo.addItems(["None", "Accepted", "Rejected"])
        self._det_state_combo.currentTextChanged.connect(self._on_detection_state_changed)
        det_row.addWidget(self._det_state_combo)
        det_row.addStretch()
        ctrl_layout.addLayout(det_row)

        # Classification state row
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Classification State:"))
        self._cls_state_combo = QComboBox()
        self._cls_state_combo.addItems(["None", "Accepted", "Rejected"])
        self._cls_state_combo.currentTextChanged.connect(self._on_classification_state_changed)
        cls_row.addWidget(self._cls_state_combo)
        cls_row.addStretch()
        ctrl_layout.addLayout(cls_row)

        # Classification override field (hidden by default)
        cls_override_row = QHBoxLayout()
        cls_override_row.addWidget(QLabel("Corrected class (rejected):"))
        self._cls_override_edit = QLineEdit()
        self._cls_override_edit.setPlaceholderText("Type alternative classification...")
        self._cls_override_edit.textChanged.connect(self._on_override_text_changed)
        cls_override_row.addWidget(self._cls_override_edit)
        self._cls_override_container = QWidget()
        self._cls_override_container.setLayout(cls_override_row)
        self._cls_override_container.setVisible(False)
        ctrl_layout.addWidget(self._cls_override_container)

        layout.addWidget(ctrl)

        # ── Export ────────────────────────────────────────────────────────
        export_row = QHBoxLayout()
        export_row.addStretch()
        export_btn = QPushButton("Export labels…")
        export_btn.clicked.connect(self._export)
        export_row.addWidget(export_btn)
        layout.addLayout(export_row)

        self._apply_theme()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self._info_label.setStyleSheet(
            f"color: {t.TEXT_SECONDARY}; font-family: monospace; font-size: {t.TEXT_XS}px;"
        )
        self._counter.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")

    def _on_theme_changed(self) -> None:
        self._apply_theme()
        self._spec.refresh_theme()
        if self._state.detected_labels:
            self._show_current()

    def _on_labels_changed(self) -> None:
        self._reload_classes()
        labels = self._state.detected_labels
        if not labels:
            self._info_label.setText("No detected labels loaded")
            self._counter.setText("— / —")
            self._spec.clear()
            self._update_counters()
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

    def _update_counters(self) -> None:
        """Update the four counters (det_accepted, det_rejected, cls_accepted, cls_rejected)."""
        labels = self._state.detected_labels
        if not labels:
            self._counter_det_accepted.setText("0")
            self._counter_det_rejected.setText("0")
            self._counter_cls_accepted.setText("0")
            self._counter_cls_rejected.setText("0")
            return

        det_accepted = sum(1 for lbl in labels if lbl.detection_state == "Accepted")
        det_rejected = sum(1 for lbl in labels if lbl.detection_state == "Rejected")
        cls_accepted = sum(1 for lbl in labels if lbl.classification_state == "Accepted")
        cls_rejected = sum(1 for lbl in labels if lbl.classification_state == "Rejected")

        self._counter_det_accepted.setText(str(det_accepted))
        self._counter_det_rejected.setText(str(det_rejected))
        self._counter_cls_accepted.setText(str(cls_accepted))
        self._counter_cls_rejected.setText(str(cls_rejected))

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

        # Sync state combos
        self._det_state_combo.blockSignals(True)
        self._det_state_combo.setCurrentText(lbl.detection_state)
        self._det_state_combo.blockSignals(False)

        self._cls_state_combo.blockSignals(True)
        self._cls_state_combo.setCurrentText(lbl.classification_state)
        self._cls_state_combo.blockSignals(False)

        # Show/hide classification override field
        if lbl.classification_state == "Rejected":
            self._cls_override_container.setVisible(True)
            self._cls_override_edit.blockSignals(True)
            self._cls_override_edit.setText(lbl.label)
            self._cls_override_edit.blockSignals(False)
        else:
            self._cls_override_container.setVisible(False)

        # Show label with 50 % padding on each side, minimum 10 ms
        pad = max(lbl.duration * 0.5, 0.01)
        t0 = max(0.0, lbl.start_time - pad)
        t1 = min(s.duration, lbl.end_time + pad)

        le = s.settings.label_edit
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

        # Enable boundary drag for the current label
        self._spec.enable_boundary_drag(
            lbl.start_time,
            lbl.end_time,
            le.spectrogram_min_freq_khz,
            le.spectrogram_max_freq_khz,
        )

        # Update counters
        self._update_counters()

    def _prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._show_current()

    def _next(self) -> None:
        if self._idx < len(self._state.detected_labels) - 1:
            self._idx += 1
            self._show_current()

    def _on_class_changed(self, text: str) -> None:
        """Called when class combo changes."""
        labels = self._state.detected_labels
        if not labels:
            return
        labels[self._idx] = replace(labels[self._idx], label=text)
        self._state.labels_changed.emit()

    def _on_detection_state_changed(self, state: str) -> None:
        """Called when detection state combo changes."""
        labels = self._state.detected_labels
        if not labels:
            return
        labels[self._idx] = replace(labels[self._idx], detection_state=state)
        self._state.labels_changed.emit()
        self._update_counters()

    def _on_classification_state_changed(self, state: str) -> None:
        """Called when classification state combo changes."""
        labels = self._state.detected_labels
        if not labels:
            return
        labels[self._idx] = replace(labels[self._idx], classification_state=state)
        self._state.labels_changed.emit()
        self._update_counters()

        # Show/hide the correction field
        if state == "Rejected":
            self._cls_override_container.setVisible(True)
            self._cls_override_edit.blockSignals(True)
            self._cls_override_edit.setText(labels[self._idx].label)
            self._cls_override_edit.blockSignals(False)
        else:
            self._cls_override_container.setVisible(False)

    def _on_override_text_changed(self, text: str) -> None:
        """Called when the classification override field changes."""
        labels = self._state.detected_labels
        if not labels:
            return
        # Update label text when user types a correction
        labels[self._idx] = replace(labels[self._idx], label=text)
        self._state.labels_changed.emit()

    def _on_boundary_dragged(self, edge: str, new_time: float) -> None:
        """Called when a label boundary is dragged in the spectrogram."""
        labels = self._state.detected_labels
        if not labels:
            return

        if edge == "start":
            labels[self._idx] = replace(labels[self._idx], start_time=new_time)
        elif edge == "end":
            labels[self._idx] = replace(labels[self._idx], end_time=new_time)

        self._state.labels_changed.emit()
        self._show_current()

    def _on_spectrogram_right_clicked(self, time: float) -> None:
        """Called when right-clicking on the spectrogram to create a new manual label."""
        labels = self._state.detected_labels
        if labels is None:
            return

        # Create a new label centered on the clicked time
        duration = self._state.settings.visualization.manual_label_length
        new_label = Label(
            start_time=max(0.0, time - duration / 2),
            end_time=min(self._state.duration, time + duration / 2),
            label="md",  # manual label marker
            detection_state="Accepted",
            classification_state="None",
        )

        # Insert sorted by start_time
        insert_idx = 0
        for i, lbl in enumerate(labels):
            if lbl.start_time < new_label.start_time:
                insert_idx = i + 1

        labels.insert(insert_idx, new_label)
        self._state.labels_changed.emit()
        self._idx = insert_idx
        self._show_current()

    def _export(self) -> None:
        labels = self._state.detected_labels
        if not labels:
            QMessageBox.warning(self, "Nothing to export", "No detected labels to export.")
            return

        # Count rejected detections and classifications
        num_rejected_detections = sum(1 for lbl in labels if lbl.detection_state == "Rejected")
        num_rejected_classifications = sum(1 for lbl in labels if lbl.classification_state == "Rejected")

        # Build suggested filename
        if self._state.wav_path:
            base_name = self._state.wav_path.stem
            suggested_name = (
                f"{base_name}_RD{num_rejected_detections}_RC{num_rejected_classifications}.txt"
            )
            default_path = str(self._state.wav_path.parent / suggested_name)
        else:
            default_path = ""

        path, _ = QFileDialog.getSaveFileName(
            self, "Export labels", default_path, "Text files (*.txt);;All files (*)"
        )
        if path:
            export_labels(path, labels)
            QMessageBox.information(self, "Exported", f"Saved {len(labels)} labels to:\n{path}")

    # ── Keyboard shortcuts ────────────────────────────────────────────────

    def accept_detection(self) -> None:
        """Accept detection for the current label (keyboard shortcut handler)."""
        labels = self._state.detected_labels
        if not labels or self._idx >= len(labels):
            return
        labels[self._idx] = replace(labels[self._idx], detection_state="Accepted")
        self._state.labels_changed.emit()
        self._update_counters()

    def reject_detection(self) -> None:
        """Reject detection for the current label (keyboard shortcut handler)."""
        labels = self._state.detected_labels
        if not labels or self._idx >= len(labels):
            return
        labels[self._idx] = replace(labels[self._idx], detection_state="Rejected")
        self._state.labels_changed.emit()
        self._update_counters()

    def accept_classification(self) -> None:
        """Accept classification for the current label (keyboard shortcut handler)."""
        labels = self._state.detected_labels
        if not labels or self._idx >= len(labels):
            return
        labels[self._idx] = replace(labels[self._idx], classification_state="Accepted")
        self._state.labels_changed.emit()
        self._update_counters()

    def reject_classification(self) -> None:
        """Reject classification for the current label (keyboard shortcut handler)."""
        labels = self._state.detected_labels
        if not labels or self._idx >= len(labels):
            return
        labels[self._idx] = replace(labels[self._idx], classification_state="Rejected")
        self._state.labels_changed.emit()
        self._update_counters()

    def accept_both_and_advance(self) -> None:
        """Accept both detection and classification, then advance to next label (space bar handler)."""
        labels = self._state.detected_labels
        if not labels or self._idx >= len(labels):
            return

        # Set both states to Accepted
        labels[self._idx] = replace(
            labels[self._idx],
            detection_state="Accepted",
            classification_state="Accepted"
        )
        self._state.labels_changed.emit()
        self._update_counters()

        # Advance to next
        if self._idx < len(labels) - 1:
            self._next()

    def move_to_previous_label(self) -> None:
        """Move to previous label (keyboard shortcut handler)."""
        self._prev()

    def move_to_next_label(self) -> None:
        """Move to next label (keyboard shortcut handler)."""
        self._next()

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.detectors.bscd import BSCDDetector
from squeak_peek.detectors.psd import PSDDetector
from squeak_peek.detectors.rbd import RBDDetector
from squeak_peek.labels.io import export_labels_detector
from squeak_peek.labels.postprocess import merge_close_labels, remove_short_labels

from . import _theme as t
from ._state import AppState

_DETECTORS = ["PSD", "BSCD", "RBD", "ML"]


class DetectionTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._status_kind = "neutral"
        self._setup_ui()
        t.signal.changed.connect(self._apply_theme)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(t.SP_4)
        layout.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)

        # ── Detector selection ────────────────────────────────────────────
        det_group = QGroupBox("Detector")
        det_layout = QVBoxLayout(det_group)
        self._det_btn_group = QButtonGroup(self)
        for i, name in enumerate(_DETECTORS):
            rb = QRadioButton(name)
            if i == 0:
                rb.setChecked(True)
            self._det_btn_group.addButton(rb, i)
            det_layout.addWidget(rb)
        layout.addWidget(det_group)

        # ── Post-processing ───────────────────────────────────────────────
        post_group = QGroupBox("Post-processing")
        post_layout = QVBoxLayout(post_group)
        self._merge_cb = QCheckBox("Merge close labels")
        self._merge_cb.setChecked(True)
        self._short_cb = QCheckBox("Remove short labels")
        self._short_cb.setChecked(True)
        post_layout.addWidget(self._merge_cb)
        post_layout.addWidget(self._short_cb)
        layout.addWidget(post_group)

        # ── Export folder ─────────────────────────────────────────────────
        export_group = QGroupBox("Export folder")
        export_row = QHBoxLayout(export_group)
        self._export_edit = QLineEdit()
        self._export_edit.setReadOnly(True)
        self._export_edit.setPlaceholderText("Same folder as WAV file")
        export_btn = QPushButton("Browse…")
        export_btn.clicked.connect(self._browse_export)
        export_row.addWidget(self._export_edit)
        export_row.addWidget(export_btn)
        layout.addWidget(export_group)

        layout.addStretch()

        # ── Run ───────────────────────────────────────────────────────────
        self._run_btn = QPushButton("Run detector")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._status = QLabel("")
        layout.addWidget(self._status)
        self._apply_theme()

    def _apply_theme(self) -> None:
        color = {"neutral": t.TEXT_SECONDARY, "success": t.SUCCESS, "danger": t.DANGER}[self._status_kind]
        self._status.setStyleSheet(f"color: {color}; font-size: {t.TEXT_XS}px;")

    def _set_status(self, text: str, kind: str) -> None:
        self._status_kind = kind
        self._apply_theme()
        self._status.setText(text)

    def _browse_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export folder")
        if path:
            self._export_edit.setText(path)

    def _run(self) -> None:
        if self._state.samples is None:
            QMessageBox.warning(self, "No file loaded", "Load a WAV file first.")
            return

        det_name = _DETECTORS[self._det_btn_group.checkedId()]
        if det_name == "ML":
            QMessageBox.information(
                self,
                "Not yet implemented",
                "The ML detector is not yet implemented (Tier 2 of PORT_PLAN.md).\n\n"
                "Use PSD, BSCD, or RBD, or load pre-computed label files via the "
                "Data Input tab.",
            )
            return

        detector = self._build_detector(det_name)

        self._run_btn.setEnabled(False)
        self._set_status(f"Running {det_name}…", "neutral")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            labels = detector.detect(self._state.samples, self._state.fs)

            post = self._state.settings.detection.post
            if self._merge_cb.isChecked():
                labels = merge_close_labels(labels, post.maxGapToMerge)
            if self._short_cb.isChecked():
                labels = remove_short_labels(labels, post.minLabelLength)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Detection failed", str(exc))
            self._set_status("Failed.", "danger")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._run_btn.setEnabled(True)

        self._state.detected_labels = labels
        self._state.labels_changed.emit()
        self._set_status(f"{det_name}: {len(labels)} events detected.", "success")

        export_dir = self._export_edit.text().strip()
        if export_dir and self._state.wav_path is not None:
            out_path = Path(export_dir) / f"{self._state.wav_path.stem}_{det_name.lower()}_detected.txt"
            export_labels_detector(out_path, labels)
            self._status.setText(self._status.text() + f"  Exported: {out_path}")

    def _build_detector(self, det_name: str) -> AbstractDetector:
        det = self._state.settings.detection
        if det_name == "PSD":
            return PSDDetector(det.psd)
        if det_name == "BSCD":
            return BSCDDetector(det.bscd)
        return RBDDetector(det.rbd)

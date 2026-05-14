from __future__ import annotations

from PyQt6.QtWidgets import (
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

from ._state import AppState

_DETECTORS = ["PSD", "BSCD", "RBD", "ML"]


class DetectionTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

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

        # ── Run ───────────────────────────────────────────────────────────
        self._run_btn = QPushButton("Run Detector")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._status = QLabel("")
        layout.addWidget(self._status)

        layout.addStretch()

    def _browse_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export folder")
        if path:
            self._export_edit.setText(path)

    def _run(self) -> None:
        if self._state.samples is None:
            QMessageBox.warning(self, "No file loaded", "Load a WAV file first.")
            return
        det = _DETECTORS[self._det_btn_group.checkedId()]
        QMessageBox.information(
            self,
            "Not yet implemented",
            f"The {det} detector is implemented in Phase 2.\n\n"
            "In the meantime, load pre-computed label files via the Data Input tab.",
        )

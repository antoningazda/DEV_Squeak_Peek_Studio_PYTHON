from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import _theme as t
from ._label_io import load_labels
from ._state import AppState


class DataInputTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        # Runtime attributes for batch mode
        self.batch_mode = False
        self.batch_usv_dir = ""
        self.batch_labels_dir = ""
        self.batch_reference_labels_dir = ""

        self._setup_ui()
        state.wav_loaded.connect(self._on_wav_loaded)
        t.signal.changed.connect(self._apply_theme)

    def _apply_theme(self) -> None:
        self._desc.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_SM}px;")
        self._info_label.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(t.SP_4)
        outer.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)

        # ── Description ───────────────────────────────────────────────────
        self._desc = QLabel(
            "Select and load USV audio, reference labels and detected labels "
            "for further analysis in single or batch mode."
        )
        self._desc.setWordWrap(True)
        outer.addWidget(self._desc)

        # ── Mode toggle (Single/Batch) ────────────────────────────────────
        mode_group = QGroupBox("Mode")
        mode_group.setToolTip("Whether to analyze one recording at a time or a whole folder at once.")
        mode_layout = QHBoxLayout(mode_group)
        self._single_mode_rb = QRadioButton("Single file")
        self._single_mode_rb.setToolTip("Load one WAV file, plus optional label files, at a time.")
        self._batch_mode_rb = QRadioButton("Batch folder")
        self._batch_mode_rb.setToolTip(
            "Load a whole folder of WAV files; matching label files are looked up by filename."
        )
        self._single_mode_rb.setChecked(True)
        self._single_mode_rb.toggled.connect(self._on_mode_changed)
        self._batch_mode_rb.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._single_mode_rb)
        mode_layout.addWidget(self._batch_mode_rb)
        mode_layout.addStretch()
        outer.addWidget(mode_group)

        # ── Single mode panel ─────────────────────────────────────────────
        self._single_panel = QGroupBox("Input files (single mode)")
        single_col = QVBoxLayout(self._single_panel)
        single_grid = QGridLayout()
        single_grid.setSpacing(t.SP_2)
        single_grid.setColumnStretch(1, 1)

        def _row(label_text: str, row: int, tooltip: str) -> tuple[QPushButton, QLineEdit]:
            btn = QPushButton(label_text)
            btn.setFixedWidth(220)
            btn.setToolTip(tooltip)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("No file selected")
            edit.setToolTip(tooltip)
            single_grid.addWidget(btn,  row, 0)
            single_grid.addWidget(edit, row, 1)
            return btn, edit

        wav_btn,  self._wav_edit  = _row(
            "Select USV (.wav)", 0, "The audio recording to analyze."
        )
        ref_btn,  self._ref_edit  = _row(
            "Select reference labels (.txt)", 1,
            "Ground-truth labels used to score detections in the Metrics tab (optional).",
        )
        det_btn,  self._det_edit  = _row(
            "Select detected labels (.txt)", 2,
            "Previously detected labels to load and continue editing in Label Edit (optional).",
        )

        wav_btn.clicked.connect(self._browse_wav)
        ref_btn.clicked.connect(self._browse_reference)
        det_btn.clicked.connect(self._browse_detected)

        single_col.addLayout(single_grid)
        outer.addWidget(self._single_panel)

        # ── Batch mode panel ──────────────────────────────────────────────
        self._batch_panel = QGroupBox("Input folders (batch mode)")
        self._batch_panel.setVisible(False)
        batch_col = QVBoxLayout(self._batch_panel)
        batch_grid = QGridLayout()
        batch_grid.setSpacing(t.SP_2)
        batch_grid.setColumnStretch(1, 1)

        def _batch_row(label_text: str, row: int, tooltip: str) -> tuple[QPushButton, QLineEdit, QLabel]:
            btn = QPushButton(label_text)
            btn.setFixedWidth(220)
            btn.setToolTip(tooltip)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("No folder selected")
            edit.setToolTip(tooltip)
            count_label = QLabel("")
            count_label.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")
            count_label.setToolTip("Number of matching files found in this folder.")
            batch_grid.addWidget(btn,  row, 0)
            batch_grid.addWidget(edit, row, 1)
            batch_grid.addWidget(count_label, row, 2)
            return btn, edit, count_label

        usv_btn, self._batch_usv_edit, self._batch_usv_count = _batch_row(
            "Select USV folder", 0, "Folder of .wav recordings to process one after another."
        )
        ref_btn_b, self._batch_ref_edit, self._batch_ref_count = _batch_row(
            "Select reference labels folder", 1,
            "Folder of ground-truth label files, matched to recordings by filename (optional).",
        )
        det_btn_b, self._batch_det_edit, self._batch_det_count = _batch_row(
            "Select detected labels folder", 2,
            "Folder of previously detected label files, matched to recordings by filename (optional).",
        )

        usv_btn.clicked.connect(self._browse_batch_usv)
        ref_btn_b.clicked.connect(self._browse_batch_reference)
        det_btn_b.clicked.connect(self._browse_batch_detected)

        batch_col.addLayout(batch_grid)
        outer.addWidget(self._batch_panel)

        # ── File info row ─────────────────────────────────────────────────
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setVisible(False)
        outer.addWidget(self._info_label)

        outer.addStretch()

        # ── Load Files button (bottom-right, primary action) ──────────────
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self._load_btn = QPushButton("Load files")
        self._load_btn.setToolTip(
            "Load the selected file(s)/folder(s) — required before Visualization, "
            "Detection or Label Edit can show anything."
        )
        self._load_btn.setObjectName("primaryBtn")
        self._load_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._load_btn.clicked.connect(self._load_files)
        bottom_row.addWidget(self._load_btn)
        outer.addLayout(bottom_row)

        # Pending paths (set by browse, committed on Load)
        self._pending_wav: str = ""
        self._pending_det: str = ""
        self._pending_ref: str = ""

        self._apply_theme()

    # ── Mode handling ────────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        self.batch_mode = self._batch_mode_rb.isChecked()
        self._single_panel.setVisible(not self.batch_mode)
        self._batch_panel.setVisible(self.batch_mode)

    # ── Browse handlers (single mode) ──────────────────────────────────────

    def _browse_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV file", "", "WAV files (*.wav *.WAV);;All files (*)"
        )
        if path:
            self._wav_edit.setText(path)
            self._pending_wav = path

    def _browse_detected(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open detected labels", "", "Text files (*.txt *.csv);;All files (*)"
        )
        if path:
            self._det_edit.setText(path)
            self._pending_det = path

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open reference labels", "", "Text files (*.txt *.csv);;All files (*)"
        )
        if path:
            self._ref_edit.setText(path)
            self._pending_ref = path

    # ── Browse handlers (batch mode) ───────────────────────────────────────

    def _browse_batch_usv(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select USV folder")
        if path:
            self._batch_usv_edit.setText(path)
            self.batch_usv_dir = path
            self._update_file_count("usv")

    def _browse_batch_detected(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select detected labels folder")
        if path:
            self._batch_det_edit.setText(path)
            self._update_file_count("detected")

    def _browse_batch_reference(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select reference labels folder")
        if path:
            self._batch_ref_edit.setText(path)
            self._update_file_count("reference")

    def _update_file_count(self, kind: str) -> None:
        """Update the file count label for a batch folder."""
        if kind == "usv":
            folder = self.batch_usv_dir
            pattern = "*.wav"
            label = self._batch_usv_count
        elif kind == "detected":
            folder = self._batch_det_edit.text().strip()
            pattern = "*.txt"
            label = self._batch_det_count
        else:  # reference
            folder = self._batch_ref_edit.text().strip()
            pattern = "*.txt"
            label = self._batch_ref_count

        if folder:
            try:
                count = len(list(Path(folder).glob(pattern)))
                label.setText(f"({count} files)")
            except Exception:
                label.setText("(error)")
        else:
            label.setText("")

    # ── Load ──────────────────────────────────────────────────────────────

    def _load_files(self) -> None:
        errors: list[str] = []

        if self._pending_wav:
            try:
                self._state.load_wav(self._pending_wav)
            except Exception as exc:
                errors.append(f"WAV: {exc}")

        if self._pending_det:
            try:
                labels = load_labels(self._pending_det)
                self._state.detected_labels = labels
                self._state.labels_changed.emit()
            except Exception as exc:
                errors.append(f"Detected labels: {exc}")

        if self._pending_ref:
            try:
                labels = load_labels(self._pending_ref)
                self._state.reference_labels = labels
                self._state.labels_changed.emit()
            except Exception as exc:
                errors.append(f"Reference labels: {exc}")

        if errors:
            QMessageBox.warning(self, "Load errors", "\n".join(errors))

    # ── Called from menu bar ──────────────────────────────────────────────

    def open_wav(self) -> None:
        self._browse_wav()
        if self._pending_wav:
            self._load_files()

    # ── State callbacks ───────────────────────────────────────────────────

    def _on_wav_loaded(self) -> None:
        s = self._state
        if s.wav_path:
            self._wav_edit.setText(str(s.wav_path))
        det_n = len(s.detected_labels)
        ref_n = len(s.reference_labels)
        self._info_label.setText(
            f"Duration: {s.duration:.3f} s   ·   "
            f"Sample rate: {s.fs:,} Hz   ·   "
            f"Detected labels: {det_n}   ·   Reference labels: {ref_n}"
        )
        self._info_label.setVisible(True)

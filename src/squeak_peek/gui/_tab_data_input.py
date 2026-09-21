from __future__ import annotations

from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
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

        # ── File rows ─────────────────────────────────────────────────────
        files_group = QGroupBox("Input files")
        files_col = QVBoxLayout(files_group)
        grid = QGridLayout()
        grid.setSpacing(t.SP_2)
        grid.setColumnStretch(1, 1)

        def _row(label_text: str, row: int) -> tuple[QPushButton, QLineEdit]:
            btn = QPushButton(label_text)
            btn.setFixedWidth(220)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("No file selected")
            grid.addWidget(btn,  row, 0)
            grid.addWidget(edit, row, 1)
            return btn, edit

        wav_btn,  self._wav_edit  = _row("Select USV (.wav)",              0)
        ref_btn,  self._ref_edit  = _row("Select reference labels (.txt)", 1)
        det_btn,  self._det_edit  = _row("Select detected labels (.txt)",  2)

        wav_btn.clicked.connect(self._browse_wav)
        ref_btn.clicked.connect(self._browse_reference)
        det_btn.clicked.connect(self._browse_detected)

        files_col.addLayout(grid)
        outer.addWidget(files_group)

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

    # ── Browse handlers ───────────────────────────────────────────────────

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

from __future__ import annotations

from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._label_io import load_labels
from ._state import AppState

_ORANGE_BTN = (
    "QPushButton { background: #F9C06E; border: 1px solid #E0A850; "
    "border-radius: 5px; padding: 8px 22px; font-weight: bold; color: #111; }"
    "QPushButton:hover { background: #FAD080; }"
    "QPushButton:pressed { background: #E8A840; }"
)


class DataInputTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        state.wav_loaded.connect(self._on_wav_loaded)

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(16)
        outer.setContentsMargins(16, 12, 16, 12)

        # ── Description ───────────────────────────────────────────────────
        desc = QLabel(
            "<i>Select and load USV audio, reference labels and detected labels "
            "for further analysis in single or batch mode.</i>"
        )
        desc.setWordWrap(True)
        outer.addWidget(desc)

        # ── File rows ─────────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)

        def _row(label_text: str, row: int) -> tuple[QPushButton, QLineEdit]:
            btn = QPushButton(label_text)
            btn.setFixedWidth(230)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("No file selected")
            grid.addWidget(btn,  row, 0)
            grid.addWidget(edit, row, 1)
            return btn, edit

        wav_btn,  self._wav_edit  = _row("Select USV (.wav)",             0)
        ref_btn,  self._ref_edit  = _row("Select Reference Labels (.txt)", 1)
        det_btn,  self._det_edit  = _row("Select Detected Labels (.txt)",  2)

        wav_btn.clicked.connect(self._browse_wav)
        ref_btn.clicked.connect(self._browse_reference)
        det_btn.clicked.connect(self._browse_detected)

        outer.addLayout(grid)

        # ── File info label ───────────────────────────────────────────────
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #555555; font-size: 12px;")
        outer.addWidget(self._info_label)

        outer.addStretch()

        # ── Load Files button (bottom-right, orange) ──────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self._load_btn = QPushButton("Load Files")
        self._load_btn.setStyleSheet(_ORANGE_BTN)
        self._load_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._load_btn.clicked.connect(self._load_files)
        bottom_row.addWidget(self._load_btn)
        outer.addLayout(bottom_row)

        # Pending paths (set by browse, committed on Load)
        self._pending_wav: str = ""
        self._pending_det: str = ""
        self._pending_ref: str = ""

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
            f"Duration: {s.duration:.3f} s  |  "
            f"Sample rate: {s.fs:,} Hz  |  "
            f"Detected labels: {det_n}  |  Reference labels: {ref_n}"
        )

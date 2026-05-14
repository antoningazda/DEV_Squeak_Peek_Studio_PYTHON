from __future__ import annotations

from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._label_io import load_labels
from ._state import AppState


class DataInputTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        state.wav_loaded.connect(self._on_wav_loaded)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── WAV file ──────────────────────────────────────────────────────
        wav_group = QGroupBox("WAV File")
        wav_form = QFormLayout(wav_group)

        self._wav_edit = QLineEdit()
        self._wav_edit.setReadOnly(True)
        self._wav_edit.setPlaceholderText("No file selected")
        wav_btn = QPushButton("Browse…")
        wav_btn.clicked.connect(self._browse_wav)
        wav_row = QHBoxLayout()
        wav_row.addWidget(self._wav_edit)
        wav_row.addWidget(wav_btn)
        wav_form.addRow("File:", wav_row)

        self._info_label = QLabel("—")
        wav_form.addRow("Info:", self._info_label)
        layout.addWidget(wav_group)

        # ── Detected labels ───────────────────────────────────────────────
        det_group = QGroupBox("Detected Labels  (optional)")
        det_form = QFormLayout(det_group)

        self._det_edit = QLineEdit()
        self._det_edit.setReadOnly(True)
        self._det_edit.setPlaceholderText("No file selected")
        det_btn = QPushButton("Browse…")
        det_btn.clicked.connect(self._browse_detected)
        det_row = QHBoxLayout()
        det_row.addWidget(self._det_edit)
        det_row.addWidget(det_btn)
        det_form.addRow("File:", det_row)

        self._det_count = QLabel("—")
        det_form.addRow("Count:", self._det_count)
        layout.addWidget(det_group)

        # ── Reference labels ──────────────────────────────────────────────
        ref_group = QGroupBox("Reference Labels  (optional)")
        ref_form = QFormLayout(ref_group)

        self._ref_edit = QLineEdit()
        self._ref_edit.setReadOnly(True)
        self._ref_edit.setPlaceholderText("No file selected")
        ref_btn = QPushButton("Browse…")
        ref_btn.clicked.connect(self._browse_reference)
        ref_row = QHBoxLayout()
        ref_row.addWidget(self._ref_edit)
        ref_row.addWidget(ref_btn)
        ref_form.addRow("File:", ref_row)

        self._ref_count = QLabel("—")
        ref_form.addRow("Count:", self._ref_count)
        layout.addWidget(ref_group)

        layout.addStretch()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _browse_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV file", "", "WAV files (*.wav *.WAV);;All files (*)"
        )
        if not path:
            return
        self._wav_edit.setText(path)
        try:
            self._state.load_wav(path)
        except Exception as exc:
            self._info_label.setText(f"<font color='red'>Error: {exc}</font>")

    def _browse_detected(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open detected labels", "", "Text files (*.txt *.csv);;All files (*)"
        )
        if not path:
            return
        self._det_edit.setText(path)
        try:
            labels = load_labels(path)
            self._state.detected_labels = labels
            self._det_count.setText(str(len(labels)))
            self._state.labels_changed.emit()
        except Exception as exc:
            self._det_count.setText(f"<font color='red'>Error: {exc}</font>")

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open reference labels", "", "Text files (*.txt *.csv);;All files (*)"
        )
        if not path:
            return
        self._ref_edit.setText(path)
        try:
            labels = load_labels(path)
            self._state.reference_labels = labels
            self._ref_count.setText(str(len(labels)))
            self._state.labels_changed.emit()
        except Exception as exc:
            self._ref_count.setText(f"<font color='red'>Error: {exc}</font>")

    def _on_wav_loaded(self) -> None:
        s = self._state
        if s.wav_path:
            self._wav_edit.setText(str(s.wav_path))
        self._info_label.setText(
            f"Duration: {s.duration:.3f} s  |  "
            f"Sample rate: {s.fs:,} Hz  |  "
            f"Samples: {len(s.samples):,}"
        )

    # ── Called from menu bar ──────────────────────────────────────────────

    def open_wav(self) -> None:
        self._browse_wav()

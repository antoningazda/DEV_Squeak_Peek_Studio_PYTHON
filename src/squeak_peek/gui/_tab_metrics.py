from __future__ import annotations

from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from squeak_peek.labels.metrics import compare_labels

from . import _theme as t
from ._state import AppState


class MetricsTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        state.labels_changed.connect(self._on_labels_changed)
        t.signal.changed.connect(self._apply_theme)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(t.SP_4)
        layout.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)

        # ── Counts ────────────────────────────────────────────────────────
        counts_group = QGroupBox("Label counts")
        counts_form = QFormLayout(counts_group)
        self._det_count = QLabel("—")
        self._ref_count = QLabel("—")
        counts_form.addRow("Detected:", self._det_count)
        counts_form.addRow("Reference:", self._ref_count)
        layout.addWidget(counts_group)

        # ── Metrics ───────────────────────────────────────────────────────
        metrics_group = QGroupBox("Detection metrics (requires reference labels)")
        metrics_form = QFormLayout(metrics_group)
        self._tp = QLabel("—")
        self._fp = QLabel("—")
        self._fn = QLabel("—")
        self._prec = QLabel("—")
        self._rec = QLabel("—")
        self._f1 = QLabel("—")
        metrics_form.addRow("True positives:", self._tp)
        metrics_form.addRow("False positives:", self._fp)
        metrics_form.addRow("False negatives:", self._fn)
        metrics_form.addRow("Precision:", self._prec)
        metrics_form.addRow("Recall:", self._rec)
        metrics_form.addRow("F1 score:", self._f1)
        layout.addWidget(metrics_group)

        layout.addStretch()

        self._compute_btn = QPushButton("Compute metrics")
        self._compute_btn.setObjectName("primaryBtn")
        self._compute_btn.setMinimumHeight(32)
        self._compute_btn.clicked.connect(self._compute)
        layout.addWidget(self._compute_btn)

        self._apply_theme()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self._f1.setStyleSheet(f"color: {t.TEXT_PRIMARY}; font-weight: 600; font-size: {t.TEXT_LG}px;")

    def _on_labels_changed(self) -> None:
        self._det_count.setText(str(len(self._state.detected_labels)))
        self._ref_count.setText(str(len(self._state.reference_labels)))

    def _compute(self) -> None:
        det = self._state.detected_labels
        ref = self._state.reference_labels
        if not ref:
            QMessageBox.warning(
                self, "No reference labels",
                "Load reference labels in the Data Input tab first."
            )
            return

        stats = compare_labels(det, ref)

        self._tp.setText(str(stats.true_positives))
        self._fp.setText(str(stats.false_positives))
        self._fn.setText(str(stats.false_negatives))
        self._prec.setText(f"{stats.precision:.3f}")
        self._rec.setText(f"{stats.recall:.3f}")
        self._f1.setText(f"{stats.f1_score:.3f}")

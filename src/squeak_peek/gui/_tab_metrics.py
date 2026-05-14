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

from ._state import AppState
from squeak_peek.labels.model import Label


class MetricsTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        state.labels_changed.connect(self._on_labels_changed)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Counts ────────────────────────────────────────────────────────
        counts_group = QGroupBox("Label Counts")
        counts_form = QFormLayout(counts_group)
        self._det_count = QLabel("—")
        self._ref_count = QLabel("—")
        counts_form.addRow("Detected:", self._det_count)
        counts_form.addRow("Reference:", self._ref_count)
        layout.addWidget(counts_group)

        # ── Metrics ───────────────────────────────────────────────────────
        metrics_group = QGroupBox("Detection Metrics  (requires reference labels)")
        metrics_form = QFormLayout(metrics_group)
        self._tp = QLabel("—")
        self._fp = QLabel("—")
        self._fn = QLabel("—")
        self._prec = QLabel("—")
        self._rec = QLabel("—")
        self._f1 = QLabel("—")
        metrics_form.addRow("True Positives:", self._tp)
        metrics_form.addRow("False Positives:", self._fp)
        metrics_form.addRow("False Negatives:", self._fn)
        metrics_form.addRow("Precision:", self._prec)
        metrics_form.addRow("Recall:", self._rec)
        metrics_form.addRow("F1 Score:", self._f1)
        layout.addWidget(metrics_group)

        self._compute_btn = QPushButton("Compute Metrics")
        self._compute_btn.clicked.connect(self._compute)
        layout.addWidget(self._compute_btn)

        layout.addStretch()

    # ── Slots ─────────────────────────────────────────────────────────────

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

        tp, fp, fn = _iou_metrics(det, ref)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        self._tp.setText(str(tp))
        self._fp.setText(str(fp))
        self._fn.setText(str(fn))
        self._prec.setText(f"{prec:.3f}")
        self._rec.setText(f"{rec:.3f}")
        self._f1.setText(f"{f1:.3f}")


def _iou_metrics(
    detected: list[Label],
    reference: list[Label],
    min_iou: float = 0.5,
) -> tuple[int, int, int]:
    """Return (TP, FP, FN) using IoU overlap criterion."""
    matched: set[int] = set()
    tp = 0
    for d in detected:
        for i, r in enumerate(reference):
            if i in matched:
                continue
            overlap = min(d.end_time, r.end_time) - max(d.start_time, r.start_time)
            union   = max(d.end_time, r.end_time) - min(d.start_time, r.start_time)
            if union > 0 and overlap / union >= min_iou:
                tp += 1
                matched.add(i)
                break
    fp = len(detected) - tp
    fn = len(reference) - tp
    return tp, fp, fn

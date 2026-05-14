from __future__ import annotations

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._state import AppState
from squeak_peek.config import AppSettings


class SettingsTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        state.settings_changed.connect(self._reload_values)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        inner = QTabWidget()
        inner.addTab(self._make_viz_tab(), "Visualization")
        inner.addTab(self._make_psd_tab(), "PSD Detector")
        inner.addTab(self._make_post_tab(), "Post-processing")
        layout.addWidget(inner)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        load_btn = QPushButton("Load settings…")
        load_btn.clicked.connect(self._load)
        save_btn = QPushButton("Save settings…")
        save_btn.clicked.connect(self._save)
        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    # ── Sub-tabs ──────────────────────────────────────────────────────────

    def _make_viz_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        vis = self._state.settings.visualization

        self._viz_window = QSpinBox()
        self._viz_window.setRange(64, 16_384)
        self._viz_window.setValue(vis.spectrogram_window)
        form.addRow("Spectrogram window (samples):", self._viz_window)

        self._viz_overlap = QSpinBox()
        self._viz_overlap.setRange(0, 16_384)
        self._viz_overlap.setValue(vis.spectrogram_overlap)
        form.addRow("Spectrogram overlap (samples):", self._viz_overlap)

        self._viz_fmin = QDoubleSpinBox()
        self._viz_fmin.setRange(0.0, 250.0)
        self._viz_fmin.setDecimals(1)
        self._viz_fmin.setSuffix(" kHz")
        self._viz_fmin.setValue(vis.spectrogram_min_freq_khz)
        form.addRow("Min display frequency:", self._viz_fmin)

        self._viz_fmax = QDoubleSpinBox()
        self._viz_fmax.setRange(0.0, 250.0)
        self._viz_fmax.setDecimals(1)
        self._viz_fmax.setSuffix(" kHz")
        self._viz_fmax.setValue(vis.spectrogram_max_freq_khz)
        form.addRow("Max display frequency:", self._viz_fmax)

        self._viz_seg_len = QDoubleSpinBox()
        self._viz_seg_len.setRange(0.01, 60.0)
        self._viz_seg_len.setDecimals(3)
        self._viz_seg_len.setSuffix(" s")
        self._viz_seg_len.setValue(vis.segment_length_seconds)
        form.addRow("Segment length:", self._viz_seg_len)

        return w

    def _make_psd_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        psd = self._state.settings.detection.psd

        self._psd_fmin = QDoubleSpinBox()
        self._psd_fmin.setRange(0, 250_000)
        self._psd_fmin.setSuffix(" Hz")
        self._psd_fmin.setValue(psd.fcutMin)
        form.addRow("fcutMin:", self._psd_fmin)

        self._psd_fmax = QDoubleSpinBox()
        self._psd_fmax.setRange(0, 250_000)
        self._psd_fmax.setSuffix(" Hz")
        self._psd_fmax.setValue(psd.fcutMax)
        form.addRow("fcutMax:", self._psd_fmax)

        self._psd_seg = QSpinBox()
        self._psd_seg.setRange(256, 65_536)
        self._psd_seg.setValue(psd.segmentLength)
        form.addRow("segmentLength (samples):", self._psd_seg)

        self._psd_overlap = QDoubleSpinBox()
        self._psd_overlap.setRange(0.0, 0.99)
        self._psd_overlap.setDecimals(3)
        self._psd_overlap.setValue(psd.overlapFactor)
        form.addRow("overlapFactor:", self._psd_overlap)

        self._psd_k = QDoubleSpinBox()
        self._psd_k.setRange(0.0, 10.0)
        self._psd_k.setDecimals(4)
        self._psd_k.setValue(psd.k)
        form.addRow("k (threshold scale):", self._psd_k)

        self._psd_w = QDoubleSpinBox()
        self._psd_w.setRange(0.0, 1.0)
        self._psd_w.setDecimals(4)
        self._psd_w.setValue(psd.w)
        form.addRow("w (noise weight):", self._psd_w)

        return w

    def _make_post_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        post = self._state.settings.detection.post

        self._post_gap = QDoubleSpinBox()
        self._post_gap.setRange(0.0, 1.0)
        self._post_gap.setDecimals(4)
        self._post_gap.setSuffix(" s")
        self._post_gap.setValue(post.maxGapToMerge)
        form.addRow("Max gap to merge:", self._post_gap)

        self._post_min_len = QDoubleSpinBox()
        self._post_min_len.setRange(0.0, 1.0)
        self._post_min_len.setDecimals(4)
        self._post_min_len.setSuffix(" s")
        self._post_min_len.setValue(post.minLabelLength)
        form.addRow("Min label length:", self._post_min_len)

        return w

    # ── Actions ───────────────────────────────────────────────────────────

    def _apply(self) -> None:
        vis = self._state.settings.visualization
        vis.spectrogram_window          = self._viz_window.value()
        vis.spectrogram_overlap         = self._viz_overlap.value()
        vis.spectrogram_min_freq_khz    = self._viz_fmin.value()
        vis.spectrogram_max_freq_khz    = self._viz_fmax.value()
        vis.segment_length_seconds      = self._viz_seg_len.value()

        psd = self._state.settings.detection.psd
        psd.fcutMin       = self._psd_fmin.value()
        psd.fcutMax       = self._psd_fmax.value()
        psd.segmentLength = self._psd_seg.value()
        psd.overlapFactor = self._psd_overlap.value()
        psd.k             = self._psd_k.value()
        psd.w             = self._psd_w.value()

        post = self._state.settings.detection.post
        post.maxGapToMerge  = self._post_gap.value()
        post.minLabelLength = self._post_min_len.value()

        self._state.settings_changed.emit()
        self._state.segment_changed.emit()

    def _save(self) -> None:
        self._apply()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save settings", "", "JSON files (*.json)"
        )
        if path:
            self._state.settings.save_json(path)
            QMessageBox.information(self, "Saved", f"Settings written to:\n{path}")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            self._state.settings = AppSettings.from_json(path)
            self._state.settings_changed.emit()
            self._state.segment_changed.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Error loading settings", str(exc))

    def _reload_values(self) -> None:
        """Sync spinboxes after settings are replaced (e.g. via Load)."""
        vis  = self._state.settings.visualization
        psd  = self._state.settings.detection.psd
        post = self._state.settings.detection.post

        for widget, value in [
            (self._viz_window,   vis.spectrogram_window),
            (self._viz_overlap,  vis.spectrogram_overlap),
            (self._viz_fmin,     vis.spectrogram_min_freq_khz),
            (self._viz_fmax,     vis.spectrogram_max_freq_khz),
            (self._viz_seg_len,  vis.segment_length_seconds),
            (self._psd_fmin,     psd.fcutMin),
            (self._psd_fmax,     psd.fcutMax),
            (self._psd_seg,      psd.segmentLength),
            (self._psd_overlap,  psd.overlapFactor),
            (self._psd_k,        psd.k),
            (self._psd_w,        psd.w),
            (self._post_gap,     post.maxGapToMerge),
            (self._post_min_len, post.minLabelLength),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from squeak_peek.config import AppSettings

from . import _theme as t
from ._state import AppState


class SettingsTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        state.settings_changed.connect(self._reload_values)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)
        layout.setSpacing(t.SP_3)

        inner = QTabWidget()
        inner.setObjectName("innerTabs")
        inner.setDocumentMode(True)
        inner.addTab(self._make_data_input_tab(), "Data Input")
        inner.addTab(self._make_viz_tab(), "Visualization")
        inner.addTab(self._make_psd_tab(), "PSD detector")
        inner.addTab(self._make_bscd_tab(), "BSCD detector")
        inner.addTab(self._make_rbd_tab(), "RBD detector")
        inner.addTab(self._make_ml_tab(), "ML detector")
        inner.addTab(self._make_post_tab(), "Post-processing")
        inner.addTab(self._make_label_edit_tab(), "Label Edit")
        inner.addTab(self._make_appearance_tab(), "Appearance")
        layout.addWidget(inner)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        load_btn = QPushButton("Load settings…")
        load_btn.clicked.connect(self._load)
        save_btn = QPushButton("Save settings…")
        save_btn.clicked.connect(self._save)
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("primaryBtn")
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
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
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

        self._viz_colormap = QComboBox()
        colormaps = ["parula", "turbo", "hsv", "hot", "cool", "spring", "summer", "autumn", "winter", "gray", "bone", "copper", "pink", "jet", "invgray"]
        self._viz_colormap.addItems(colormaps)
        idx = self._viz_colormap.findText(vis.colormap)
        if idx >= 0:
            self._viz_colormap.setCurrentIndex(idx)
        form.addRow("Colormap:", self._viz_colormap)

        label_colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "white"]

        self._viz_label_color = QComboBox()
        self._viz_label_color.addItems(label_colors)
        idx = self._viz_label_color.findText(vis.label_color)
        if idx >= 0:
            self._viz_label_color.setCurrentIndex(idx)
        form.addRow("Label color:", self._viz_label_color)

        self._viz_ref_label_color = QComboBox()
        self._viz_ref_label_color.addItems(label_colors)
        idx = self._viz_ref_label_color.findText(vis.reference_label_color)
        if idx >= 0:
            self._viz_ref_label_color.setCurrentIndex(idx)
        form.addRow("Reference label color:", self._viz_ref_label_color)

        self._viz_manual_label_len = QDoubleSpinBox()
        self._viz_manual_label_len.setRange(0.0, 10.0)
        self._viz_manual_label_len.setDecimals(4)
        self._viz_manual_label_len.setSuffix(" s")
        self._viz_manual_label_len.setValue(vis.manual_label_length)
        form.addRow("Manual label length:", self._viz_manual_label_len)

        self._viz_show_labels = QComboBox()
        self._viz_show_labels.addItem("Show", True)
        self._viz_show_labels.addItem("Hide", False)
        self._viz_show_labels.setCurrentIndex(0 if vis.show_labels else 1)
        form.addRow("Show detected labels:", self._viz_show_labels)

        self._viz_show_ref_labels = QComboBox()
        self._viz_show_ref_labels.addItem("Show", True)
        self._viz_show_ref_labels.addItem("Hide", False)
        self._viz_show_ref_labels.setCurrentIndex(0 if vis.show_reference_labels else 1)
        form.addRow("Show reference labels:", self._viz_show_ref_labels)

        self._viz_show_loading = QComboBox()
        self._viz_show_loading.addItem("Show", True)
        self._viz_show_loading.addItem("Hide", False)
        self._viz_show_loading.setCurrentIndex(0 if vis.show_loading_dialog else 1)
        form.addRow("Show loading dialog:", self._viz_show_loading)

        self._viz_sonif_st = QDoubleSpinBox()
        self._viz_sonif_st.setRange(-100.0, 100.0)
        self._viz_sonif_st.setDecimals(1)
        self._viz_sonif_st.setSuffix(" semitones")
        self._viz_sonif_st.setValue(vis.sonification_st)
        form.addRow("Sonification semitones:", self._viz_sonif_st)

        self._viz_sonif_slowdown = QSpinBox()
        self._viz_sonif_slowdown.setRange(1, 100)
        self._viz_sonif_slowdown.setValue(vis.sonification_slowdown)
        form.addRow("Sonification slowdown factor:", self._viz_sonif_slowdown)

        return w

    def _make_data_input_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
        data_input = self._state.settings.data_input

        self._di_usv_single = QLineEdit()
        self._di_usv_single.setText(data_input.default_usv_single)
        form.addRow("Default USV (single file):", self._di_usv_single)

        self._di_label_single = QLineEdit()
        self._di_label_single.setText(data_input.default_label_single)
        form.addRow("Default labels (single file):", self._di_label_single)

        self._di_ref_label_single = QLineEdit()
        self._di_ref_label_single.setText(data_input.default_reference_label_single)
        form.addRow("Default reference labels (single file):", self._di_ref_label_single)

        self._di_batch_mode = QComboBox()
        self._di_batch_mode.addItem("Single file", False)
        self._di_batch_mode.addItem("Batch (folders)", True)
        self._di_batch_mode.setCurrentIndex(1 if data_input.batch_mode else 0)
        form.addRow("Mode:", self._di_batch_mode)

        self._di_usv_batch = QLineEdit()
        self._di_usv_batch.setText(data_input.default_usv_batch)
        form.addRow("Default USV folder (batch):", self._di_usv_batch)

        self._di_label_batch = QLineEdit()
        self._di_label_batch.setText(data_input.default_label_batch)
        form.addRow("Default labels folder (batch):", self._di_label_batch)

        self._di_ref_label_batch = QLineEdit()
        self._di_ref_label_batch.setText(data_input.default_reference_label_batch)
        form.addRow("Default reference labels folder (batch):", self._di_ref_label_batch)

        return w

    def _make_psd_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
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

    def _make_bscd_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
        bscd = self._state.settings.detection.bscd

        self._bscd_fmin = QDoubleSpinBox()
        self._bscd_fmin.setRange(0, 250_000)
        self._bscd_fmin.setSuffix(" Hz")
        self._bscd_fmin.setValue(bscd.fcutMin)
        form.addRow("fcutMin:", self._bscd_fmin)

        self._bscd_fmax = QDoubleSpinBox()
        self._bscd_fmax.setRange(0, 250_000)
        self._bscd_fmax.setSuffix(" Hz")
        self._bscd_fmax.setValue(bscd.fcutMax)
        form.addRow("fcutMax:", self._bscd_fmax)

        self._bscd_wlen = QDoubleSpinBox()
        self._bscd_wlen.setRange(0.0, 1.0)
        self._bscd_wlen.setDecimals(4)
        self._bscd_wlen.setSuffix(" s")
        self._bscd_wlen.setValue(bscd.wlen)
        form.addRow("Window length (wlen):", self._bscd_wlen)

        self._bscd_ma = QSpinBox()
        self._bscd_ma.setRange(1, 100_000)
        self._bscd_ma.setValue(bscd.maWindow)
        form.addRow("Moving average window:", self._bscd_ma)

        self._bscd_noise = QSpinBox()
        self._bscd_noise.setRange(1, 100_000)
        self._bscd_noise.setValue(bscd.noiseWindow)
        form.addRow("Noise window:", self._bscd_noise)

        self._bscd_local = QSpinBox()
        self._bscd_local.setRange(1, 100_000)
        self._bscd_local.setValue(bscd.localWindow)
        form.addRow("Local window:", self._bscd_local)

        self._bscd_k = QDoubleSpinBox()
        self._bscd_k.setRange(0.0, 10.0)
        self._bscd_k.setDecimals(4)
        self._bscd_k.setValue(bscd.k)
        form.addRow("k (threshold scale):", self._bscd_k)

        self._bscd_w = QDoubleSpinBox()
        self._bscd_w.setRange(0.0, 1.0)
        self._bscd_w.setDecimals(4)
        self._bscd_w.setValue(bscd.w)
        form.addRow("w (noise weight):", self._bscd_w)

        return w

    def _make_rbd_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
        rbd = self._state.settings.detection.rbd

        self._rbd_fmin = QDoubleSpinBox()
        self._rbd_fmin.setRange(0, 250_000)
        self._rbd_fmin.setSuffix(" Hz")
        self._rbd_fmin.setValue(rbd.fcutMin)
        form.addRow("fcutMin:", self._rbd_fmin)

        self._rbd_fmax = QDoubleSpinBox()
        self._rbd_fmax.setRange(0, 250_000)
        self._rbd_fmax.setSuffix(" Hz")
        self._rbd_fmax.setValue(rbd.fcutMax)
        form.addRow("fcutMax:", self._rbd_fmax)

        self._rbd_wlen = QDoubleSpinBox()
        self._rbd_wlen.setRange(0.0, 1.0)
        self._rbd_wlen.setDecimals(4)
        self._rbd_wlen.setSuffix(" s")
        self._rbd_wlen.setValue(rbd.wlen)
        form.addRow("Window length (wlen):", self._rbd_wlen)

        self._rbd_ar_left = QSpinBox()
        self._rbd_ar_left.setRange(0, 50)
        self._rbd_ar_left.setValue(rbd.AR_order_left)
        form.addRow("AR order (left):", self._rbd_ar_left)

        self._rbd_ar_right = QSpinBox()
        self._rbd_ar_right.setRange(0, 50)
        self._rbd_ar_right.setValue(rbd.AR_order_right)
        form.addRow("AR order (right):", self._rbd_ar_right)

        self._rbd_bayes_order = QSpinBox()
        self._rbd_bayes_order.setRange(0, 50)
        self._rbd_bayes_order.setValue(rbd.Bayesian_Evidence_order)
        form.addRow("Bayesian Evidence order:", self._rbd_bayes_order)

        self._rbd_dynamic_scaling = QDoubleSpinBox()
        self._rbd_dynamic_scaling.setRange(0.0, 10.0)
        self._rbd_dynamic_scaling.setDecimals(4)
        self._rbd_dynamic_scaling.setValue(rbd.dynamicScaling)
        form.addRow("Dynamic scaling:", self._rbd_dynamic_scaling)

        self._rbd_smooth_rbd = QDoubleSpinBox()
        self._rbd_smooth_rbd.setRange(0.0, 1.0)
        self._rbd_smooth_rbd.setDecimals(4)
        self._rbd_smooth_rbd.setSuffix(" s")
        self._rbd_smooth_rbd.setValue(rbd.smoothingWindowRBD)
        form.addRow("Smoothing window (RBD):", self._rbd_smooth_rbd)

        self._rbd_smooth_thr = QDoubleSpinBox()
        self._rbd_smooth_thr.setRange(0.0, 1.0)
        self._rbd_smooth_thr.setDecimals(4)
        self._rbd_smooth_thr.setSuffix(" s")
        self._rbd_smooth_thr.setValue(rbd.smoothingWindowThr)
        form.addRow("Smoothing window (threshold):", self._rbd_smooth_thr)

        self._rbd_amplitude = QDoubleSpinBox()
        self._rbd_amplitude.setRange(0.0, 1.0)
        self._rbd_amplitude.setDecimals(4)
        self._rbd_amplitude.setValue(rbd.amplitudeThreshold)
        form.addRow("Amplitude threshold:", self._rbd_amplitude)

        return w

    def _make_ml_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
        ml = self._state.settings.detection.ml

        # Model path with browse button
        model_row = QHBoxLayout()
        self._ml_model_path = QLineEdit()
        self._ml_model_path.setText(ml.modelPath)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_ml_model)
        model_row.addWidget(self._ml_model_path)
        model_row.addWidget(browse_btn)
        form.addRow("Model path (.joblib):", model_row)

        self._ml_sensitivity = QDoubleSpinBox()
        self._ml_sensitivity.setRange(0.0, 1.0)
        self._ml_sensitivity.setDecimals(4)
        self._ml_sensitivity.setValue(ml.sensitivity)
        form.addRow("Sensitivity (0-1):", self._ml_sensitivity)

        self._ml_min_duration = QDoubleSpinBox()
        self._ml_min_duration.setRange(0.0, 10.0)
        self._ml_min_duration.setDecimals(4)
        self._ml_min_duration.setSuffix(" s")
        self._ml_min_duration.setValue(ml.minEventDuration)
        form.addRow("Min event duration:", self._ml_min_duration)

        return w

    def _browse_ml_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ML model", "", "Joblib files (*.joblib)"
        )
        if path:
            self._ml_model_path.setText(path)

    def _make_post_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
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

    def _make_label_edit_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
        label_edit = self._state.settings.label_edit

        self._le_window = QSpinBox()
        self._le_window.setRange(64, 16_384)
        self._le_window.setValue(label_edit.spectrogram_window)
        form.addRow("Spectrogram window (samples):", self._le_window)

        self._le_overlap = QSpinBox()
        self._le_overlap.setRange(0, 16_384)
        self._le_overlap.setValue(label_edit.spectrogram_overlap)
        form.addRow("Spectrogram overlap (samples):", self._le_overlap)

        self._le_fmin = QDoubleSpinBox()
        self._le_fmin.setRange(0.0, 250.0)
        self._le_fmin.setDecimals(1)
        self._le_fmin.setSuffix(" kHz")
        self._le_fmin.setValue(label_edit.spectrogram_min_freq_khz)
        form.addRow("Min display frequency:", self._le_fmin)

        self._le_fmax = QDoubleSpinBox()
        self._le_fmax.setRange(0.0, 250.0)
        self._le_fmax.setDecimals(1)
        self._le_fmax.setSuffix(" kHz")
        self._le_fmax.setValue(label_edit.spectrogram_max_freq_khz)
        form.addRow("Max display frequency:", self._le_fmax)

        self._le_colormap = QComboBox()
        colormaps = ["parula", "turbo", "hsv", "hot", "cool", "spring", "summer", "autumn", "winter", "gray", "bone", "copper", "pink", "jet", "invgray"]
        self._le_colormap.addItems(colormaps)
        idx = self._le_colormap.findText(label_edit.colormap)
        if idx >= 0:
            self._le_colormap.setCurrentIndex(idx)
        form.addRow("Colormap:", self._le_colormap)

        self._le_classifications = QLineEdit()
        self._le_classifications.setText(label_edit.classifications)
        form.addRow("Classifications (comma-separated):", self._le_classifications)

        return w

    def _make_appearance_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)

        self._theme_combo = QComboBox()
        self._theme_combo.addItem("System", "system")
        self._theme_combo.addItem("Light", "light")
        self._theme_combo.addItem("Dark", "dark")
        self._theme_combo.setCurrentIndex(max(self._theme_combo.findData(t.get_mode()), 0))
        self._theme_combo.currentIndexChanged.connect(self._on_theme_mode_changed)
        form.addRow("Color mode:", self._theme_combo)

        # WP27 THEME DECISION (NOT IMPLEMENTED):
        # config.ThemeSettings (MATLAB Light/Gray/Custom plus 4 raw RGB color pickers)
        # is intentionally NOT wired into this UI. The MATLAB app had Light/Gray/Custom
        # theme presets with custom RGB color pickers. Since landing WP27, we have a
        # newer design-token dark-mode system (System/Light/Dark toggle above, not
        # related to ThemeSettings). Reconciling these two overlapping theme systems
        # is a product decision for a human to make, not a mechanical port. The settings
        # round-trip correctly through save/load JSON (config.ThemeSettings fields are
        # preserved even if not shown in UI), so no data is lost. A future WP27b could
        # either (a) add a 4th "Custom" mode with RGB pickers, or (b) deprecate the raw
        # ThemeSettings in favor of the cleaner design-token approach. Choose one.

        return w

    def _on_theme_mode_changed(self, index: int) -> None:
        t.set_mode(self._theme_combo.itemData(index))

    # ── Actions ───────────────────────────────────────────────────────────

    def _apply(self) -> None:
        # Data Input
        data_input = self._state.settings.data_input
        data_input.default_usv_single = self._di_usv_single.text()
        data_input.default_label_single = self._di_label_single.text()
        data_input.default_reference_label_single = self._di_ref_label_single.text()
        data_input.batch_mode = self._di_batch_mode.currentData()
        data_input.default_usv_batch = self._di_usv_batch.text()
        data_input.default_label_batch = self._di_label_batch.text()
        data_input.default_reference_label_batch = self._di_ref_label_batch.text()

        # Visualization
        vis = self._state.settings.visualization
        vis.spectrogram_window          = self._viz_window.value()
        vis.spectrogram_overlap         = self._viz_overlap.value()
        vis.spectrogram_min_freq_khz    = self._viz_fmin.value()
        vis.spectrogram_max_freq_khz    = self._viz_fmax.value()
        vis.segment_length_seconds      = self._viz_seg_len.value()
        vis.colormap                    = self._viz_colormap.currentText()
        vis.label_color                 = self._viz_label_color.currentText()
        vis.reference_label_color       = self._viz_ref_label_color.currentText()
        vis.manual_label_length         = self._viz_manual_label_len.value()
        vis.show_labels                 = self._viz_show_labels.currentData()
        vis.show_reference_labels       = self._viz_show_ref_labels.currentData()
        vis.show_loading_dialog         = self._viz_show_loading.currentData()
        vis.sonification_st             = self._viz_sonif_st.value()
        vis.sonification_slowdown       = self._viz_sonif_slowdown.value()

        # PSD
        psd = self._state.settings.detection.psd
        psd.fcutMin       = self._psd_fmin.value()
        psd.fcutMax       = self._psd_fmax.value()
        psd.segmentLength = self._psd_seg.value()
        psd.overlapFactor = self._psd_overlap.value()
        psd.k             = self._psd_k.value()
        psd.w             = self._psd_w.value()

        # BSCD
        bscd = self._state.settings.detection.bscd
        bscd.fcutMin       = self._bscd_fmin.value()
        bscd.fcutMax       = self._bscd_fmax.value()
        bscd.wlen          = self._bscd_wlen.value()
        bscd.maWindow      = self._bscd_ma.value()
        bscd.noiseWindow   = self._bscd_noise.value()
        bscd.localWindow   = self._bscd_local.value()
        bscd.k             = self._bscd_k.value()
        bscd.w             = self._bscd_w.value()

        # RBD
        rbd = self._state.settings.detection.rbd
        rbd.fcutMin                  = self._rbd_fmin.value()
        rbd.fcutMax                  = self._rbd_fmax.value()
        rbd.wlen                     = self._rbd_wlen.value()
        rbd.AR_order_left            = self._rbd_ar_left.value()
        rbd.AR_order_right           = self._rbd_ar_right.value()
        rbd.Bayesian_Evidence_order  = self._rbd_bayes_order.value()
        rbd.dynamicScaling           = self._rbd_dynamic_scaling.value()
        rbd.smoothingWindowRBD       = self._rbd_smooth_rbd.value()
        rbd.smoothingWindowThr       = self._rbd_smooth_thr.value()
        rbd.amplitudeThreshold       = self._rbd_amplitude.value()

        # ML
        ml = self._state.settings.detection.ml
        ml.modelPath       = self._ml_model_path.text()
        ml.sensitivity     = self._ml_sensitivity.value()
        ml.minEventDuration = self._ml_min_duration.value()

        # Post-processing
        post = self._state.settings.detection.post
        post.maxGapToMerge  = self._post_gap.value()
        post.minLabelLength = self._post_min_len.value()

        # Label Edit
        label_edit = self._state.settings.label_edit
        label_edit.spectrogram_window    = self._le_window.value()
        label_edit.spectrogram_overlap   = self._le_overlap.value()
        label_edit.spectrogram_min_freq_khz = self._le_fmin.value()
        label_edit.spectrogram_max_freq_khz = self._le_fmax.value()
        label_edit.colormap              = self._le_colormap.currentText()
        label_edit.classifications       = self._le_classifications.text()

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
        """Sync all widgets after settings are replaced (e.g. via Load)."""
        data_input = self._state.settings.data_input
        vis = self._state.settings.visualization
        psd = self._state.settings.detection.psd
        bscd = self._state.settings.detection.bscd
        rbd = self._state.settings.detection.rbd
        ml = self._state.settings.detection.ml
        post = self._state.settings.detection.post
        label_edit = self._state.settings.label_edit

        # Data Input
        spinbox_updates = [
            (self._di_usv_single, data_input.default_usv_single, "setText"),
            (self._di_label_single, data_input.default_label_single, "setText"),
            (self._di_ref_label_single, data_input.default_reference_label_single, "setText"),
            (self._di_usv_batch, data_input.default_usv_batch, "setText"),
            (self._di_label_batch, data_input.default_label_batch, "setText"),
            (self._di_ref_label_batch, data_input.default_reference_label_batch, "setText"),
        ]
        for widget, value, method in spinbox_updates:
            widget.blockSignals(True)
            getattr(widget, method)(value)
            widget.blockSignals(False)

        self._di_batch_mode.blockSignals(True)
        self._di_batch_mode.setCurrentIndex(1 if data_input.batch_mode else 0)
        self._di_batch_mode.blockSignals(False)

        # Visualization
        for widget, value in [
            (self._viz_window,   vis.spectrogram_window),
            (self._viz_overlap,  vis.spectrogram_overlap),
            (self._viz_fmin,     vis.spectrogram_min_freq_khz),
            (self._viz_fmax,     vis.spectrogram_max_freq_khz),
            (self._viz_seg_len,  vis.segment_length_seconds),
            (self._viz_manual_label_len, vis.manual_label_length),
            (self._viz_sonif_st, vis.sonification_st),
            (self._viz_sonif_slowdown, vis.sonification_slowdown),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        for widget, value in [
            (self._viz_colormap, vis.colormap),
            (self._viz_label_color, vis.label_color),
            (self._viz_ref_label_color, vis.reference_label_color),
        ]:
            widget.blockSignals(True)
            idx = widget.findText(value)
            if idx >= 0:
                widget.setCurrentIndex(idx)
            widget.blockSignals(False)

        self._viz_show_labels.blockSignals(True)
        self._viz_show_labels.setCurrentIndex(0 if vis.show_labels else 1)
        self._viz_show_labels.blockSignals(False)

        self._viz_show_ref_labels.blockSignals(True)
        self._viz_show_ref_labels.setCurrentIndex(0 if vis.show_reference_labels else 1)
        self._viz_show_ref_labels.blockSignals(False)

        self._viz_show_loading.blockSignals(True)
        self._viz_show_loading.setCurrentIndex(0 if vis.show_loading_dialog else 1)
        self._viz_show_loading.blockSignals(False)

        # PSD
        for widget, value in [
            (self._psd_fmin,     psd.fcutMin),
            (self._psd_fmax,     psd.fcutMax),
            (self._psd_seg,      psd.segmentLength),
            (self._psd_overlap,  psd.overlapFactor),
            (self._psd_k,        psd.k),
            (self._psd_w,        psd.w),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        # BSCD
        for widget, value in [
            (self._bscd_fmin,     bscd.fcutMin),
            (self._bscd_fmax,     bscd.fcutMax),
            (self._bscd_wlen,     bscd.wlen),
            (self._bscd_k,        bscd.k),
            (self._bscd_w,        bscd.w),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        for widget, value in [
            (self._bscd_ma,       bscd.maWindow),
            (self._bscd_noise,    bscd.noiseWindow),
            (self._bscd_local,    bscd.localWindow),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        # RBD
        for widget, value in [
            (self._rbd_fmin,            rbd.fcutMin),
            (self._rbd_fmax,            rbd.fcutMax),
            (self._rbd_wlen,            rbd.wlen),
            (self._rbd_dynamic_scaling, rbd.dynamicScaling),
            (self._rbd_smooth_rbd,      rbd.smoothingWindowRBD),
            (self._rbd_smooth_thr,      rbd.smoothingWindowThr),
            (self._rbd_amplitude,       rbd.amplitudeThreshold),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        for widget, value in [
            (self._rbd_ar_left,     rbd.AR_order_left),
            (self._rbd_ar_right,    rbd.AR_order_right),
            (self._rbd_bayes_order, rbd.Bayesian_Evidence_order),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        # ML
        self._ml_model_path.blockSignals(True)
        self._ml_model_path.setText(ml.modelPath)
        self._ml_model_path.blockSignals(False)

        for widget, value in [
            (self._ml_sensitivity, ml.sensitivity),
            (self._ml_min_duration, ml.minEventDuration),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        # Post-processing
        for widget, value in [
            (self._post_gap,     post.maxGapToMerge),
            (self._post_min_len, post.minLabelLength),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        # Label Edit
        for widget, value in [
            (self._le_fmin, label_edit.spectrogram_min_freq_khz),
            (self._le_fmax, label_edit.spectrogram_max_freq_khz),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        for widget, value in [
            (self._le_window, label_edit.spectrogram_window),
            (self._le_overlap, label_edit.spectrogram_overlap),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        self._le_colormap.blockSignals(True)
        idx = self._le_colormap.findText(label_edit.colormap)
        if idx >= 0:
            self._le_colormap.setCurrentIndex(idx)
        self._le_colormap.blockSignals(False)

        self._le_classifications.blockSignals(True)
        self._le_classifications.setText(label_edit.classifications)
        self._le_classifications.blockSignals(False)

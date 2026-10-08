from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import squeak_peek.classifiers  # noqa: F401  (registers built-in classifiers)
import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek.audio.colormaps import COLORMAP_NAMES
from squeak_peek.classifiers.base import AbstractClassifier
from squeak_peek.config import AppSettings
from squeak_peek.detectors.base import AbstractDetector

from . import _plugin_form as pf
from . import _shortcuts as shortcuts
from . import _theme as t
from ._numeric_indicator import RangeIndicator, bind_range, bind_value
from ._state import AppState


class SettingsTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._detector_widgets: dict[str, dict[str, QWidget]] = {}
        self._classifier_widgets: dict[str, dict[str, QWidget]] = {}
        self._setup_ui()
        state.settings_changed.connect(self._reload_values)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)
        layout.setSpacing(t.SP_3)

        sub_tabs = [
            (self._make_data_input_tab(), "Data Input",
             "Default files/folders auto-loaded at startup, and single-vs-batch mode."),
            (self._make_viz_tab(), "Visualization",
             "Spectrogram rendering, overlays and sonification for the Visualization tab."),
        ]
        pre_tab, self._pre_widgets = self._make_plugin_tab(self._state.settings.detection.pre)
        sub_tabs.append((pre_tab, "Pre-processing",
             "Optional noise suppression applied to the audio before the classical detectors run."))
        for detector_cls in AbstractDetector.all():
            params = self._state.settings.detection.params_for(detector_cls.id)
            tab, widgets = self._make_plugin_tab(params)
            self._detector_widgets[detector_cls.id] = widgets
            sub_tabs.append((tab, f"{detector_cls.display_name} detector", detector_cls.description))
        sub_tabs.append((self._make_post_tab(), "Post-processing",
             "Merging and filtering rules applied to detections after any detector runs."))
        for classifier_cls in AbstractClassifier.all():
            params = self._state.settings.classification.params_for(classifier_cls.id)
            tab, widgets = self._make_plugin_tab(params)
            self._classifier_widgets[classifier_cls.id] = widgets
            sub_tabs.append((tab, f"{classifier_cls.display_name} classifier", classifier_cls.description))
        sub_tabs += [
            (self._make_label_edit_tab(), "Label Edit",
             "Spectrogram rendering and call-type list for the Label Edit tab."),
            (self._make_video_tab(), "Video",
             "Finger-snap sync-click detection used when importing a behavior video."),
            (self._make_appearance_tab(), "Appearance",
             "The app's color scheme."),
            (self._make_shortcuts_tab(), "Shortcuts",
             "View and customize keyboard shortcuts."),
        ]
        # Section list on the left instead of a tab bar: with one sub-tab per
        # detector/classifier plugin there were too many to fit in one row.
        body = QHBoxLayout()
        body.setSpacing(t.SP_4)
        self._nav = QListWidget()
        self._nav.setObjectName("settingsNav")
        self._nav.setFixedWidth(230)
        self._pages = QStackedWidget()
        for widget, title, tooltip in sub_tabs:
            item = QListWidgetItem(title)
            item.setToolTip(tooltip)
            self._nav.addItem(item)
            self._pages.addWidget(self._scroll_page(widget))
        self._nav.currentRowChanged.connect(self._pages.setCurrentIndex)
        self._nav.setCurrentRow(0)
        body.addWidget(self._nav)
        body.addWidget(self._pages, 1)
        layout.addLayout(body, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        load_btn = QPushButton("Load settings…")
        load_btn.setToolTip("Replace all settings below by loading them from a JSON file.")
        load_btn.clicked.connect(self._load)
        save_btn = QPushButton("Save settings…")
        save_btn.setToolTip("Write the current settings (including unapplied edits) to a JSON file.")
        save_btn.clicked.connect(self._save)
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("primaryBtn")
        apply_btn.setDefault(True)
        apply_btn.setToolTip("Apply the edited values above to the running app.")
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _scroll_page(widget: QWidget) -> QScrollArea:
        """Scrollable page, its form capped at a readable width."""
        widget.setMaximumWidth(920)
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(widget, 1)
        row.addStretch(0)
        lay.addLayout(row)
        lay.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(holder)
        return area

    # ── Row-building helpers ─────────────────────────────────────────────

    @staticmethod
    def _new_form(w: QWidget) -> QFormLayout:
        form = QFormLayout(w)
        form.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        form.setVerticalSpacing(t.SP_2)
        form.setHorizontalSpacing(t.SP_3)
        return form

    @staticmethod
    def _caption(text: str) -> QLabel:
        cap = QLabel(text)
        cap.setWordWrap(True)
        cap.setStyleSheet(f"color: {t.TEXT_MUTED}; font-size: {t.TEXT_XS}px;")
        # A margin, not QSS padding: padding on a word-wrapped label breaks
        # its height-for-width, clipping the second line.
        cap.setContentsMargins(0, 0, 0, t.SP_2)
        return cap

    def _field(
        self,
        form: QFormLayout,
        label_text: str,
        widget: QWidget,
        tooltip: str,
        caption: str | None = None,
        indicator: bool = True,
    ) -> QWidget:
        """Add one settings row: label + control, both carrying ``tooltip``
        as a hover description, an optional muted caption line beneath it,
        and — for spin boxes — a small bar showing the value's position
        within its allowed range.
        """
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(label, widget)

        if caption:
            form.addRow(self._caption(caption))

        if indicator and isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            ind = RangeIndicator(unit=widget.suffix())
            bind_value(widget, ind)
            form.addRow(ind)

        return widget

    def _range_indicator(
        self,
        form: QFormLayout,
        spin_lo: QSpinBox | QDoubleSpinBox,
        spin_hi: QSpinBox | QDoubleSpinBox,
    ) -> None:
        """A single combined bar spanning both bounds of a lo/hi pair
        (e.g. a frequency band), shown once under the second field."""
        ind = RangeIndicator(spin_lo.minimum(), spin_hi.maximum(), unit=spin_hi.suffix())
        bind_range(spin_lo, spin_hi, ind)
        form.addRow(ind)

    # ── Sub-tabs ──────────────────────────────────────────────────────────

    def _make_viz_tab(self) -> QWidget:
        w = QWidget()
        form = self._new_form(w)
        vis = self._state.settings.visualization

        self._viz_window = QSpinBox()
        self._viz_window.setRange(64, 16_384)
        self._viz_window.setValue(vis.spectrogram_window)
        self._field(
            form, "Spectrogram window (samples):", self._viz_window,
            "Number of audio samples per FFT window used to draw the spectrogram.",
            "Larger windows sharpen frequency detail but blur fast time changes; "
            "smaller windows do the opposite.",
        )

        self._viz_overlap = QSpinBox()
        self._viz_overlap.setRange(0, 16_384)
        self._viz_overlap.setValue(vis.spectrogram_overlap)
        self._field(
            form, "Spectrogram overlap (samples):", self._viz_overlap,
            "Samples shared between consecutive FFT windows. Must be smaller than the window size.",
            "Higher overlap gives a smoother-looking spectrogram at the cost of more computation.",
        )

        self._viz_fmin = QDoubleSpinBox()
        self._viz_fmin.setRange(0.0, 250.0)
        self._viz_fmin.setDecimals(1)
        self._viz_fmin.setSuffix(" kHz")
        self._viz_fmin.setValue(vis.spectrogram_min_freq_khz)
        self._field(
            form, "Min display frequency:", self._viz_fmin,
            "Lower bound of the frequency axis shown in the spectrogram.",
            indicator=False,
        )

        self._viz_fmax = QDoubleSpinBox()
        self._viz_fmax.setRange(0.0, 250.0)
        self._viz_fmax.setDecimals(1)
        self._viz_fmax.setSuffix(" kHz")
        self._viz_fmax.setValue(vis.spectrogram_max_freq_khz)
        self._field(
            form, "Max display frequency:", self._viz_fmax,
            "Upper bound of the frequency axis shown in the spectrogram.",
            "Does not affect detection — only what part of the spectrum is drawn.",
            indicator=False,
        )
        self._range_indicator(form, self._viz_fmin, self._viz_fmax)

        self._viz_seg_start = QDoubleSpinBox()
        self._viz_seg_start.setRange(0.0, 99_999.0)
        self._viz_seg_start.setDecimals(3)
        self._viz_seg_start.setSuffix(" s")
        self._viz_seg_start.setValue(vis.segment_start_seconds)
        self._field(
            form, "Initial segment start:", self._viz_seg_start,
            "Segment start time jumped to whenever a new WAV file is loaded.",
            indicator=False,
        )

        self._viz_seg_len = QDoubleSpinBox()
        self._viz_seg_len.setRange(0.01, 60.0)
        self._viz_seg_len.setDecimals(3)
        self._viz_seg_len.setSuffix(" s")
        self._viz_seg_len.setValue(vis.segment_length_seconds)
        self._field(
            form, "Segment length:", self._viz_seg_len,
            "Length of the audio segment shown at once in the Visualization tab.",
            "Shorter segments zoom in on detail; longer segments show more context per view.",
        )

        self._viz_colormap = QComboBox()
        self._viz_colormap.addItems(COLORMAP_NAMES)
        idx = self._viz_colormap.findText(vis.colormap)
        if idx >= 0:
            self._viz_colormap.setCurrentIndex(idx)
        self._field(
            form, "Colormap:", self._viz_colormap,
            "Color palette used to render spectrogram intensity.",
        )

        label_colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "white"]

        self._viz_label_color = QComboBox()
        self._viz_label_color.addItems(label_colors)
        idx = self._viz_label_color.findText(vis.label_color)
        if idx >= 0:
            self._viz_label_color.setCurrentIndex(idx)
        self._field(
            form, "Label color:", self._viz_label_color,
            "Color used to draw detected-label boxes on the spectrogram.",
        )

        self._viz_ref_label_color = QComboBox()
        self._viz_ref_label_color.addItems(label_colors)
        idx = self._viz_ref_label_color.findText(vis.reference_label_color)
        if idx >= 0:
            self._viz_ref_label_color.setCurrentIndex(idx)
        self._field(
            form, "Reference label color:", self._viz_ref_label_color,
            "Color used to draw reference-label boxes on the spectrogram.",
        )

        self._viz_manual_label_len = QDoubleSpinBox()
        self._viz_manual_label_len.setRange(0.0, 10.0)
        self._viz_manual_label_len.setDecimals(4)
        self._viz_manual_label_len.setSuffix(" s")
        self._viz_manual_label_len.setValue(vis.manual_label_length)
        self._field(
            form, "Manual label length:", self._viz_manual_label_len,
            "Default duration given to a label you create by right-clicking the spectrogram.",
            "The new label is centered on the click point; drag its edges afterward to adjust it.",
        )

        self._viz_manual_label_marker = QLineEdit()
        self._viz_manual_label_marker.setText(vis.manual_label_marker)
        self._field(
            form, "Manual label text:", self._viz_manual_label_marker,
            "Label text given to a label you create by right-clicking the spectrogram.",
        )

        self._viz_show_labels = QComboBox()
        self._viz_show_labels.addItem("Show", True)
        self._viz_show_labels.addItem("Hide", False)
        self._viz_show_labels.setCurrentIndex(0 if vis.show_labels else 1)
        self._field(
            form, "Show detected labels:", self._viz_show_labels,
            "Whether detected-label boxes are drawn on the spectrogram by default.",
        )

        self._viz_show_ref_labels = QComboBox()
        self._viz_show_ref_labels.addItem("Show", True)
        self._viz_show_ref_labels.addItem("Hide", False)
        self._viz_show_ref_labels.setCurrentIndex(0 if vis.show_reference_labels else 1)
        self._field(
            form, "Show reference labels:", self._viz_show_ref_labels,
            "Whether reference-label boxes are drawn on the spectrogram by default.",
        )

        self._viz_show_loading = QComboBox()
        self._viz_show_loading.addItem("Show", True)
        self._viz_show_loading.addItem("Hide", False)
        self._viz_show_loading.setCurrentIndex(0 if vis.show_loading_dialog else 1)
        self._field(
            form, "Show loading dialog:", self._viz_show_loading,
            "Whether a progress dialog appears while a large WAV file is loading.",
        )

        self._viz_sonif_st = QDoubleSpinBox()
        self._viz_sonif_st.setRange(-100.0, 100.0)
        self._viz_sonif_st.setDecimals(1)
        self._viz_sonif_st.setSuffix(" semitones")
        self._viz_sonif_st.setValue(vis.sonification_st)
        self._field(
            form, "Sonification semitones:", self._viz_sonif_st,
            "Pitch shift applied when sonifying a segment so ultrasonic calls become audible.",
            "More negative values shift the pitch down further, bringing high-frequency "
            "calls deeper into human hearing range. −36 is exactly three octaves "
            "(a division by 8), which puts 50 kHz calls at about 6 kHz.",
        )

        self._viz_sonif_natural = QComboBox()
        self._viz_sonif_natural.addItem("Match the pitch shift (best quality)", True)
        self._viz_sonif_natural.addItem("Use the slowdown factor below", False)
        self._viz_sonif_natural.setCurrentIndex(0 if vis.sonification_natural_speed else 1)
        self._field(
            form, "Sonification playback speed:", self._viz_sonif_natural,
            "How far sonified playback is stretched out in time.",
            "Matching the pitch shift makes sonification a pure tape-speed transform — "
            "exactly what a time-expansion bat detector does. Nothing has to be "
            "resynthesised, so there are no stretching artefacts at all. Choosing an "
            "independent slowdown factor costs one phase-vocoder pass and sounds "
            "slightly less clean.",
        )

        self._viz_sonif_slowdown = QSpinBox()
        self._viz_sonif_slowdown.setRange(1, 100)
        self._viz_sonif_slowdown.setValue(vis.sonification_slowdown)
        self._field(
            form, "Sonification slowdown factor:", self._viz_sonif_slowdown,
            "Factor by which sonified playback is slowed down.",
            "Only used when the playback speed above is set to use this factor. "
            "Higher values stretch playback out longer.",
        )

        self._viz_sonif_denoise = QComboBox()
        self._viz_sonif_denoise.addItem("On", True)
        self._viz_sonif_denoise.addItem("Off", False)
        self._viz_sonif_denoise.setCurrentIndex(0 if vis.sonification_denoise else 1)
        self._field(
            form, "Sonification denoising:", self._viz_sonif_denoise,
            "Suppress the recording's broadband noise floor before shifting it down.",
            "An ultrasonic recording is mostly noise; shifted into the audible range "
            "that noise becomes a wall of hiss that buries the calls. A spectral gate "
            "keyed on the per-bin noise floor leaves the calls standing out of a quiet "
            "background. Turn it off to hear the raw, unmodified signal.",
        )

        return w

    def _make_data_input_tab(self) -> QWidget:
        w = QWidget()
        form = self._new_form(w)
        data_input = self._state.settings.data_input

        self._di_usv_single = QLineEdit()
        self._di_usv_single.setText(data_input.default_usv_single)
        self._field(
            form, "Default USV (single file):", self._di_usv_single,
            "WAV file automatically loaded at startup when in single-file mode.",
            indicator=False,
        )

        self._di_label_single = QLineEdit()
        self._di_label_single.setText(data_input.default_label_single)
        self._field(
            form, "Default labels (single file):", self._di_label_single,
            "Detected-label file automatically loaded at startup when in single-file mode.",
            indicator=False,
        )

        self._di_ref_label_single = QLineEdit()
        self._di_ref_label_single.setText(data_input.default_reference_label_single)
        self._field(
            form, "Default reference labels (single file):", self._di_ref_label_single,
            "Reference-label file automatically loaded at startup when in single-file mode.",
            indicator=False,
        )

        self._di_batch_mode = QComboBox()
        self._di_batch_mode.addItem("Single file", False)
        self._di_batch_mode.addItem("Batch (folders)", True)
        self._di_batch_mode.setCurrentIndex(1 if data_input.batch_mode else 0)
        self._field(
            form, "Mode:", self._di_batch_mode,
            "Whether the Data Input tab starts in single-file or batch (folder) mode.",
        )

        self._di_usv_batch = QLineEdit()
        self._di_usv_batch.setText(data_input.default_usv_batch)
        self._field(
            form, "Default USV folder (batch):", self._di_usv_batch,
            "Folder of WAV files automatically loaded at startup when in batch mode.",
            indicator=False,
        )

        self._di_label_batch = QLineEdit()
        self._di_label_batch.setText(data_input.default_label_batch)
        self._field(
            form, "Default labels folder (batch):", self._di_label_batch,
            "Folder of detected-label files automatically loaded at startup when in batch mode.",
            indicator=False,
        )

        self._di_ref_label_batch = QLineEdit()
        self._di_ref_label_batch.setText(data_input.default_reference_label_batch)
        self._field(
            form, "Default reference labels folder (batch):", self._di_ref_label_batch,
            "Folder of reference-label files automatically loaded at startup when in batch mode.",
            indicator=False,
        )

        return w

    def _make_plugin_tab(self, params) -> tuple[QWidget, dict[str, QWidget]]:
        """Build a Settings sub-tab for one detector/classifier's Params model
        (auto-generated — see gui._plugin_form.build_params_form)."""
        w = QWidget()
        form = self._new_form(w)
        widgets = pf.build_params_form(form, params, self._field, self._range_indicator)
        return w, widgets

    def _make_post_tab(self) -> QWidget:
        w = QWidget()
        form = self._new_form(w)
        detection = self._state.settings.detection
        post = detection.post

        export_row = QWidget()
        export_row_layout = QHBoxLayout(export_row)
        export_row_layout.setContentsMargins(0, 0, 0, 0)
        self._det_export_path = QLineEdit()
        self._det_export_path.setReadOnly(True)
        self._det_export_path.setText(detection.export_path)
        self._det_export_path.setPlaceholderText("Same folder as WAV file")
        export_browse_btn = QPushButton("Browse…")
        export_browse_btn.clicked.connect(self._browse_det_export_path)
        export_clear_btn = QPushButton("Clear")
        export_clear_btn.clicked.connect(lambda: self._det_export_path.setText(""))
        export_row_layout.addWidget(self._det_export_path)
        export_row_layout.addWidget(export_browse_btn)
        export_row_layout.addWidget(export_clear_btn)
        self._field(
            form, "Default export folder:", export_row,
            "Default folder the Detection tab's export folder is set to on startup. "
            "Leave blank to default to the loaded WAV file's own folder.",
            indicator=False,
        )

        self._post_gap = QDoubleSpinBox()
        self._post_gap.setRange(0.0, 1.0)
        self._post_gap.setDecimals(4)
        self._post_gap.setSuffix(" s")
        self._post_gap.setValue(post.maxGapToMerge)
        self._field(
            form, "Max gap to merge:", self._post_gap,
            "Detections separated by a gap shorter than this are merged into one label.",
            "Used by the \"Merge Close Labels\" post-processing step in the Detection tab.",
        )

        self._post_min_len = QDoubleSpinBox()
        self._post_min_len.setRange(0.0, 1.0)
        self._post_min_len.setDecimals(4)
        self._post_min_len.setSuffix(" s")
        self._post_min_len.setValue(post.minLabelLength)
        self._field(
            form, "Min label length:", self._post_min_len,
            "Labels shorter than this are discarded after merging.",
            "Used by the \"Remove Short Labels\" post-processing step in the Detection tab.",
        )

        return w

    def _make_label_edit_tab(self) -> QWidget:
        w = QWidget()
        form = self._new_form(w)
        label_edit = self._state.settings.label_edit

        self._le_window = QSpinBox()
        self._le_window.setRange(64, 16_384)
        self._le_window.setValue(label_edit.spectrogram_window)
        self._field(
            form, "Spectrogram window (samples):", self._le_window,
            "Number of audio samples per FFT window used to draw each label's spectrogram.",
            "Larger windows sharpen frequency detail but blur fast time changes; "
            "smaller windows do the opposite.",
        )

        self._le_overlap = QSpinBox()
        self._le_overlap.setRange(0, 16_384)
        self._le_overlap.setValue(label_edit.spectrogram_overlap)
        self._field(
            form, "Spectrogram overlap (samples):", self._le_overlap,
            "Samples shared between consecutive FFT windows. Must be smaller than the window size.",
            "Higher overlap gives a smoother-looking spectrogram at the cost of more computation.",
        )

        self._le_fmin = QDoubleSpinBox()
        self._le_fmin.setRange(0.0, 250.0)
        self._le_fmin.setDecimals(1)
        self._le_fmin.setSuffix(" kHz")
        self._le_fmin.setValue(label_edit.spectrogram_min_freq_khz)
        self._field(
            form, "Min display frequency:", self._le_fmin,
            "Lower bound of the frequency axis shown for each label.",
            indicator=False,
        )

        self._le_fmax = QDoubleSpinBox()
        self._le_fmax.setRange(0.0, 250.0)
        self._le_fmax.setDecimals(1)
        self._le_fmax.setSuffix(" kHz")
        self._le_fmax.setValue(label_edit.spectrogram_max_freq_khz)
        self._field(
            form, "Max display frequency:", self._le_fmax,
            "Upper bound of the frequency axis shown for each label.",
            indicator=False,
        )
        self._range_indicator(form, self._le_fmin, self._le_fmax)

        self._le_colormap = QComboBox()
        self._le_colormap.addItems(COLORMAP_NAMES)
        idx = self._le_colormap.findText(label_edit.colormap)
        if idx >= 0:
            self._le_colormap.setCurrentIndex(idx)
        self._field(
            form, "Colormap:", self._le_colormap,
            "Color palette used to render spectrogram intensity.",
        )

        self._le_classifications = QLineEdit()
        self._le_classifications.setText(label_edit.classifications)
        self._field(
            form, "Classifications (comma-separated):", self._le_classifications,
            "Valid call-type labels offered in the Label Edit tab's class dropdown.",
            "Example: d,sk,5,5t,5w,c5 — edit freely, just keep the values comma-separated.",
            indicator=False,
        )

        return w

    def _make_video_tab(self) -> QWidget:
        w = QWidget()
        form = self._new_form(w)
        video = self._state.settings.video

        self._video_snap_min = QDoubleSpinBox()
        self._video_snap_min.setRange(0.0, 250_000.0)
        self._video_snap_min.setDecimals(0)
        self._video_snap_min.setSuffix(" Hz")
        self._video_snap_min.setValue(video.snap_band_min_hz)
        self._field(
            form, "Snap band min:", self._video_snap_min,
            "Lower bound of the band searched for a finger-snap transient in the "
            "video's own audio track.",
            "Used to sync a video's timeline when no 'sk' label is available.",
            indicator=False,
        )

        self._video_snap_max = QDoubleSpinBox()
        self._video_snap_max.setRange(0.0, 250_000.0)
        self._video_snap_max.setDecimals(0)
        self._video_snap_max.setSuffix(" Hz")
        self._video_snap_max.setValue(video.snap_band_max_hz)
        self._field(
            form, "Snap band max:", self._video_snap_max,
            "Upper bound of the band searched for a finger-snap transient in the "
            "video's own audio track.",
            indicator=False,
        )
        self._range_indicator(form, self._video_snap_min, self._video_snap_max)

        self._video_snap_threshold = QDoubleSpinBox()
        self._video_snap_threshold.setRange(0.1, 100.0)
        self._video_snap_threshold.setDecimals(1)
        self._video_snap_threshold.setValue(video.snap_threshold_factor)
        self._field(
            form, "Snap threshold factor:", self._video_snap_threshold,
            "How many times the noise-floor energy a window must exceed to be "
            "flagged as the snap transient.",
        )

        return w

    def _make_appearance_tab(self) -> QWidget:
        w = QWidget()
        form = self._new_form(w)

        self._theme_combo = QComboBox()
        self._theme_combo.addItem("System", "system")
        self._theme_combo.addItem("Light", "light")
        self._theme_combo.addItem("Dark", "dark")
        self._theme_combo.setCurrentIndex(max(self._theme_combo.findData(t.get_mode()), 0))
        self._theme_combo.currentIndexChanged.connect(self._on_theme_mode_changed)
        self._field(
            form, "Color mode:", self._theme_combo,
            "Choose the app's color scheme.",
            "\"System\" follows your OS's light/dark setting and updates automatically if it changes.",
        )

        # WP27 THEME DECISION (resolved): MATLAB's Light/Gray/Custom presets with
        # 4 raw RGB color pickers (formerly config.ThemeSettings) were deliberately
        # not ported. The design-token System/Light/Dark system above is the app's
        # only theme mechanism now; config.ThemeSettings has been removed.

        return w

    def _browse_det_export_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select default export folder")
        if path:
            self._det_export_path.setText(path)

    def _on_theme_mode_changed(self, index: int) -> None:
        t.set_mode(self._theme_combo.itemData(index))

    # ── Shortcuts sub-tab ─────────────────────────────────────────────────

    def _make_shortcuts_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(t.SP_4, t.SP_4, t.SP_4, t.SP_4)
        outer.setSpacing(t.SP_3)

        intro = QLabel(
            "Click a shortcut field, then press the new key combination — it saves "
            "immediately and takes effect app-wide. Shortcuts that clash with another "
            "action are flagged below; only one will actually fire."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")
        outer.addWidget(intro)

        grid = QGridLayout()
        grid.setHorizontalSpacing(t.SP_3)
        grid.setVerticalSpacing(t.SP_2)
        grid.setColumnStretch(4, 1)

        header_style = f"color: {t.TEXT_SECONDARY}; font-weight: 600; font-size: {t.TEXT_XS}px;"
        for col, text in enumerate(["Action", "Where", "Shortcut", "", ""]):
            h = QLabel(text)
            h.setStyleSheet(header_style)
            grid.addWidget(h, 0, col)

        self._shortcut_edits: dict[str, QKeySequenceEdit] = {}
        self._shortcut_warnings: dict[str, QLabel] = {}

        for row, spec in enumerate(shortcuts.SHORTCUTS, start=1):
            name_label = QLabel(spec.label)
            name_label.setToolTip(spec.description)

            ctx_label = QLabel(spec.context)
            ctx_label.setToolTip(spec.description)
            ctx_label.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")

            edit = QKeySequenceEdit(shortcuts.get_shortcut(spec.id))
            edit.setMaximumSequenceLength(1)
            edit.setToolTip(f"{spec.description}\n\nClick, then press the new key combination.")
            edit.keySequenceChanged.connect(
                lambda seq, sid=spec.id: self._on_shortcut_edited(sid, seq)
            )

            reset_btn = QPushButton("Reset")
            reset_btn.setFixedWidth(64)
            reset_btn.setToolTip(f"Restore the default shortcut ({spec.default}).")
            reset_btn.clicked.connect(lambda _checked, sid=spec.id: self._on_shortcut_reset(sid))

            warn = QLabel("")
            warn.setStyleSheet(f"color: {t.DANGER}; font-size: {t.TEXT_XS}px;")
            warn.setWordWrap(True)

            grid.addWidget(name_label, row, 0)
            grid.addWidget(ctx_label, row, 1)
            grid.addWidget(edit, row, 2)
            grid.addWidget(reset_btn, row, 3)
            grid.addWidget(warn, row, 4)

            self._shortcut_edits[spec.id] = edit
            self._shortcut_warnings[spec.id] = warn

        outer.addLayout(grid)
        outer.addStretch()

        reset_all_row = QHBoxLayout()
        reset_all_row.addStretch()
        reset_all_btn = QPushButton("Reset all shortcuts to defaults")
        reset_all_btn.setToolTip("Restore every shortcut above to its shipped default.")
        reset_all_btn.clicked.connect(self._on_shortcuts_reset_all)
        reset_all_row.addWidget(reset_all_btn)
        outer.addLayout(reset_all_row)

        self._refresh_shortcut_conflicts()
        return w

    def _on_shortcut_edited(self, action_id: str, seq) -> None:
        shortcuts.set_shortcut(action_id, seq)
        self._refresh_shortcut_conflicts()

    def _on_shortcut_reset(self, action_id: str) -> None:
        shortcuts.reset_shortcut(action_id)
        edit = self._shortcut_edits[action_id]
        edit.blockSignals(True)
        edit.setKeySequence(shortcuts.get_shortcut(action_id))
        edit.blockSignals(False)
        self._refresh_shortcut_conflicts()

    def _on_shortcuts_reset_all(self) -> None:
        shortcuts.reset_all()
        for sid, edit in self._shortcut_edits.items():
            edit.blockSignals(True)
            edit.setKeySequence(shortcuts.get_shortcut(sid))
            edit.blockSignals(False)
        self._refresh_shortcut_conflicts()

    def _refresh_shortcut_conflicts(self) -> None:
        for sid, warn in self._shortcut_warnings.items():
            seq = self._shortcut_edits[sid].keySequence()
            conflicting = shortcuts.conflicts_with(sid, seq)
            if conflicting:
                names = ", ".join(shortcuts.label_for(c) for c in conflicting)
                warn.setText(f"⚠ also bound to: {names}")
            else:
                warn.setText("")

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
        vis.segment_start_seconds       = self._viz_seg_start.value()
        vis.segment_length_seconds      = self._viz_seg_len.value()
        vis.colormap                    = self._viz_colormap.currentText()
        vis.label_color                 = self._viz_label_color.currentText()
        vis.reference_label_color       = self._viz_ref_label_color.currentText()
        vis.manual_label_length         = self._viz_manual_label_len.value()
        vis.manual_label_marker         = self._viz_manual_label_marker.text().strip() or "md"
        vis.show_labels                 = self._viz_show_labels.currentData()
        vis.show_reference_labels       = self._viz_show_ref_labels.currentData()
        vis.show_loading_dialog         = self._viz_show_loading.currentData()
        vis.sonification_st             = self._viz_sonif_st.value()
        vis.sonification_slowdown       = self._viz_sonif_slowdown.value()
        vis.sonification_natural_speed  = self._viz_sonif_natural.currentData()
        vis.sonification_denoise        = self._viz_sonif_denoise.currentData()

        # Detectors (one Params sub-tab per registered plugin — see _plugin_form)
        for detector_id, widgets in self._detector_widgets.items():
            params_cls = AbstractDetector.get(detector_id).Params
            self._state.settings.detection.set_params(detector_id, pf.read_params_form(params_cls, widgets))

        # Classifiers
        for classifier_id, widgets in self._classifier_widgets.items():
            params_cls = AbstractClassifier.get(classifier_id).Params
            self._state.settings.classification.set_params(classifier_id, pf.read_params_form(params_cls, widgets))

        # Pre-processing / Post-processing / Detection export
        detection = self._state.settings.detection
        detection.pre = pf.read_params_form(type(detection.pre), self._pre_widgets)
        detection.export_path = self._det_export_path.text()
        post = detection.post
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

        # Video
        video = self._state.settings.video
        video.snap_band_min_hz       = self._video_snap_min.value()
        video.snap_band_max_hz       = self._video_snap_max.value()
        video.snap_threshold_factor  = self._video_snap_threshold.value()

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
            (self._viz_seg_start, vis.segment_start_seconds),
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

        self._viz_manual_label_marker.blockSignals(True)
        self._viz_manual_label_marker.setText(vis.manual_label_marker)
        self._viz_manual_label_marker.blockSignals(False)

        self._viz_show_labels.blockSignals(True)
        self._viz_show_labels.setCurrentIndex(0 if vis.show_labels else 1)
        self._viz_show_labels.blockSignals(False)

        self._viz_show_ref_labels.blockSignals(True)
        self._viz_show_ref_labels.setCurrentIndex(0 if vis.show_reference_labels else 1)
        self._viz_show_ref_labels.blockSignals(False)

        self._viz_show_loading.blockSignals(True)
        self._viz_show_loading.setCurrentIndex(0 if vis.show_loading_dialog else 1)
        self._viz_show_loading.blockSignals(False)

        self._viz_sonif_natural.blockSignals(True)
        self._viz_sonif_natural.setCurrentIndex(0 if vis.sonification_natural_speed else 1)
        self._viz_sonif_natural.blockSignals(False)

        self._viz_sonif_denoise.blockSignals(True)
        self._viz_sonif_denoise.setCurrentIndex(0 if vis.sonification_denoise else 1)
        self._viz_sonif_denoise.blockSignals(False)

        # Detectors / classifiers
        for detector_id, widgets in self._detector_widgets.items():
            pf.reload_params_form(self._state.settings.detection.params_for(detector_id), widgets)
        for classifier_id, widgets in self._classifier_widgets.items():
            pf.reload_params_form(self._state.settings.classification.params_for(classifier_id), widgets)

        # Pre-processing / Post-processing / Detection export
        detection = self._state.settings.detection
        pf.reload_params_form(detection.pre, self._pre_widgets)
        self._det_export_path.blockSignals(True)
        self._det_export_path.setText(detection.export_path)
        self._det_export_path.blockSignals(False)

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

        # Video
        video = self._state.settings.video
        for widget, value in [
            (self._video_snap_min, video.snap_band_min_hz),
            (self._video_snap_max, video.snap_band_max_hz),
            (self._video_snap_threshold, video.snap_threshold_factor),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import squeak_peek.classifiers  # noqa: F401  (registers built-in classifiers)
import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek.audio.denoise import preprocess
from squeak_peek.audio.io import load_wav
from squeak_peek.classifiers.base import AbstractClassifier
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.io import export_labels, export_labels_detector
from squeak_peek.labels.postprocess import POST_STEPS, apply_post_processing

from . import _theme as t
from ._detector_training import DetectorTrainingPage
from ._state import AppState


class DetectionTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._status_kind = "neutral"
        self._data_input_tab = None  # Will be set by app
        self._progress_dialog = None
        self._cancel_requested = False
        self._setup_ui()
        t.signal.changed.connect(self._apply_theme)
        state.settings_changed.connect(self._on_settings_changed)

    def set_data_input_tab(self, tab) -> None:
        """Called by app to link the data input tab for batch mode access."""
        self._data_input_tab = tab
        self._training_page.set_data_input_tab(tab)

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)
        outer.setSpacing(t.SP_3)
        self._desc = QLabel(
            "Run automatic call detectors on the loaded recording (or the whole Data Input batch "
            "folder) and export their labels — or, in the second sub-tab, tune any detector's "
            "parameters or train the ML / CNN detector on your own labeled recordings."
        )
        self._desc.setWordWrap(True)
        outer.addWidget(self._desc)

        inner = QTabWidget()
        inner.setObjectName("innerTabs")
        inner.setDocumentMode(True)
        inner.addTab(self._build_run_page(), "Run detectors")
        self._training_page = DetectorTrainingPage(self._state)
        self._training_page.set_detection_tab(self)
        self._training_page.model_applied.connect(self._on_model_applied)
        inner.addTab(self._training_page, "Train detector")
        inner.setTabToolTip(0, "Detect calls with one or more detectors and export label files.")
        inner.setTabToolTip(1, "Tune a detector's parameters, or train the ML / CNN detector, from labeled recordings.")
        self._inner = inner
        outer.addWidget(inner, 1)
        self._apply_theme()

    @staticmethod
    def _check_list(tooltip: str) -> QListWidget:
        lst = QListWidget()
        lst.setObjectName("checkList")
        lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        lst.setToolTip(tooltip)
        lst.itemClicked.connect(  # click anywhere on the row, not just the box
            lambda item: item.setCheckState(
                Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
        )
        return lst

    @staticmethod
    def _add_check_item(lst: QListWidget, text: str, tooltip: str, checked: bool) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setToolTip(tooltip)
        lst.addItem(item)

    @staticmethod
    def _checked(lst: QListWidget) -> list[str]:
        return [lst.item(i).data(Qt.ItemDataRole.UserRole) or lst.item(i).text()
                for i in range(lst.count()) if lst.item(i).checkState() == Qt.CheckState.Checked]

    def _build_run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, t.SP_3, 0, 0)
        layout.setSpacing(t.SP_4)

        lists = QHBoxLayout()
        lists.setSpacing(t.SP_4)

        # ── Detector selection ────────────────────────────────────────────
        det_group = QGroupBox("Detectors")
        det_group.setToolTip("Each checked detector runs independently and exports its own label file.")
        det_layout = QVBoxLayout(det_group)
        self._det_list = self._check_list("Check one or more detectors.")
        for i, detector_cls in enumerate(AbstractDetector.all()):
            self._add_check_item(self._det_list, detector_cls.id, detector_cls.description, i == 0)
            self._det_list.item(i).setText(f"{detector_cls.id}  —  {self._short(detector_cls.description)}")
            self._det_list.item(i).setData(Qt.ItemDataRole.UserRole, detector_cls.id)
        det_layout.addWidget(self._det_list)
        lists.addWidget(det_group, 3, Qt.AlignmentFlag.AlignTop)

        right = QVBoxLayout()
        right.setSpacing(t.SP_4)

        # ── Classifier selection (optional) ───────────────────────────────
        cls_group = QGroupBox("Classifiers (optional)")
        cls_group.setToolTip(
            "Each checked classifier assigns call types to every detector's (post-processed) "
            "output, producing one exported file per detector × classifier pair. Leave all "
            "unchecked to export unclassified detections."
        )
        cls_layout = QVBoxLayout(cls_group)
        self._cls_list = self._check_list("Check classifiers to assign call types after detection.")
        for i, classifier_cls in enumerate(AbstractClassifier.all()):
            self._add_check_item(self._cls_list, classifier_cls.display_name, classifier_cls.description, False)
            self._cls_list.item(i).setData(Qt.ItemDataRole.UserRole, classifier_cls.id)
        cls_layout.addWidget(self._cls_list)
        right.addWidget(cls_group)

        # ── Pre-processing ────────────────────────────────────────────────
        pre_group = QGroupBox("Pre-processing: denoise input")
        pre_group.setToolTip(
            "Suppress stationary background noise before the checked detectors run "
            "(Settings → Pre-processing). ML / CNN detectors use the denoising their model "
            "was trained with. Export, Label Edit and classification keep the original audio."
        )
        pre_layout = QVBoxLayout(pre_group)
        self._pre_checks: dict[str, QCheckBox] = {}
        for detector_cls in AbstractDetector.all():
            if not detector_cls.uses_pipeline_preprocessing or "denoise" not in detector_cls.Params.model_fields:
                continue
            cb = QCheckBox(detector_cls.display_name)
            cb.setToolTip(detector_cls.Params.model_fields["denoise"].description or "")
            cb.toggled.connect(lambda checked, det_id=detector_cls.id: self._on_denoise_toggled(det_id, checked))
            pre_layout.addWidget(cb)
            self._pre_checks[detector_cls.id] = cb
        self._sync_denoise_checks()
        right.addWidget(pre_group)

        # ── Post-processing ───────────────────────────────────────────────
        post_group = QGroupBox("Post-processing")
        post_group.setToolTip(
            "Applied to every detector's output before export, in the listed order. "
            "Thresholds are set in Settings → Post-processing."
        )
        post_layout = QVBoxLayout(post_group)
        self._post_list = self._check_list("Steps run top to bottom.")
        _POST_TOOLTIPS = {
            "Filter Broadband": "Discard detections whose energy is spread across the band "
                                 "instead of concentrated in a narrowband USV whistle — drops "
                                 "cage knocks and rustle (Settings → Post-processing → Min tonality).",
            "Merge Close Labels": "Merge detections separated by a small gap "
                                   "(Settings → Post-processing → Max gap to merge).",
            "Remove Short Labels": "Discard detections shorter than a minimum duration "
                                    "(Settings → Post-processing → Min label length).",
        }
        for name in POST_STEPS:  # Filter Broadband is opt-in
            self._add_check_item(self._post_list, name, _POST_TOOLTIPS[name], name != "Filter Broadband")
        post_layout.addWidget(self._post_list)
        right.addWidget(post_group)
        lists.addLayout(right, 2)
        layout.addLayout(lists)

        row_h = self.fontMetrics().height() + 2 * t.SP_1 + 4
        for lst in (self._det_list, self._cls_list, self._post_list):
            lst.setFixedHeight(max(lst.count(), 1) * row_h + 4)
        right.addStretch()

        # ── Export folder ─────────────────────────────────────────────────
        export_group = QGroupBox("Export folder")
        export_group.setToolTip("Where the resulting label file(s) are written.")
        export_row = QHBoxLayout(export_group)
        self._export_edit = QLineEdit()
        self._export_edit.setReadOnly(True)
        self._export_edit.setText(self._state.settings.detection.export_path)
        self._export_edit.setPlaceholderText("Same folder as the WAV file")
        self._export_edit.setToolTip(
            "Folder where exported label files are written. Leave blank to use the "
            "WAV file's own folder."
        )
        export_btn = QPushButton("Browse…")
        export_btn.setToolTip("Choose an export folder.")
        export_btn.clicked.connect(self._browse_export)
        clear_btn = QPushButton("Reset")
        clear_btn.setToolTip("Export next to the WAV file again.")
        clear_btn.clicked.connect(lambda: self._set_export_path(""))
        export_row.addWidget(self._export_edit)
        export_row.addWidget(export_btn)
        export_row.addWidget(clear_btn)
        layout.addWidget(export_group)

        layout.addStretch(1)

        # ── Run ───────────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._status = QLabel("")
        self._status.setWordWrap(True)
        run_row.addWidget(self._status, 1)
        self._run_btn = QPushButton("Run detectors")
        self._run_btn.setToolTip(
            "Run the checked detector(s) — on the loaded file, or on every file in the "
            "batch folder if Data Input is in batch mode — then export label files."
        )
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn)
        layout.addLayout(run_row)
        return page

    @staticmethod
    def _short(description: str) -> str:
        first = description.split(". ")[0].rstrip(".")
        return first if len(first) <= 90 else first[:87] + "…"

    def _on_model_applied(self, det_id: str, _path: str, _sensitivity) -> None:
        for i in range(self._det_list.count()):
            item = self._det_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == det_id:
                item.setCheckState(Qt.CheckState.Checked)

    def _on_settings_changed(self) -> None:
        self._export_edit.setText(self._state.settings.detection.export_path)
        self._sync_denoise_checks()

    def _sync_denoise_checks(self) -> None:
        """Mirror each detector's own ``denoise`` parameter."""
        detection = self._state.settings.detection
        for det_id, cb in self._pre_checks.items():
            cb.blockSignals(True)
            cb.setChecked(bool(getattr(detection.params_for(det_id), "denoise", False)))
            cb.blockSignals(False)

    def _on_denoise_toggled(self, det_id: str, checked: bool) -> None:
        detection = self._state.settings.detection
        params = detection.params_for(det_id).model_copy(update={"denoise": checked})
        detection.set_params(det_id, params)
        self._state.settings_changed.emit()

    def _apply_theme(self) -> None:
        self._desc.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_SM}px;")
        color = {"neutral": t.TEXT_SECONDARY, "success": t.SUCCESS, "danger": t.DANGER}[self._status_kind]
        self._status.setStyleSheet(f"color: {color}; font-size: {t.TEXT_XS}px;")

    def _set_status(self, text: str, kind: str) -> None:
        self._status_kind = kind
        self._apply_theme()
        self._status.setText(text)

    def _browse_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export folder")
        if path:
            self._set_export_path(path)

    def _set_export_path(self, path: str) -> None:
        """Shared with Settings → Post-processing → Export path."""
        self._export_edit.setText(path)
        self._state.settings.detection.export_path = path
        self._state.settings_changed.emit()

    def _run(self) -> None:
        # Get selected detectors
        selected_detectors = self._checked(self._det_list)

        if not selected_detectors:
            QMessageBox.warning(self, "No detector selected", "Check at least one detector.")
            return

        selected_classifiers = self._checked(self._cls_list)
        selected_post = set(self._checked(self._post_list))

        post_processing = [p for p in POST_STEPS if p in selected_post]

        # Determine if batch or single mode
        is_batch = self._data_input_tab and self._data_input_tab.batch_mode

        if is_batch:
            self._run_batch_mode(selected_detectors, selected_classifiers, post_processing)
        else:
            self._run_single_mode(selected_detectors, selected_classifiers, post_processing)

    def _run_single_mode(
        self, detectors: list[str], classifiers: list[str], post_processing: list[str],
    ) -> None:
        """Run detectors on a single loaded WAV file."""
        if self._state.samples is None:
            QMessageBox.warning(self, "No file loaded", "Load a WAV file first.")
            return

        wav_path = self._state.wav_path
        if wav_path is None:
            QMessageBox.warning(self, "No file loaded", "Load a WAV file first.")
            return

        # Create progress dialog
        self._progress_dialog = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._cancel_requested = False
        self._progress_dialog.canceled.connect(self._on_progress_canceled)
        self._progress_dialog.show()
        QApplication.processEvents()

        self._run_btn.setEnabled(False)
        export_dir = self._export_edit.text().strip()
        if not export_dir:
            export_dir = str(wav_path.parent)

        base_name = wav_path.stem
        file_count = 1
        total_files = 1

        try:
            self._run_detectors_on_signal(
                self._state.samples,
                self._state.fs,
                base_name,
                export_dir,
                detectors,
                classifiers,
                post_processing,
                file_count,
                total_files,
            )

            if not self._cancel_requested:
                self._set_status(f"All detectors finished: {', '.join(detectors)}.", "success")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Detection failed", str(exc))
            self._set_status("Failed.", "danger")
        finally:
            self._run_btn.setEnabled(True)
            if self._progress_dialog:
                self._progress_dialog.close()
                self._progress_dialog = None

    def _run_batch_mode(
        self, detectors: list[str], classifiers: list[str], post_processing: list[str],
    ) -> None:
        """Run detectors on all WAV files in the batch folder."""
        if not self._data_input_tab:
            QMessageBox.warning(self, "Error", "Data Input tab not available.")
            return

        usv_dir = self._data_input_tab.batch_usv_dir
        if not usv_dir:
            QMessageBox.warning(self, "No folder selected", "Select a USV folder in batch mode.")
            return

        usv_path = Path(usv_dir)
        wav_files = sorted(usv_path.glob("*.wav"))

        if not wav_files:
            QMessageBox.warning(self, "No files found", f"No .wav files found in {usv_dir}.")
            return

        # Create progress dialog
        self._progress_dialog = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._cancel_requested = False
        self._progress_dialog.canceled.connect(self._on_progress_canceled)
        self._progress_dialog.show()
        QApplication.processEvents()

        self._run_btn.setEnabled(False)
        export_dir = self._export_edit.text().strip()
        if not export_dir:
            export_dir = str(usv_path.parent)

        try:
            for file_idx, wav_file in enumerate(wav_files, 1):
                if self._cancel_requested:
                    self._progress_dialog.setLabelText("Canceled.")
                    self._set_status("Canceled.", "neutral")
                    break

                base_name = wav_file.stem
                self._progress_dialog.setWindowTitle(
                    f"File {file_idx}/{len(wav_files)} — Initializing..."
                )
                self._progress_dialog.setLabelText(f"Loading {base_name}...")
                QApplication.processEvents()

                try:
                    samples, fs = self._load_wav_file(wav_file)
                except Exception as exc:  # noqa: BLE001
                    self._progress_dialog.setLabelText(f"Error loading {base_name}: {exc}")
                    QApplication.processEvents()
                    continue

                self._run_detectors_on_signal(
                    samples,
                    fs,
                    base_name,
                    export_dir,
                    detectors,
                    classifiers,
                    post_processing,
                    file_idx,
                    len(wav_files),
                )

                if self._cancel_requested:
                    break

            if not self._cancel_requested:
                self._set_status(f"Batch complete: {len(wav_files)} file(s) processed.", "success")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Batch processing failed", str(exc))
            self._set_status("Failed.", "danger")
        finally:
            self._run_btn.setEnabled(True)
            if self._progress_dialog:
                self._progress_dialog.close()
                self._progress_dialog = None

    def _run_detectors_on_signal(
        self,
        samples,
        fs,
        base_name: str,
        export_dir: str,
        detectors: list[str],
        classifiers: list[str],
        post_processing: list[str],
        file_idx: int,
        total_files: int,
    ) -> None:
        """Run all selected detectors (and optional classifiers) on a signal,
        applying post-processing to each detector's output before export."""
        total_detectors = len(detectors)
        pre = self._state.settings.detection.pre
        denoised = None  # computed once per file, only if a detector wants it

        for det_idx, det_name in enumerate(detectors, 1):
            if self._cancel_requested:
                if self._progress_dialog:
                    self._progress_dialog.setLabelText("Canceled.")
                return

            if self._progress_dialog:
                title = f"File {file_idx}/{total_files} — {det_name} ({det_idx}/{total_detectors})"
                self._progress_dialog.setWindowTitle(title)
                self._progress_dialog.setLabelText(f"Running {det_name} on {base_name}...")
                self._progress_dialog.setValue(int((det_idx - 1) / total_detectors * 100))
                QApplication.processEvents()

            try:
                detector = self._build_detector(det_name)
                det_input = samples
                if detector.wants_denoise:
                    if denoised is None:
                        if self._progress_dialog:
                            self._progress_dialog.setLabelText(f"Denoising {base_name}...")
                            QApplication.processEvents()
                        denoised = preprocess(samples, fs, pre)
                    det_input = denoised
                labels = detector.detect(det_input, fs)

                if post_processing:
                    labels = self.post_process(labels, samples, fs, det_name, post_processing)

                export_path = Path(export_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if classifiers:
                    display_labels = labels
                    for cls_name in classifiers:
                        if self._cancel_requested:
                            return
                        classifier = self._build_classifier(cls_name)
                        classified = classifier.classify(labels, samples, fs)
                        filename = f"{base_name}_{det_name}_{cls_name}_{timestamp}_detected.txt"
                        out_file = export_path / filename
                        # Full format: export_labels_detector would overwrite
                        # every call type with the placeholder "d".
                        export_labels(out_file, classified)
                        display_labels = classified
                    summary = f"Finished {det_name}: exported {len(classifiers)} classified file(s)."
                else:
                    filename = f"{base_name}_{det_name}_{timestamp}_detected.txt"
                    out_file = export_path / filename
                    export_labels_detector(out_file, labels)
                    display_labels = labels
                    summary = f"Finished {det_name}: {len(labels)} events. Exported to {out_file.name}"

                if self._progress_dialog:
                    self._progress_dialog.setLabelText(summary)
                    self._progress_dialog.setValue(int(det_idx / total_detectors * 100))
                    QApplication.processEvents()

                # For single file mode, update state with last detector/classifier result
                if total_files == 1:
                    self._state.detected_labels = display_labels
                    self._state.labels_changed.emit()

            except Exception as exc:  # noqa: BLE001
                if self._progress_dialog:
                    self._progress_dialog.setLabelText(f"Error in {det_name}: {exc}")
                    QApplication.processEvents()
                raise

    def checked_post_steps(self) -> list[str]:
        return self._checked(self._post_list)

    def post_process(self, labels, samples, fs, det_id: str, steps: list[str]):
        """Run detectors' post-processing, with thresholds from Settings."""
        post = self._state.settings.detection.post
        fcut_min, fcut_max = self._detector_band(det_id)
        return apply_post_processing(
            labels, samples, fs, steps,
            max_gap=post.maxGapToMerge, min_length=post.minLabelLength,
            min_tonality=post.minTonality or 0.5, fcut_min=fcut_min, fcut_max=fcut_max,
        )

    def _on_progress_canceled(self) -> None:
        """Called when progress dialog cancel button is clicked."""
        self._cancel_requested = True

    def _load_wav_file(self, path: Path):
        """Load WAV file and return samples and sample rate."""
        return load_wav(path)

    def _detector_band(self, det_id: str) -> tuple[float, float]:
        """The detector's own analysis band, for band-limited post-processing."""
        try:
            params = self._state.settings.detection.params_for(det_id)
        except KeyError:
            return 40_000.0, 120_000.0
        return getattr(params, "fcutMin", 40_000.0), getattr(params, "fcutMax", 120_000.0)

    def _build_detector(self, det_id: str) -> AbstractDetector:
        cls = AbstractDetector.get(det_id)
        params = self._state.settings.detection.params_for(det_id)
        return cls(params)

    def _build_classifier(self, cls_id: str) -> AbstractClassifier:
        cls = AbstractClassifier.get(cls_id)
        params = self._state.settings.classification.params_for(cls_id)
        return cls(params)

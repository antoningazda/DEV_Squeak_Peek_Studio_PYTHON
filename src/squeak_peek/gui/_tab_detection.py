from __future__ import annotations

from datetime import datetime
from pathlib import Path

import soundfile as sf
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
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
    QVBoxLayout,
    QWidget,
)

import squeak_peek.classifiers  # noqa: F401  (registers built-in classifiers)
import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek.classifiers.base import AbstractClassifier
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.io import export_labels_detector
from squeak_peek.labels.postprocess import merge_close_labels, remove_short_labels

from . import _theme as t
from ._state import AppState

_POSTPROCESSING = ["None", "Merge Close Labels", "Remove Short Labels"]


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

    def set_data_input_tab(self, tab) -> None:
        """Called by app to link the data input tab for batch mode access."""
        self._data_input_tab = tab

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(t.SP_4)
        layout.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)

        # ── Detector selection (multi-select) ─────────────────────────────
        det_group = QGroupBox("Detectors (select one or more)")
        det_group.setToolTip("Each selected detector runs independently and exports its own label file.")
        det_layout = QVBoxLayout(det_group)
        self._det_list = QListWidget()
        self._det_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for i, detector_cls in enumerate(AbstractDetector.all()):
            item = QListWidgetItem(detector_cls.id)
            item.setToolTip(detector_cls.description)
            if i == 0:
                item.setSelected(True)
            self._det_list.addItem(item)
        self._det_list.setToolTip("Ctrl/Cmd-click or Shift-click to select more than one detector.")
        det_layout.addWidget(self._det_list)
        layout.addWidget(det_group)

        # ── Classifier selection (multi-select, optional) ──────────────────
        cls_group = QGroupBox("Classifiers (optional — assign call types after detection)")
        cls_group.setToolTip(
            "Each selected classifier runs on every selected detector's (post-processed) output, "
            "producing one exported file per detector × classifier pair. Leave empty to export "
            "each detector's output unclassified, as before."
        )
        cls_layout = QVBoxLayout(cls_group)
        self._cls_list = QListWidget()
        self._cls_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for classifier_cls in AbstractClassifier.all():
            item = QListWidgetItem(classifier_cls.id)
            item.setToolTip(classifier_cls.description)
            self._cls_list.addItem(item)
        self._cls_list.setToolTip("Ctrl/Cmd-click or Shift-click to select more than one classifier.")
        cls_layout.addWidget(self._cls_list)
        layout.addWidget(cls_group)

        # ── Post-processing (multi-select) ────────────────────────────────
        post_group = QGroupBox("Post-processing (applied in order: None → Merge → Remove Short)")
        post_group.setToolTip(
            "Steps applied to every detector's output before export. "
            "Thresholds are set in Settings → Post-processing."
        )
        post_layout = QVBoxLayout(post_group)
        self._post_list = QListWidget()
        self._post_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        _POST_TOOLTIPS = {
            "None": "Export detections exactly as the detector produced them.",
            "Merge Close Labels": "Merge detections separated by a small gap "
                                   "(Settings → Post-processing → Max gap to merge).",
            "Remove Short Labels": "Discard detections shorter than a minimum duration "
                                    "(Settings → Post-processing → Min label length).",
        }
        for i, name in enumerate(_POSTPROCESSING):
            item = QListWidgetItem(name)
            item.setToolTip(_POST_TOOLTIPS.get(name, ""))
            if i in (1, 2):  # Merge and Remove Short
                item.setSelected(True)
            self._post_list.addItem(item)
        post_layout.addWidget(self._post_list)
        layout.addWidget(post_group)

        # ── Export folder ─────────────────────────────────────────────────
        export_group = QGroupBox("Export folder")
        export_group.setToolTip("Where the resulting label file(s) are written.")
        export_row = QHBoxLayout(export_group)
        self._export_edit = QLineEdit()
        self._export_edit.setReadOnly(True)
        self._export_edit.setPlaceholderText("Same folder as WAV file")
        self._export_edit.setToolTip(
            "Folder where exported label files are written. Leave blank to use the "
            "WAV file's own folder."
        )
        export_btn = QPushButton("Browse…")
        export_btn.setToolTip("Choose an export folder.")
        export_btn.clicked.connect(self._browse_export)
        export_row.addWidget(self._export_edit)
        export_row.addWidget(export_btn)
        layout.addWidget(export_group)

        layout.addStretch()

        # ── Run ───────────────────────────────────────────────────────────
        self._run_btn = QPushButton("Run detectors")
        self._run_btn.setToolTip(
            "Run the selected detector(s) — on the loaded file, or on every file in the "
            "batch folder if Data Input is in batch mode — then export label files."
        )
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._status = QLabel("")
        layout.addWidget(self._status)
        self._apply_theme()

    def _apply_theme(self) -> None:
        color = {"neutral": t.TEXT_SECONDARY, "success": t.SUCCESS, "danger": t.DANGER}[self._status_kind]
        self._status.setStyleSheet(f"color: {color}; font-size: {t.TEXT_XS}px;")

    def _set_status(self, text: str, kind: str) -> None:
        self._status_kind = kind
        self._apply_theme()
        self._status.setText(text)

    def _browse_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export folder")
        if path:
            self._export_edit.setText(path)

    def _run(self) -> None:
        # Get selected detectors
        selected_detectors = []
        for item in self._det_list.selectedItems():
            selected_detectors.append(item.text())

        if not selected_detectors:
            QMessageBox.warning(self, "No detector selected", "Select at least one detector.")
            return

        selected_classifiers = [item.text() for item in self._cls_list.selectedItems()]

        # Get selected post-processing (in fixed order: None, Merge, Remove Short)
        selected_post = set()
        for item in self._post_list.selectedItems():
            selected_post.add(item.text())

        # Apply in fixed order: None, Merge, Remove Short
        post_order = ["None", "Merge Close Labels", "Remove Short Labels"]
        post_processing = [p for p in post_order if p in selected_post]

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
                labels = detector.detect(samples, fs)

                # Apply post-processing in fixed order
                if post_processing:
                    for post_name in post_processing:
                        if self._cancel_requested:
                            return
                        if post_name == "Merge Close Labels":
                            post = self._state.settings.detection.post
                            labels = merge_close_labels(labels, post.maxGapToMerge)
                        elif post_name == "Remove Short Labels":
                            post = self._state.settings.detection.post
                            labels = remove_short_labels(labels, post.minLabelLength)

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
                        export_labels_detector(out_file, classified)
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

    def _on_progress_canceled(self) -> None:
        """Called when progress dialog cancel button is clicked."""
        self._cancel_requested = True

    def _load_wav_file(self, path: Path):
        """Load WAV file and return samples and sample rate."""
        samples, fs = sf.read(path)
        return samples, fs

    def _build_detector(self, det_id: str) -> AbstractDetector:
        cls = AbstractDetector.get(det_id)
        params = self._state.settings.detection.params_for(det_id)
        return cls(params)

    def _build_classifier(self, cls_id: str) -> AbstractClassifier:
        cls = AbstractClassifier.get(cls_id)
        params = self._state.settings.classification.params_for(cls_id)
        return cls(params)

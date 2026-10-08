"""
Train detector page (Detection tab) — from labeled recordings (the loaded
file, the Data Input batch folders, or hand-picked WAV + label pairs):

  - train the ML (Random Forest) or CNN (Faster R-CNN) detector's model;
    the ML model can optionally be calibrated on a held-out recording
    (best Sensitivity, optionally best NoiseRatio);
  - or tune any detector's parameters (PSD, BSCD, RBD, …) for the best F1
    (squeak_peek.detectors.tuning).

Work runs on a worker thread; "Use for detection" stores the result in the
detector's Settings.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _theme as t
from ._label_io import load_labels
from ._state import AppState
from ._tab_classification import (
    _browse_row,
    _muted,
    _open_folder,
    _ro_item,
    _RunBar,
    _scroll,
    _table,
    _Worker,
)

_KINDS = {
    "TUNE": ("Tune detector parameters",
             "Search a detector's parameters (thresholds, window lengths, …) for the best F1 "
             "against your labels — works for every detector, e.g. PSD, BSCD and RBD. "
             "No model file is written; the best values go straight to Settings.",
             "", ""),
    "ML": ("Train ML model (Random Forest)",
           "Fast to train on a CPU (minutes). Classifies short frames as call/noise from "
           "acoustic features. Good default; works well with a few labeled recordings.",
           ".joblib", "Joblib model (*.joblib)"),
    "CNN": ("Train CNN model (Faster R-CNN)",
            "Finds each call as a time-frequency box on spectrogram tiles. Needs more labeled "
            "calls and is much slower to train (a GPU helps), but also predicts frequency bounds.",
            ".pt", "PyTorch checkpoint (*.pt)"),
}
_NOISE_RATIOS = (1, 3, 5, 10, 15)


def _dspin(lo: float, hi: float, value: float, decimals: int, suffix: str, tip: str, step: float = 1.0):
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setSuffix(suffix)
    spin.setToolTip(tip)
    return spin


def _ispin(lo: int, hi: int, value: int, tip: str, suffix: str = "") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(value)
    spin.setSuffix(suffix)
    spin.setToolTip(tip)
    return spin


def _form(group: QWidget) -> QFormLayout:
    form = QFormLayout(group)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setVerticalSpacing(t.SP_2)
    return form


class DetectorTrainingPage(QWidget):
    """Left: inputs and parameters. Right: training results."""

    # (detector id, model path, calibrated sensitivity or None)
    model_applied = pyqtSignal(str, str, object)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._data_input_tab = None
        self._worker: _Worker | None = None
        self._result: dict | None = None
        self._rows: list[tuple[Path, Path, int]] = []   # (wav, labels, n calls)
        self._detection_tab = None
        self._setup_ui()
        t.signal.changed.connect(self._apply_theme)

    def set_data_input_tab(self, tab) -> None:
        self._data_input_tab = tab

    def set_detection_tab(self, tab) -> None:
        """Source of the Run detectors post-processing steps used while tuning."""
        self._detection_tab = tab

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, t.SP_2, t.SP_2, 0)
        col.setSpacing(t.SP_3)

        # Detector type
        kind_group = QGroupBox("What to do")
        kcol = QVBoxLayout(kind_group)
        krow = QHBoxLayout()
        self._kind_buttons = QButtonGroup(self)
        for i, (kid, (title, _desc, _ext, _flt)) in enumerate(_KINDS.items()):
            rb = QRadioButton(title)
            rb.setProperty("kind", kid)
            rb.setChecked(i == 0)
            self._kind_buttons.addButton(rb, i)
            krow.addWidget(rb)
        krow.addStretch()
        kcol.addLayout(krow)
        self._kind_desc = _muted("")
        kcol.addWidget(self._kind_desc)
        col.addWidget(kind_group)

        # Recordings
        rec_group = QGroupBox("Training recordings")
        rec_group.setToolTip("WAV recordings paired with ground-truth label files.")
        rcol = QVBoxLayout(rec_group)
        self._table = _table(
            ["Recording (WAV)", "Ground-truth labels", "Calls"],
            "Each recording's label file must mark every real call in it — unlabeled calls are "
            "learned as noise. Detections rejected in Label Edit are ignored.",
        )
        rcol.addWidget(self._table)
        btns = QHBoxLayout()
        for text, tip, slot in (
            ("Add loaded", "Add the WAV selected in Data Input (single file) with its reference "
                           "labels (or, if none, its detected labels).", self._add_loaded),
            ("Add batch", "Add every WAV of the Data Input batch folder, paired by filename with the "
                          "batch reference-labels folder (or, if empty, the detected-labels folder).",
             self._add_batch),
            ("Add files…", "Pick a WAV and its ground-truth label file.", self._add_files),
            ("Remove", "Remove the selected rows.", self._remove),
            ("Clear", "Remove all rows.", self._clear),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch()
        rcol.addLayout(btns)
        rcol.addWidget(_muted(
            "Use reference labels (or detections you checked and corrected in Label Edit) — "
            "the detector learns exactly what these files mark as a call."
        ))
        col.addWidget(rec_group)

        # Parameters (one page per detector type)
        par_group = QGroupBox("Parameters")
        pcol = QVBoxLayout(par_group)
        self._params = QStackedWidget()
        self._params.addWidget(self._build_tune_params())
        self._params.addWidget(self._build_ml_params())
        self._params.addWidget(self._build_cnn_params())
        pcol.addWidget(self._params)
        col.addWidget(par_group)

        # Calibration (ML only)
        self._cal_group = QGroupBox("Calibration (optional)")
        self._cal_group.setToolTip(
            "Score the trained model against one labeled recording and pick the Sensitivity "
            "with the best F1 — the value 'Use for detection' then stores in Settings."
        )
        cform = _form(self._cal_group)
        self._cal_combo = QComboBox()
        self._cal_combo.setToolTip(
            "Recording used to choose the Sensitivity. With two or more recordings it is held "
            "out of training, so the score is honest; with only one, the model is calibrated "
            "on its own training data (optimistic)."
        )
        cform.addRow("Calibrate on:", self._cal_combo)
        self._refresh_calibration()
        self._cal_noise = QCheckBox("Also tune the noise ratio (" + ", ".join(map(str, _NOISE_RATIOS)) + ")")
        self._cal_noise.setToolTip(
            "Train one model per noise ratio and keep the one with the best F1 on the "
            "calibration recording (about 5× longer)."
        )
        cform.addRow("", self._cal_noise)
        col.addWidget(self._cal_group)

        # Output
        out_group = QGroupBox("Output")
        oform = _form(out_group)
        row, self._out_edit = _browse_row(
            "New file next to the first WAV", "Where the trained model is saved.", self._browse_out,
        )
        oform.addRow("Model file:", row)
        self._train_denoise = QCheckBox("Denoise the training audio")
        self._train_denoise.setToolTip(
            "Suppress stationary background noise before training (Settings → Pre-processing). "
            "Stored in the model: the detector applies the same denoising when it runs."
        )
        oform.addRow("", self._train_denoise)
        self._out_group = out_group
        col.addWidget(out_group)
        col.addStretch()

        self._runbar = _RunBar()
        self._runbar.cancel_clicked.connect(self._cancel)
        col.addWidget(self._runbar)
        run_row = QHBoxLayout()
        self._status = _muted("")
        run_row.addWidget(self._status, 1)
        self._run_btn = QPushButton("Train detector")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setToolTip("Train the selected detector on the listed recordings.")
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn)
        col.addLayout(run_row)

        # Results (right)
        right = QGroupBox("Training results")
        rcol2 = QVBoxLayout(right)
        self._summary = _muted("Add labeled recordings, then tune or train a detector to see results here.")
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rcol2.addWidget(self._summary)
        self._sweep = _table(["Setting", "Precision", "Recall", "F1"], "Calibration sweep (best first).")
        self._sweep.setMinimumHeight(0)
        self._sweep.setVisible(False)
        rcol2.addWidget(self._sweep, 1)
        rcol2.addStretch()
        brow = QHBoxLayout()
        brow.addStretch()
        self._open_btn = QPushButton("Show model file")
        self._open_btn.setEnabled(False)
        self._open_btn.setToolTip("Open the folder containing the trained model.")
        self._open_btn.clicked.connect(lambda: self._result and _open_folder(Path(self._result["path"]).parent))
        brow.addWidget(self._open_btn)
        self._use_btn = QPushButton("Use for detection")
        self._use_btn.setEnabled(False)
        self._use_btn.setToolTip(
            "Point the detector's Settings at the new model (and calibrated Sensitivity, if any) "
            "and select it in Run detectors."
        )
        self._use_btn.clicked.connect(self._use_model)
        brow.addWidget(self._use_btn)
        rcol2.addLayout(brow)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(_scroll(left))
        right_wrap = QWidget()
        rlay = QVBoxLayout(right_wrap)
        rlay.setContentsMargins(t.SP_2, t.SP_2, 0, 0)
        rlay.addWidget(right)
        split.addWidget(right_wrap)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)

        self._kind_buttons.idToggled.connect(lambda _i, on: on and self._on_kind_changed())
        self._on_kind_changed()
        self._apply_theme()

    def _build_tune_params(self) -> QWidget:
        from squeak_peek.detectors.base import AbstractDetector

        w = QWidget()
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(t.SP_2)
        top = QWidget()
        form = _form(top)
        form.setContentsMargins(0, 0, 0, 0)
        self._tune_det = QComboBox()
        for cls in AbstractDetector.all():
            self._tune_det.addItem(f"{cls.id} — {cls.display_name}" if cls.display_name != cls.id else cls.id, cls.id)
        idx = self._tune_det.findData("PSD")
        self._tune_det.setCurrentIndex(max(idx, 0))
        self._tune_det.setToolTip("Detector whose parameters are tuned (starting from its current Settings).")
        self._tune_det.currentIndexChanged.connect(lambda _i: self._rebuild_tune_table())
        form.addRow("Detector:", self._tune_det)
        self._tune_trials = _ispin(1, 5000, 40, "Parameter combinations to try. Each runs the detector on "
                                                "every listed recording, so time grows with both.")
        form.addRow("Trials:", self._tune_trials)
        self._tune_seconds = _dspin(0, 100_000, 30, 0, " s",
                                    "Use only the first N seconds of each recording (0 = whole recording). "
                                    "Shorter is faster; make sure it contains enough labeled calls.", 10)
        self._tune_seconds.setSpecialValueText("whole recording")
        form.addRow("Analyse first:", self._tune_seconds)
        self._tune_post = QCheckBox("Apply the post-processing checked in Run detectors")
        self._tune_post.setChecked(True)
        self._tune_post.setToolTip("Score the detections exactly as Run detectors would export them.")
        form.addRow("", self._tune_post)
        self._tune_seed = _ispin(0, 2**31 - 1, 0, "Random seed of the search.")
        form.addRow("Seed:", self._tune_seed)
        col.addWidget(top)

        self._tune_table = _table(
            ["Parameter", "Current", "Search from", "Search to"],
            "Check the parameters to tune and adjust their search ranges. Unchecked parameters "
            "keep their current value.",
        )
        self._tune_table.setMinimumHeight(220)
        col.addWidget(self._tune_table)
        col.addWidget(_muted(
            "Ranges default to ÷4 … ×4 around the current value (±0.25 for fractions). "
            "The current values are always tried first, so tuning never makes the F1 worse."
        ))
        self._state.settings_changed.connect(self._rebuild_tune_table)
        self._rebuild_tune_table()
        return w

    def _tune_params(self):
        det_id = self._tune_det.currentData()
        return det_id, self._state.settings.detection.params_for(det_id)

    def _rebuild_tune_table(self) -> None:
        from squeak_peek.detectors.tuning import default_ranges

        if self._worker is not None and self._worker.isRunning():
            return
        _det_id, params = self._tune_params()
        table = self._tune_table
        table.setRowCount(0)
        self._tune_ranges = default_ranges(params)
        for r, rng in enumerate(self._tune_ranges):
            table.insertRow(r)
            name = QTableWidgetItem(rng.name)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if rng.enabled else Qt.CheckState.Unchecked)
            info = type(params).model_fields[rng.name]
            name.setToolTip(info.description or rng.name)
            table.setItem(r, 0, name)
            table.setItem(r, 1, _ro_item(f"{getattr(params, rng.name):g}"))
            for c, value in ((2, rng.lo), (3, rng.hi)):
                spin = QDoubleSpinBox()
                mag = max(abs(rng.lo), abs(rng.hi), 1e-12)
                spin.setDecimals(0 if rng.integer else min(8, max(0, 3 - math.floor(math.log10(mag)))))
                spin.setRange(-1e12, 1e12)
                spin.setValue(value)
                spin.setFrame(False)
                table.setCellWidget(r, c, spin)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    def _tune_ranges_from_table(self):
        from dataclasses import replace

        ranges = []
        for r, rng in enumerate(self._tune_ranges):
            enabled = self._tune_table.item(r, 0).checkState() == Qt.CheckState.Checked
            lo = self._tune_table.cellWidget(r, 2).value()
            hi = self._tune_table.cellWidget(r, 3).value()
            ranges.append(replace(rng, lo=min(lo, hi), hi=max(lo, hi), enabled=enabled))
        return ranges

    def _build_ml_params(self) -> QWidget:
        w = QWidget()
        form = _form(w)
        form.setContentsMargins(0, 0, 0, 0)
        self._ml_fmin = _dspin(0, 500, 40, 1, " kHz", "Lower edge of the analysis band.")
        self._ml_fmax = _dspin(0, 500, 120, 1, " kHz", "Upper edge of the analysis band.")
        form.addRow("Band from:", self._ml_fmin)
        form.addRow("Band to:", self._ml_fmax)
        self._ml_frame = _dspin(1, 500, 20, 1, " ms", "Length of each analysis frame.")
        self._ml_hop = _dspin(0.5, 500, 5, 1, " ms", "Step between frames (time resolution of detections).")
        form.addRow("Frame length:", self._ml_frame)
        form.addRow("Frame hop:", self._ml_hop)
        self._ml_trees = _ispin(1, 10_000, 200, "Number of Random Forest trees.")
        form.addRow("Trees:", self._ml_trees)
        self._ml_leaf = _ispin(1, 1000, 3, "Minimum frames per leaf — higher values give a smoother model.")
        form.addRow("Min. leaf size:", self._ml_leaf)
        self._ml_noise = _dspin(0.1, 100, 3, 1, " × calls",
                                "Noise frames kept per call frame when balancing the training set.", 0.5)
        form.addRow("Noise ratio:", self._ml_noise)
        self._ml_min_event = _dspin(0, 1000, 3, 1, " ms",
                                    "Shortest event counted when scoring the calibration sweep.")
        form.addRow("Min. event duration:", self._ml_min_event)
        self._ml_seed = _ispin(0, 2**31 - 1, 42, "Random seed — same seed and data, same model.")
        form.addRow("Seed:", self._ml_seed)
        return w

    def _build_cnn_params(self) -> QWidget:
        w = QWidget()
        form = _form(w)
        form.setContentsMargins(0, 0, 0, 0)
        self._cnn_backbone = QComboBox()
        self._cnn_backbone.addItem("MobileNet v3 (fast, CPU-friendly)", "mobilenet")
        self._cnn_backbone.addItem("ResNet-50 (heavier, wants a GPU)", "resnet50")
        self._cnn_backbone.setToolTip("Feature-extractor network of the Faster R-CNN.")
        form.addRow("Backbone:", self._cnn_backbone)
        self._cnn_pretrained = QCheckBox("Start from ImageNet weights (downloaded once)")
        self._cnn_pretrained.setChecked(True)
        self._cnn_pretrained.setToolTip(
            "Transfer learning: much better results from few labels. Needs internet the first time."
        )
        form.addRow("", self._cnn_pretrained)
        self._cnn_epochs = _ispin(1, 1000, 10, "Passes over the training tiles.")
        form.addRow("Epochs:", self._cnn_epochs)
        self._cnn_batch = _ispin(1, 256, 4, "Tiles per training step.")
        form.addRow("Batch size:", self._cnn_batch)
        self._cnn_lr = _dspin(1e-6, 1.0, 1e-4, 6, "", "AdamW learning rate.", 1e-5)
        form.addRow("Learning rate:", self._cnn_lr)
        self._cnn_window = _dspin(0.05, 10, 1.0, 2, " s", "Length of each spectrogram tile.", 0.1)
        self._cnn_hop = _dspin(0.01, 10, 0.5, 2, " s", "Step between training tiles.", 0.1)
        form.addRow("Tile length:", self._cnn_window)
        form.addRow("Tile hop:", self._cnn_hop)
        self._cnn_fmin = _dspin(0, 500, 40, 1, " kHz", "Lower edge of the spectrogram tiles.")
        self._cnn_fmax = _dspin(0, 500, 120, 1, " kHz", "Upper edge of the spectrogram tiles.")
        form.addRow("Band from:", self._cnn_fmin)
        form.addRow("Band to:", self._cnn_fmax)
        self._cnn_val = _dspin(0, 0.9, 0.15, 2, "", "Fraction at the end of each recording held out for validation.", 0.05)
        form.addRow("Validation fraction:", self._cnn_val)
        self._cnn_device = QComboBox()
        self._cnn_device.addItems(["auto", "cpu", "mps", "cuda"])
        self._cnn_device.setToolTip("'auto' picks CUDA, then Apple GPU (mps), then CPU.")
        form.addRow("Device:", self._cnn_device)
        self._cnn_seed = _ispin(0, 2**31 - 1, 42, "Random seed for tile sampling.")
        form.addRow("Seed:", self._cnn_seed)
        return w

    def _apply_theme(self) -> None:
        for label in self.findChildren(QLabel):
            if label.property("muted"):
                label.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")
        self._color_status()

    def _color_status(self) -> None:
        kind = self._status.property("kind") or "neutral"
        color = {"neutral": t.TEXT_SECONDARY, "success": t.SUCCESS, "danger": t.DANGER}[kind]
        self._status.setStyleSheet(f"color: {color}; font-size: {t.TEXT_XS}px;")

    def _set_status(self, text: str, kind: str = "neutral") -> None:
        self._status.setProperty("kind", kind)
        self._status.setText(text)
        self._color_status()

    # ── Detector type ─────────────────────────────────────────────────────

    def _kind(self) -> str:
        return self._kind_buttons.checkedButton().property("kind")

    def _on_kind_changed(self) -> None:
        kind = self._kind()
        self._kind_desc.setText(_KINDS[kind][1])
        idx = list(_KINDS).index(kind)
        self._params.setCurrentIndex(idx)
        # A QStackedWidget is as tall as its tallest page; let only the
        # visible one count so the ML form doesn't trail empty space.
        for i in range(self._params.count()):
            policy = QSizePolicy.Policy.Preferred if i == idx else QSizePolicy.Policy.Ignored
            self._params.widget(i).setSizePolicy(QSizePolicy.Policy.Preferred, policy)
        self._params.adjustSize()
        self._cal_group.setVisible(kind == "ML")
        self._out_group.setVisible(kind != "TUNE")
        self._run_btn.setText("Tune parameters" if kind == "TUNE" else "Train detector")
        out = self._out_edit.text().strip()
        if out and kind != "TUNE":   # keep the extension in step with the chosen detector
            self._out_edit.setText(str(Path(out).with_suffix(_KINDS[kind][2])))

    # ── Recordings ────────────────────────────────────────────────────────

    def _append(self, wav: Path, labels: Path) -> bool:
        try:
            n = sum(1 for lbl in load_labels(labels) if lbl.detection_state != "Rejected")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Cannot read {labels.name}: {exc}", "danger")
            return False
        self._rows.append((wav, labels, n))
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, _ro_item(wav.name, str(wav)))
        self._table.setItem(r, 1, _ro_item(labels.name, str(labels)))
        self._table.setItem(r, 2, _ro_item(str(n)))
        self._table.resizeColumnsToContents()
        self._refresh_calibration()
        return True

    def _add_loaded(self) -> None:
        di = self._data_input_tab
        wav, ref, det = di.single_paths() if di else ("", "", "")
        labels = ref or det
        if not wav or not labels:
            QMessageBox.warning(self, "Nothing selected",
                                "Select a WAV and its reference (or corrected detected) labels in "
                                "Data Input → Single file first.")
            return
        self._append(Path(wav), Path(labels))

    def _add_batch(self) -> None:
        from squeak_peek.usv_classifier.api import find_matching_file

        di = self._data_input_tab
        usv, ref, det = di.batch_paths() if di else ("", "", "")
        folder = ref or det
        if not usv or not folder:
            QMessageBox.warning(self, "No batch folders",
                                "Select the USV folder and the reference-labels folder in "
                                "Data Input → Batch folder first.")
            return
        skipped, added = [], 0
        for wav in sorted(Path(usv).glob("*.wav")):
            labels = find_matching_file(folder, wav)
            if labels is None:
                skipped.append(wav.name)
            elif self._append(wav, labels):
                added += 1
        msg = f"Added {added} recording(s) from the batch folder."
        if skipped:
            msg += f" No labels for (skipped): {', '.join(skipped)}"
        self._set_status(msg, "neutral")

    def _add_files(self) -> None:
        wav, _ = QFileDialog.getOpenFileName(self, "Select WAV", "", "WAV files (*.wav *.WAV);;All files (*)")
        if not wav:
            return
        labels, _ = QFileDialog.getOpenFileName(self, f"Ground-truth labels for {Path(wav).name}",
                                                str(Path(wav).parent), "Text files (*.txt *.csv);;All files (*)")
        if labels:
            self._append(Path(wav), Path(labels))

    def _remove(self) -> None:
        for r in sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True):
            self._table.removeRow(r)
            del self._rows[r]
        self._refresh_calibration()

    def _clear(self) -> None:
        self._table.setRowCount(0)
        self._rows.clear()
        self._refresh_calibration()

    def _refresh_calibration(self) -> None:
        current = self._cal_combo.currentData()
        self._cal_combo.clear()
        self._cal_combo.addItem("No calibration — keep the current Sensitivity", None)
        for i, (wav, _labels, _n) in enumerate(self._rows):
            self._cal_combo.addItem(wav.name, i)
        idx = self._cal_combo.findData(current)
        self._cal_combo.setCurrentIndex(max(idx, 0))

    # ── Output ────────────────────────────────────────────────────────────

    def _default_out(self) -> Path:
        kind = self._kind()
        first = self._rows[0][0]
        return first.parent / f"{kind.lower()}_detector_{datetime.now():%Y%m%d_%H%M%S}{_KINDS[kind][2]}"

    def _browse_out(self) -> None:
        kind = self._kind()
        start = self._out_edit.text().strip() or (str(self._default_out()) if self._rows else "")
        path, _ = QFileDialog.getSaveFileName(self, "Save trained model as", start, _KINDS[kind][3])
        if path:
            self._out_edit.setText(str(Path(path).with_suffix(_KINDS[kind][2])))

    # ── Run ───────────────────────────────────────────────────────────────

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._runbar.cancel_btn.setEnabled(False)
            self._runbar.message.setText("Cancelling after the current step…")

    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Wait for the current training to finish (or cancel it).")
            return
        if not self._rows:
            QMessageBox.warning(self, "No recordings", "Add at least one labeled recording first.")
            return
        if sum(n for _w, _l, n in self._rows) < 5:
            QMessageBox.warning(self, "Too few calls", "The label files contain fewer than 5 calls in total.")
            return
        kind = self._kind()
        pairs = [(w, lbl) for w, lbl, _n in self._rows]
        if kind == "TUNE":
            job = self._tune_job(pairs)
        else:
            out = Path(self._out_edit.text().strip() or self._default_out())
            if out.exists() and QMessageBox.question(
                self, "Overwrite model?", f"{out.name} already exists. Overwrite it?"
            ) != QMessageBox.StandardButton.Yes:
                return
            job = self._ml_job(pairs, out) if kind == "ML" else self._cnn_job(pairs, out)
        if job is None:
            return

        self._set_status("")
        self._run_btn.setEnabled(False)
        self._use_btn.setEnabled(False)
        self._runbar.start()
        worker = _Worker(job, self)
        worker.progressed.connect(self._runbar.show_progress)

        def finish() -> None:
            self._runbar.stop()
            self._run_btn.setEnabled(True)

        def ok(result) -> None:
            finish()
            self._done(result)

        def failed(message: str, tb: str) -> None:
            finish()
            self._set_status(f"Failed: {message}", "danger")
            title = "Parameter tuning failed" if self._kind() == "TUNE" else "Detector training failed"
            box = QMessageBox(QMessageBox.Icon.Critical, title, message, parent=self)
            box.setDetailedText(tb)
            box.exec()

        def cancelled() -> None:
            finish()
            self._set_status("Cancelled.", "neutral")

        worker.succeeded.connect(ok)
        worker.failed.connect(failed)
        worker.was_cancelled.connect(cancelled)
        self._worker = worker
        worker.start()

    @staticmethod
    def _reporter(progress, cancelled):
        from squeak_peek.usv_classifier.api import Cancelled

        def report(fraction, message) -> None:
            if cancelled():
                raise Cancelled
            progress(fraction, message)
        return report

    def _training_denoise(self):
        """Denoising for the training audio (Settings → Pre-processing), or None.
        Stored in the model, so the detector reapplies it at inference."""
        if not self._train_denoise.isChecked():
            return None
        return self._state.settings.detection.pre.model_copy()

    def _tune_job(self, pairs):
        ranges = self._tune_ranges_from_table()
        if not any(r.enabled for r in ranges):
            QMessageBox.warning(self, "Nothing to tune", "Check at least one parameter to tune.")
            return None
        det_id, params = self._tune_params()
        steps = self._detection_tab.checked_post_steps() if (
            self._detection_tab is not None and self._tune_post.isChecked()) else []
        post = (lambda labels, samples, fs: self._detection_tab.post_process(labels, samples, fs, det_id, steps)) \
            if steps else None
        n_trials, seed = self._tune_trials.value(), self._tune_seed.value()
        max_seconds = self._tune_seconds.value() or None
        from squeak_peek.detectors.base import AbstractDetector

        pre = None
        if AbstractDetector.get(det_id)(params).wants_denoise:
            from squeak_peek.audio.denoise import spectral_denoise

            denoise = self._state.settings.detection.pre.model_copy()
            pre = lambda samples, fs: spectral_denoise(samples, fs, denoise)  # noqa: E731
        reporter = self._reporter

        def job(progress, cancelled):
            from squeak_peek.detectors.tuning import tune_detector

            report = reporter(progress, cancelled)
            result = tune_detector(det_id, params, pairs, ranges, n_trials=n_trials, max_seconds=max_seconds,
                                   pre_process=pre,
                                   post_process=post, seed=seed, progress=report)
            return {"kind": "TUNE", "detector": det_id, "tuning": result, "steps": steps,
                    "n_recordings": len(pairs), "max_seconds": max_seconds}

        return job

    def _ml_job(self, pairs, out: Path):
        fmin, fmax = self._ml_fmin.value() * 1000, self._ml_fmax.value() * 1000
        if fmin >= fmax:
            QMessageBox.warning(self, "Invalid band", "'Band from' must be below 'Band to'.")
            return None
        common = dict(
            denoise=self._training_denoise(),
            fcut_min=fmin, fcut_max=fmax,
            frame_len_s=self._ml_frame.value() / 1000, hop_len_s=self._ml_hop.value() / 1000,
        )
        fit = dict(n_trees=self._ml_trees.value(), min_leaf_size=self._ml_leaf.value(), seed=self._ml_seed.value())
        noise_ratio = self._ml_noise.value()
        min_event = self._ml_min_event.value() / 1000
        cal_idx = self._cal_combo.currentData()
        tune_noise = self._cal_noise.isChecked() and cal_idx is not None
        val = pairs[cal_idx] if cal_idx is not None else None
        held_out = val is not None and len(pairs) > 1
        train_pairs = [p for i, p in enumerate(pairs) if not (held_out and i == cal_idx)]
        reporter = self._reporter

        def job(progress, cancelled):
            from squeak_peek.ml.optimize import sweep_noise_ratio, sweep_sensitivity
            from squeak_peek.ml.train import collect_training_data, save_model, train_from_features

            report = reporter(progress, cancelled)
            result = {"kind": "ML", "path": str(out), "n_recordings": len(train_pairs),
                      "calibrated_on": Path(val[0]).name if val else None, "held_out": held_out}
            if tune_noise:
                sweep = sweep_noise_ratio(
                    train_pairs, val[0], val[1], noise_ratio_range=_NOISE_RATIOS,
                    min_event_duration=min_event, progress=report, **common, **fit,
                )
                model = sweep["best_model"]
                result["sensitivity"] = sweep["best_sensitivity"]
                result["stats"] = sweep["best_stats"]
                result["sweep"] = [(f"noise ratio {e['noise_ratio']} · sens. {e['best_sensitivity']:.2f}",
                                    e["best_stats"]) for e in sweep["sweep"]]
            else:
                X, y = collect_training_data(train_pairs, progress=report, **common)
                report(None, "Training the Random Forest… (this step cannot be cancelled)")
                model = train_from_features(X, y, noise_ratio=noise_ratio, **common, **fit)
                if val is not None:
                    report(None, f"Calibrating sensitivity on {Path(val[0]).name}…")
                    sweep = sweep_sensitivity(model, val[0], val[1], min_event_duration=min_event)
                    result["sensitivity"] = sweep["best_sensitivity"]
                    result["stats"] = sweep["best_stats"]
                    result["sweep"] = [(f"sensitivity {e['sensitivity']:.2f}", e["stats"]) for e in sweep["sweep"]]
            report(None, "Saving model…")
            save_model(model, out)
            result["info"] = model["training_info"]
            return result

        return job

    def _cnn_job(self, pairs, out: Path):
        fmin, fmax = self._cnn_fmin.value() * 1000, self._cnn_fmax.value() * 1000
        if fmin >= fmax:
            QMessageBox.warning(self, "Invalid band", "'Band from' must be below 'Band to'.")
            return None
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError:
            QMessageBox.warning(self, "PyTorch missing",
                                "CNN training needs torch and torchvision (pip install torch torchvision).")
            return None
        device = self._cnn_device.currentText()
        kwargs = dict(
            denoise=self._training_denoise(),
            window_s=self._cnn_window.value(), hop_s=self._cnn_hop.value(),
            fcut_min=fmin, fcut_max=fmax,
            backbone=self._cnn_backbone.currentData(), pretrained_backbone=self._cnn_pretrained.isChecked(),
            epochs=self._cnn_epochs.value(), batch_size=self._cnn_batch.value(), lr=self._cnn_lr.value(),
            val_fraction=self._cnn_val.value(), device=None if device == "auto" else device,
            seed=self._cnn_seed.value(),
        )
        reporter = self._reporter

        def job(progress, cancelled):
            from squeak_peek.cnn.train import save_checkpoint, train_cnn

            report = reporter(progress, cancelled)
            checkpoint = train_cnn(pairs, progress_callback=report, **kwargs)
            report(None, "Saving model…")
            save_checkpoint(checkpoint, out)
            return {"kind": "CNN", "path": str(out), "n_recordings": len(pairs),
                    "info": checkpoint["training_info"]}

        return job

    # ── Results ───────────────────────────────────────────────────────────

    def _done(self, result: dict) -> None:
        self._result = result
        if result["kind"] == "TUNE":
            self._tune_done(result)
            return
        self._open_btn.setVisible(True)
        info = result["info"]
        lines = [f"Model saved to {result['path']}"]
        if result["kind"] == "ML":
            lines.append(
                f"Trained on {result['n_recordings']} recording(s): {info['n_usv_total']:,} call frames and "
                f"{info['n_noise_total']:,} noise frames ({info['n_frames_trained']:,} used after balancing, "
                f"noise ratio {info['noise_ratio']:g})."
            )
            lines.append(f"Out-of-bag frame accuracy: {info['oob_accuracy']:.3f}")
            stats = result.get("stats")
            if stats is not None:
                where = "held-out" if result["held_out"] else "training (optimistic — add a second recording to hold one out)"
                lines.append(
                    f"Best sensitivity {result['sensitivity']:.2f} on {result['calibrated_on']} ({where}): "
                    f"precision {stats.precision:.3f} · recall {stats.recall:.3f} · F1 {stats.f1_score:.3f} "
                    f"({stats.true_positives} TP, {stats.false_positives} FP, {stats.false_negatives} FN)"
                )
        else:
            losses = info.get("epoch_losses") or []
            lines.append(
                f"Trained on {info['n_tiles_train']} tiles from {result['n_recordings']} recording(s) "
                f"({info['n_tiles_val']} held out for validation)."
            )
            if losses:
                lines.append("Loss per epoch: " + " → ".join(f"{v:.3f}" for v in losses))
            if info.get("val_ground_truth_boxes"):
                lines.append(
                    f"Validation: {info['val_detections_at_0.5']} detections (score > 0.5) vs. "
                    f"{info['val_ground_truth_boxes']} labeled calls."
                )
        self._summary.setText("\n".join(lines))
        self._show_sweep(result.get("sweep"))
        self._open_btn.setEnabled(True)
        self._use_btn.setEnabled(True)
        self._set_status("Training finished.", "success")

    def _tune_done(self, result: dict) -> None:
        res = result["tuning"]
        base, best = res.baseline, res.best

        def fmt(trial) -> str:
            st = trial.stats
            if st is None:
                return f"failed ({trial.error})"
            return (f"F1 {st.f1_score:.3f} · precision {st.precision:.3f} · recall {st.recall:.3f} "
                    f"({st.true_positives} TP, {st.false_positives} FP, {st.false_negatives} FN)")

        span = f"first {result['max_seconds']:g} s of " if result["max_seconds"] else ""
        lines = [
            f"Tuned {res.detector_id} on {span}{result['n_recordings']} recording(s), "
            f"{len(res.trials)} trials, post-processing: {', '.join(result['steps']) or 'none'}.",
            f"Current settings: {fmt(base)}",
            f"Best found:       {fmt(best)}",
        ]
        changed = [f"{k}: {base.params[k]:g} → {v:g}" for k, v in best.params.items() if v != base.params[k]]
        lines.append("Changes: " + ("; ".join(changed) if changed else "none — the current settings were best."))
        n_failed = sum(tr.stats is None for tr in res.trials)
        if n_failed:
            lines.append(f"{n_failed} combination(s) failed and were skipped.")
        self._summary.setText("\n".join(lines))
        ok = sorted((tr for tr in res.trials if tr.stats is not None), key=lambda tr: tr.f1, reverse=True)[:25]
        self._show_sweep([
            (", ".join(f"{k}={v:.4g}" for k, v in tr.params.items()) + ("  (current)" if tr is base else ""),
             tr.stats) for tr in ok
        ])
        self._open_btn.setVisible(False)
        self._use_btn.setEnabled(best.f1 > base.f1)
        self._set_status("Tuning finished." if best.f1 > base.f1 else
                         "Tuning finished — no better parameters found.", "success")

    def _show_sweep(self, sweep) -> None:
        table = self._sweep
        table.setRowCount(0)
        if not sweep:
            table.setVisible(False)
            return
        rows = sorted(sweep, key=lambda e: e[1].f1_score, reverse=True)
        table.setRowCount(len(rows))
        for r, (name, stats) in enumerate(rows):
            table.setItem(r, 0, _ro_item(name))
            table.setItem(r, 1, _ro_item(f"{stats.precision:.3f}"))
            table.setItem(r, 2, _ro_item(f"{stats.recall:.3f}"))
            table.setItem(r, 3, _ro_item(f"{stats.f1_score:.3f}"))
        table.resizeColumnsToContents()
        table.setVisible(True)

    def _use_model(self) -> None:
        if not self._result:
            return
        if self._result["kind"] == "TUNE":
            res = self._result["tuning"]
            self._state.settings.detection.set_params(res.detector_id, res.best_params)
            self._state.settings_changed.emit()
            self.model_applied.emit(res.detector_id, "", None)
            self._use_btn.setEnabled(False)
            self._set_status(f"{res.detector_id} now uses the tuned parameters (see Settings).", "success")
            return
        kind, path = self._result["kind"], self._result["path"]
        sensitivity = self._result.get("sensitivity")
        detection = self._state.settings.detection
        update = {"modelPath": path}
        if sensitivity is not None:
            update["sensitivity"] = float(sensitivity)
        detection.set_params(kind, detection.params_for(kind).model_copy(update=update))
        self._state.settings_changed.emit()
        self.model_applied.emit(kind, path, sensitivity)
        extra = f" with sensitivity {sensitivity:.2f}" if sensitivity is not None else ""
        self._set_status(f"{kind} detector now uses {Path(path).name}{extra}.", "success")

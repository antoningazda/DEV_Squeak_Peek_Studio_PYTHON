"""
Classification tab — classify detected calls with a trained USV model, or
train a new one from labeled recordings.

The model is the two-stage pipeline in squeak_peek.usv_classifier (CNN
USV/NOISE, then Random Forest call types with calibrated UNCERTAIN
rejection). Long work runs on a worker thread; inputs are snapshotted when
a run starts, so the rest of the app stays usable meanwhile.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    QStandardPaths,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from squeak_peek.labels.model import Label

from . import _theme as t
from ._state import AppState

PLUGIN_ID = "USV_MODEL"
_DEVICES = ["cpu", "auto", "mps"]
_DEVICE_TIP = "Where the CNN runs. 'auto' uses a CUDA GPU when available; 'mps' is Apple-GPU (experimental)."


def _feature_cache_dir() -> Path:
    """Persistent, validated feature cache shared by every run."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    return Path(base or tempfile.gettempdir()) / "usv_features"


def _open_folder(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _default_out(first_wav: Path, prefix: str) -> Path:
    return first_wav.parent / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


def _csv_list(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


# ═══════════════════════════════════════════════════════════════════════════
# Worker thread
# ═══════════════════════════════════════════════════════════════════════════

class _Worker(QThread):
    """Runs ``fn(progress, cancelled)`` off the GUI thread."""

    progressed = pyqtSignal(float, str)   # fraction (<0 = indeterminate), message
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)         # message, traceback
    was_cancelled = pyqtSignal()

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from squeak_peek.usv_classifier.api import Cancelled

        def progress(fraction, message) -> None:
            self.progressed.emit(-1.0 if fraction is None else float(fraction), message)

        try:
            result = self._fn(progress, lambda: self._cancel)
        except Cancelled:
            self.was_cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc) or type(exc).__name__, traceback.format_exc())
        else:
            self.succeeded.emit(result)


class _RunBar(QWidget):
    """Progress bar + message + Cancel, shown while a worker runs."""

    cancel_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(t.SP_1)
        row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop after the current recording / step.")
        self.cancel_btn.clicked.connect(self.cancel_clicked)
        row.addWidget(self.bar, 1)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        lay.addWidget(self.message)
        self.setVisible(False)

    def start(self) -> None:
        self.bar.setRange(0, 0)
        self.message.setText("Starting…")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel")
        self.setVisible(True)

    def show_progress(self, fraction: float, message: str) -> None:
        if fraction < 0:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(fraction * 1000))
        self.message.setText(message)
        if "cannot be cancelled" in message:
            self.cancel_btn.setEnabled(False)

    def stop(self) -> None:
        self.setVisible(False)


# ═══════════════════════════════════════════════════════════════════════════
# Predictions table model
# ═══════════════════════════════════════════════════════════════════════════

_RESULT_COLUMNS = [
    ("RecordingID", "Recording"),
    ("Start_s", "Start (s)"),
    ("End_s", "End (s)"),
    ("ClusterPred", "Prediction"),
    ("Stage1", "USV/NOISE"),
    ("pUSV", "p(USV)"),
    ("RawClusterPred", "RF guess"),
    ("RFMaxProbability", "RF prob."),
    ("UncertainReason", "Why uncertain"),
]


class _PredictionsModel(QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._df = pd.DataFrame(columns=[c for c, _ in _RESULT_COLUMNS])

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        cols = [c for c, _ in _RESULT_COLUMNS]
        self._df = df.reindex(columns=cols).reset_index(drop=True)
        self.endResetModel()

    def row(self, i: int) -> pd.Series:
        return self._df.iloc[i]

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(_RESULT_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return _RESULT_COLUMNS[section][1]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.UserRole:  # sort key
            if pd.isna(value):
                return None
            return value.item() if hasattr(value, "item") else value
        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(value) or value == "":
                return ""
            if isinstance(value, float):
                return f"{value:.4f}" if index.column() in (1, 2) else f"{value:.3f}"
            return str(value)
        return None


class _PredictionFilter(QSortFilterProxyModel):
    """Filter on the Prediction column; sort numerically where possible."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._value = ""
        self.setSortRole(Qt.ItemDataRole.UserRole)

    def set_value(self, value: str) -> None:
        self._value = value
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent) -> bool:
        if not self._value:
            return True
        pred = str(self.sourceModel().row(row).ClusterPred)
        if self._value == "__uncertain__":
            return pred in ("UNCERTAIN", "PROCESSING_ERROR")
        return pred == self._value

    def lessThan(self, left, right) -> bool:
        a, b = left.data(Qt.ItemDataRole.UserRole), right.data(Qt.ItemDataRole.UserRole)
        if a is None:
            return b is not None
        if b is None:
            return False
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


# ═══════════════════════════════════════════════════════════════════════════
# Recording lists
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _ClassifyRow:
    wav: Path
    labels_path: Path | None      # None = the loaded recording's in-memory labels
    labels: list[Label]
    from_state: bool = False


def _table(headers: list[str], tooltip: str) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setToolTip(tooltip)
    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(False)
    table.setWordWrap(False)
    table.setMinimumHeight(140)
    header = table.horizontalHeader()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.resizeColumnsToContents()
    return table


def _ro_item(text: str, tooltip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    item.setToolTip(tooltip or text)
    return item


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("muted", True)
    return label


def _browse_row(placeholder: str, tooltip: str, on_browse) -> tuple[QHBoxLayout, QLineEdit]:
    row = QHBoxLayout()
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setToolTip(tooltip)
    btn = QPushButton("Browse…")
    btn.setToolTip(tooltip)
    btn.clicked.connect(on_browse)
    row.addWidget(edit, 1)
    row.addWidget(btn)
    return row, edit


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setWidget(widget)
    return area


# ═══════════════════════════════════════════════════════════════════════════
# Tab
# ═══════════════════════════════════════════════════════════════════════════

class ClassificationTab(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._data_input_tab = None
        self._classify_rows: list[_ClassifyRow] = []
        self._worker: _Worker | None = None
        self._last_classify_out: Path | None = None
        self._last_train_out: Path | None = None
        self._trained_model: Path | None = None
        self._result_wavs: dict[str, Path] = {}
        self._setup_ui()
        self._sync_model_from_settings()
        state.settings_changed.connect(self._sync_model_from_settings)
        t.signal.changed.connect(self._apply_theme)

    def set_data_input_tab(self, tab) -> None:
        self._data_input_tab = tab

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(t.SP_5, t.SP_4, t.SP_5, t.SP_4)
        outer.setSpacing(t.SP_3)
        self._desc = QLabel(
            "Assign call types to detected calls with a trained model: a CNN first separates "
            "real USVs from noise, then a Random Forest picks the call type and flags calls it "
            "is unsure about as UNCERTAIN. Train a new model from your labeled recordings in "
            "the second sub-tab."
        )
        self._desc.setWordWrap(True)
        outer.addWidget(self._desc)

        inner = QTabWidget()
        inner.setObjectName("innerTabs")
        inner.setDocumentMode(True)
        inner.addTab(self._build_classify_page(), "Classify")
        inner.addTab(self._build_train_page(), "Train model")
        inner.setTabToolTip(0, "Run a trained model on detected calls of one or more recordings.")
        inner.setTabToolTip(1, "Train a new CNN + Random Forest model from labeled recordings.")
        self._inner = inner
        outer.addWidget(inner, 1)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._desc.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_SM}px;")
        for label in self.findChildren(QLabel):
            if label.property("muted"):
                label.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_XS}px;")
        for status in (getattr(self, "_cls_status", None), getattr(self, "_trn_status", None)):
            if status is not None:
                self._color_status(status)

    def _color_status(self, label: QLabel) -> None:
        kind = label.property("kind") or "neutral"
        color = {"neutral": t.TEXT_SECONDARY, "success": t.SUCCESS, "danger": t.DANGER}[kind]
        label.setStyleSheet(f"color: {color}; font-size: {t.TEXT_XS}px;")

    def _set_status(self, label: QLabel, text: str, kind: str = "neutral") -> None:
        label.setProperty("kind", kind)
        label.setText(text)
        self._color_status(label)

    # ── Classify page ─────────────────────────────────────────────────────

    def _build_classify_page(self) -> QWidget:
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, t.SP_2, t.SP_2, 0)
        col.setSpacing(t.SP_3)

        # Model
        model_group = QGroupBox("Model")
        model_group.setToolTip(
            "A model folder produced by Train model (<output>/run/model) or by the standalone "
            "USV_Klasifikace tool: manifest.json + rf.joblib + cnn/cnn.pt."
        )
        mcol = QVBoxLayout(model_group)
        row, self._model_edit = _browse_row(
            "No model selected", "Folder of a trained model (contains manifest.json).", self._browse_model,
        )
        self._model_edit.setReadOnly(True)
        mcol.addLayout(row)
        self._model_info = _muted("")
        mcol.addWidget(self._model_info)
        col.addWidget(model_group)

        # Recordings
        rec_group = QGroupBox("Recordings to classify")
        rec_group.setToolTip("Each recording's detected calls (segments) are classified.")
        rcol = QVBoxLayout(rec_group)
        self._cls_table = _table(
            ["Recording (WAV)", "Detected calls"],
            "Recordings and the detected-label file whose segments get classified.",
        )
        rcol.addWidget(self._cls_table)
        btns = QHBoxLayout()
        for text, tip, slot in (
            ("Add loaded", "Add the recording loaded in Data Input, with its current detected labels "
                           "(including unsaved Label Edit changes).", self._cls_add_loaded),
            ("Add batch", "Add every WAV of the Data Input batch folder, paired by filename with "
                          "the batch detected-labels folder.", self._cls_add_batch),
            ("Add files…", "Pick a WAV and its detected-label file.", self._cls_add_files),
            ("Remove", "Remove the selected rows.", self._cls_remove),
            ("Clear", "Remove all rows.", self._cls_clear),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch()
        rcol.addLayout(btns)
        col.addWidget(rec_group)

        # Options
        opt_group = QGroupBox("Options")
        form = QFormLayout(opt_group)
        self._cls_device = QComboBox()
        self._cls_device.addItems(_DEVICES)
        self._cls_device.setToolTip(_DEVICE_TIP)
        form.addRow("Device:", self._cls_device)
        row, self._cls_out_edit = _browse_row(
            "New folder next to the first WAV",
            "Results folder; must not exist yet (results are never overwritten).",
            lambda: self._browse_dir(self._cls_out_edit, "Select results folder"),
        )
        form.addRow("Output folder:", row)
        self._cls_write_back = QComboBox()
        for text, mode in (("Results folder only", "none"),
                           ("Also beside each label file (…_classified.txt)", "beside"),
                           ("Also overwrite the detected-label files", "overwrite")):
            self._cls_write_back.addItem(text, mode)
        self._cls_write_back.setToolTip(
            "Where else the classified labels are saved. 'Beside' writes <label file>_classified.txt "
            "next to each recording's detected-label file; 'Overwrite' replaces that file. "
            "Recordings added with 'Add loaded' have no label file and get <WAV>_classified.txt "
            "next to the WAV instead."
        )
        form.addRow("Save labels:", self._cls_write_back)
        self._cls_update_loaded = QCheckBox("Show results on the loaded recording")
        self._cls_update_loaded.setToolTip(
            "Replace the loaded recording's detected labels with the classified ones, so "
            "Visualization and Label Edit show the call types."
        )
        self._cls_update_loaded.setChecked(True)
        self._cls_drop_noise = QCheckBox("Leave NOISE calls out of the classified labels")
        self._cls_drop_noise.setToolTip(
            "Otherwise they stay in, labeled 'NOISE' (predictions.csv always keeps every call)."
        )
        self._cls_guess = QCheckBox("Label UNCERTAIN calls with the best guess (e.g. '5t?')")
        self._cls_guess.setToolTip(
            "The Random Forest's most likely type plus '?', instead of the plain word UNCERTAIN."
        )
        self._cls_guess.setChecked(True)
        for cb in (self._cls_update_loaded, self._cls_drop_noise, self._cls_guess):
            form.addRow("", cb)
        col.addWidget(opt_group)
        col.addStretch()

        # Run
        self._cls_runbar = _RunBar()
        self._cls_runbar.cancel_clicked.connect(self._cancel)
        col.addWidget(self._cls_runbar)
        run_row = QHBoxLayout()
        self._cls_status = _muted("")
        run_row.addWidget(self._cls_status, 1)
        self._cls_run_btn = QPushButton("Classify")
        self._cls_run_btn.setObjectName("primaryBtn")
        self._cls_run_btn.setMinimumHeight(36)
        self._cls_run_btn.setToolTip("Classify every listed recording's detected calls.")
        self._cls_run_btn.clicked.connect(self._run_classify)
        run_row.addWidget(self._cls_run_btn)
        col.addLayout(run_row)

        # Results (right)
        right = QGroupBox("Results")
        rcol2 = QVBoxLayout(right)
        self._cls_summary = _muted("Run a classification to see results here.")
        rcol2.addWidget(self._cls_summary)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Show:"))
        self._cls_filter = QComboBox()
        self._cls_filter.setToolTip("Filter rows by prediction.")
        self._cls_filter.currentIndexChanged.connect(
            lambda _: self._cls_proxy.set_value(self._cls_filter.currentData() or "")
        )
        frow.addWidget(self._cls_filter, 1)
        rcol2.addLayout(frow)
        self._cls_model = _PredictionsModel(self)
        self._cls_proxy = _PredictionFilter(self)
        self._cls_proxy.setSourceModel(self._cls_model)
        self._cls_view = QTableView()
        self._cls_view.setModel(self._cls_proxy)
        self._cls_view.setSortingEnabled(True)
        self._cls_view.verticalHeader().setVisible(False)
        self._cls_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._cls_view.horizontalHeader().setStretchLastSection(True)
        self._cls_view.setToolTip("Double-click a call of the loaded recording to jump to it in Visualization.")
        self._cls_view.doubleClicked.connect(self._jump_to_call)
        rcol2.addWidget(self._cls_view, 1)
        self._cls_open_btn = QPushButton("Open results folder")
        self._cls_open_btn.setToolTip(
            "predictions.csv (every call, with scores), expert_review_queue.csv (uncertain calls "
            "plus an audit sample), calls_features.csv and labels/<recording>_classified.txt."
        )
        self._cls_open_btn.setEnabled(False)
        self._cls_open_btn.clicked.connect(lambda: self._last_classify_out and _open_folder(self._last_classify_out))
        rcol2.addWidget(self._cls_open_btn, 0, Qt.AlignmentFlag.AlignRight)

        return self._split(left, right)

    def _split(self, left: QWidget, right: QWidget) -> QSplitter:
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(_scroll(left))
        right_wrap = QWidget()
        rlay = QVBoxLayout(right_wrap)
        rlay.setContentsMargins(t.SP_2, t.SP_2, 0, 0)
        rlay.addWidget(right)
        split.addWidget(right_wrap)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        return split

    # ── Train page ────────────────────────────────────────────────────────

    def _build_train_page(self) -> QWidget:
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, t.SP_2, t.SP_2, 0)
        col.setSpacing(t.SP_3)

        rec_group = QGroupBox("Training recordings")
        rec_group.setToolTip(
            "Labeled recordings. Calls take their call type from the labels file; detections "
            "that overlap no labeled call become NOISE examples."
        )
        rcol = QVBoxLayout(rec_group)
        self._trn_table = _table(
            ["Recording (WAV)", "Call-type labels", "Detected labels (NOISE)", "Group", "Split"],
            "Group: recordings of the same animal(s) must share a group (or list the animals, "
            "e.g. A01;A02 — shared animals join groups). Whole groups go to one split.\n"
            "Split: 'auto' freezes a ~60/20/20 train/calibration/test split of groups "
            "(needs at least 5 groups).",
        )
        rcol.addWidget(self._trn_table)
        btns = QHBoxLayout()
        for text, tip, slot in (
            ("Add loaded", "Add the loaded recording with its Data Input label files "
                           "(reference labels = call types, detected labels = NOISE source).",
             self._trn_add_loaded),
            ("Add batch", "Add every WAV of the Data Input batch folder, paired by filename with "
                          "the batch reference-labels (call types) and detected-labels folders.",
             self._trn_add_batch),
            ("Add files…", "Pick a WAV, its call-type label file and (optionally) a detected-label file.",
             self._trn_add_files),
            ("Remove", "Remove the selected rows.", lambda: self._remove_rows(self._trn_table)),
            ("Clear", "Remove all rows.", lambda: self._trn_table.setRowCount(0)),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch()
        rcol.addLayout(btns)
        col.addWidget(rec_group)

        lab_group = QGroupBox("Labels")
        form = QFormLayout(lab_group)
        self._trn_generic = QLineEdit("d")
        self._trn_generic.setToolTip(
            "Labels that mark a real USV without a call type (comma-separated), e.g. the "
            "detector placeholder 'd'. They train the USV/NOISE stage only."
        )
        form.addRow("USV without type:", self._trn_generic)
        self._trn_noise = QLineEdit("noise")
        self._trn_noise.setToolTip("Labels meaning NOISE (comma-separated, case-insensitive).")
        form.addRow("NOISE labels:", self._trn_noise)
        self._trn_ignore = QLineEdit("")
        self._trn_ignore.setPlaceholderText("none")
        self._trn_ignore.setToolTip("Labels to leave out of training entirely (comma-separated).")
        form.addRow("Ignore labels:", self._trn_ignore)
        self._trn_rejected = QCheckBox("Rejected detections (Label Edit) are NOISE")
        self._trn_rejected.setChecked(True)
        self._trn_rejected.setToolTip("A call whose detection was rejected in Label Edit is a NOISE example.")
        form.addRow("", self._trn_rejected)
        self._trn_unmatched = QCheckBox("Detections overlapping no labeled call are NOISE")
        self._trn_unmatched.setChecked(True)
        self._trn_unmatched.setToolTip(
            "Use the detector's false positives as NOISE examples — only when the labels file "
            "is complete, i.e. every real call in it is labeled."
        )
        form.addRow("", self._trn_unmatched)
        self._trn_min = QSpinBox()
        self._trn_min.setRange(1, 10_000)
        self._trn_min.setValue(10)
        self._trn_min.setToolTip(
            "Call types with fewer usable examples in the training split (or none in "
            "calibration) are learned as plain USV instead of as a type."
        )
        form.addRow("Min. examples per type:", self._trn_min)
        col.addWidget(lab_group)

        par_group = QGroupBox("Training")
        pform = QFormLayout(par_group)
        from squeak_peek.usv_classifier.core.schema import feature_sets
        self._trn_features = QComboBox()
        self._trn_features.addItems(list(feature_sets()))
        self._trn_features.setToolTip("Acoustic features the Random Forest uses (Compact = the validated default).")
        pform.addRow("Feature set:", self._trn_features)
        self._trn_epochs = self._spin(1, 1000, 12, "CNN training epochs (fixed; no early stopping).")
        pform.addRow("CNN epochs:", self._trn_epochs)
        self._trn_trees = self._spin(1, 10_000, 200, "Number of Random Forest trees.")
        pform.addRow("RF trees:", self._trn_trees)
        self._trn_batch = self._spin(1, 4096, 64, "CNN batch size.")
        pform.addRow("Batch size:", self._trn_batch)
        self._trn_seed = self._spin(0, 2**31 - 1, 7, "Random seed (splits, CNN, RF) — same seed, same model.")
        pform.addRow("Seed:", self._trn_seed)
        self._trn_k = self._spin(0, 100, 0, "Clusters suggested for review in preparation/ (0 = automatic).")
        self._trn_k.setSpecialValueText("auto")
        pform.addRow("Clusters (k):", self._trn_k)
        self._trn_device = QComboBox()
        self._trn_device.addItems(_DEVICES)
        self._trn_device.setToolTip(_DEVICE_TIP)
        pform.addRow("Device:", self._trn_device)
        row, self._trn_out_edit = _browse_row(
            "New folder next to the first WAV",
            "Training output folder; must not exist yet. The model is written to <folder>/run/model.",
            lambda: self._browse_dir(self._trn_out_edit, "Select training output folder"),
        )
        pform.addRow("Output folder:", row)
        col.addWidget(par_group)
        col.addStretch()

        self._trn_runbar = _RunBar()
        self._trn_runbar.cancel_clicked.connect(self._cancel)
        col.addWidget(self._trn_runbar)
        run_row = QHBoxLayout()
        self._trn_status = _muted("")
        run_row.addWidget(self._trn_status, 1)
        self._trn_preview_btn = QPushButton("Check data")
        self._trn_preview_btn.setToolTip("Count examples per label and split from the label files (no audio).")
        self._trn_preview_btn.clicked.connect(self._preview_training)
        run_row.addWidget(self._trn_preview_btn)
        self._trn_run_btn = QPushButton("Train model")
        self._trn_run_btn.setObjectName("primaryBtn")
        self._trn_run_btn.setMinimumHeight(36)
        self._trn_run_btn.setToolTip(
            "Extract features, train the CNN and Random Forest on the train split, calibrate "
            "thresholds on the calibration split, then evaluate on the held-out test split."
        )
        self._trn_run_btn.clicked.connect(self._run_training)
        run_row.addWidget(self._trn_run_btn)
        col.addLayout(run_row)

        right = QGroupBox("Training results")
        rcol2 = QVBoxLayout(right)
        self._trn_summary = _muted("Check the data or train a model to see results here.")
        self._trn_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rcol2.addWidget(self._trn_summary)
        self._trn_counts = _table(["Label"], "Examples per label and split.")
        self._trn_counts.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        rcol2.addWidget(self._trn_counts, 1)
        brow = QHBoxLayout()
        brow.addStretch()
        self._trn_open_btn = QPushButton("Open output folder")
        self._trn_open_btn.setEnabled(False)
        self._trn_open_btn.setToolTip(
            "run/model (the model), run/evaluation (test metrics), run/classification "
            "(test predictions), training_data (segments + labels used)."
        )
        self._trn_open_btn.clicked.connect(lambda: self._last_train_out and _open_folder(self._last_train_out))
        brow.addWidget(self._trn_open_btn)
        self._trn_use_btn = QPushButton("Use for classification")
        self._trn_use_btn.setEnabled(False)
        self._trn_use_btn.setToolTip("Select the new model in Classify and for the Detection tab.")
        self._trn_use_btn.clicked.connect(self._use_trained_model)
        brow.addWidget(self._trn_use_btn)
        rcol2.addLayout(brow)
        return self._split(left, right)

    @staticmethod
    def _spin(lo: int, hi: int, value: int, tip: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        spin.setToolTip(tip)
        spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return spin

    # ── Shared helpers ────────────────────────────────────────────────────

    def _browse_dir(self, edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            edit.setText(path)

    def _remove_rows(self, table: QTableWidget) -> list[int]:
        rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
        for r in rows:
            table.removeRow(r)
        return rows

    def _paths(self, kind: str) -> tuple[str, str, str]:
        """(wav-or-usv-folder, reference, detected) from Data Input."""
        di = self._data_input_tab
        if di is None:
            return "", "", ""
        return di.single_paths() if kind == "single" else di.batch_paths()

    def _busy(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Wait for the current run to finish (or cancel it).")
            return True
        return False

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            for bar in (self._cls_runbar, self._trn_runbar):
                bar.cancel_btn.setEnabled(False)
                bar.message.setText("Cancelling after the current step…")

    def _start(self, fn, bar: _RunBar, on_success, on_fail_status: QLabel) -> None:
        self._cls_run_btn.setEnabled(False)
        self._trn_run_btn.setEnabled(False)
        bar.start()
        worker = _Worker(fn, self)
        worker.progressed.connect(bar.show_progress)

        def finish() -> None:
            bar.stop()
            self._cls_run_btn.setEnabled(True)
            self._trn_run_btn.setEnabled(True)

        def ok(result) -> None:
            finish()
            on_success(result)

        def failed(message: str, tb: str) -> None:
            finish()
            self._set_status(on_fail_status, f"Failed: {message}", "danger")
            box = QMessageBox(QMessageBox.Icon.Critical, "USV classification failed", message, parent=self)
            box.setDetailedText(tb)
            box.exec()

        def cancelled() -> None:
            finish()
            self._set_status(on_fail_status, "Cancelled.", "neutral")

        worker.succeeded.connect(ok)
        worker.failed.connect(failed)
        worker.was_cancelled.connect(cancelled)
        self._worker = worker
        worker.start()

    # ── Model selection ───────────────────────────────────────────────────

    def _browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select model folder (contains manifest.json)")
        if path:
            self._select_model(Path(path))

    def _select_model(self, path: Path) -> bool:
        """Show a model's summary and store it for the Detection-tab plugin."""
        from squeak_peek.usv_classifier.api import manifest_path, read_model_info

        try:
            info = read_model_info(path)
        except Exception as exc:  # noqa: BLE001
            self._model_info.setText(f"Not a usable model: {exc}")
            return False
        self._model_edit.setText(str(manifest_path(path).parent))
        text = info.summary()
        if not (info.rf_target_met and info.cnn_target_met):
            text += ". Calibration targets not met: review its predictions with extra care."
        self._model_info.setText(text)
        settings = self._state.settings.classification
        params = settings.params_for(PLUGIN_ID)
        if params.modelPath != str(info.path):
            settings.set_params(PLUGIN_ID, params.model_copy(update={"modelPath": str(info.path)}))
            self._state.settings_changed.emit()
        return True

    def _sync_model_from_settings(self) -> None:
        try:
            path = self._state.settings.classification.params_for(PLUGIN_ID).modelPath
        except KeyError:
            return
        if not path:
            return
        from squeak_peek.usv_classifier.api import manifest_path
        current = self._model_edit.text().strip()
        if current and manifest_path(current) == Path(path):
            return
        if Path(path).exists():
            self._select_model(Path(path))

    # ── Classify: recordings ──────────────────────────────────────────────

    def _cls_append(self, row: _ClassifyRow, source: str) -> None:
        self._classify_rows.append(row)
        r = self._cls_table.rowCount()
        self._cls_table.insertRow(r)
        self._cls_table.setItem(r, 0, _ro_item(row.wav.name, str(row.wav)))
        self._cls_table.setItem(r, 1, _ro_item(f"{source} ({len(row.labels)} calls)",
                                               str(row.labels_path or "Loaded recording's labels")))
        self._cls_table.resizeColumnToContents(0)

    def _cls_add_loaded(self) -> None:
        s = self._state
        if s.wav_path is None or not s.detected_labels:
            QMessageBox.warning(self, "Nothing loaded",
                                "Load a WAV with detected labels in Data Input (or run Detection) first.")
            return
        self._cls_append(_ClassifyRow(s.wav_path, None, list(s.detected_labels), from_state=True),
                         "loaded labels")

    def _cls_add_batch(self) -> None:
        from squeak_peek.usv_classifier.api import find_matching_file, read_label_file

        usv, _ref, det = self._paths("batch")
        if not usv or not det:
            QMessageBox.warning(self, "No batch folders",
                                "Select the USV folder and the detected-labels folder in Data Input → "
                                "Batch folder first.")
            return
        missing = []
        for wav in sorted(Path(usv).glob("*.wav")):
            txt = find_matching_file(det, wav)
            if txt is None:
                missing.append(wav.name)
                continue
            self._cls_append(_ClassifyRow(wav, txt, read_label_file(txt)), txt.name)
        if missing:
            self._set_status(self._cls_status, f"No detected labels for: {', '.join(missing)}", "danger")

    def _cls_add_files(self) -> None:
        from squeak_peek.usv_classifier.api import read_label_file

        wav, _ = QFileDialog.getOpenFileName(self, "Select WAV", "", "WAV files (*.wav *.WAV);;All files (*)")
        if not wav:
            return
        txt, _ = QFileDialog.getOpenFileName(self, f"Detected labels for {Path(wav).name}",
                                             str(Path(wav).parent), "Text files (*.txt);;All files (*)")
        if not txt:
            return
        self._cls_append(_ClassifyRow(Path(wav), Path(txt), read_label_file(txt)), Path(txt).name)

    def _cls_remove(self) -> None:
        for r in self._remove_rows(self._cls_table):
            del self._classify_rows[r]

    def _cls_clear(self) -> None:
        self._cls_table.setRowCount(0)
        self._classify_rows.clear()

    # ── Classify: run ─────────────────────────────────────────────────────

    def _run_classify(self) -> None:
        from squeak_peek.usv_classifier import api

        if self._busy():
            return
        model = self._model_edit.text().strip()
        if not model:
            QMessageBox.warning(self, "No model", "Select a trained model folder first.")
            return
        if not self._classify_rows:
            QMessageBox.warning(self, "No recordings", "Add at least one recording to classify.")
            return
        inputs = []
        for row in self._classify_rows:
            labels = row.labels
            if row.from_state and self._state.wav_path == row.wav and self._state.detected_labels:
                labels = list(self._state.detected_labels)   # pick up later Label Edit changes
            inputs.append(api.ClassifyInput(row.wav, labels, labels_file=row.labels_path))
        out = Path(self._cls_out_edit.text().strip() or _default_out(inputs[0].wav, "classification"))
        if out.exists():
            QMessageBox.warning(self, "Output exists", f"{out} already exists — choose a new folder.")
            return
        write_back = self._cls_write_back.currentData()
        existing = [p for p in (api.write_back_target(i, write_back) for i in inputs) if p and p.exists()]
        if existing:
            names = "\n".join(p.name for p in existing[:10]) + ("\n…" if len(existing) > 10 else "")
            if QMessageBox.question(
                self, "Overwrite label files?",
                f"Classifying will overwrite {len(existing)} existing label file(s):\n\n{names}",
            ) != QMessageBox.StandardButton.Yes:
                return
        device = self._cls_device.currentText()
        drop_noise, guess = self._cls_drop_noise.isChecked(), self._cls_guess.isChecked()
        cache = _feature_cache_dir()
        self._set_status(self._cls_status, "")

        def job(progress, cancelled):
            return api.classify_recordings(
                inputs, model, out, device=device, cache_dir=cache,
                drop_noise=drop_noise, guess_uncertain=guess, write_back=write_back,
                progress=progress, cancelled=cancelled,
            )

        self._start(job, self._cls_runbar, self._classify_done, self._cls_status)

    def _classify_done(self, result) -> None:
        self._last_classify_out = result.out_dir
        self._result_wavs = {rid: Path(w).resolve() for rid, w in result.wavs.items()}
        self._cls_open_btn.setEnabled(True)
        self._cls_model.set_frame(result.predictions)
        self._cls_view.resizeColumnsToContents()

        counts = result.counts()
        self._cls_filter.blockSignals(True)
        self._cls_filter.clear()
        self._cls_filter.addItem(f"All calls ({len(result.predictions)})", "")
        n_unc = int(counts.get("UNCERTAIN", 0) + counts.get("PROCESSING_ERROR", 0))
        self._cls_filter.addItem(f"Needs review — uncertain / errors ({n_unc})", "__uncertain__")
        for name, n in counts.items():
            self._cls_filter.addItem(f"{name} ({n})", name)
        self._cls_filter.blockSignals(False)
        self._cls_proxy.set_value("")

        types = {k: v for k, v in counts.items() if k not in ("NOISE", "UNCERTAIN", "PROCESSING_ERROR")}
        parts = [f"{len(result.predictions)} calls in {len(result.wavs)} recording(s)",
                 f"NOISE {int(counts.get('NOISE', 0))}",
                 f"UNCERTAIN {int(counts.get('UNCERTAIN', 0))}"]
        if counts.get("PROCESSING_ERROR", 0):
            parts.append(f"errors {int(counts['PROCESSING_ERROR'])}")
        parts.append("types: " + (", ".join(f"{k} {v}" for k, v in types.items()) or "—"))
        saved = f"\nSaved to {result.out_dir}"
        if result.noise_free_files:
            saved += " (labels/ holds both the full set and a NOISE-free copy)"
        if result.written_back:
            saved += f" and {len(result.written_back)} label file(s) written back"
        self._cls_summary.setText(" · ".join(parts) + saved)

        applied = ""
        loaded = self._state.wav_path.resolve() if self._state.wav_path else None
        if self._cls_update_loaded.isChecked() and loaded is not None:
            for rid, wav in self._result_wavs.items():
                if wav == loaded:
                    self._state.detected_labels = result.labels[rid]
                    # Point Data Input at the file these labels came from —
                    # the write-back copy when there is one, else the run's own.
                    self._state.detected_labels_path = (
                        result.written_back.get(rid) or result.label_files[rid])
                    self._add_call_types(result.labels[rid])
                    self._state.labels_changed.emit()
                    applied = " Loaded recording updated."
                    break
        self._set_status(self._cls_status, f"Done.{applied}", "success")

    def _add_call_types(self, labels: list[Label]) -> None:
        """Make predicted types selectable in Label Edit."""
        le = self._state.settings.label_edit
        known = le.classification_list
        new = sorted({lbl.label for lbl in labels if lbl.label and not lbl.label.endswith("?")} - set(known))
        if new:
            le.classifications = ",".join(known + new)
            self._state.settings_changed.emit()

    def _jump_to_call(self, index) -> None:
        row = self._cls_model.row(self._cls_proxy.mapToSource(index).row())
        wav = self._result_wavs.get(str(row.RecordingID))
        s = self._state
        if wav is None or s.wav_path is None or wav != s.wav_path.resolve():
            self._set_status(self._cls_status, "That call is not in the loaded recording.", "neutral")
            return
        center = (float(row.Start_s) + float(row.End_s)) / 2
        s.segment_start = max(0.0, min(center - s.segment_length / 2, max(0.0, s.duration - s.segment_length)))
        s.segment_changed.emit()
        self._set_status(self._cls_status, f"Visualization moved to {center:.3f} s.", "neutral")

    # ── Train: recordings ─────────────────────────────────────────────────

    def _trn_append(self, wav: Path, labels: Path, detected: Path | None, group: str = "") -> None:
        table = self._trn_table
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, _ro_item(wav.name, str(wav)))
        table.item(r, 0).setData(Qt.ItemDataRole.UserRole, str(wav))
        table.setItem(r, 1, _ro_item(labels.name, str(labels)))
        table.item(r, 1).setData(Qt.ItemDataRole.UserRole, str(labels))
        det_item = _ro_item(detected.name if detected else "—", str(detected) if detected else "No NOISE source")
        det_item.setData(Qt.ItemDataRole.UserRole, str(detected) if detected else "")
        table.setItem(r, 2, det_item)
        group_item = QTableWidgetItem(group or wav.stem)
        group_item.setToolTip("Editable. Recordings of the same animal(s) must share a group.")
        table.setItem(r, 3, group_item)
        split = QComboBox()
        from squeak_peek.usv_classifier.api import SPLITS
        split.addItems(list(SPLITS))
        split.setToolTip("'auto' for all rows = automatic ~60/20/20 split of groups.")
        table.setCellWidget(r, 4, split)
        table.resizeColumnsToContents()

    def _trn_add_loaded(self) -> None:
        wav, ref, det = self._paths("single")
        if not wav or not (ref or det):
            QMessageBox.warning(self, "Nothing loaded",
                                "Select a WAV and a labels file in Data Input first. Reference labels "
                                "give the call types; detected labels add NOISE examples.")
            return
        labels, detected = (Path(ref), Path(det) if det else None) if ref else (Path(det), None)
        self._trn_append(Path(wav), labels, detected)

    def _trn_add_batch(self) -> None:
        from squeak_peek.usv_classifier.api import find_matching_file

        usv, ref, det = self._paths("batch")
        if not usv or not (ref or det):
            QMessageBox.warning(self, "No batch folders",
                                "Select the USV folder and the reference-labels (call types) folder in "
                                "Data Input → Batch folder first.")
            return
        skipped = []
        for wav in sorted(Path(usv).glob("*.wav")):
            r = find_matching_file(ref, wav) if ref else None
            d = find_matching_file(det, wav) if det else None
            if r is None and d is None:
                skipped.append(wav.name)
                continue
            self._trn_append(wav, r or d, d if r else None)
        if skipped:
            self._set_status(self._trn_status, f"No labels for (skipped): {', '.join(skipped)}", "neutral")

    def _trn_add_files(self) -> None:
        wav, _ = QFileDialog.getOpenFileName(self, "Select WAV", "", "WAV files (*.wav *.WAV);;All files (*)")
        if not wav:
            return
        folder = str(Path(wav).parent)
        labels, _ = QFileDialog.getOpenFileName(self, f"Call-type labels for {Path(wav).name}", folder,
                                                "Text files (*.txt);;All files (*)")
        if not labels:
            return
        det, _ = QFileDialog.getOpenFileName(
            self, "Detected labels (optional — Cancel to skip)", folder, "Text files (*.txt);;All files (*)")
        self._trn_append(Path(wav), Path(labels), Path(det) if det else None)

    def _training_inputs(self):
        from squeak_peek.usv_classifier.api import TrainingInput

        table = self._trn_table
        inputs = []
        for r in range(table.rowCount()):
            det = table.item(r, 2).data(Qt.ItemDataRole.UserRole)
            inputs.append(TrainingInput(
                wav=Path(table.item(r, 0).data(Qt.ItemDataRole.UserRole)),
                labels=Path(table.item(r, 1).data(Qt.ItemDataRole.UserRole)),
                detected=Path(det) if det else None,
                group=(table.item(r, 3).text() if table.item(r, 3) else "").strip(),
                split=table.cellWidget(r, 4).currentText(),
            ))
        return inputs

    def _training_options(self):
        from squeak_peek.usv_classifier.api import TrainingOptions

        return TrainingOptions(
            generic_labels=_csv_list(self._trn_generic.text()),
            noise_labels=_csv_list(self._trn_noise.text()),
            ignore_labels=_csv_list(self._trn_ignore.text()),
            rejected_as_noise=self._trn_rejected.isChecked(),
            unmatched_detections_as_noise=self._trn_unmatched.isChecked(),
            min_examples=self._trn_min.value(),
            feature_set=self._trn_features.currentText(),
            epochs=self._trn_epochs.value(),
            trees=self._trn_trees.value(),
            batch_size=self._trn_batch.value(),
            seed=self._trn_seed.value(),
            k=self._trn_k.value() or None,
            device=self._trn_device.currentText(),
        )

    def _show_counts(self, counts: pd.DataFrame) -> None:
        table = self._trn_counts
        cols = list(counts.columns)
        table.clear()
        table.setColumnCount(len(cols) + 2)
        table.setHorizontalHeaderLabels(["Label", *cols, "Total"])
        table.setRowCount(len(counts))
        order = counts.sum(axis=1).sort_values(ascending=False).index
        for r, label in enumerate(order):
            table.setItem(r, 0, _ro_item(str(label)))
            for c, split in enumerate(cols, 1):
                table.setItem(r, c, _ro_item(str(int(counts.loc[label, split]))))
            table.setItem(r, len(cols) + 1, _ro_item(str(int(counts.loc[label].sum()))))
        table.resizeColumnsToContents()

    # ── Train: preview / run ──────────────────────────────────────────────

    def _preview_training(self) -> None:
        from squeak_peek.usv_classifier.api import build_training_set

        inputs = self._training_inputs()
        if not inputs:
            QMessageBox.warning(self, "No recordings", "Add labeled recordings first.")
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ts = build_training_set(inputs, Path(tmp), self._training_options())
        except Exception as exc:  # noqa: BLE001
            self._set_status(self._trn_status, str(exc), "danger")
            return
        self._show_counts(ts.counts())
        groups = ts.manifest.groupby("Split").GroupID.nunique().to_dict()
        lines = [
            f"{len(ts.review)} examples from {len(ts.manifest)} recording(s). Groups per split: "
            + ", ".join(f"{k} {v}" for k, v in groups.items()) + ".",
            "'USV' rows train only the USV/NOISE stage. Types rare in train or absent from "
            "calibration are demoted to USV when training starts.",
            *ts.notes,
        ]
        self._trn_summary.setText("\n".join(lines))
        self._set_status(self._trn_status, "Data OK.", "success")

    def _run_training(self) -> None:
        from squeak_peek.usv_classifier import api

        if self._busy():
            return
        inputs = self._training_inputs()
        if not inputs:
            QMessageBox.warning(self, "No recordings", "Add labeled recordings first.")
            return
        options = self._training_options()
        out = Path(self._trn_out_edit.text().strip() or _default_out(inputs[0].wav, "usv_model"))
        if out.exists():
            QMessageBox.warning(self, "Output exists", f"{out} already exists — choose a new folder.")
            return
        cache = _feature_cache_dir()
        self._set_status(self._trn_status, "")

        def job(progress, cancelled):
            return api.train_model(inputs, out, options, cache_dir=cache,
                                   progress=progress, cancelled=cancelled)

        self._start(job, self._trn_runbar, self._training_done, self._trn_status)

    def _training_done(self, result) -> None:
        self._last_train_out = result.out_dir
        self._trn_open_btn.setEnabled(True)
        self._show_counts(result.training_set.counts())
        lines = [f"Model saved to {result.model_dir}"]
        if result.info is not None:
            lines.append(result.info.summary())
        m = result.metrics
        if m:
            s1 = m.get("Stage1", {})
            lines.append(
                f"Held-out test ({m.get('NGroups', '?')} group(s), {m.get('N', '?')} typed calls): "
                f"accuracy {m['Accuracy']:.3f} · balanced accuracy {m['BalancedAccuracy']:.3f} · "
                f"macro-F1 {m['MacroF1']:.3f} · sent to review {m.get('ReviewFraction', 0):.1%}"
            )
            if s1:
                lines.append(
                    f"USV/NOISE stage on test: accuracy {s1['Accuracy']:.3f} · "
                    f"balanced accuracy {s1['BalancedAccuracy']:.3f}"
                )
            ci = m.get("MacroF1GroupBootstrap95CI")
            if ci:
                lines.append(f"Macro-F1 95% CI over test groups: {ci[0]:.3f}–{ci[1]:.3f}")
        else:
            lines.append("No labeled test calls — no held-out evaluation.")
        if result.dropped_types:
            lines.append("Learned as plain USV (too few examples): " + ", ".join(result.dropped_types))
        lines += [n for n in result.training_set.notes if not n.startswith("Too few")]
        lines.append(
            "Test scores come from groups the model never saw. They are not a biological "
            "validation — check the model on independently annotated recordings."
        )
        self._trn_summary.setText("\n".join(lines))
        self._trained_model = result.model_dir
        self._trn_use_btn.setEnabled(result.model_dir is not None)
        self._set_status(self._trn_status, "Training finished.", "success")

    def _use_trained_model(self) -> None:
        if self._trained_model and self._select_model(self._trained_model):
            self._inner.setCurrentIndex(0)

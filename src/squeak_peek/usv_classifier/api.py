"""
App-facing bridge between Squeak Peek Studio and the vendored USV pipeline.

The pipeline (``core``) classifies already-detected segments in two stages:
a compact CNN decides USV vs NOISE from a spectrogram image, then a Random
Forest assigns the call type from acoustic features, with calibrated
rejection (UNCERTAIN) when it is not confident. This module adapts it to the
app's world — WAV files plus 2-line label files / in-memory ``Label`` lists —
without changing any of its numerics:

* classify_recordings() mirrors ``core.workflows.master_inference`` (same
  output files, same predictions) but runs per recording so the GUI can show
  progress and cancel between recordings.
* train_model() turns labeled recordings (call-type labels, plus NOISE from
  rejected / unmatched detections) into the expert review the pipeline
  expects, then calls the tool's own ``train_and_classify``.
* classify_signal() classifies one in-memory signal (Detection-tab plugin).

Everything is synchronous; the GUI runs it on a worker thread.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from squeak_peek.labels.model import Label

logger = logging.getLogger(__name__)

# progress(fraction or None for indeterminate, message)
ProgressFn = Callable[[float | None, str], None]
CancelFn = Callable[[], bool]

#: ClusterPred values that are not call types.
NOT_A_TYPE = {"NOISE", "UNCERTAIN", "PROCESSING_ERROR"}

SPLITS = ("auto", "train", "calibration", "test")


class Cancelled(Exception):
    """Raised between pipeline steps when the user cancels."""


class TorchMissingError(RuntimeError):
    pass


def _noop_progress(_fraction: float | None, _message: str) -> None:
    pass


def _never() -> bool:
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Interop with the standalone tool
# ═══════════════════════════════════════════════════════════════════════════

def _install_usv_alias() -> None:
    """Make pickled models interchangeable with the standalone tool.

    Its rf.joblib pickles ``usv.rf.RFModel``; ours are saved under the same
    module names, so a model trained in either place loads in the other.
    """
    if "usv" in sys.modules:
        return
    import importlib

    from . import core
    sys.modules["usv"] = core
    for name in ("common", "config", "schema", "labels", "metrics", "splits", "rf", "clustering"):
        sys.modules[f"usv.{name}"] = importlib.import_module(f"{core.__name__}.{name}")
    rf = sys.modules["usv.rf"]
    clustering = sys.modules["usv.clustering"]
    for cls in (rf.RFModel, rf.DistanceModel):
        cls.__module__ = "usv.rf"
    clustering.ClusterModel.__module__ = "usv.clustering"


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise TorchMissingError(
            "USV classification needs PyTorch, which is not installed in "
            f"this Python environment ({sys.executable}). Install with: "
            f'"{sys.executable}" -m pip install torch torchvision'
        ) from exc


def _core():
    """Import the pipeline lazily (torch/sklearn are heavy)."""
    _require_torch()
    _install_usv_alias()
    from . import core
    from .core import entry, pipeline  # noqa: F401  (attribute access below)
    return core


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    path: Path                      # the manifest.json
    bundle_id: str
    classes: list[str]              # RF call types (NOISE is the CNN stage)
    feature_columns: list[str]
    rf_target_met: bool
    cnn_target_met: bool
    demo_only: bool
    environment: dict = field(default_factory=dict)
    split_groups: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.bundle_id[:12]

    def summary(self) -> str:
        targets = []
        targets.append("RF calibration target met" if self.rf_target_met else "RF calibration target NOT met")
        targets.append("CNN calibration target met" if self.cnn_target_met else "CNN calibration target NOT met")
        text = (
            f"Model {self.short_id} · call types: {', '.join(self.classes) or '—'} · "
            + " · ".join(targets)
        )
        if self.demo_only:
            text += " · DEMO ONLY (synthetic labels)"
        return text


def manifest_path(model: str | Path) -> Path:
    p = Path(model)
    return p / "manifest.json" if p.is_dir() else p


def read_model_info(model: str | Path) -> ModelInfo:
    """Read a trained model's manifest (and RF classes). Validates checksums."""
    import joblib

    path = manifest_path(model)
    if not path.is_file():
        raise FileNotFoundError(f"No manifest.json in {model}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "USV_PY_BUNDLE_1":
        raise ValueError(f"{path} is not a USV classifier model (format {manifest.get('format')!r})")
    _install_usv_alias()
    rf_path = (path.parent / manifest["models"]["rf"]["path"]).resolve()
    from .core.common import file_hash
    if file_hash(rf_path) != manifest["models"]["rf"]["sha256"]:
        raise ValueError("Model checksum mismatch: rf")
    rf = joblib.load(rf_path)  # trusted, locally produced bundles only
    return ModelInfo(
        path=path,
        bundle_id=manifest.get("bundle_id", ""),
        classes=[str(c) for c in rf.forest.classes_],
        feature_columns=list(manifest.get("feature_columns", [])),
        rf_target_met=bool(manifest.get("rf_target_met")),
        cnn_target_met=bool(manifest.get("cnn_target_met")),
        demo_only=bool(manifest.get("demo_only", False)),
        environment=manifest.get("environment", {}),
        split_groups=manifest.get("split_groups", {}),
    )


_bundle_cache: dict[tuple, tuple] = {}


def load_model(model: str | Path, device: str = "cpu") -> tuple:
    """core.pipeline.load_bundle, cached per (manifest, mtimes, device)."""
    core = _core()
    path = manifest_path(model).resolve()
    files = [path]
    try:
        declared = json.loads(path.read_text(encoding="utf-8")).get("models", {})
        files += [path.parent / m["path"] for m in declared.values()]
    except (OSError, ValueError, KeyError, TypeError):
        pass  # load_bundle reports the real problem
    stamps = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
    key = (str(path), stamps, device)
    if key not in _bundle_cache:
        _bundle_cache.clear()
        _bundle_cache[key] = core.pipeline.load_bundle(path, device)
    return _bundle_cache[key]


# ═══════════════════════════════════════════════════════════════════════════
# Label files ↔ segment tables
# ═══════════════════════════════════════════════════════════════════════════

def read_label_file(path: str | Path) -> list[Label]:
    """Read the app's 2-line label format, falling back to plain
    ``start end`` lines (the standalone tool's segment TXT)."""
    from squeak_peek.labels.io import import_labels

    labels = import_labels(path)
    if labels:
        return labels
    from .core.io import read_segments_txt
    seg = read_segments_txt(path)
    return [Label(float(r.Start_s), float(r.End_s)) for r in seg.itertuples()]


def write_segments(labels: Sequence[Label], path: Path) -> None:
    """One ``start<TAB>end`` line per label, so SourceRow == list index + 1.

    repr() round-trips floats exactly, so segment bounds are bit-identical
    to the Label objects they came from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for lbl in labels:
            fh.write(f"{float(lbl.start_time)!r}\t{float(lbl.end_time)!r}\n")


def unique_ids(stems: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for s in stems:
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.append(s if n == 0 else f"{s}_{n + 1}")
    return out


def find_matching_file(folder: str | Path, wav: str | Path, pattern: str = "*.txt") -> Path | None:
    """Label file in ``folder`` whose name starts with the WAV's stem
    (newest wins, e.g. several timestamped detector exports)."""
    folder, stem = Path(folder), Path(wav).stem
    if not folder.is_dir():
        return None
    hits = [p for p in folder.glob(pattern) if p.stem == stem or p.stem.startswith(stem)]
    if not hits:
        return None
    return max(hits, key=lambda p: (p.stem == stem, p.stat().st_mtime))


# ═══════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClassifyInput:
    wav: Path
    labels: list[Label]             # segments to classify (detector output)
    recording_id: str = ""


@dataclass
class ClassificationResult:
    out_dir: Path
    predictions: pd.DataFrame
    labels: dict[str, list[Label]]           # RecordingID -> classified labels
    label_files: dict[str, Path]             # RecordingID -> exported .txt
    wavs: dict[str, Path]                    # RecordingID -> WAV

    def counts(self) -> pd.Series:
        return self.predictions.ClusterPred.astype(str).value_counts()


def label_text(row, *, guess_uncertain: bool) -> str:
    pred = str(row.ClusterPred)
    raw = "" if pd.isna(row.RawClusterPred) else str(row.RawClusterPred)
    if guess_uncertain and pred == "UNCERTAIN" and raw:
        return f"{raw}?"
    return pred


def predictions_to_labels(
    predictions: pd.DataFrame,
    source: Sequence[Label],
    *,
    drop_noise: bool = False,
    guess_uncertain: bool = True,
) -> list[Label]:
    """Copy each source label with ``.label`` set to its prediction.

    Rows are matched by SourceRow (1-based index into ``source``, see
    write_segments). Segments the pipeline dropped as unparsable intervals
    keep their original label.
    """
    by_row = {int(r.SourceRow): r for r in predictions.itertuples()}
    out = []
    for i, lbl in enumerate(source, 1):
        row = by_row.get(i)
        if row is None:
            out.append(lbl)
            continue
        if drop_noise and str(row.ClusterPred) == "NOISE":
            continue
        out.append(replace(lbl, label=label_text(row, guess_uncertain=guess_uncertain)))
    return out


def classify_recordings(
    inputs: Sequence[ClassifyInput],
    model: str | Path,
    out_dir: str | Path,
    *,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    drop_noise: bool = False,
    guess_uncertain: bool = True,
    progress: ProgressFn = _noop_progress,
    cancelled: CancelFn = _never,
) -> ClassificationResult:
    """Classify detected segments of one or more recordings.

    Writes the same files as the standalone tool's ``classify`` —
    calls_features.csv (+ .diagnostics.json), predictions.csv,
    expert_review_queue.csv, inference_manifest.json — plus inputs/ (the
    exact segments used) and labels/<id>_classified.txt in the app's label
    format.
    """
    _core()
    from .core.common import new_directory, read_table, write_csv, write_json
    from .core.config import FeatureConfig
    from .core.io import iter_recordings
    from .core.pipeline import classify_calls
    from .core.review import export_expert_review_queue

    if not inputs:
        raise ValueError("No recordings to classify.")
    progress(None, "Loading model…")
    loaded = load_model(model, device)
    config = FeatureConfig(**loaded[0]["feature_config"])
    out = new_directory(Path(out_dir).resolve())

    ids = unique_ids([i.recording_id or Path(i.wav).stem for i in inputs])
    rows, sources, wavs = [], {}, {}
    for rid, item in zip(ids, inputs):
        if not item.labels:
            raise ValueError(f"{Path(item.wav).name}: no segments to classify.")
        txt = out / "inputs" / f"{rid}.txt"
        write_segments(item.labels, txt)
        rows.append({"RecordingID": rid, "WavFile": str(Path(item.wav).resolve()), "TxtFile": str(txt)})
        sources[rid] = list(item.labels)
        wavs[rid] = Path(item.wav)
    manifest = pd.DataFrame(rows)
    write_csv(manifest, out / "inputs" / "recordings.csv")

    # ── Feature extraction: core.io.extract_recordings, one recording at a time
    features_csv = out / "calls_features.csv"
    fd, temp = tempfile.mkstemp(prefix=".extract_", dir=out)
    diagnostics, first, n = [], True, len(manifest)
    progress(0.0, f"Extracting features: {manifest.RecordingID.iloc[0]}…")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            for i, (features, diag) in enumerate(iter_recordings(manifest, config, cache_dir)):
                features.to_csv(f, index=False, header=first)
                first = False
                diagnostics.append(diag)
                if cancelled():
                    raise Cancelled
                if i + 1 < n:
                    progress((i + 1) / n * 0.5, f"Extracting features: {manifest.RecordingID.iloc[i + 1]}…")
        os.replace(temp, features_csv)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    write_json(features_csv.with_suffix(".diagnostics.json"), diagnostics)

    # ── CNN + RF: core.pipeline.classify_calls, one recording at a time
    table = read_table(features_csv)
    parts = []
    groups = list(table.groupby("WavFile", sort=False))
    for i, (_wav, sub) in enumerate(groups):
        rid = str(sub.RecordingID.iloc[0])
        progress(0.5 + i / len(groups) * 0.5, f"Classifying {rid} ({len(sub)} calls)…")
        parts.append(classify_calls(sub, loaded, device))
        if cancelled():
            raise Cancelled
    prediction = pd.concat(parts, ignore_index=True)
    write_csv(prediction, out / "predictions.csv")
    write_csv(export_expert_review_queue(prediction), out / "expert_review_queue.csv")
    write_json(out / "inference_manifest.json",
               dict(ModelBundleID=loaded[0]["bundle_id"], NCalls=len(prediction)))

    # ── Back to the app's label format
    from squeak_peek.labels.io import export_labels
    labels, files = {}, {}
    for rid, src in sources.items():
        pred = prediction[prediction.RecordingID.astype(str) == rid]
        labels[rid] = predictions_to_labels(pred, src, drop_noise=drop_noise, guess_uncertain=guess_uncertain)
        files[rid] = out / "labels" / f"{rid}_classified.txt"
        files[rid].parent.mkdir(parents=True, exist_ok=True)
        export_labels(files[rid], labels[rid])
    progress(1.0, f"Classified {len(prediction)} calls.")
    return ClassificationResult(out, prediction, labels, files, wavs)


def classify_signal(
    labels: Sequence[Label],
    signal: np.ndarray,
    fs: int,
    model: str | Path,
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Classify segments of an in-memory signal (no files written).

    Same steps as classify_recordings for one recording; features come from
    the float64 cast of ``signal`` (identical to reading a 16-bit or float32
    WAV from disk).
    """
    _core()
    from .core.config import FeatureConfig
    from .core.features import extract_features
    from .core.io import assign_identity
    from .core.pipeline import classify_calls

    if not labels:
        return pd.DataFrame()
    loaded = load_model(model, device)
    config = FeatureConfig(**loaded[0]["feature_config"])
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    segments = pd.DataFrame(
        [{"SourceRow": i, "Start_s": float(lbl.start_time), "End_s": float(lbl.end_time),
          "Duration_s": float(lbl.end_time) - float(lbl.start_time)}
         for i, lbl in enumerate(labels, 1) if lbl.end_time > lbl.start_time],
        columns=["SourceRow", "Start_s", "End_s", "Duration_s"],
    )
    segments = assign_identity(segments, "signal")
    key = "<in-memory signal>"
    features = extract_features(x, fs, segments, key, config)
    features["WavFile"] = key
    return classify_calls(features, loaded, device, audio={key: (x, fs)})


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingInput:
    wav: Path
    labels: Path                    # call-type labels (e.g. reference labels)
    detected: Path | None = None    # detector output — unmatched detections become NOISE
    group: str = ""                 # independent group; "A01;A02" = animal members
    split: str = "auto"             # auto / train / calibration / test


@dataclass
class TrainingOptions:
    generic_labels: tuple[str, ...] = ("d",)       # USV without a call type
    noise_labels: tuple[str, ...] = ("noise",)     # labels meaning NOISE (case-insensitive)
    ignore_labels: tuple[str, ...] = ()            # dropped entirely
    rejected_as_noise: bool = True                 # Label Edit "detection rejected" → NOISE
    unmatched_detections_as_noise: bool = True
    min_examples: int = 10                         # per call type, in the training split
    feature_set: str = "Compact"
    epochs: int = 12
    trees: int = 200
    batch_size: int = 64
    seed: int = 7
    k: int | None = None
    device: str = "cpu"


@dataclass
class TrainingSet:
    manifest: pd.DataFrame          # RecordingID, WavFile, TxtFile, GroupID, Split
    review: pd.DataFrame            # RecordingID, CallID, Start_s, End_s, Expert*
    notes: list[str]

    def counts(self) -> pd.DataFrame:
        """Label × split table of training examples."""
        r = self.review.merge(self.manifest[["RecordingID", "Split"]], on="RecordingID")
        t = pd.crosstab(r.ExpertFinalLabel, r.Split)
        return t.reindex(columns=[c for c in ("train", "calibration", "test") if c in t.columns])


@dataclass
class TrainingResult:
    out_dir: Path
    status: str
    artifacts: dict
    model_dir: Path | None
    metrics: dict | None
    info: ModelInfo | None
    training_set: TrainingSet
    dropped_types: list[str]


def _overlaps(a: Label, others_start: np.ndarray, others_end: np.ndarray) -> bool:
    return bool(np.any((others_start < a.end_time) & (others_end > a.start_time)))


def build_training_set(
    inputs: Sequence[TrainingInput], out_dir: Path, options: TrainingOptions,
) -> TrainingSet:
    """Turn labeled recordings into a recording manifest + expert review.

    Per recording, every segment comes from:
      * the call-type labels file: its label is the call type; NOISE labels,
        or (optionally) rejected detections, are NOISE; generic labels ("d")
        are USV without a type;
      * optionally the detected-labels file: detections overlapping no
        labeled segment are NOISE (detector false positives).
    Writes one segment TXT per recording into ``out_dir`` (SourceRow ==
    line number, so CallIDs are the pipeline's own).
    """
    from .core.io import assign_identity, read_segments_txt
    from .core.legacy import connected_pair_groups
    from .core.splits import freeze_splits

    if not inputs:
        raise ValueError("Add at least one labeled recording.")
    notes: list[str] = []
    noise = {s.lower() for s in options.noise_labels}
    generic = {s.lower() for s in options.generic_labels}
    ignore = {s.lower() for s in options.ignore_labels}

    ids = unique_ids([Path(i.wav).stem for i in inputs])
    rows, reviews, unmatched = [], [], []
    for rid, item in zip(ids, inputs):
        labeled = read_label_file(item.labels)
        segs: list[tuple[Label, str]] = []
        for lbl in labeled:
            text = lbl.label.strip()
            if text.lower() in ignore:
                continue
            if text.lower() in noise or (options.rejected_as_noise and lbl.detection_state == "Rejected"):
                segs.append((lbl, "NOISE"))
            elif not text or text.lower() in generic or lbl.classification_state == "Rejected":
                segs.append((lbl, "USV"))
            else:
                segs.append((lbl, text))
        n_noise_det = 0
        if item.detected and options.unmatched_detections_as_noise:
            starts = np.array([lbl.start_time for lbl in labeled], float)
            ends = np.array([lbl.end_time for lbl in labeled], float)
            for det in read_label_file(item.detected):
                if not _overlaps(det, starts, ends):
                    segs.append((det, "NOISE"))
                    n_noise_det += 1
        segs.sort(key=lambda s: (s[0].start_time, s[0].end_time))
        if not segs:
            raise ValueError(f"{Path(item.labels).name}: no usable labels.")
        txt = out_dir / "segments" / f"{rid}.txt"
        write_segments([s[0] for s in segs], txt)
        identity = assign_identity(read_segments_txt(txt), rid)
        if len(identity) != len(segs):
            notes.append(f"{rid}: {len(segs) - len(identity)} invalid interval(s) skipped.")
        by_row = {i: s[1] for i, s in enumerate(segs, 1)}
        review = identity[["RecordingID", "CallID", "Start_s", "End_s"]].copy()
        review["ExpertFinalLabel"] = [by_row[int(r)] for r in identity.SourceRow]
        review["ExpertBinaryLabel"] = np.where(review.ExpertFinalLabel.eq("NOISE"), "NOISE", "USV")
        reviews.append(review)
        if n_noise_det:
            unmatched.append(f"{rid} {n_noise_det}")
        group = item.group.strip() or rid
        rows.append({"RecordingID": rid, "WavFile": str(Path(item.wav).resolve()), "TxtFile": str(txt),
                     "GroupID": group, "Split": (item.split or "auto").lower()})

    if unmatched:
        notes.append("Unmatched detections used as NOISE: " + ", ".join(unmatched) + ".")
    manifest = pd.DataFrame(rows)
    if manifest.GroupID.str.contains(";").any():
        manifest["AnimalMembers"] = manifest.GroupID
        manifest = connected_pair_groups(manifest).drop(columns="AnimalMembers")
    auto = manifest.Split.eq("auto")
    if auto.all():
        manifest = freeze_splits(manifest.drop(columns="Split"), "GroupID", seed=options.seed)
    elif auto.any():
        raise ValueError("Set Split for every recording, or leave all of them on 'auto'.")
    else:
        manifest = freeze_splits(manifest, "GroupID", seed=options.seed)  # validates
    return TrainingSet(manifest, pd.concat(reviews, ignore_index=True), notes)


def _restrict_types(ts: TrainingSet, features: pd.DataFrame, options: TrainingOptions) -> list[str]:
    """Demote call types the Random Forest cannot learn and calibrate to
    plain USV (they still train the USV/NOISE CNN).

    A type is kept when it has >= min_examples fit-eligible calls in the
    training split and at least one usable call in calibration — the
    pipeline's own class-coverage rule, checked up front so a rare label
    does not abort training.
    """
    from .core.common import bool_values
    from .core.schema import feature_sets, matrix, requires_contour

    columns = feature_sets()[options.feature_set]
    # Recording metadata (GroupID, Split) is already attached by iter_recordings.
    f = features.merge(ts.review[["CallID", "ExpertFinalLabel"]], on="CallID")
    usable = f.ProcessingStatus.eq("OK").to_numpy() & ~bool_values(f.BoundsClipped)
    eligible = usable & np.isfinite(matrix(f, columns)).all(axis=1)
    if requires_contour(columns):
        eligible &= bool_values(f.ContourValid)
    types = sorted(set(ts.review.ExpertFinalLabel) - {"NOISE", "USV"})
    kept = [
        c for c in types
        if ((f.ExpertFinalLabel == c) & f.Split.eq("train") & eligible).sum() >= options.min_examples
        and ((f.ExpertFinalLabel == c) & f.Split.eq("calibration") & usable).sum() >= 1
    ]
    dropped = [c for c in types if c not in kept]
    if dropped:
        ts.review.loc[ts.review.ExpertFinalLabel.isin(dropped), "ExpertFinalLabel"] = "USV"
        ts.notes.append(
            "Too few examples in train/calibration to learn as call types (kept as generic USV): "
            + ", ".join(dropped)
        )
    if len(kept) < 2:
        raise ValueError(
            f"The Random Forest needs at least two call types with >= {options.min_examples} "
            f"usable training examples and calibration examples; found {kept or 'none'}. "
            "Add labeled recordings, lower 'Min. examples per type', or adjust the splits."
        )
    split_of = ts.manifest.set_index("RecordingID").Split
    binary = ts.review.ExpertBinaryLabel.groupby(ts.review.RecordingID.map(split_of)).agg(set)
    for split in ("train", "calibration"):
        missing = {"USV", "NOISE"} - binary.get(split, set())
        if missing:
            raise ValueError(
                f"The {split} split has no {'/'.join(sorted(missing))} examples. The USV/NOISE "
                "CNN needs both: add detected-label files (unmatched detections become NOISE), "
                "reject false detections in Label Edit, or label segments 'NOISE'."
            )
    return dropped


def train_model(
    inputs: Sequence[TrainingInput],
    out_dir: str | Path,
    options: TrainingOptions | None = None,
    *,
    cache_dir: str | Path | None = None,
    progress: ProgressFn = _noop_progress,
    cancelled: CancelFn = _never,
) -> TrainingResult:
    """Train a new CNN + RF model from labeled recordings.

    Output layout is the standalone tool's (preparation/, model/,
    classification/, evaluation/) plus training_data/ (segments + review
    built from the label files). The model to use later is out_dir/model.
    """
    core = _core()
    import torch

    from .core.common import new_directory, write_csv, write_json
    from .core.config import FeatureConfig
    from .core.io import iter_recordings
    from .core.schema import feature_sets

    options = options or TrainingOptions()
    if options.feature_set not in feature_sets():
        raise ValueError(f"Unknown feature set {options.feature_set!r}")
    out = new_directory(Path(out_dir).resolve())
    data_dir = out / "training_data"
    cache_dir = cache_dir or out / "feature_cache"   # extraction below is reused by the pipeline

    progress(None, "Reading labels…")
    ts = build_training_set(inputs, data_dir, options)

    # Extract features up front (cached) — cancellable, and tells us which
    # call types have enough usable examples before committing to training.
    config = FeatureConfig()
    n = len(ts.manifest)
    chunks = []
    progress(0.0, f"Extracting features: {ts.manifest.RecordingID.iloc[0]}…")
    for i, (features, diag) in enumerate(iter_recordings(ts.manifest, config, cache_dir)):
        chunks.append(features)
        if cancelled():
            raise Cancelled
        if i + 1 < n:
            progress((i + 1) / n, f"Extracting features: {ts.manifest.RecordingID.iloc[i + 1]}…")
    dropped = _restrict_types(ts, pd.concat(chunks, ignore_index=True), options)
    write_csv(ts.manifest, data_dir / "recordings.csv")
    write_csv(ts.review, data_dir / "review.csv")
    write_json(data_dir / "notes.json", ts.notes)
    if cancelled():
        raise Cancelled

    # Pipeline-owned from here (not interruptible). It sets global torch
    # state for reproducible training; restore it for the rest of the app.
    progress(None, "Training CNN + Random Forest (this cannot be cancelled)…")
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    handler = _ProgressLogHandler(progress)
    core_logger = logging.getLogger(core.__name__)
    level = core_logger.level
    core_logger.setLevel(logging.INFO)
    core_logger.addHandler(handler)
    try:
        result = core.entry.train_and_classify(
            out / "run", recordings=ts.manifest, review=ts.review,
            feature_set=options.feature_set, epochs=options.epochs, trees=options.trees,
            seed=options.seed, batch_size=options.batch_size, k=options.k,
            cache=cache_dir, device=options.device, group_column="GroupID",
        )
    finally:
        core_logger.removeHandler(handler)
        core_logger.setLevel(level)
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)

    metrics = None
    if result.get("artifacts", {}).get("metrics"):
        metrics = json.loads(Path(result["artifacts"]["metrics"]).read_text(encoding="utf-8"))
    model_dir = Path(result["artifacts"]["model"]).parent if "model" in result.get("artifacts", {}) else None
    info = read_model_info(model_dir) if model_dir else None
    progress(1.0, "Training finished.")
    return TrainingResult(out, result["status"], result.get("artifacts", {}), model_dir,
                          metrics, info, ts, dropped)


class _ProgressLogHandler(logging.Handler):
    """Forward the pipeline's per-recording log lines as progress messages."""

    def __init__(self, progress: ProgressFn) -> None:
        super().__init__(logging.INFO)
        self._progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._progress(None, record.getMessage())
        except Exception:  # noqa: BLE001
            pass

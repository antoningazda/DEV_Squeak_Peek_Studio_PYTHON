from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import importlib.metadata
import numpy as np
import pandas as pd


def digest(value):
    if is_dataclass(value):
        value = asdict(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    atomic_write(path, json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_csv(table, path):
    atomic_write(path, table.to_csv(index=False))


def read_table(value):
    if isinstance(value, pd.DataFrame):
        table = value.copy()
    else:
        # Preserve string IDs, subtype labels, leading zeroes and literal NA labels.
        string_cols = {x: "string" for x in ("RecordingID", "CallID", "AnimalID", "GroupID", "Group", "AnimalMembers", "PairID", "Split", "Cluster_Clean", "ExpertFinalLabel", "ExpertBinaryLabel", "FinalLabel", "ClusterPred", "Cluster", "Stage1", "WavFile", "TxtFile", "ContourStatus", "ProcessingStatus", "ProcessingReason")}
        import csv
        with open(value, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        if len(set(header)) != len(header):
            raise ValueError("Duplicate CSV column names")
        table = pd.read_csv(value, dtype={k:v for k,v in string_cols.items() if k in header}, keep_default_na=False, na_values=[""],low_memory=False)
    if table.columns.duplicated().any():
        raise ValueError("Duplicate table column names")
    return table


def require_columns(table, columns):
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate requested feature/column")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def bool_values(series):
    mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False, "": False}
    s = series.fillna("").astype(str).str.strip().str.lower()
    if not s.isin(mapping).all():
        raise ValueError("Boolean column contains unsupported values")
    return s.map(mapping).to_numpy(bool)


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    import platform
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("rat-usv", "numpy", "scipy", "pandas", "scikit-learn", "torch", "Pillow", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions

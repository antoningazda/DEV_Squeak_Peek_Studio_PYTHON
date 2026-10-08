"""Strict pairing, stable identities and recording-level streaming extraction."""
from pathlib import Path
import logging
import time
import json
import re
import numpy as np
import pandas as pd
import soundfile as sf
from . import FEATURE_VERSION
from .common import digest, file_hash, read_table, require_columns, write_csv, write_json
from .config import FeatureConfig


def pair_txt_wav(directory, prefix_chars=None, allow_unmatched=False):
    root = Path(directory).resolve()
    if prefix_chars is not None and (int(prefix_chars) != prefix_chars or prefix_chars < 1):
        raise ValueError("prefix_chars must be a positive integer")
    def collect(extension):
        result = {}
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() == extension:
                key = p.stem if prefix_chars is None else p.stem[:prefix_chars]
                if key in result:
                    raise ValueError(f"Ambiguous {extension} pairing for {key}: {result[key]} and {p}")
                result[key] = p
        return result
    wav, txt = collect(".wav"), collect(".txt")
    unmatched = sorted(set(wav) ^ set(txt))
    if unmatched and not allow_unmatched:
        raise ValueError(f"Unmatched WAV/TXT keys: {unmatched[:20]}; use an explicit recording manifest or prefix_chars")
    keys = sorted(set(wav) & set(txt))
    if not keys:
        raise ValueError(f"No WAV/TXT pairs in {root}")
    table = pd.DataFrame([{"RecordingID":k, "WavFile":str(wav[k]), "TxtFile":str(txt[k])} for k in keys])
    table.attrs["unmatched"] = unmatched
    return table


def recording_manifest(value):
    table = read_table(value)
    require_columns(table, ["RecordingID", "WavFile", "TxtFile"])
    root = Path(value).resolve().parent if not isinstance(value, pd.DataFrame) else Path.cwd()
    for col in ("RecordingID", "WavFile", "TxtFile"):
        if table[col].isna().any() or table[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty {col}")
    if table.RecordingID.duplicated().any():
        raise ValueError("RecordingID must be unique")
    for col in ("WavFile", "TxtFile"):
        table[col] = [str((root/str(v)).resolve()) for v in table[col]]
        if table[col].duplicated().any():
            raise ValueError(f"Duplicate {col} in manifest")
    return table


def read_segments_txt(path):
    rows, diagnostics = [], []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith(("\\", "#", "%")):
            continue
        parts = re.split(r"[\s,;]+", text)
        try:
            start, end = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            diagnostics.append({"SourceRow":line_no, "Reason":"NON_NUMERIC_HEADER_OR_LINE", "Text":text})
            continue
        if not np.isfinite([start, end]).all() or end <= start:
            diagnostics.append({"SourceRow":line_no, "Reason":"INVALID_INTERVAL", "Text":text})
            continue
        # Physical line number remains stable even when headers/invalid rows occur.
        rows.append({"SourceRow":line_no, "Start_s":start, "End_s":end, "Duration_s":end-start})
    result = pd.DataFrame(rows, columns=["SourceRow", "Start_s", "End_s", "Duration_s"])
    result.attrs["diagnostics"] = diagnostics
    return result


def sample_bounds(start, end, fs, n_samples):
    if not np.isfinite([start, end, fs]).all() or fs <= 0 or end <= start:
        raise ValueError("Invalid interval or sample rate")
    a, b = max(0, int(np.floor(start*fs))), min(n_samples, int(np.ceil(end*fs)))
    if b-a < 2:
        raise ValueError("Interval lies outside recording or has fewer than two samples")
    return a, b


def read_audio(path):
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite waveform")
    return x.mean(axis=1), fs


def assign_identity(segments, recording_id):
    t = segments.copy()
    t["RecordingID"] = str(recording_id)
    t["CallID"] = [digest([str(recording_id), int(r.SourceRow), float(r.Start_s), float(r.End_s)]) for r in t.itertuples()]
    validate_identity(t)
    return t


def validate_identity(table):
    require_columns(table, ["RecordingID", "CallID"])
    for c in ("RecordingID", "CallID"):
        if table[c].isna().any() or table[c].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {c}")
    if table.duplicated(["RecordingID", "CallID"]).any() or table.CallID.duplicated().any():
        raise ValueError("Duplicate call identity")


def feature_cache_key(wav_hash, segments, config):
    # Animal labels/splits are attached after reading the cache, never cached.
    cols = ["RecordingID", "CallID", "SourceRow", "Start_s", "End_s"]
    return digest({"version":FEATURE_VERSION, "waveform":wav_hash, "segments":segments[cols].to_dict("records"), "config":__import__("dataclasses").asdict(config)})


def iter_recordings(manifest, config=None, cache_dir=None):
    from .features import extract_features, error_rows
    config = config or FeatureConfig()
    recordings = recording_manifest(manifest)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    for record in recordings.to_dict("records"):
        started=time.perf_counter()
        logging.getLogger(__name__).info("Extracting %s",record["RecordingID"])
        segments = assign_identity(read_segments_txt(record["TxtFile"]), record["RecordingID"])
        diagnostics = segments.attrs.get("diagnostics", [])
        waveform_hash, key, cache_hit = "", "", False
        try:
            waveform_hash = file_hash(record["WavFile"])
            key = feature_cache_key(waveform_hash, segments, config)
            cached = cache / (key+".csv") if cache else None
            cache_manifest=cached.with_suffix(".json") if cached else None
            if cached and cached.is_file() and cache_manifest.is_file() and json.loads(cache_manifest.read_text()).get("sha256")==file_hash(cached):
                features = read_table(cached)
                validate_identity(features)
                if features.CallID.tolist()!=segments.CallID.tolist() or not features.FeatureConfigHash.eq(digest(config)).all():
                    raise ValueError("Invalid cached feature identity/configuration")
                cache_hit = True
            else:
                x, fs = read_audio(record["WavFile"])
                features = extract_features(x, fs, segments, record["WavFile"], config)
                if cached and not features.ProcessingStatus.eq("PROCESSING_ERROR").any():
                    write_csv(features, cached)
                    write_json(cache_manifest,{"sha256":file_hash(cached),"key":key})
        except (OSError, ValueError, RuntimeError) as exc:
            features = error_rows(segments, record["WavFile"], str(exc), config)
        features["WavFile"] = record["WavFile"]
        features["WaveformSHA256"] = waveform_hash
        features["FeatureCacheKey"] = key
        for name,value in record.items():
            if name in ("RecordingID","WavFile","TxtFile"):continue
            if name in features:raise ValueError(f"Recording metadata would overwrite extracted column: {name}")
            features[name]=value
        seconds=time.perf_counter()-started
        logging.getLogger(__name__).info("Finished %s: %s calls in %.2f s (cache=%s)",record["RecordingID"],len(features),seconds,cache_hit)
        yield features, {"RecordingID":record["RecordingID"], "CacheHit":cache_hit, "NCalls":len(features), "Seconds":seconds,"LineDiagnostics":diagnostics}


def extract_recordings(manifest, out_csv, config=None, cache_dir=None):
    """Stream one recording at a time to disk; no whole-cohort accumulation."""
    import os
    import tempfile
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".extract_", dir=out_csv.parent)
    diagnostics, first = [], True
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            for features, diag in iter_recordings(manifest, config, cache_dir):
                features.to_csv(f, index=False, header=first)
                first = False
                diagnostics.append(diag)
        os.replace(temp, out_csv)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    write_json(out_csv.with_suffix(".diagnostics.json"), diagnostics)
    return out_csv


def add_features(table, config=None):
    """Recompute from source bounds and join by identity; never use padded bounds."""
    from .features import extract_features, error_rows
    table = read_table(table)
    validate_identity(table)
    require_columns(table, ["WavFile", "Start_s", "End_s", "SourceRow"])
    config = config or FeatureConfig()
    chunks = []
    for wav, sub in table.groupby("WavFile", sort=False):
        try:
            x, fs = read_audio(wav)
            chunk = extract_features(x, fs, sub, wav, config)
        except (OSError, ValueError, RuntimeError) as exc:
            chunk = error_rows(sub, wav, str(exc), config)
        chunk["WaveformSHA256"] = file_hash(wav) if Path(wav).is_file() else ""
        chunks.append(chunk)
    if not chunks:
        raise ValueError("No calls to recompute")
    result = pd.concat(chunks, ignore_index=True).set_index("CallID")
    result = result.loc[table.CallID].reset_index()
    for c in table:
        if c not in result:
            result[c] = table[c].to_numpy()
    return result

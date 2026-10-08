"""
Detect USVs in every WAV of a folder and score each against its reference
labels — one CSV row per recording plus a pooled total.

Uses the Python API directly, with the same detection + post-processing
pipeline as ``squeak-peek-cli detect``.

    python examples/evaluate_cohort.py recordings/ reference/ -d psd --min-tonality 0.5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek.audio.io import load_wav
from squeak_peek.config import AppSettings
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.io import export_labels_detector, import_labels
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.model import Label
from squeak_peek.labels.postprocess import (
    filter_broadband_labels,
    merge_close_labels,
    remove_short_labels,
)


def detect(wav: Path, detector_id: str, settings: AppSettings) -> list[Label]:
    """One recording through detector + post-processing (tonality, merge, min length)."""
    signal, fs = load_wav(wav)
    params = settings.detection.params_for(detector_id)
    labels = AbstractDetector.get(detector_id)(params).detect(signal, fs)

    post = settings.detection.post
    if post.minTonality > 0:  # before merging, so gaps don't dilute tonality
        labels = filter_broadband_labels(
            labels, signal, fs, post.minTonality,
            fcut_min=getattr(params, "fcutMin", 40_000.0),
            fcut_max=getattr(params, "fcutMax", 120_000.0),
        )
    labels = merge_close_labels(labels, post.maxGapToMerge)
    return remove_short_labels(labels, post.minLabelLength)


def find_reference(ref_dir: Path, wav: Path) -> Path | None:
    """``rec01.txt`` or ``rec01<anything>.txt`` (e.g. ``rec01-annotator.txt``)."""
    hits = sorted(ref_dir.glob(f"{wav.stem}*.txt"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wav_dir", type=Path)
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("-d", "--detector", default="psd")
    ap.add_argument("-s", "--settings", type=Path, default=Path("settings/default.json"))
    ap.add_argument("--min-tonality", type=float, default=None)
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("detected"))
    args = ap.parse_args()

    settings = AppSettings.from_json(args.settings)
    if args.min_tonality is not None:
        settings.detection.post.minTonality = args.min_tonality
    detector_id = args.detector.upper()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for wav in sorted(args.wav_dir.glob("*.wav")):
        ref = find_reference(args.ref_dir, wav)
        if ref is None:
            print(f"skip {wav.name}: no reference in {args.ref_dir}")
            continue
        labels = detect(wav, detector_id, settings)
        export_labels_detector(args.output_dir / f"{wav.stem}_{args.detector.lower()}_detected.txt", labels)
        s = compare_labels(labels, import_labels(ref))
        rows.append(dict(recording=wav.stem, detected=s.total_detected_labels,
                         reference=s.total_provided_labels, tp=s.true_positives,
                         fp=s.false_positives, fn=s.false_negatives,
                         precision=round(s.precision, 3), recall=round(s.recall, 3),
                         f1=round(s.f1_score, 3)))
        print(f"{wav.stem}: P={s.precision:.3f} R={s.recall:.3f} F1={s.f1_score:.3f}")

    if not rows:
        return
    # Pooled over all recordings (micro-average), not a mean of per-file F1.
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    rows.append(dict(recording="TOTAL", detected=sum(x["detected"] for x in rows),
                     reference=sum(x["reference"] for x in rows), tp=tp, fp=fp, fn=fn,
                     precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3)))
    print(f"TOTAL: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    summary = args.output_dir / f"summary_{args.detector.lower()}.csv"
    with open(summary, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

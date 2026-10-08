"""
Sweep one post-processing parameter (minTonality) on a single recording and
print precision / recall / F1 for each value — to pick a threshold before a
batch run. Detection runs once; only the post-processing is repeated.

    python examples/tonality_sweep.py rec.wav reference.txt -d bscd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek.audio.io import load_wav
from squeak_peek.config import AppSettings
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.io import import_labels
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.postprocess import (
    filter_broadband_labels,
    merge_close_labels,
    remove_short_labels,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wav", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("-d", "--detector", default="psd")
    ap.add_argument("-s", "--settings", type=Path, default=Path("settings/default.json"))
    ap.add_argument("--values", type=float, nargs="+",
                    default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    args = ap.parse_args()

    settings = AppSettings.from_json(args.settings)
    detector_id = args.detector.upper()
    params = settings.detection.params_for(detector_id)
    post = settings.detection.post

    signal, fs = load_wav(args.wav)
    raw = AbstractDetector.get(detector_id)(params).detect(signal, fs)
    reference = import_labels(args.reference)
    band = getattr(params, "fcutMin", 40_000.0), getattr(params, "fcutMax", 120_000.0)

    print(f"{detector_id}: {len(raw)} raw detections, {len(reference)} reference labels\n")
    print("minTonality  detected  precision  recall     F1")
    for t in args.values:
        labels = raw
        if t > 0:
            labels = filter_broadband_labels(labels, signal, fs, t, fcut_min=band[0], fcut_max=band[1])
        labels = remove_short_labels(merge_close_labels(labels, post.maxGapToMerge), post.minLabelLength)
        s = compare_labels(labels, reference)
        print(f"{t:11.2f}  {len(labels):8d}  {s.precision:9.3f}  {s.recall:6.3f}  {s.f1_score:5.3f}")


if __name__ == "__main__":
    main()

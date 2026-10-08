r"""
Build the RECORDINGS_CSV for ``squeak-peek-cli train-classifier`` from three
folders: recordings, call-type labels, and (optionally) detector output.
Files are paired by name prefix: ``rec01.wav`` ↔ ``rec01*.txt``.

    python examples/make_training_csv.py recordings/ labels/ training.csv \
        --detected detected/ --group-pattern "(m[A-Z]\d+)"

``--group-pattern`` is a regex whose first group extracts the animal ID from
the WAV name, so recordings of the same animal never land in different
train/calibration/test splits.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


def match(folder: Path, stem: str) -> Path | None:
    hits = sorted(folder.glob(f"{stem}*.txt"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wav_dir", type=Path)
    ap.add_argument("label_dir", type=Path)
    ap.add_argument("output_csv", type=Path)
    ap.add_argument("--detected", type=Path, default=None, help="Folder with detector output.")
    ap.add_argument("--group-pattern", default=None, help="Regex; group 1 = animal ID.")
    args = ap.parse_args()

    root = args.output_csv.absolute().parent  # paths in the CSV are relative to it

    def rel(p: Path) -> str:
        return Path(os.path.relpath(p.absolute(), root)).as_posix()

    rows = []
    for wav in sorted(args.wav_dir.glob("*.wav")):
        labels = match(args.label_dir, wav.stem)
        if labels is None:
            print(f"skip {wav.name}: no labels")
            continue
        detected = match(args.detected, wav.stem) if args.detected else None
        group = ""
        if args.group_pattern and (m := re.search(args.group_pattern, wav.stem)):
            group = m.group(1)
        rows.append({"WavFile": rel(wav), "LabelFile": rel(labels),
                     "DetectedFile": rel(detected) if detected else "",
                     "GroupID": group, "Split": "auto"})

    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["WavFile", "LabelFile", "DetectedFile", "GroupID", "Split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} recordings -> {args.output_csv}")


if __name__ == "__main__":
    main()

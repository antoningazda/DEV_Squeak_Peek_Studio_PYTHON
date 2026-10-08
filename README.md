# Squeak Peek Studio — Python Port

Python reimplementation of [Squeak Peek Studio](../DEV_Squeak-Peek-Studio), a MATLAB application for visualization, segmentation, and sonification of ultrasonic vocalizations (USVs) of laboratory rats.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation & Infrastructure | ✅ Done |
| 2 | Signal Processing & Detection Engines (PSD, BSCD, RBD) | ✅ Detectors ported (Tier 1); see `PORT_PLAN.md` |
| 3 | ML Feature Extraction & Detector | ✅ Feature extraction (Tier 1) + Random Forest detector, training, calibration (Tier 2) ported |
| 4 | Label System & Evaluation | ✅ Label I/O, post-processing, metrics ported (Tier 1); see `PORT_PLAN.md` |
| 5 | GUI (PyQt6) | ✅ Detection/Metrics tabs wired to PSD/BSCD/RBD/ML + compare_labels (Tier 3) |
| 6 | Testing, Validation & Packaging | ✅ CI green (3.11/3.12 × macOS/Ubuntu/Windows); unit + CLI test coverage across all detectors |

All of Tiers 1–3 in `PORT_PLAN.md` are complete and merged, including the ML
detector (WP6: `squeak_peek.detectors.ml.MLDetector`) and its training/
calibration pipeline (WP7: `squeak_peek.ml.train`, `squeak_peek.ml.optimize`).
Train a model with `squeak-peek-cli train <wav> --labels <labels> --output <path>`,
then point `Detection.ML.modelPath` at it in your settings JSON to use
`--detector ml` / the ML radio button (the Settings tab doesn't expose this
field yet — edit the JSON directly, or set `AppState.settings.detection.ml.modelPath`
before launching the GUI).

There is also a CNN (Faster R-CNN) object-detection alternative to the ML
detector — `squeak_peek.detectors.cnn.CNNDetector`, trained by
`squeak_peek.cnn.train`. Unlike MLDetector's frame classifier, it predicts a
call's (start_time, end_time, start_frequency, end_frequency) box directly,
the way DeepSqueak's own detector works. See **Training a CNN detector**
below.

## Tonality post-processing (optional)

Rodent USVs are narrowband FM whistles, while most false positives from the
energy/change-point detectors are broadband (cage knocks, bedding rustle,
scratching). `--min-tonality` drops detections whose energy isn't
concentrated around a moving peak frequency:

```bash
squeak-peek-cli detect audio.wav --detector bscd --min-tonality 0.5
```

It is **off by default** (`Detection.POST.minTonality = 0`) because it trades
recall for precision. Measured on held-out USVSEG mouse recordings, with the
threshold chosen on separate training recordings:

| detector | | precision | recall | F1 |
|---|---|---|---|---|
| PSD  | off | 0.465 | 0.707 | 0.561 |
| PSD  | `--min-tonality 0.5` | 0.934 | 0.675 | **0.784** |
| BSCD | off | 0.628 | 0.827 | 0.714 |
| BSCD | `--min-tonality 0.5` | 0.899 | 0.763 | **0.825** |

It runs before merging, so a merged detection spanning the gap between two
calls is not penalised. Detections too short to score are always kept.

## Training a CNN detector

torch + torchvision are part of the standard install (`pip install -e .`).

### 1. Get labeled training data

DeepSqueak's own training corpus was never publicly released. The
[USVSEG dataset](https://zenodo.org/records/3428024) (Zenodo, gerbil/mouse/rat
recordings with hand-scored call times) is: download a species zip, extract
it, then convert it into this app's label format:

```bash
squeak-peek-cli convert-usvseg path/to/extracted_dir/
```

This writes a `<name>_labels.txt` next to each `<name>.wav`. USVSEG's CSVs
only give call start/end times, not frequency bounds — the converter derives
each call's frequency band from the real signal energy around it (see
`squeak_peek.cnn.convert_usvseg.estimate_freq_band`), not a guess.

Any (wav, label) pair in this app's own label format works too — including
GUI-annotated files and PSD/BSCD/RBD/ML detector output.

### 2. Train

```bash
squeak-peek-cli train-cnn rec1.wav rec2.wav \
    --labels rec1_labels.txt --labels rec2_labels.txt \
    --output data/models/cnn_detector_model.pt
```

`--backbone mobilenet` (default) is fast and CPU-friendly; `--backbone
resnet50` is the heavier, more accurate network DeepSqueak itself used —
wants a GPU. Point `Detection.CNN.modelPath` in your settings JSON at the
saved checkpoint to use `--detector cnn`.

## Classifying call types (Classification tab)

Detectors find *where* calls are; the **Classification** tab decides *what*
they are, using the two-stage USV pipeline ported from the standalone
`USV_Klasifikace` tool (`squeak_peek.usv_classifier`; PyTorch is part of the standard install):

1. a compact CNN looks at each call's spectrogram and decides **USV vs NOISE**,
2. a Random Forest assigns the **call type** from acoustic features, and marks
   calls it is not confident about as **UNCERTAIN** (thresholds calibrated on
   held-out recordings).

**Classify** — pick a model folder, add recordings with their detected labels
(the loaded recording, the Data Input batch folders, or files), and run.
Results show in a filterable table (double-click jumps Visualization to the
call), the loaded recording's labels get the call types for Label Edit, and
the output folder holds the tool's own files (`predictions.csv`,
`expert_review_queue.csv`, `calls_features.csv`) plus
`labels/<recording>_classified.txt`. The same model also runs from the
Detection tab as the **USV_MODEL** classifier.

**Train model** — add labeled recordings: a call-type label file (e.g. your
reference labels) and, optionally, the detector's output for the same WAV —
detections that overlap no labeled call become NOISE examples, as do calls
rejected in Label Edit. Recordings sharing animals must share a *Group*;
whole groups are split ~60/20/20 into train / calibration / test (at least 5
groups), so the reported test scores come from animals the model never saw.
Call types with too few examples are learned as plain USV. The model is
written to `<output>/run/model` (`manifest.json`, `rf.joblib`, `cnn/cnn.pt`)
and is interchangeable with the standalone tool in both directions.

Headless equivalents:

```bash
squeak-peek-cli train-classifier training.csv model_run/   # WavFile,LabelFile[,DetectedFile,GroupID,Split]
squeak-peek-cli classify-calls model_run/run/model results/ -r rec.wav rec_detected.txt
```

## Requirements

- Python 3.11+
- See `pyproject.toml` for full dependency list

## Installation

```bash
# Clone the repo
git clone <url> squeak-peek-python
cd squeak-peek-python

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
pytest --cov=squeak_peek    # with coverage report
```

## Project Structure

```
src/squeak_peek/       # Main installable package
    config.py          # Pydantic settings (mirrors settings/default.json)
    audio/             # WAV I/O and signal processing
    detectors/         # PSD, BSCD, RBD, ML, CNN detectors
    features/          # 12-D acoustic feature extraction
    labels/            # Label I/O, post-processing, metrics
    ml/                # RF training & hyperparameter optimization
    cnn/                # Faster R-CNN training, dataset conversion (optional 'cnn' extra)
    usv_classifier/    # CNN + RF call-type model (core/ = vendored USV_Klasifikace pipeline)
    gui/               # PyQt6 desktop application
cli/                   # Click-based headless CLI
tests/                 # pytest test suite
settings/              # default.json (shared with MATLAB project)
data/                  # Example WAV files and labels
```

## Reference

Original MATLAB application: Ing. Antonín Gazda, Master's Thesis, CTU Prague FEL, May 2025.

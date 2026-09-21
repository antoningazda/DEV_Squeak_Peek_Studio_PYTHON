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
    detectors/         # PSD, BSCD, RBD, ML detectors
    features/          # 12-D acoustic feature extraction
    labels/            # Label I/O, post-processing, metrics
    ml/                # RF training & hyperparameter optimization
    gui/               # PyQt6 desktop application
cli/                   # Click-based headless CLI
tests/                 # pytest test suite
settings/              # default.json (shared with MATLAB project)
data/                  # Example WAV files and labels
```

## Reference

Original MATLAB application: Bc. Antonín Gazda, Master's Thesis, CTU Prague FEE, May 2025.

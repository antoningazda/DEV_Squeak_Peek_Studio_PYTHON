# Squeak Peek Studio — Python Port

Python reimplementation of [Squeak Peek Studio](../DEV_Squeak-Peek-Studio), a MATLAB application for visualization, segmentation, and sonification of ultrasonic vocalizations (USVs) of laboratory rats.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation & Infrastructure | ✅ In progress |
| 2 | Signal Processing & Detection Engines | ⏳ Pending |
| 3 | ML Feature Extraction & Detector | ⏳ Pending |
| 4 | Label System & Evaluation | ⏳ Pending |
| 5 | GUI (PyQt6) | ⏳ Pending |
| 6 | Testing, Validation & Packaging | ⏳ Pending |

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

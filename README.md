# Squeak Peek Studio — Python Port

Python reimplementation of [Squeak Peek Studio](../DEV_Squeak-Peek-Studio), a MATLAB application for visualization, segmentation, and sonification of ultrasonic vocalizations (USVs) of laboratory rats.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation & Infrastructure | ✅ Done |
| 2 | Signal Processing & Detection Engines (PSD, BSCD, RBD) | ✅ Detectors ported (Tier 1); see `PORT_PLAN.md` |
| 3 | ML Feature Extraction & Detector | 🟡 Feature extraction ported (Tier 1); ML detector/training pending (Tier 2) — see `PORT_PLAN.md` |
| 4 | Label System & Evaluation | ✅ Label I/O, post-processing, metrics ported (Tier 1); see `PORT_PLAN.md` |
| 5 | GUI (PyQt6) | 🟡 Shell built (7 tabs, spectrogram view, label edit/export); not yet wired to Tier 1 detectors (Tier 3) |
| 6 | Testing, Validation & Packaging | 🟡 CI configured; Tier 1 has unit test coverage, ML/GUI coverage pending Tier 2–3 |

See `PORT_PLAN.md` for the detailed work breakdown for Tiers 2–3.

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

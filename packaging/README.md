# Squeak Peek Studio

A desktop app for visualizing, detecting, and labeling ultrasonic
vocalizations (USVs) of laboratory rats — a Python rewrite of the original
MATLAB thesis application.

## What it does

- Load ultrasonic (250 kHz) audio recordings and view them as spectrograms
- Automatically detect USV calls (PSD, BSCD, or RBD algorithms)
- Manually review, classify, and export call labels
- Compare detected calls against reference annotations (precision/recall/F1)

## Status

Early release. Core detection and labeling are working; ML-based detection
is not yet available.

- ✅ Signal detection (PSD, BSCD, RBD)
- ✅ Spectrogram visualization and label editing
- ✅ Detection accuracy metrics
- ⏳ ML-based detector — coming in a future release

## Installing

Download the installer for your platform from the latest release:

- **macOS** — `.dmg`
- **Windows** — `-Setup.exe`
- **Linux** — `.tar.gz`

## Origin

Python port of Squeak Peek Studio, originally built in MATLAB for
Antonín Gazda's Master's Thesis (CTU Prague FEE, 2025).

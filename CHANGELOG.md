# Changelog

All notable user-facing changes to Squeak Peek Studio. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

- Added optional pre-detection denoising: stationary background noise is
  estimated per frequency bin from the recording itself and subtracted
  before the classical detectors run (Detection → Pre-processing, or
  `--denoise` on the CLI). Detection only — export, review, sonification and
  classification keep the original audio. ML/CNN models record the
  denoising they were trained with and reapply it at inference, and
  parameter tuning fits the denoised pipeline.
- Added undo/redo for label edits (Edit menu, Ctrl+Z / Ctrl+Shift+Z),
  covering accept/reject, call-type corrections, boundary drags and manual
  labels, in both Label Edit and Visualization.
- Mouse drag/wheel now pans and zooms the spectrogram's time axis, with the
  visible range re-rendered rather than stretched; the Visualization tab
  adopts the new range as its current segment.
- Pitch trace: frames are gated on peak prominence over a local background
  and followed with a Viterbi path that penalises frequency jumps, so the
  trace no longer zigzags between parallel bands or traces pure noise.
- Rewrote the BSCD detector as a literal port of the original `bscd.m`, and
  added a `thresholdMode` parameter: `mean` (the original global threshold,
  now the default) or `adaptive` (the local noise-floor scheme, which is
  what `noiseWindow`/`localWindow`/`k`/`w` drive).
- Fixed PSD's power envelope: it is now summed straight from the power
  spectrum instead of via decibels, whose `+eps` floor biased the noise
  floor — and so the threshold — on quiet recordings. Single-frame
  detections are kept and left to Remove Short Labels, as in the MATLAB
  original.
- BSCD/RBD smoothing now matches MATLAB's `movmean`, whose window shrinks at
  the edges instead of padding.
- The call type given to a manually created label is configurable
  (Settings → Visualization → Manual label text, default `md`).
- Release builds are signed, notarized and stapled when the corresponding
  repository secrets are present, and produce the same unsigned artifacts as
  before when they are not (see `packaging/SIGNING.md`).
- Documented the above on the documentation site and added generated
  screenshots of every tab (`tools/make_docs_screenshots.py`).
- Added a Train/Tune UI for the ML and CNN detectors, a rat app icon, and a
  general UI polish pass.
- Bundled PyTorch/torchvision as core dependencies (CNN detector and
  Classification tab no longer need a separate install step).
- Published the public documentation site.
- Added the Classification tab: two-stage CNN + Random Forest call-type
  model, with its own training workflow.
- Retuned BSCD/RBD detection thresholds on a labeled batch dataset.
- Capped pitch-trace dominant-frequency picking at 100 kHz (stray
  high-frequency noise was otherwise winning the per-frame argmax and
  making the trace jump wildly).
- Added an MIT license.

## [0.0.8] — 2026-09-22

- Added video import, auto-sync to the loaded recording, sonified playback,
  and labeled-video export.
- Added the CNN (Faster R-CNN) detector, its training pipeline, and a
  USVSEG-dataset converter.
- Exposed the CNN detector and an opt-in tonality post-processing filter
  (drops broadband false positives like cage knocks) in the GUI.
- Rebuilt sonification around a tape-speed pitch/time shift.
- Composited a scrolling labeled spectrogram panel into exported videos.
- Fixed several spectrogram/label-overlay rendering bugs (invisible
  reference labels, invisible outlines, missing pitch trace on unlabeled
  calls) and a PyInstaller build issue that dropped detector/classifier
  plugins from packaged builds.

## [0.0.7] — 2026-09-22

- Completed the MATLAB GUI parity pass: keyboard shortcuts, startup
  auto-load, draggable label boundaries + manual label creation, spectrogram
  colormap and label-color selection, pitch-tracking overlay, batch mode,
  multi-detector/multi-post-processing run pipeline, remaining detector
  settings tabs (BSCD/RBD/ML/Label Edit/Data Input), and the Info tab
  (links, logos, Easter egg).
- Reworked the Label Edit tab around independent detection/classification
  accept-reject state (rejecting no longer deletes a label).
- Added the design-token light/dark theme system.
- Made detectors and classifiers self-registering plugins.
- Made `squeak_peek.__version__` the single source of truth for the app
  version.

## [0.0.6] — 2026-09-21

- Added the ML (Random Forest) detector, with training and probability
  calibration.

## [0.0.5] — 2026-09-21

- Internal: Tier 1/3 port-status documentation updates; no user-facing
  changes.

## [0.0.4] — 2026-09-21

- Releases are now published to a separate public repository.

## [0.0.3] — 2026-09-21

- Added a Windows installer (Inno Setup) to the release build.

## [0.0.2] — 2026-09-21

- Added a macOS `.dmg` installer (replacing a raw `.zip`).

## [0.0.1] — 2026-09-21

- First packaged release: PSD/BSCD/RBD detectors, acoustic feature
  extraction, label I/O, spectrogram visualization and label editing,
  detection accuracy metrics (precision/recall/F1 vs. reference labels).

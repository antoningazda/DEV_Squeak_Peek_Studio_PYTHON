# Squeak Peek Studio

A desktop app for visualizing, detecting, and labeling ultrasonic
vocalizations (USVs) of laboratory rats — a Python rewrite of the original
MATLAB thesis application.

📖 **[Full documentation](https://antoningazda.github.io/squeak-peek-studio-releases/)**

## What it does

- Load ultrasonic (250 kHz) audio recordings and view them as spectrograms,
  single-file or batch (whole folder)
- Automatically detect USV calls (PSD, BSCD, RBD, a trainable Random Forest
  detector, or a trainable CNN object detector), optionally running several
  detectors/post-processing steps in one pass
- Classify call types with a two-stage CNN + Random Forest model (also
  trainable from your own labeled recordings)
- Manually review, drag-resize, and label-edit calls, with keyboard shortcuts
  and independent accept/reject state for detection vs. classification
- Sonify segments (pitch-shift/time-stretch) for audible playback of
  ultrasonic calls, with video import/sync
- Compare detected calls against reference annotations (precision/recall/F1)
- Export labeled/annotated video clips

## Status

Actively developed. Full feature set, documentation and release notes are
kept at the documentation site linked above — this file only covers
installing and opening the downloaded build.

## Installing

Download the installer for your platform from the latest release:

- **macOS** — `.dmg`
- **Windows** — `-Setup.exe`
- **Linux** — `.tar.gz`

## Opening it the first time

These builds are not yet signed with a paid developer certificate, so both
macOS and Windows show a warning the first time you open the app. Nothing is
wrong with the download — the warning is about the missing certificate, not
about the file.

**macOS** — if you see *"Squeak Peek Studio is damaged and can't be opened"*
or *"unidentified developer"*:

1. Drag the app into your **Applications** folder.
2. **Right-click** (or Control-click) it and choose **Open**.
3. Click **Open** in the dialog that appears.

You only need to do this once per installed version; double-clicking works
normally afterwards. If the dialog gives you no Open button, run this once in
Terminal:

```bash
xattr -dr com.apple.quarantine "/Applications/Squeak Peek Studio.app"
```

**Windows** — if you see *"Windows protected your PC"*, click **More info**,
then **Run anyway**.

**Linux** — no warning; extract the archive and run `SqueakPeekStudio`.

## Origin

Python port of Squeak Peek Studio, originally built in MATLAB for
Antonín Gazda's Master's Thesis (CTU Prague FEL, 2025).

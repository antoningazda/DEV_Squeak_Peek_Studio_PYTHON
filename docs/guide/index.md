# User guide

The application is organised as a row of tabs, roughly in the order you work
through them. State is shared: load a recording once in **Data Input** and
every other tab sees it; run a detector and the resulting labels appear
immediately in **Visualization**, **Label Edit** and **Metrics**.

| Tab | What it is for |
|---|---|
| [Data Input](data-input.md) | Choose the recording(s) and label files to work on |
| [Visualization](visualization.md) | Scroll the spectrogram, overlay labels, sonify |
| [Detection](detection.md) | Run one or more detectors with post-processing |
| [Classification](classification.md) | Assign call types with a trained model; train one |
| [Label Edit](label-edit.md) | Accept, reject and correct calls one at a time |
| [Video](video.md) | Sync a behaviour video and export it with sonified audio |
| [Metrics](metrics.md) | Score detections against reference labels |
| [Settings](settings.md) | Every parameter, saved to a JSON file you can share |
| Info | Version, links, credits |

## The shared state

Three things are global to the session and set in **Data Input**:

- **the WAV recording** — its samples, sample rate and duration;
- **detected labels** — the working set, overwritten whenever you run a
  detector or classifier, and edited in Label Edit;
- **reference labels** — ground truth, read-only, used only for scoring.

Anything that changes one of these updates every tab at once. That is why
you can run a detector, flip to Visualization to see the boxes land on the
spectrogram, and flip to Metrics to score them, without re-loading anything.

## Two modes

**Single file** mode works on one recording. Everything is interactive.

**Batch folder** mode works on a folder of recordings, matching label files
by filename. Detection and Classification will process every recording in
turn; Visualization and Label Edit still show the one recording you have
loaded. See [Data Input](data-input.md#batch-folder-mode).

## Keyboard

Review work is keyboard-driven. See
[Keyboard shortcuts](shortcuts.md) — every binding is rebindable in
Settings.

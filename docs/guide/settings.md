# Settings

Every parameter in the application, in one place, saved to a JSON file you
can version, share with a collaborator, or attach to a paper as your
analysis protocol.

The tab is a set of sections, listed on the left. The detector and classifier sections are
**generated automatically** from the registered plugins — add a detector to
the codebase and its parameter form appears here with no extra work.

| Sub-tab | Covers |
|---|---|
| Data Input | Default files/folders auto-loaded at startup, single-vs-batch mode |
| Visualization | Spectrogram rendering, overlays and sonification |
| PSD / BSCD / RBD / ML / CNN detector | One sub-tab per detector, with its own parameters |
| Post-processing | Merging and filtering rules applied after any detector |
| USV model / Duration classifier | One sub-tab per classifier |
| Label Edit | Spectrogram rendering and the call-type list |
| Video | Finger-snap sync-click detection |
| Appearance | Colour scheme |
| [Shortcuts](shortcuts.md) | View and customise keyboard shortcuts |

**Apply** pushes the edited values to the running app. **Save settings…**
writes them to JSON; **Load settings…** replaces everything from a file.

!!! tip "One settings file per experiment"

    Save a settings file alongside each dataset. It records the exact
    detector parameters a result came from, which is the difference between
    a reproducible analysis and a number you cannot explain six months
    later.

---

## Data Input

Defaults loaded at startup, so you do not re-pick the same paths every day.

| Setting | Key | Default |
|---|---|---|
| Default USV (single file) | `DataInput.DefaultUSVFieldSingle` | example recording |
| Default labels (single file) | `DataInput.DefaultLabelFieldSingle` | example labels |
| Default reference labels (single file) | `DataInput.DefaultReferenceLabelFieldSingle` | example reference |
| Batch mode | `DataInput.BatchMode` | `false` |
| Default USV folder (batch) | `DataInput.DefaultUSVFieldBatch` | — |
| Default labels folder (batch) | `DataInput.DefaultLabelFieldBatch` | — |
| Default reference labels folder (batch) | `DataInput.DefaultReferenceLabelFieldBatch` | — |

## Visualization

| Setting | Key | Default | Notes |
|---|---|---|---|
| Show loading dialog | `ShowLoadingDialog` | `false` | Progress dialog while a large WAV loads |
| Initial segment start | `SegmentStartSeconds` | `0.0` | Jumped to whenever a WAV is loaded |
| Segment length | `SegmentLengthSeconds` | `1.0` | Shorter zooms in; longer gives context |
| Spectrogram window | `SpectrogramWindow` | `1024` | Larger windows sharpen frequency, blur time |
| Spectrogram overlap | `SpectrogramOverlap` | `512` | Must be smaller than the window. Higher = smoother, slower |
| Min / Max display frequency | `SpectrogramMinFrequency` / `Max` | `40` / `120` kHz | **Display only — does not affect detection** |
| Colormap | `SpectrogramColormap` | `parula` | 15 options |
| Show detected / reference labels | `ShowLabels` / `ShowReferenceLabels` | `true` | Default overlay state |
| Label / reference label colour | `LabelColor` / `ReferenceLabelColor` | `cyan` / `white` | |
| Manual label length | `ManualLabelLength` | `0.075` s | Duration of a right-click-created label |
| Sonification semitones | `SonificationST` | `-35` | More negative shifts further down. −36 is exactly three octaves |
| Sonification slowdown factor | `SonificationSlowdown` | `4` | Only used when playback speed is set to use it |
| Sonification playback speed | — | Match the pitch shift | See below |
| Sonification denoising | — | on | Spectral gate on the noise floor |

!!! note "Leave playback speed on “Match the pitch shift”"

    Matching makes sonification a pure tape-speed transform: nothing is
    resynthesised, so there are no stretching artefacts at all. Choosing an
    independent slowdown factor costs one phase-vocoder pass and sounds
    slightly less clean.

**Denoising** matters more than it sounds. An ultrasonic recording is mostly
broadband noise; shifted down, that becomes a wall of hiss that buries the
calls. The spectral gate leaves calls standing out of a quiet background.
Turn it off to hear the raw signal.

## Post-processing

| Setting | Key | Default | Used by |
|---|---|---|---|
| Max gap to merge | `Detection.POST.maxGapToMerge` | `0.005` s | Merge Close Labels |
| Min label length | `Detection.POST.minLabelLength` | `0.001` s | Remove Short Labels |
| Min tonality | `Detection.POST.minTonality` | `0.0` (off) | Filter Broadband |

Set **Min tonality** to `0.5` to enable broadband filtering — see
[Tonality filtering](../methods.md#tonality-filtering).

## Detector sub-tabs

Each detector has its own parameter page. Every detector shares a frequency
band (`fcutMin` / `fcutMax`, default 40–120 kHz) — unlike the Visualization
frequency range, **these do affect detection**. The rest is
detector-specific and documented in [Methods](../methods.md).

| Detector | Key prefix | Tuning guide |
|---|---|---|
| PSD | `Detection.PSD.*` | [Methods → PSD](../methods.md#psd) |
| BSCD | `Detection.BSCD.*` | [Methods → BSCD](../methods.md#bscd) |
| RBD | `Detection.RBD.*` | [Methods → RBD](../methods.md#rbd) |
| ML | `Detection.ML.*` | [Methods → ML](../methods.md#ml-random-forest) |
| CNN | `Detection.CNN.*` | [Methods → CNN](../methods.md#cnn-faster-r-cnn) |

The ML and CNN pages are where you set the *Model file* — the trained model
each one needs. **Detection → Train detector → Use for detection** fills it
in for you (and, for tuning, writes the tuned values into these pages).

## Label Edit

| Setting | Key | Default |
|---|---|---|
| Spectrogram window | `LabelEdit.SpectrogramWindow` | `1024` |
| Spectrogram overlap | `LabelEdit.SpectrogramOverlap` | `512` |
| Min / Max display frequency | `LabelEdit.SpectrogramMinFrequency` / `Max` | `40` / `120` kHz |
| Colormap | `LabelEdit.SpectrogramColormap` | `invgray` |
| Classifications | `LabelEdit.Classifications` | `d,sk,5,5t,5w,c5` |

**Classifications** is the call-type dropdown in
[Label Edit](label-edit.md). Edit it freely to match your taxonomy; just
keep the values comma-separated.

## Video

Finger-snap sync detection, used when no `sk` label is available.

| Setting | Meaning |
|---|---|
| Snap band min / max | The band searched for the snap transient in the *video's* audio track |
| Snap threshold factor | How many times the noise-floor energy a window must exceed to be flagged as the snap |

See [Video](video.md#how-the-ultrasonic-snap-is-found).

## Appearance

The colour scheme. Stored by the OS rather than in the JSON file, so it
follows you across settings files.

## Export folder

| Setting | Key | Default |
|---|---|---|
| Default export folder | `Detection.ExportPath` | `data/export` |

Leave blank to default to the loaded WAV's own folder.

---

→ [The settings file format](../file-formats.md#settings-json)

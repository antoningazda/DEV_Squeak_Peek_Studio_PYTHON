# Detection

Find the calls. Pick one or more detectors, optionally attach a call-type
classifier, choose post-processing, and run — all in the **Run detectors**
sub-tab. The **Train detector** sub-tab
[tunes or trains detectors](#train-detector) on your own labeled recordings.

<figure markdown>
  ![The Detection tab's Run detectors sub-tab](../assets/screenshots/detection.png#only-light){ .spk-shot }
  ![The Detection tab's Run detectors sub-tab](../assets/screenshots/detection-dark.png#only-dark){ .spk-shot }
  <figcaption>Run detectors: detectors on the left, optional classifiers and the
  pre/post-processing chain on the right, export folder below.</figcaption>
</figure>

## Detectors

Tick one or more. **Each selected detector runs independently and exports
its own label file**, which makes it easy to compare two algorithms on the
same recording in a single pass.

| Detector | One-line summary |
|---|---|
| **PSD** | Power Spectral Density — thresholds band power against an adaptive noise floor. Fast and robust on clear, high-SNR recordings. |
| **BSCD** | Bayesian Sequential Change Detection — flags points where the signal's statistics shift abruptly. Good for call onsets/offsets in noisier audio. |
| **RBD** | Relative Bayesian Difference — compares autoregressive models on either side of a candidate boundary. More precise boundaries, more compute per call. |
| **PITCH** | Pitch trace — marks every stretch where a coherent frequency contour stands out from the background, i.e. exactly where [Visualization](visualization.md) draws its orange trace. Ignores broadband knocks and rustle; misses calls too faint for the contour to lock on. |
| **ML** | Random Forest sliding-window classifier over acoustic features. Needs a trained model file. |
| **CNN** | Faster R-CNN predicting a time/frequency box per call. Needs a trained checkpoint. |

The list is alphabetical and the first entry, **BSCD**, is ticked when the
app starts. Start with **PSD** if your recordings are clean; if they are
noisy or PSD keeps clipping call onsets, stay with **BSCD**. See [Methods](../methods.md) for what each one
actually computes, how to tune it, and when to prefer it.

!!! tip "PITCH only reports what you can already see"

    Because it reports exactly the stretches Visualization traces in orange,
    you can judge it before you run it: open a recording, look at the trace,
    and that is the detector's output. It is also the only detector that
    writes a real frequency per call, which is what its **Min/Max call
    frequency** filter acts on — see
    [Settings](settings.md#detector-sub-tabs).

!!! note "ML and CNN need a model first"

    Both are empty shells until you point them at a trained model
    (Settings → ML detector / CNN detector). See
    [Training models](../training.md).

## Classifiers (optional)

Leave this empty and each detector simply exports its detections untyped.

Select a classifier and it runs on **every** selected detector's
post-processed output, producing one exported file per
detector × classifier pair.

| Classifier | What it does |
|---|---|
| **USV_MODEL** | The trained two-stage model: a CNN separates real USVs from noise, then a Random Forest assigns the call type, flagging low-confidence calls as `UNCERTAIN`. Pick or train the model in the [Classification tab](classification.md). |
| **DURATION** | A baseline example that assigns a type purely from event duration. Not a validated model — it exists as a template for writing your own. |

A classifier never invents new events. It only fills in the call type of
events a detector already found.

## Pre-processing

One checkbox per classical detector: **denoise the audio before that
detector runs**. Each box is the detector's own `denoise` setting
([Settings](settings.md#detector-sub-tabs) → *detector* → Denoise), so the
two always agree.

| Detector | Default | Why |
|---|---|---|
| **PSD** | **on** | On raw recordings the in-band noise floor is most of PSD's envelope; denoising roughly doubles its precision |
| **BSCD** | off | Its mean threshold is fitted to raw audio; denoising adds false positives |
| **RBD** | off | Its own bandpass already does the work |
| **PITCH** | off | It references every frequency bin to its own median, which removes stationary noise lines by itself |

Denoising estimates each frequency bin's background level from the
recording itself and subtracts it (Settings →
[Pre-processing](settings.md#pre-processing) holds the shared algorithm
settings). See [Pre-detection denoising](../methods.md#pre-detection-denoising)
for the algorithm and the measurements behind the defaults.

Three things worth knowing:

- **ML** and **CNN** have no checkbox: they apply whatever denoising their
  own model was trained with, so training and inference always see the
  same kind of audio.
- It is for **detection only**. Export, [Label Edit](label-edit.md),
  sonification and [classification](classification.md) all keep working on
  the original recording — denoising never ends up in anything you listen
  to, look at or ship.
- Changing a detector's box changes what its thresholds are fitted to;
  score the change on a recording you have reference labels for (or
  re-tune) before trusting it.

The denoised audio is computed once per recording and reused by every
selected detector that has its box checked.

!!! tip "From the command line"

    `squeak-peek-cli detect recording.wav -d bscd --denoise` (also on
    `batch`) overrides the chosen detector's setting; `--no-denoise` turns
    it off. See the [CLI reference](../cli.md#detect).

## Post-processing

Applied to every detector's output before export, always in this order:

```
Filter Broadband → Merge Close Labels → Remove Short Labels
```

Leave all three unchecked to export detections exactly as the detector
produced them.

| Step | Default | What it does |
|---|---|---|
| **Filter Broadband** | off | Discard detections whose energy is spread across the band instead of concentrated in a narrowband whistle — drops cage knocks and rustle |
| **Merge Close Labels** | **on** | Merge detections separated by a gap smaller than *Max gap to merge* |
| **Remove Short Labels** | **on** | Discard detections shorter than *Min label length* |

Thresholds for both stages live in Settings —
[Pre-processing](settings.md#pre-processing) and
[Post-processing](settings.md#post-processing):

| Threshold | Default | Used by |
|---|---|---|
| `Min tonality` | `0` (off) | Filter Broadband |
| `Max gap to merge` | `0.005` s | Merge Close Labels |
| `Min label length` | `0.001` s | Remove Short Labels |

!!! tip "Filter Broadband is the single biggest precision win"

    It is off by default because it trades recall for precision. On
    benchmark recordings, turning it on with `Min tonality = 0.5` took PSD
    from F1 0.56 → 0.78 and BSCD from 0.71 → 0.83, almost entirely by
    killing false positives. See
    [Tonality filtering](../methods.md#tonality-filtering) for the full
    numbers and the caveats.

    It runs *before* merging, so a merged detection that spans the gap
    between two real calls is not penalised, and detections too short to
    score are always kept.

## Export folder

Where the label files go. Leave it empty to write next to the WAV file.
The default comes from `Detection.ExportPath` in
[Settings](settings.md#export-folder).

Filenames are:

```
<recording>_<DETECTOR>_<timestamp>_detected.txt
<recording>_<DETECTOR>_<CLASSIFIER>_<timestamp>_detected.txt
```

The timestamp means a re-run never silently overwrites an earlier result —
you always have the previous attempt to compare against.

## Running

Click **Run detectors**. A progress dialog appears; long recordings with
RBD or CNN can take a while.

In **single file** mode the result is written *and* loaded as the current
detected labels, so it appears immediately in
[Visualization](visualization.md) and [Label Edit](label-edit.md).

In **batch folder** mode every WAV in the folder is processed in turn, one
file per recording per detector (per classifier).

## Train detector

<figure markdown>
  ![The Detection tab's Train detector sub-tab](../assets/screenshots/detection-train.png#only-light){ .spk-shot }
  ![The Detection tab's Train detector sub-tab](../assets/screenshots/detection-train-dark.png#only-dark){ .spk-shot }
  <figcaption>Train detector, with the loaded recording added as a training pair.</figcaption>
</figure>

Fit a detector to *your* recordings. Add labeled recordings with **Add
loaded** (the Data Input single file + its reference labels), **Add batch**
(the Data Input batch folders, paired by filename) or **Add files…**, then
choose what to do:

| Mode | What it does | Result |
|---|---|---|
| **Tune detector parameters** | Searches the numeric parameters of any detector (PSD, BSCD, RBD, or the ML/CNN sensitivity) for the best F1 against your labels. Check which parameters to tune and adjust their search ranges; the current values are always tried first, so tuning never makes things worse. | Best parameters, applied to Settings with **Use for detection** |
| **Train ML model** | Trains the Random Forest frame classifier. Optionally hold one recording out to calibrate the sensitivity (and noise ratio). | `.joblib` model file |
| **Train CNN model** | Trains the Faster R-CNN box detector (slow without a GPU). | `.pt` checkpoint |

Labels must mark **every** call in the analysed audio — unlabeled calls
count as false positives. Detections rejected in [Label Edit](label-edit.md)
are ignored. Tuning scores detections after the post-processing checked in
Run detectors, and by default only analyses the first 30 s of each
recording to keep each trial fast — make sure that part contains calls.

**Use for detection** writes the tuned parameters (or the new model path and
calibrated sensitivity) into the detector's Settings and checks it in Run
detectors. Save your settings to keep them.

!!! note "Denoising in training and tuning"

    - **Tuning** follows the tuned detector's own `denoise` setting: if it
      is on, each recording is denoised once and every trial is scored on
      that audio, so the parameters fit how you will actually run it.
    - **Training** an ML or CNN model denoises the training audio only when
      **Denoise the training audio** (under *Output*) is ticked, and records
      that in the model file; the detector reapplies exactly that at
      inference.

---

**Next:** [Classification →](classification.md) ·
[How the detectors work →](../methods.md)

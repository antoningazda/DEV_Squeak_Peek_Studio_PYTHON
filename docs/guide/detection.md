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
| **ML** | Random Forest sliding-window classifier over acoustic features. Needs a trained model file. |
| **CNN** | Faster R-CNN predicting a time/frequency box per call. Needs a trained checkpoint. |

The list is alphabetical and the first entry, **BSCD**, is ticked when the
app starts. Start with **PSD** if your recordings are clean; if they are
noisy or PSD keeps clipping call onsets, stay with **BSCD**. See [Methods](../methods.md) for what each one
actually computes, how to tune it, and when to prefer it.

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

One checkbox, applied to the **audio** before any detector sees it:

| Step | Default | What it does |
|---|---|---|
| **Denoise (suppress stationary background noise)** | off | Estimates each frequency bin's background level from the recording itself and subtracts it, so a detector's envelope is driven by the calls rather than by the room |

An ultrasonic recording is mostly background — fans, electronics,
microphone hiss — that barely changes over a session, while calls are
sparse and brief. Energy-based detectors see that constant in-band noise as
a large part of their envelope, which costs precision. See
[Pre-detection denoising](../methods.md#pre-detection-denoising) for the
algorithm and its parameters.

Three things worth knowing:

- It applies to the **classical detectors** (PSD, BSCD, RBD). **ML** and
  **CNN** ignore it and apply whatever denoising their own model was
  trained with, so training and inference always see the same kind of
  audio.
- It is for **detection only**. Export, [Label Edit](label-edit.md),
  sonification and [classification](classification.md) all keep working on
  the original recording — denoising never ends up in anything you listen
  to, look at or ship.
- Thresholds tuned on raw audio **may need retuning** after you turn it on;
  the envelope it feeds the detector is quieter and flatter. Score the
  change on a recording you have reference labels for before trusting it.

The denoised audio is computed once per recording and reused by every
selected classical detector in the same run.

!!! tip "From the command line"

    `squeak-peek-cli detect recording.wav --denoise` (also on `batch`).
    See the [CLI reference](../cli.md#detect).

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

!!! note "Denoising follows into training and tuning"

    Whatever **Pre-processing → Denoise** is set to in Run detectors is
    what this sub-tab uses:

    - **Tuning** denoises each recording once and searches parameters
      against that audio, so the values you get are the values that fit how
      you will actually run the detector.
    - **Training** an ML or CNN model records the denoising in the model
      file, and the detector reapplies exactly that at inference — which is
      why ML and CNN ignore the checkbox at detection time.

    Change the checkbox and the result is a different model, or a different
    set of tuned parameters. Tune or train with it set the way you will run.

---

**Next:** [Classification →](classification.md) ·
[How the detectors work →](../methods.md)

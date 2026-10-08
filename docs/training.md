# Training models

Three trainable models, for two different jobs — plus parameter tuning for
the rule-based detectors.

| Model | Job | Where to train it |
|---|---|---|
| [Detector parameter tuning](#tuning-detector-parameters) | Fit PSD / BSCD / RBD to your recordings | GUI (Detection → Train detector) |
| [ML detector](#ml-detector-random-forest) | Find calls | GUI (Detection → Train detector) or CLI |
| [CNN detector](#cnn-detector-faster-r-cnn) | Find calls, with frequency extent | GUI (Detection → Train detector) or CLI |
| [Call-type classifier](#call-type-classifier) | Say what kind of call each one is | GUI or CLI |

!!! tip "Train on your own recordings"

    Microphone, cage, room acoustics and strain all leave fingerprints in
    the data. A model trained on fifty of your own recordings will
    outperform any general-purpose model on your setup — often by a lot.
    This is the main reason to train at all.

---

## Getting labelled data

Training needs recordings with known call positions. Three ways to get
them:

### Your own reference labels

The best option. If you have hand-scored recordings, you already have a
training set. Use them directly.

### Bootstrap from a detector

Run [PSD or BSCD](methods.md) with
[tonality filtering](methods.md#tonality-filtering) for high precision,
then curate the output in [Label Edit](guide/label-edit.md). Reviewing an
80%-correct detector output is much faster than labelling from scratch, and
the rejected detections become NOISE training examples for free.

### The USVSEG dataset

DeepSqueak's own training corpus was never publicly released. The
[USVSEG dataset](https://zenodo.org/records/3428024) on Zenodo is — gerbil,
mouse and rat recordings with hand-scored call times.

```bash
# download and extract a species zip from Zenodo, then:
squeak-peek-cli convert-usvseg path/to/extracted_dir/
```

This converts the matching `<name>.wav` / `<name>.csv` pairs into this
app's [label format](file-formats.md#label-files), ready for `train` or
`train-cnn`.

!!! warning "Species matters"

    Mouse USVs are not rat USVs: different frequency ranges, different
    durations, different repertoires. A model trained on USVSEG mouse data
    is a reasonable starting point for mouse work and a poor one for rats.

---

## Tuning detector parameters

PSD, BSCD and RBD have no model, but their thresholds and window lengths
decide how well they work on your setup. **Detection → Train detector →
Tune detector parameters** searches them for the best F1 against your
reference labels (the successor of the MATLAB app's Bayesian PSD
optimisation, now for every detector).

- Trial 1 is your current settings, so the result is never worse.
- About a third of the trials sample the search box at random; the rest
  refine around the best point found so far.
- F1 is pooled over all listed recordings, after the post-processing
  checked in Run detectors.
- Each trial runs the detector on every recording; limit **Analyse first**
  (default 30 s) for speed and add more recordings rather than longer ones.

Tuned on few calls, the parameters can overfit — check the result on a
recording you did not tune on (Metrics tab).

---

## ML detector (Random Forest)

A frame-wise Random Forest over
[12 acoustic features](methods.md#ml-random-forest). Fast to train, runs on
CPU, no extra dependencies.

In the GUI: **Detection → Train detector → Train ML model**. Pick a
recording under *Calibrate on* to choose the best `sensitivity` on it
(held out of training when you list two or more recordings). From the
command line:

```bash
squeak-peek-cli train rec1.wav rec2.wav rec3.wav \
  -l labels1.txt -l labels2.txt -l labels3.txt \
  -o models/colony_a.joblib \
  --noise-ratio 3.0 \
  --n-trees 200
```

Label files must be given in the **same order and count** as the WAV files.

| Parameter | Default | Effect |
|---|---|---|
| `--noise-ratio` | `3.0` | Noise:USV frame ratio for class balancing. Calls are a tiny fraction of any recording; without balancing the model learns to say "noise" always |
| `--n-trees` | `200` | More trees = marginally better, linearly slower. Rarely worth going above 500 |

### Using it

1. [Settings → ML detector](guide/settings.md#detector-sub-tabs) → set
   `modelPath` to the `.joblib` file.
2. Select **ML** in the [Detection tab](guide/detection.md), or pass
   `--detector ml` to the CLI.
3. Tune `sensitivity` (default `0.42`) — the frame-probability cutoff. This
   is a precision/recall dial that needs **no retraining**.

### Hyperparameter optimisation

`squeak_peek.ml.optimize` wraps Optuna for automated tuning. See the
[source repository](https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON).

---

## CNN detector (Faster R-CNN)

Predicts a full time/frequency box per call rather than classifying frames.
PyTorch comes with the standard install, so there is nothing extra to set
up. Train it in **Detection → Train detector → Train CNN model**, or:

```bash
squeak-peek-cli train-cnn rec1.wav rec2.wav \
  -l labels1.txt -l labels2.txt \
  -o models/cnn_colony_a.pt \
  --epochs 10 \
  --backbone mobilenet
```

| Parameter | Default | Notes |
|---|---|---|
| `--epochs` | `10` | |
| `--batch-size` | `4` | Raise it if you have GPU memory to spare |
| `--lr` | `0.0001` | |
| `--window-s` | `1.0` | Spectrogram tile length in seconds |
| `--hop-s` | `0.5` | Tile hop. Smaller = more training tiles from the same data |
| `--backbone` | `mobilenet` | `mobilenet` is fast and CPU-friendly; `resnet50` is more accurate and wants a GPU |

**Start with `mobilenet`.** It trains in a reasonable time on a laptop. Move
to `resnet50` only if you have a GPU and mobilenet has plateaued.

### Using it

Set `Detection.CNN.modelPath` to the `.pt` checkpoint, then select **CNN**
in the Detection tab or pass `--detector cnn`. `sensitivity` (default `0.5`)
is the box-score cutoff.

---

## Call-type classifier

The two-stage model: a CNN for USV-vs-NOISE, then a Random Forest for the
call type, with calibrated confidence thresholds that flag uncertain calls
for review.

This one has a **full GUI workflow** —
[Classification → Train model](guide/classification.md#train-model) — which
is the recommended path, because the data-checking step catches mistakes
that are expensive to discover after an hour of training.

### The data you need

Per recording:

- the **WAV**;
- a **call-type label file** — each real call with its type;
- optionally the **detector's output** on the same WAV, which supplies
  NOISE examples;
- a **Group**.

### Groups are the important part

**Recordings of the same animal must share a group.** Whole groups go to
one split — roughly 60% train, 20% calibration, 20% test — which needs at
least five groups.

Without this, two recordings of the same rat can land in train and test,
the model memorises that individual's voice, and your test score measures
nothing. Group-wise splitting is the difference between a number you can
report and a number that is wrong.

### CLI

```bash
squeak-peek-cli train-classifier training.csv model_run/
```

With `training.csv`:

```csv
WavFile,LabelFile,DetectedFile,GroupID,Split
rec01.wav,labels/rec01.txt,detected/rec01.txt,A01,auto
rec02.wav,labels/rec02.txt,detected/rec02.txt,A01,auto
rec03.wav,labels/rec03.txt,detected/rec03.txt,A02,auto
rec04.wav,labels/rec04.txt,,A03,auto
```

Paths are relative to the CSV. → [Full format](file-formats.md#training-csv)

| Parameter | Default | Notes |
|---|---|---|
| `--epochs` | `12` | CNN epochs, fixed — no early stopping |
| `--trees` | `200` | Random Forest size |
| `--seed` | `7` | Same seed, same model |
| `--min-examples` | `10` | Below this, a call type is learned as plain USV |
| `--generic-labels` | `d` | Labels meaning "a USV with no type" |
| `--device` | `cpu` | `mps` on Apple Silicon |

### Reading the results

The run reports, on the **held-out test groups**:

| Metric | What it tells you |
|---|---|
| Accuracy | Fraction correct. Misleading when call types are imbalanced |
| Balanced accuracy | Averaged over classes |
| **Macro-F1** | The number to watch — weights rare types as heavily as common ones |
| Macro-F1 95% CI | Bootstrapped over test groups. A wide interval means too few groups |
| Review fraction | How many calls the model sends to you as UNCERTAIN |

A model with good accuracy and poor macro-F1 has learned your two commonest
call types and given up on the rest.

!!! warning "A test score is not a biological validation"

    These numbers say how well the model reproduces **your labelling** on
    animals it never heard. They say nothing about whether your call-type
    taxonomy is the right one for the question you are asking.

### Output

```
model_run/run/model/           manifest.json, rf.joblib, cnn/cnn.pt
model_run/run/evaluation/      test metrics
model_run/run/classification/  the run's own predictions
model_run/preparation/         clustering output for review
```

Point the [Classify](guide/classification.md#classify) sub-tab or
`classify-calls` at `model_run/run/model`.

Model folders are interchangeable in both directions with the standalone
`USV_Klasifikace` tool.

---

## Practical advice

**Check your data before training.** The GUI's **Check data** button counts
examples per label and split from the label files alone — no audio, so it
takes seconds. Run it every time.

**Watch for the NOISE trap.** Supply a detected-labels file only when the
call-type file is *complete*. NOISE examples are built from detections that
match no labelled call, so a real call missing from your labels gets taught
to the model as noise.

**More groups beats more recordings.** Five recordings from five animals
tell you more about generalisation than fifty from one.

**Rare call types need examples or they vanish.** Types below
`--min-examples` are folded into plain USV. If a rare type matters to you,
collect more of it rather than lowering the threshold — a type learned from
four examples is noise with a name.

**Keep the seed.** Same seed, same model. Record it with your results.

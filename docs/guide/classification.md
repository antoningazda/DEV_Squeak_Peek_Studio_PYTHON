# Classification

A detector tells you *where* the calls are. The classifier tells you *what
kind* they are — and which ones are not calls at all.

The tab has two sub-tabs: **Classify** (run a trained model) and
**Train model** (make one from your own labelled recordings).

<figure markdown>
  ![The Classification tab](../assets/screenshots/classification.png#only-light){ .spk-shot }
  ![The Classification tab](../assets/screenshots/classification-dark.png#only-dark){ .spk-shot }
  <figcaption>The Classify sub-tab: model, the recordings to run it over, options, and the
  results table that fills in as calls are typed.</figcaption>
</figure>

## How the model works

Two stages, run in order on each detected call:

1. **A CNN decides USV vs NOISE.** It looks at the call's spectrogram patch.
   This is what removes the detector's false positives — cage knocks,
   rustle, scratching.
2. **A Random Forest assigns the call type** from acoustic features. Where
   its confidence falls below a threshold calibrated on held-out
   recordings, the call is marked **`UNCERTAIN`** instead of being given a
   type it does not deserve.

The result is three kinds of outcome per call: a confident call type, a
`NOISE` verdict, or `UNCERTAIN` — which is a request for you to look at it.

This pipeline is ported from the standalone `USV_Klasifikace` tool, and
models are interchangeable between the two in both directions.

!!! warning "Test scores are not a biological validation"

    Reported accuracy tells you how well the model reproduces *your
    labelling* on animals it never saw. It says nothing about whether the
    call-type taxonomy itself is the right one for your question.

---

## Classify

### 1. Pick a model

Under **Model**, browse to a model folder — a directory containing
`manifest.json`, produced by **Train model** (at `<output>/run/model`) or by
the standalone tool.

The tab shows a summary of the model and stores it, which also makes it
available as the **USV_MODEL** classifier in the
[Detection tab](detection.md#classifiers-optional).

### 2. Add recordings

Each row is a WAV plus the detected-label file whose segments get
classified. Three ways to add rows:

| Button | Adds |
|---|---|
| **Add loaded** | The recording currently loaded in Data Input, with its current detected labels |
| **Add batch** | Every WAV in the Data Input batch folder, paired by filename with the batch detected-labels folder |
| **Add files…** | Pick a WAV and its detected-label file by hand |

You can remove selected rows or clear the list.

### 3. Options

| Option | Default | Effect |
|---|---|---|
| **Output folder** | New folder next to the first WAV | Results go here. **It must not exist yet** — results are never overwritten |
| **Leave NOISE calls out of the classified labels** | off | Drop rejected calls from the exported label files. `predictions.csv` always keeps every call either way |
| **Label UNCERTAIN calls with the best guess** | off | Write e.g. `5t?` instead of the plain word `UNCERTAIN` |
| **Show results on the loaded recording** | on | Replace the loaded recording's detected labels with the classified ones, so the predicted types are selectable in [Label Edit](label-edit.md) |

### 4. Run and read the results

Click **Classify**. A progress bar runs; **Cancel** stops after the current
recording.

The results table is filterable by prediction — **All calls**, **NOISE**,
**UNCERTAIN**, **Needs review** — and sortable. **Double-click a call of
the loaded recording to jump straight to it in
[Visualization](visualization.md).**

The output folder contains:

| File | Contents |
|---|---|
| `predictions.csv` | Every call, with its scores |
| `expert_review_queue.csv` | The uncertain calls plus an audit sample — your review worklist |
| `calls_features.csv` | The extracted acoustic features |
| `labels/<recording>_classified.txt` | A label file per recording, ready to load anywhere in the app |

**Open results folder** opens it in your file browser.

---

## Train model

A general-purpose model will never beat one trained on your own colony,
microphones and room. This sub-tab trains one.

### 1. Add labelled recordings

Each row needs:

| Column | Required | What it is |
|---|---|---|
| **Recording (WAV)** | Yes | The audio |
| **Call-type labels** | Yes | A label file giving each real call its type — typically your reference labels |
| **Detected labels (NOISE)** | No | The detector's output on the same WAV. Detections overlapping no labelled call become NOISE examples |
| **Group** | Yes | See below |

!!! danger "Only add a detected-labels file if the call-type file is complete"

    The NOISE examples are built from detections that match no labelled
    call. If your call-type file is missing real calls, those calls get
    taught to the model as noise. Only supply a detected-labels file when
    every real call in the recording is labelled.

Calls rejected in [Label Edit](label-edit.md) also become NOISE examples if
you leave **Rejected detections (Label Edit) are NOISE** ticked.

### 2. Groups and the split

**Recordings of the same animal(s) must share a Group.** You can also list
the animals directly (`A01;A02`) and recordings sharing an animal are joined
into one group automatically.

Whole groups — never individual recordings — go to one split:
approximately **60% train / 20% calibration / 20% test**, which needs at
least five groups. This is the point of the whole mechanism: the reported
test scores come from animals the model has never heard, so they are not
inflated by the model memorising an individual's voice.

Set **Split** to `auto` on every row for the automatic split, or freeze a
specific assignment per row.

### 3. Label handling

| Field | Meaning |
|---|---|
| **NOISE labels** | Labels that mean NOISE (comma-separated, case-insensitive) |
| **USV without type** | Labels marking a real USV with no call type — e.g. the detector placeholder `d`. These train the USV/NOISE stage only |
| **Ignore labels** | Labels left out of training entirely |
| **Min. examples per type** | Below this count, a call type is learned as plain USV instead of as its own type |

### 4. Check the data first

**Check data** counts examples per label and split, reading only the label
files — no audio, so it is fast. Run it before committing to a training run
and you will catch a mis-grouped animal or a typo'd call type in seconds
rather than after an hour of training.

### 5. Training parameters

| Parameter | Default | Notes |
|---|---|---|
| **CNN epochs** | `12` | Fixed; no early stopping |
| **Batch size** | — | CNN batch size |
| **RF trees** | `200` | Random Forest size |
| **Feature set** | Compact | Compact is the validated default |
| **Seed** | `7` | Same seed, same model |
| **Clusters (k)** | `0` | Clusters suggested for review in `preparation/`; 0 = automatic |

### 6. Train, then read the scores

**Train model** extracts features, trains the CNN and Random Forest on the
train split, calibrates the confidence thresholds on the calibration split,
and evaluates on the held-out test split.

The results panel reports, for the test groups:

- accuracy, balanced accuracy and **macro-F1**
- a 95% confidence interval for macro-F1, bootstrapped over test groups
- the fraction of calls sent to review
- which call types had too few examples and were learned as plain USV

Macro-F1 is the number to watch — it weights rare call types as heavily as
common ones, so a model that gets the 50-Hz calls right and ignores
everything else cannot hide behind a good accuracy score.

Output layout:

```
<output>/run/model/           manifest.json, rf.joblib, cnn/cnn.pt
<output>/run/evaluation/      test metrics
<output>/run/classification/  the training run's own predictions
<output>/preparation/         clustering output for review
```

**Use for classification** selects the new model in the Classify sub-tab and
registers it for the Detection tab's **USV_MODEL** plugin.

---

## From the command line

```bash
# train: a CSV with WavFile,LabelFile[,DetectedFile,GroupID,Split]
squeak-peek-cli train-classifier training.csv model_run/

# classify
squeak-peek-cli classify-calls model_run/run/model results/ \
  -r rec.wav rec_detected.txt
```

→ [CLI reference](../cli.md#train-classifier) ·
[Training models in depth](../training.md#call-type-classifier)

---

**Next:** [Label Edit →](label-edit.md)

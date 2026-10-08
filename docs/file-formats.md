# File formats

Everything Squeak Peek Studio reads and writes is plain text or standard
formats. Nothing is locked in a proprietary container.

## Label files

Tab-separated text, **two lines per label**, compatible with Audacity-style
label tracks and with the original MATLAB application.

```
<start_time>	<end_time>	<label>
\	<start_frequency>	<end_frequency>
```

A real example:

```
1.145008	1.198752	d
\	0.000000	0.000000
1.284100	1.331900	5t
\	62000.000000	71500.000000
```

| Field | Unit | Notes |
|---|---|---|
| `start_time`, `end_time` | seconds | From the start of the recording |
| `label` | text | The call type, e.g. `d`, `sk`, `5`, `5t`, `5w`, `c5` |
| `start_frequency`, `end_frequency` | Hz | `0` when the detector does not estimate frequency extent |

The second line always begins with a literal backslash. Detectors that work
purely in time (PSD, BSCD, RBD) write zeros for the frequencies; the
[CNN detector](methods.md#cnn-faster-r-cnn) fills them in.

### Call types

`d` is the generic "a call was detected here, type unknown" placeholder
every detector emits. `sk` conventionally marks a
[video sync click](guide/video.md). The rest are your taxonomy — the
shipped default list is `d, sk, 5, 5t, 5w, c5`, editable in
[Settings → Label Edit](guide/settings.md#label-edit).

A classifier may also write `NOISE` or `UNCERTAIN`, or — with the "best
guess" option on — a type followed by `?`, e.g. `5t?`.

### Review state suffixes

Files exported from [Label Edit](guide/label-edit.md) encode your
accept/reject decisions as a **two-character suffix** on the label text,
separated by an underscore:

```
d_Dc
```

| Position | Character | Meaning |
|---|---|---|
| 1st — detection | `D` | Detection accepted |
| | `d` | Detection rejected |
| | `x` | No decision |
| 2nd — classification | `C` | Classification accepted |
| | `c` | Classification rejected |
| | `x` | No decision |

So `d_Dc` means: label `d`, detection **accepted**, classification
**rejected**. A file with no suffixes is simply one nobody has reviewed yet;
both forms load fine.

The [USV model classifier](guide/classification.md) writes these too: a
predicted call type gets `C`, while `NOISE`, `UNCERTAIN` and best guesses
(`5t?`) get `c`, marking them for review.

This is what lets [classifier training](guide/classification.md#train-model)
use your rejected detections as NOISE examples.

---

## Settings JSON

The settings file is a nested JSON document mirroring the
[Settings tab](guide/settings.md) and shared with the original MATLAB
project. A source checkout ships `settings/default.json`.

An abridged example — every key below is valid, but a shipped file only
carries the ones it overrides:

```json
{
  "Detection": {
    "PSD": {
      "fcutMin": 40000,
      "fcutMax": 120000,
      "segmentLength": 8192,
      "k": 0.023,
      "w": 0.994,
      "denoise": true
    },
    "BSCD": {
      "wlen": 0.008,
      "maWindow": 7500,
      "thresholdMode": "mean",
      "denoise": false
    },
    "PRE": {
      "nfft": 1024,
      "noisePercentile": 50.0,
      "oversubtraction": 1.25,
      "maxReductionDb": 30.0
    },
    "POST": {
      "maxGapToMerge": 0.005,
      "minLabelLength": 0.001,
      "minTonality": 0.0
    }
  },
  "Visualization": {
    "SpectrogramWindow": 1024,
    "SpectrogramColormap": "invgray",
    "ManualLabelLength": 0.075,
    "ManualLabelMarker": "md"
  }
}
```

Top-level sections: `DataInput`, `Visualization`, `LabelEdit`, `Detection`
(one sub-object per detector, plus `PRE` for
[pre-detection denoising](methods.md#pre-detection-denoising) and `POST`
for post-processing), `Classification` and `Video`.

Two properties worth knowing:

- **Missing keys fall back to defaults.** A partial settings file is valid —
  you can write one containing only the handful of parameters you changed.
- **Unknown keys round-trip.** Per-plugin parameters are stored as raw
  dictionaries, so a settings file from a version with an extra detector
  loads without error and keeps those values intact.

Pass one to the CLI with `--settings`:

```bash
squeak-peek-cli detect rec.wav --settings my_protocol.json
```

!!! tip "Treat it as part of your method"

    Save the settings file alongside the results it produced. It is the
    complete record of how a detection run was configured.

---

## Classifier output

A [classification run](guide/classification.md#classify) writes a folder:

| File | Contents |
|---|---|
| `predictions.csv` | Every call with its predicted type and scores. Always complete, even when NOISE calls are dropped from the label files |
| `expert_review_queue.csv` | Uncertain calls plus an audit sample — your review worklist |
| `calls_features.csv` | The extracted acoustic features per call |
| `labels/<recording>_classified.txt` | One label file per recording, in the format above |
| `labels/<recording>_classified_no_noise.txt` | The same, without the calls classified `NOISE` |

Columns you will see in `predictions.csv` include `RecordingID`,
`Prediction`, `RFMaxProbability`, `UncertainReason`, `ClusterPred` and
`RawClusterPred`.

---

## Model folders

A trained call-type model is a **folder**, not a single file:

```
model/
  manifest.json      what the model is, its classes and thresholds
  rf.joblib          the Random Forest
  cnn/cnn.pt         the USV/NOISE CNN
```

`manifest.json` is what the app looks for when you browse to a model, so
point the file picker at the folder containing it.

These folders are **interchangeable in both directions** with the
standalone `USV_Klasifikace` tool.

A full training run produces more:

```
<output>/run/model/           the model folder above
<output>/run/evaluation/      held-out test metrics
<output>/run/classification/  the training run's own predictions
<output>/preparation/         clustering output for review
```

Detector models are single files instead: `.joblib` for
[ML](methods.md#ml-random-forest), `.pt` for
[CNN](methods.md#cnn-faster-r-cnn).

---

## Training CSV

`squeak-peek-cli train-classifier` takes a CSV. Paths are **relative to the
CSV file**.

| Column | Required | Meaning |
|---|---|---|
| `WavFile` | yes | The recording |
| `LabelFile` | yes | Call-type labels |
| `DetectedFile` | no | Detector output; detections matching no labelled call become NOISE |
| `GroupID` | no | Recordings sharing animals must share a group |
| `Split` | no | `auto`, or an explicit `train` / `calibration` / `test` |

```csv
WavFile,LabelFile,DetectedFile,GroupID,Split
rec01.wav,labels/rec01.txt,detected/rec01.txt,A01,auto
rec02.wav,labels/rec02.txt,detected/rec02.txt,A01,auto
rec03.wav,labels/rec03.txt,,A02,auto
```

→ [Training models](training.md#call-type-classifier)

---

## Audio

Standard WAV (`.wav`), read with libsndfile. Designed around **250 kHz**
mono recordings but any sample rate covering your band works — the
detectors derive their frame sizes from the actual rate.

A two-minute 250 kHz recording is roughly 70 MB, and files are loaded fully
into memory.

## Video

Import: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v` — **with an audio track**,
which is required for [automatic sync](guide/video.md).

Export: `.mp4`, with the video stream copied unchanged and the audio
replaced by the sonified ultrasound.

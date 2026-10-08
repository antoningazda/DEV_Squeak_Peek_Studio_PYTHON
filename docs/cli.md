# CLI reference

```
squeak-peek-cli [OPTIONS] COMMAND [ARGS]...
```

Headless batch processing — everything the GUI does, scriptable. Installed
alongside the `squeak-peek` desktop command when you
[install from source](install.md#install-from-source).

| Command | Purpose |
|---|---|
| [`detect`](#detect) | Run a detector on one WAV file |
| [`batch`](#batch) | Run a detector on every WAV in a directory |
| [`evaluate`](#evaluate) | Score detected labels against a reference |
| [`train`](#train) | Train an ML (Random Forest) detector |
| [`train-cnn`](#train-cnn) | Train a CNN (Faster R-CNN) detector |
| [`convert-usvseg`](#convert-usvseg) | Convert the USVSEG dataset to this app's label format |
| [`classify-calls`](#classify-calls) | Assign call types with a trained model |
| [`train-classifier`](#train-classifier) | Train a call-type model |

Global options: `--version`, `--help`. Every command takes `--help`.

For end-to-end recipes — cohort scoring, threshold sweeps, detection →
classification, building a training set — see [CLI workflows](workflows.md).

---

## `detect`

Run a detector on a single WAV file.

```bash
squeak-peek-cli detect [OPTIONS] WAV_PATH
```

| Option | Default | Meaning |
|---|---|---|
| `-d, --detector [bscd\|cnn\|ml\|psd\|rbd]` | `psd` | Detection algorithm |
| `-s, --settings FILE` | `settings/default.json` | Settings JSON |
| `-o, --output PATH` | next to the WAV | Output label file |
| `--min-tonality FLOAT` | from settings | Drop broadband detections below this score, 0–1. Overrides `Detection.POST.minTonality`. Try `0.5` |
| `--denoise / --no-denoise` | from settings | Suppress stationary background noise before detection. Overrides the detector's own `denoise` setting (on for PSD by default) |

```bash
squeak-peek-cli detect recording.wav
squeak-peek-cli detect recording.wav -d bscd --min-tonality 0.5
squeak-peek-cli detect recording.wav -d psd --denoise
squeak-peek-cli detect recording.wav -s protocols/noisy_room.json -o out.txt
```

`--denoise` / `--no-denoise` affect the classical detectors (PSD, BSCD, RBD)
only: ML and CNN apply whatever denoising their model was trained with, so
that training and inference always see the same kind of audio. The exported
labels always refer to the original recording.

→ [Methods](methods.md) · [Tonality filtering](methods.md#tonality-filtering) ·
[Pre-detection denoising](methods.md#pre-detection-denoising)

---

## `batch`

Run a detector on all WAV files in a directory.

```bash
squeak-peek-cli batch [OPTIONS] WAV_DIR
```

| Option | Default |
|---|---|
| `-d, --detector [bscd\|cnn\|ml\|psd\|rbd]` | `psd` |
| `-s, --settings FILE` | `settings/default.json` |
| `-o, --output-dir PATH` | next to each WAV |
| `--min-tonality FLOAT` | from settings |
| `--denoise / --no-denoise` | from settings |

```bash
squeak-peek-cli batch recordings/ -d bscd --min-tonality 0.5 -o detected/
```

---

## `evaluate`

Compare detected labels against a reference file (TP / FP / FN / F1).

```bash
squeak-peek-cli evaluate DETECTED REFERENCE
```

```bash
squeak-peek-cli evaluate detected/rec01.txt reference/rec01.txt
```

Matching is midpoint-based and one-to-one — see
[Metrics](guide/metrics.md#how-matching-works) for what that implies.

### Scoring a whole cohort

```bash
for f in recordings/*.wav; do
  base=$(basename "$f" .wav)
  squeak-peek-cli detect "$f" -d psd --min-tonality 0.5 -o "detected/$base.txt"
  echo -n "$base: "
  squeak-peek-cli evaluate "detected/$base.txt" "reference/$base.txt"
done
```

For a summary CSV with a pooled total, use
[`examples/evaluate_cohort.py`](workflows.md#2-scoring-a-cohort-against-reference-labels).

---

## `train`

Train an ML detector model from WAV + label file pairs.

```bash
squeak-peek-cli train [OPTIONS] WAV_PATHS...
```

| Option | Default | Meaning |
|---|---|---|
| `-l, --labels FILE` | **required** | Label file for each WAV, same order and count as `WAV_PATHS` |
| `-o, --output PATH` | `data/models/ml_detector_model.joblib` | Where to save the model |
| `--noise-ratio FLOAT` | `3.0` | Noise:USV frame ratio for class balancing |
| `--n-trees INTEGER` | `200` | Random Forest tree count |

```bash
squeak-peek-cli train rec1.wav rec2.wav \
  -l labels1.txt -l labels2.txt \
  -o models/colony_a.joblib
```

Then point `Detection.ML.modelPath` at the result.

→ [Training an ML detector](training.md#ml-detector-random-forest)

---

## `train-cnn`

Train a CNN (Faster R-CNN) detector.

```bash
squeak-peek-cli train-cnn [OPTIONS] WAV_PATHS...
```

| Option | Default | Meaning |
|---|---|---|
| `-l, --labels FILE` | **required** | Label file per WAV, same order and count |
| `-o, --output PATH` | `data/models/cnn_detector_model.pt` | Checkpoint path |
| `--epochs INTEGER` | `10` | Training epochs |
| `--batch-size INTEGER` | `4` | |
| `--lr FLOAT` | `0.0001` | Learning rate |
| `--window-s FLOAT` | `1.0` | Spectrogram tile length (s) |
| `--hop-s FLOAT` | `0.5` | Tile hop during training (s) |
| `--backbone [mobilenet\|resnet50]` | `mobilenet` | `mobilenet` = fast, CPU-friendly. `resnet50` = heavier, more accurate, wants a GPU |

→ [Training a CNN detector](training.md#cnn-detector-faster-r-cnn)

---

## `convert-usvseg`

Convert an extracted [USVSEG dataset](https://zenodo.org/records/3428024)
directory (matching `<name>.wav` / `<name>.csv` pairs) into this app's label
format, ready for `train` or `train-cnn`.

```bash
squeak-peek-cli convert-usvseg [OPTIONS] SRC_DIR
```

| Option | Default |
|---|---|
| `-o, --output-dir PATH` | next to each WAV |

```bash
squeak-peek-cli convert-usvseg ~/datasets/usvseg_mouse/
```

---

## `classify-calls`

Classify detected calls with a trained USV model (CNN USV/NOISE + Random
Forest call types).

```bash
squeak-peek-cli classify-calls [OPTIONS] MODEL OUTPUT
```

`MODEL` is a model folder containing `manifest.json`. `OUTPUT` **must be a
new folder** — results are never overwritten.

| Option | Default | Meaning |
|---|---|---|
| `-r, --recording FILE...` | **required** | A WAV and its detected-label file. Repeat for more recordings |
| `--device [cpu\|auto\|mps]` | `cpu` | `mps` = Apple Silicon GPU |
| `--drop-noise` | off | Leave NOISE calls out of the classified label files |

```bash
squeak-peek-cli classify-calls model_run/run/model results/ \
  -r rec01.wav detected/rec01.txt \
  -r rec02.wav detected/rec02.txt \
  --device mps
```

→ [Classifier output](file-formats.md#classifier-output)

---

## `train-classifier`

Train a USV call-type model.

```bash
squeak-peek-cli train-classifier [OPTIONS] RECORDINGS_CSV OUTPUT
```

`RECORDINGS_CSV` has columns `WavFile` and `LabelFile`, optionally
`DetectedFile`, `GroupID` and `Split`; paths are relative to the CSV.
`OUTPUT` **must be a new folder**; the model lands in `OUTPUT/run/model`.

| Option | Default | Meaning |
|---|---|---|
| `--epochs INTEGER` | `12` | CNN epochs |
| `--trees INTEGER` | `200` | Random Forest trees |
| `--seed INTEGER` | `7` | Same seed, same model |
| `--min-examples INTEGER` | `10` | Below this, a call type is learned as plain USV |
| `--generic-labels TEXT` | `d` | Labels meaning "USV without a type" |
| `--device [cpu\|auto\|mps]` | `cpu` | |

```bash
squeak-peek-cli train-classifier training.csv model_run/ --device mps
```

→ [Training CSV format](file-formats.md#training-csv) ·
[Training a classifier](training.md#call-type-classifier)

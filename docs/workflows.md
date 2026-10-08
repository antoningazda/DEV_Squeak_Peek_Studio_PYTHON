# CLI workflows

Recipes for running whole studies without the GUI: batch detection, cohort
scoring, threshold tuning, classification and training. Each one is either
a short shell snippet or a script from the
[`examples/`](https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON/tree/main/examples)
folder of a source checkout, shown here in full.

Every option used below is documented in the [CLI reference](cli.md).

!!! info "Before you start"

    - Run commands from the repository root, so the default
      `--settings settings/default.json` resolves — or pass `-s` explicitly.
    - The Python scripts need the same environment as `squeak-peek-cli`
      ([install from source](install.md#install-from-source)).
    - **Save the settings file with the results.** It is the full record of
      how a run was configured; the scripts below copy it for you.

## The typical pipeline

```
 recordings/*.wav
        │  batch  (detector + post-processing)
        ▼
 detected/*_psd_detected.txt ──── evaluate ───▶ precision / recall / F1
        │                          (vs. reference labels)
        │  classify-calls  (trained call-type model)
        ▼
 classified/labels/*_classified.txt  +  predictions.csv
```

Detector output files are always named `<recording>_<detector>_detected.txt`
(e.g. `rec01_psd_detected.txt`), which is what lets the steps below find
each recording's labels.

---

## 1. Batch detection

One detector over a whole folder:

```bash
squeak-peek-cli batch recordings/ -d psd --min-tonality 0.5 -o detected/
```

Recordings spread over subfolders (`batch` only looks at one level):

```bash
find recordings/ -name '*.wav' -print0 |
  while IFS= read -r -d '' wav; do
    rel=$(dirname "${wav#recordings/}")
    mkdir -p "detected/$rel"
    squeak-peek-cli detect "$wav" -d psd --min-tonality 0.5 \
      -o "detected/$rel/$(basename "$wav" .wav)_psd_detected.txt"
  done
```

Every detector on the same data, to compare them side by side:

```bash
for det in psd bscd rbd; do
  squeak-peek-cli batch recordings/ -d "$det" -o "detected_$det/"
done
```

=== "Windows (PowerShell)"

    ```powershell
    foreach ($det in "psd", "bscd", "rbd") {
        squeak-peek-cli batch recordings\ -d $det -o "detected_$det\"
    }
    ```

### One protocol per condition

Settings files may be **partial** — only the keys you change, everything
else falls back to defaults ([Settings JSON](file-formats.md#settings-json)).
Keep one small file per recording condition:

```json title="protocols/noisy_room.json"
{
  "Detection": {
    "PSD":  { "fcutMin": 45000, "k": 0.03 },
    "POST": { "minTonality": 0.6, "maxGapToMerge": 0.01 }
  }
}
```

```bash
squeak-peek-cli batch cage_A/ -s protocols/quiet_room.json -o detected/cage_A/
squeak-peek-cli batch cage_B/ -s protocols/noisy_room.json -o detected/cage_B/
```

---

## 2. Scoring a cohort against reference labels

`evaluate` scores one file. For a whole cohort with a summary table, use
`examples/evaluate_cohort.py`: it detects each WAV, pairs it with its
reference file by name prefix (`rec01.wav` ↔ `rec01*.txt`), writes the
detected labels, and saves a CSV with one row per recording plus a pooled
total.

```bash
python examples/evaluate_cohort.py recordings/ reference/ \
  -d psd --min-tonality 0.5 -o detected/
```

```
rec01: P=0.742 R=0.893 F1=0.810
TOTAL: P=0.742 R=0.893 F1=0.810
Summary: detected/summary_psd.csv
```

(Output for the bundled `data/example/single/USV_Example_Short.wav` and its
reference labels, renamed to `rec01`.)

The `TOTAL` row pools TP / FP / FN over all recordings (micro-average) —
recordings with more calls weigh more, unlike a mean of per-file F1.
Matching is midpoint-based; see [Metrics](guide/metrics.md#how-matching-works).

??? example "examples/evaluate_cohort.py"

    ```python
    --8<-- "examples/evaluate_cohort.py"
    ```

The `detect()` function in this script is the same pipeline
`squeak-peek-cli detect` runs — detector, then tonality filter, merge of
close events and removal of short ones — so its label files are identical
to the CLI's. Copy it when you need detection inside your own analysis.

---

## 3. Tuning a threshold before a batch run

Pick `minTonality` on one representative, hand-labelled recording first.
`examples/tonality_sweep.py` runs the detector **once** and re-applies only
the post-processing for each value:

```bash
python examples/tonality_sweep.py rec01.wav reference/rec01.txt -d bscd
```

```
BSCD: 392 raw detections, 270 reference labels

minTonality  detected  precision  recall     F1
       0.00       383      0.632   0.896  0.741
       0.20       383      0.632   0.896  0.741
       0.30       358      0.676   0.896  0.771
       0.40       352      0.688   0.896  0.778
       0.50       349      0.693   0.896  0.782
       0.60       342      0.705   0.893  0.788
       0.70       320      0.738   0.874  0.800
```

(Bundled example recording.) Higher values trade recall for precision —
here recall holds until about 0.6 and then starts to fall. Take the F1
peak, or the highest value that keeps recall acceptable for your question,
and check it on a second recording before committing to it.

??? example "examples/tonality_sweep.py"

    ```python
    --8<-- "examples/tonality_sweep.py"
    ```

The same loop works for any detector parameter: change it on the `params`
object (e.g. `params.k = value` for PSD) and re-run `detect` inside the
loop instead of only the post-processing.

---

## 4. Detection → call-type classification

`classify-calls` takes each recording as a `-r WAV LABELS` pair.
`examples/detect_and_classify.sh` batch-detects a folder and passes every
recording to a single classifier run:

```bash
examples/detect_and_classify.sh recordings/ models/colony_a/run/model results/
```

Tune it with environment variables:

```bash
DETECTOR=bscd MIN_TONALITY=0.6 DEVICE=mps \
  examples/detect_and_classify.sh recordings/ models/colony_a/run/model results/
```

??? example "examples/detect_and_classify.sh"

    ```bash
    --8<-- "examples/detect_and_classify.sh"
    ```

The output folder must not exist yet — runs are never overwritten. Results
are described in [Classifier output](file-formats.md#classifier-output);
`expert_review_queue.csv` is the list of calls worth checking by hand.

### Summarising the predictions

`predictions.csv` holds every call, so per-recording call-type counts are
one `pandas` pivot away:

```python
import pandas as pd

pred = pd.read_csv("results/classified/predictions.csv")
counts = pd.crosstab(pred.RecordingID, pred.ClusterPred)
counts["USV total"] = counts.drop(columns=["NOISE"], errors="ignore").sum(axis=1)
counts.to_csv("results/call_type_counts.csv")
print(counts)
```

### From Python

The same run without the CLI, e.g. inside a notebook:

```python
from pathlib import Path

from squeak_peek.usv_classifier import api

recordings = sorted(Path("recordings").glob("*.wav"))
inputs = [
    api.ClassifyInput(wav, api.read_label_file(f"detected/{wav.stem}_psd_detected.txt"))
    for wav in recordings
]
result = api.classify_recordings(
    inputs, "models/colony_a/run/model", "results/classified",
    device="cpu", drop_noise=True,
    progress=lambda _fraction, message: print(message),
)
print(result.counts())             # calls per predicted type
print(result.label_files)          # RecordingID -> classified label file
df = result.predictions            # the full predictions.csv as a DataFrame
```

---

## 5. Training a call-type model

Training needs a CSV listing each recording with its call-type labels
([Training CSV](file-formats.md#training-csv)).
`examples/make_training_csv.py` builds it from folders, pairing files by
name prefix:

```bash
# detector output is optional but recommended: detections that match no
# labelled call become NOISE examples for the USV/NOISE stage
squeak-peek-cli batch recordings/ -d psd -o detected/

python examples/make_training_csv.py recordings/ labels/ training.csv \
  --detected detected/ --group-pattern '(m[A-Z]\d+)'

squeak-peek-cli train-classifier training.csv model_run/ --device mps
```

`--group-pattern` is a regex whose first group is the animal ID in the
file name (`LPS-SI2homo-mH02-I04-USV.wav` → `mH02`). Recordings sharing a
group always land in the same train / calibration / test split, so the
test score is not inflated by having heard the same animal in training.

??? example "examples/make_training_csv.py"

    ```python
    --8<-- "examples/make_training_csv.py"
    ```

The trained model ends up in `model_run/run/model` — pass that folder to
`classify-calls`. Held-out test metrics are in `model_run/run/evaluation/`.

→ [Training a classifier](training.md#call-type-classifier)

### Detector models

The ML and CNN detectors train from WAV + label pairs passed on the command
line. With many recordings, build the argument list in a loop:

```bash
args=(); labels=()
for wav in train/*.wav; do
  args+=("$wav")
  labels+=(-l "labels/$(basename "$wav" .wav).txt")
done
squeak-peek-cli train "${args[@]}" "${labels[@]}" -o models/colony_a.joblib
```

Then set `Detection.ML.modelPath` in a settings file and detect with
`-d ml -s that_file.json`.

→ [Training an ML detector](training.md#ml-detector-random-forest)

---

## 6. Hands-off processing on a server

Long batches survive a closed SSH session with `nohup` and a log:

```bash
nohup examples/detect_and_classify.sh /data/2026_cohort/ \
  models/colony_a/run/model /results/2026_cohort/ \
  > /results/2026_cohort.log 2>&1 &
tail -f /results/2026_cohort.log
```

Or process each day's recordings automatically with `cron`:

```cron
# 02:00 every night: detect yesterday's folder
0 2 * * * cd /opt/squeak-peek && .venv/bin/squeak-peek-cli batch \
  /data/$(date -d yesterday +\%F)/ -d psd --min-tonality 0.5 \
  -o /results/$(date -d yesterday +\%F)/ >> /var/log/squeak-peek.log 2>&1
```

Recordings are loaded fully into memory (a two-minute 250 kHz file is
~70 MB), so running several batches in parallel is limited by RAM, not CPU.

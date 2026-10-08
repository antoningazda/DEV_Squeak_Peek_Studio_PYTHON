# Examples

Runnable workflow scripts for Squeak Peek Studio. Run them from the
repository root (so `settings/default.json` resolves), in the environment
where `squeak-peek-cli` is installed.

| Script | What it does |
|---|---|
| `evaluate_cohort.py` | Detect every WAV in a folder, score each against its reference labels, write a summary CSV |
| `tonality_sweep.py` | Precision / recall / F1 for a range of `minTonality` values on one recording |
| `detect_and_classify.sh` | Batch detection followed by one `classify-calls` run over all recordings |
| `make_training_csv.py` | Build the recordings CSV for `train-classifier` from folders of WAVs and labels |

Each script takes `--help`. The walkthrough, with sample output, is the
[CLI workflows](../docs/workflows.md) page of the documentation.

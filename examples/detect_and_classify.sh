#!/usr/bin/env bash
# Batch-detect every WAV in a folder, then assign call types to all of them
# in one classifier run.
#
#   examples/detect_and_classify.sh recordings/ path/to/model results/
#
# Writes:
#   results/detected/<rec>_<detector>_detected.txt    detector output
#   results/settings.json                             the settings used (keep it!)
#   results/classified/                               classify-calls output folder
#   results/classified/labels/<rec>_classified.txt    one label file per recording
set -euo pipefail

WAV_DIR=${1:?usage: $0 WAV_DIR MODEL_DIR OUT_DIR}
MODEL=${2:?usage: $0 WAV_DIR MODEL_DIR OUT_DIR}
OUT=${3:?usage: $0 WAV_DIR MODEL_DIR OUT_DIR}

DETECTOR=${DETECTOR:-psd}
SETTINGS=${SETTINGS:-settings/default.json}
MIN_TONALITY=${MIN_TONALITY:-0.5}
DEVICE=${DEVICE:-cpu}            # mps on Apple Silicon

mkdir -p "$OUT/detected"
cp "$SETTINGS" "$OUT/settings.json"

# 1. Detection
squeak-peek-cli batch "$WAV_DIR" -d "$DETECTOR" -s "$SETTINGS" \
  --min-tonality "$MIN_TONALITY" -o "$OUT/detected"

# 2. Classification — one -r WAV LABELS pair per recording
args=()
for wav in "$WAV_DIR"/*.wav; do
  base=$(basename "$wav" .wav)
  args+=(-r "$wav" "$OUT/detected/${base}_${DETECTOR}_detected.txt")
done

squeak-peek-cli classify-calls "$MODEL" "$OUT/classified" "${args[@]}" --device "$DEVICE"

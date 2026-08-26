#!/usr/bin/env bash
#
# Run the full CFI experiment pipeline end-to-end in replay (CPU) mode.
#
# Usage:
#   bash scripts/cfi/run_experiment.sh
#   bash scripts/cfi/run_experiment.sh --mode real        # GPU re-inference
#
# Replay mode is fast (seconds–minutes) and is the default published path. The
# headline CSVs/figures land under results/cfi/ and the full per-variant
# artefacts under experiments/h2_cfi/.

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/cfi.yaml"
MODE="replay"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --mode)   MODE="$2"; shift 2 ;;
    *)        echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p experiments/h2_cfi/results experiments/h2_cfi/figures experiments/h2_cfi/votes results/cfi

LOG_DIR="experiments/h2_cfi/results"

echo "[1/4] building CFI manifest"
python scripts/cfi/build_manifest.py --config "${CONFIG}" \
  > "${LOG_DIR}/script05.log" 2>&1

echo "[2/4] running CFI votes (mode=${MODE})"
python scripts/cfi/run_bank_replay.py --config "${CONFIG}" --mode "${MODE}" \
  > "${LOG_DIR}/script06.log" 2>&1

echo "[3/4] analyzing CFI"
python scripts/cfi/analyze_replay.py --config "${CONFIG}" \
  > "${LOG_DIR}/script07.log" 2>&1

echo "[4/4] plotting CFI"
python scripts/cfi/plot.py --config "${CONFIG}" \
  > "${LOG_DIR}/script08.log" 2>&1

echo "done."
echo "headline csvs/figures: results/cfi/"
echo "full artefacts:        experiments/h2_cfi/"

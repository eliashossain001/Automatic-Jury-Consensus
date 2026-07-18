#!/usr/bin/env bash
#
# Launch the full H1 voting pass split across two TITAN RTX cards.
#
# GPU 0:   Llama-3.1-8B (pair, likert), Mistral-7B-v0.3 (pair, likert), Phi-3.5-mini (pair)
# GPU 1:   Qwen-2.5-7B (pair, likert), Gemma-2-9B (pair, likert), Phi-3.5-mini (likert)
#
# Both cards get two 7-8B-class judges plus one smaller judge so the wall time
# is balanced. The vote cache is per-logical-judge parquet, so the two
# processes do not contend; existing cached items are skipped automatically.
#
# Usage:
#   bash scripts/run_bank_parallel.sh          # full calibration set
#   bash scripts/run_bank_parallel.sh 200      # smoke (first 200 items)
#
# Logs land in experiments/h1_measurement/results/full_bank_gpu{0,1}.log.

set -euo pipefail

cd "$(dirname "$0")/.."

LIMIT_ARG=""
if [[ $# -ge 1 ]]; then
  LIMIT_ARG="--limit $1"
fi

mkdir -p experiments/h1_measurement/results

GPU0_JUDGES=(
  "llama-3.1-8b::pairwise"
  "llama-3.1-8b::likert"
  "mistral-7b-v0.3::pairwise"
  "mistral-7b-v0.3::likert"
  "phi-3.5-mini::pairwise"
)

GPU1_JUDGES=(
  "qwen-2.5-7b::pairwise"
  "qwen-2.5-7b::likert"
  "gemma-2-9b::pairwise"
  "gemma-2-9b::likert"
  "phi-3.5-mini::likert"
)

build_args() {
  local out=""
  for jid in "$@"; do
    out+=" --judge-id ${jid}"
  done
  echo "${out}"
}

GPU0_LOG="experiments/h1_measurement/results/full_bank_gpu0.log"
GPU1_LOG="experiments/h1_measurement/results/full_bank_gpu1.log"

echo "GPU 0 → ${GPU0_JUDGES[*]}"
echo "GPU 1 → ${GPU1_JUDGES[*]}"

CUDA_VISIBLE_DEVICES=0 python3 scripts/02_run_judges.py \
  $(build_args "${GPU0_JUDGES[@]}") ${LIMIT_ARG} \
  > "${GPU0_LOG}" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python3 scripts/02_run_judges.py \
  $(build_args "${GPU1_JUDGES[@]}") ${LIMIT_ARG} \
  > "${GPU1_LOG}" 2>&1 &
PID1=$!

echo "GPU 0 process: ${PID0}  (log: ${GPU0_LOG})"
echo "GPU 1 process: ${PID1}  (log: ${GPU1_LOG})"
echo "waiting for both..."
wait "${PID0}" "${PID1}"
echo "both GPU processes complete"

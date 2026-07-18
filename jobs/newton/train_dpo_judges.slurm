#!/usr/bin/env bash
#SBATCH --job-name=train_dpo_judges
#SBATCH --partition=normal
#SBATCH --account=nyousefi
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
# Bucket 2 — DPO-train one judge per array task into a LoRA adapter.
set -euo pipefail
source "$(git rev-parse --show-toplevel 2>/dev/null || echo .)/jobs/newton/env.sh" 2>/dev/null || source jobs/newton/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|qwen-2.5-7b|qwen|7B|"
  "mistralai/Mistral-7B-Instruct-v0.3|mistral-7b-v0.3|mistral|7B|"
  "microsoft/Phi-3.5-mini-instruct|phi-3.5-mini|phi|3.8B|--trust-remote-code"
)
IFS='|' read -r MODEL BASE_ID FAMILY SCALE EXTRA <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
OUT="$RESULTS/dpo_judges/adapters/${BASE_ID}_seed${SEED}"
DONE="$RESULTS/.done/train_dpo_${BASE_ID}"; mkdir -p "$(dirname "$DONE")"
if [ -f "$OUT/adapter_model.safetensors" ]; then echo "[resume] $BASE_ID adapter exists; skip"; touch "$DONE"; exit 0; fi

echo "### training DPO judge: $MODEL ###"
$PY scripts/train_judges_dpo.py --model "$MODEL" --base-id "$BASE_ID" --family "$FAMILY" \
  --scale "$SCALE" --seed "$SEED" --output "$OUT" --epochs 1 --beta 0.1 --lr 5e-6 \
  $(prec_flag) $EXTRA
touch "$DONE"
echo "TRAIN_DPO_JUDGE_DONE $BASE_ID"

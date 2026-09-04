#!/usr/bin/env bash
#SBATCH --job-name=eval_dpo_judges
#SBATCH --partition=normal
#SBATCH --account=CHANGE_ME   # set to your SLURM account
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Bucket 2 — run DPO-judge inference on RewardBench, then base-vs-DPO dependence table.
set -euo pipefail
source jobs/slurm/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"

echo "### [1/2] DPO judge inference (6 logical judges) ###"
$PY scripts/calibration/run_judge_bank.py --bank configs/dpo_judge.yaml --calibration configs/calibration.yaml --sort-by-length

echo "### [2/2] dependence + CorrFilter gain: DPO vs matched base ###"
mkdir -p outputs/dpo_judges
$PY scripts/calibration/evaluate_judge_bank.py --trained-bank configs/dpo_judge.yaml --base-bank configs/judge_bank.yaml \
  --label dpo --manifest "$CAL_MANIFEST" --out outputs/dpo_judges/table_dpo_vs_base.csv \
  --trained-id qwen-2.5-7b-dpo::pairwise --trained-id qwen-2.5-7b-dpo::likert \
  --trained-id mistral-7b-dpo::pairwise --trained-id mistral-7b-dpo::likert \
  --trained-id phi-3.5-mini-dpo::pairwise --trained-id phi-3.5-mini-dpo::likert \
  --base-id qwen-2.5-7b::pairwise --base-id qwen-2.5-7b::likert \
  --base-id mistral-7b-v0.3::pairwise --base-id mistral-7b-v0.3::likert \
  --base-id phi-3.5-mini::pairwise --base-id phi-3.5-mini::likert
touch "$RESULTS/.done/eval_dpo_judges"
echo "EVAL_DPO_JUDGES_DONE"

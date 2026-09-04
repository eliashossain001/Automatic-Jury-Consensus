#!/usr/bin/env bash
#SBATCH --job-name=eval_extended_bank
#SBATCH --partition=normal
#SBATCH --account=CHANGE_ME   # set to your SLURM account
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Bucket 4 — inference over the extended (up to 20-judge) bank, then n_eff saturation.
set -euo pipefail
source jobs/slurm/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"

echo "### [1/2] extended-bank inference (cached votes are skipped automatically) ###"
$PY scripts/calibration/run_judge_bank.py --bank configs/judge_bank_extended.yaml \
  --calibration configs/calibration.yaml --sort-by-length

echo "### [2/2] dependence vs nominal-n + saturation figure ###"
$PY scripts/calibration/sweep_extended_bank.py --bank configs/judge_bank_extended.yaml \
  --manifest "$CAL_MANIFEST" --out-dir outputs/extended_bank
touch "$RESULTS/.done/eval_extended_bank"
echo "EVAL_EXTENDED_BANK_DONE"

#!/usr/bin/env bash
#SBATCH --job-name=train_policy_majority
#SBATCH --partition=normal
#SBATCH --account=nyousefi
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Bucket 3 — DPO-train the majority-filtered policy, then eval on the clean holdout.
set -euo pipefail
source jobs/newton/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"
DD="$RESULTS/downstream/data"; MD="$RESULTS/downstream/models/majority"; RD="$RESULTS/downstream/results"
mkdir -p "$RD"

# build filtered datasets once (idempotent; CPU) — needs the GRPO judge votes present
if [ ! -f "$DD/dpo_majority.jsonl" ]; then
  $PY scripts/build_filtered_dpo_data.py --bank configs/grpo_judge_bank.yaml \
    --manifest "$CAL_MANIFEST" --retention "$RETENTION" --holdout-frac 0.20 --out-dir "$DD"
fi

echo "### DPO-train majority policy ($POLICY_BASE) ###"
$PY scripts/train_policy_dpo.py --data "$DD/dpo_majority.jsonl" --base-model "$POLICY_BASE" \
  --output "$MD" --epochs 2 --beta 0.1 --lr 5e-6 --seed "$SEED" $(prec_flag)

echo "### eval majority policy on clean holdout ###"
$PY scripts/eval_policy.py --base-model "$POLICY_BASE" --adapter "$MD" \
  --manifest "$DD/eval_clean.jsonl" --n 0 --tag majority \
  --out "$RD/eval_majority.csv" $(prec_flag)
touch "$RESULTS/.done/train_policy_majority"
echo "TRAIN_POLICY_majority_DONE"

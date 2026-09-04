#!/usr/bin/env bash
#SBATCH --job-name=train_policy_corrfilter
#SBATCH --partition=normal
#SBATCH --account=CHANGE_ME   # set to your SLURM account
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Bucket 3 — DPO-train the corrfilter-filtered policy, then eval on the clean holdout.
set -euo pipefail
source jobs/slurm/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"
DD="$RESULTS/downstream/data"; MD="$RESULTS/downstream/models/corrfilter"; RD="$RESULTS/downstream/results"
mkdir -p "$RD"

# build filtered datasets once (idempotent; CPU) — needs the GRPO judge votes present
if [ ! -f "$DD/dpo_corrfilter.jsonl" ]; then
  $PY scripts/downstream/build_filtered_dpo_data.py --bank configs/grpo_judge_bank.yaml \
    --manifest "$CAL_MANIFEST" --retention "$RETENTION" --holdout-frac 0.20 --out-dir "$DD"
fi

echo "### DPO-train corrfilter policy ($POLICY_BASE) ###"
$PY scripts/downstream/train_policy_dpo.py --data "$DD/dpo_corrfilter.jsonl" --base-model "$POLICY_BASE" \
  --output "$MD" --epochs 2 --beta 0.1 --lr 5e-6 --seed "$SEED" $(prec_flag)

echo "### eval corrfilter policy on clean holdout ###"
$PY scripts/downstream/evaluate_policy.py --base-model "$POLICY_BASE" --adapter "$MD" \
  --manifest "$DD/eval_clean.jsonl" --n 0 --tag corrfilter \
  --out "$RD/eval_corrfilter.csv" $(prec_flag)
touch "$RESULTS/.done/train_policy_corrfilter"
echo "TRAIN_POLICY_corrfilter_DONE"

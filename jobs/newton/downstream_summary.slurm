#!/usr/bin/env bash
#SBATCH --job-name=downstream_summary
#SBATCH --partition=normal
#SBATCH --account=nyousefi
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# Bucket 3 — combine the two policy evals into the reviewer table (paired CI). CPU-only.
set -euo pipefail
source jobs/newton/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"
RD="$RESULTS/downstream/results"; mkdir -p "$RD"
$PY scripts/summarize_downstream.py \
  --majority-eval "$RD/eval_majority.csv" --corrfilter-eval "$RD/eval_corrfilter.csv" \
  --dataset-stats "$RESULTS/downstream/data/dataset_stats.csv" \
  --out outputs/downstream/table_policy_comparison.csv
echo "DOWNSTREAM_SUMMARY_DONE"

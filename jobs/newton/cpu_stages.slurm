#!/usr/bin/env bash
#SBATCH --job-name=cpu_stages
#SBATCH --partition=normal
#SBATCH --account=nyousefi
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Buckets 1/5/6 (reproduce, qualitative, summary figure) — CPU, but on a COMPUTE node
# because the login node's memory cgroup OOMs on pandas/sklearn imports.
set -euo pipefail
source jobs/newton/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"

echo "### Bucket 1: reproduce current paper results ###"
bash scripts/reproduce_all.sh

echo "### Bucket 5: qualitative failure examples ###"
$PY scripts/extract_failure_examples.py --bank configs/judge_bank.yaml --out-dir outputs/qualitative || true

echo "### Bucket 6: summary figure (GRPO) ###"
$PY scripts/figures/make_summary_figure.py --method-label GRPO \
  --trained-label grpo_trained_6 --filters-trained-label grpo || true

touch "$RESULTS/.done/cpu_stages"
echo "CPU_STAGES_DONE"

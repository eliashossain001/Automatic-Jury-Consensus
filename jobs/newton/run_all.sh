#!/usr/bin/env bash
# =============================================================================
# SINGLE NEWTON ENTRYPOINT — runs every strengthening-phase bucket.
#   CPU stages (reproduce / qualitative / summary figure) run inline (fast).
#   GPU stages are submitted as a SLURM DAG with dependencies (resumable).
# Usage:   bash jobs/newton/run_all.sh [--cpu-only] [--no-submit]
# Override anything via env, e.g.:  SLURM_GRES=gpu:nvidia_h100_80gb_hbm3:1 bash jobs/newton/run_all.sh
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"
mkdir -p "$RESULTS/.done" outputs/tables outputs/figures outputs/downstream outputs/qualitative
CPU_ONLY=0; NO_SUBMIT=0; LOCAL=0
for a in "$@"; do
  [ "$a" = "--cpu-only" ] && CPU_ONLY=1
  [ "$a" = "--no-submit" ] && NO_SUBMIT=1
  [ "$a" = "--local" ] && LOCAL=1     # run CPU stages inline (only on a box w/ enough RAM, e.g. no SLURM)
done

SB() { sbatch --parsable --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
              --qos="$SLURM_QOS" --time="$SLURM_TIME" "$@"; }

echo "########## Buckets 1/5/6: reproduce + qualitative + summary figure (CPU) ##########"
# IMPORTANT: these import pandas/sklearn/matplotlib and OOM on the Newton login node's
# memory cgroup, so they are submitted as a COMPUTE-node job unless --local is given.
J_CPU=""
if [ "$LOCAL" = 1 ] || ! command -v sbatch >/dev/null; then
  bash scripts/pipelines/reproduce_all.sh
  $PY scripts/robustness/extract_failure_examples.py --bank configs/judge_bank.yaml --out-dir outputs/qualitative || true
  $PY scripts/figures/make_training_summary_figure.py --method-label GRPO \
    --trained-label grpo_trained_6 --filters-trained-label grpo || true
else
  J_CPU=$(SB jobs/newton/cpu_stages.slurm)
  echo "  submitted cpu_stages job=$J_CPU"
fi

if [ "$CPU_ONLY" = 1 ]; then echo "[run_all] --cpu-only: skipping GPU stages (cpu job=$J_CPU)"; else
if command -v sbatch >/dev/null && [ "$NO_SUBMIT" = 0 ]; then
  echo "########## GPU DAG via SLURM ##########"
  # Bucket 2: DPO judges (train array -> eval)
  J_DPOTR=$(SB --gres="$SLURM_GRES" jobs/newton/train_dpo_judges.slurm)
  J_DPOEV=$(SB --gres="$SLURM_GRES" --dependency=afterok:"$J_DPOTR" jobs/newton/eval_dpo_judges.slurm)
  # Bucket 3: downstream policies (independent) -> summary
  J_PMAJ=$(SB --gres="$SLURM_GRES" jobs/newton/train_policy_majority.slurm)
  J_PCF=$(SB  --gres="$SLURM_GRES" jobs/newton/train_policy_corrfilter.slurm)
  J_SUM=$(SB  --dependency=afterok:"$J_PMAJ":"$J_PCF" jobs/newton/downstream_summary.slurm)
  # Bucket 4: extended bank (independent)
  J_EXT=$(SB --gres="$SLURM_GRES" jobs/newton/eval_extended_bank.slurm)
  echo "submitted jobs: dpo_train=$J_DPOTR dpo_eval=$J_DPOEV policy_maj=$J_PMAJ policy_cf=$J_PCF summary=$J_SUM extended=$J_EXT"
  echo "watch: squeue -u \$USER ; tail -f logs/*.out"
  SUBMITTED="dpo_train=$J_DPOTR,dpo_eval=$J_DPOEV,policy_maj=$J_PMAJ,policy_cf=$J_PCF,summary=$J_SUM,extended=$J_EXT"
else
  echo "[run_all] no sbatch (or --no-submit): GPU stages NOT launched. Run each jobs/newton/*.slurm manually."
  SUBMITTED="none"
fi
fi

echo "########## manifest.json ##########"
$PY scripts/calibration/write_run_manifest.py --out "$RESULTS/manifest.json" --stage run_all \
  --config configs/judge_bank.yaml --config configs/grpo_judge_bank.yaml \
  --config configs/dpo_judge.yaml --config configs/judge_bank_extended.yaml \
  --dataset "$CAL_MANIFEST" --dataset "$UF_MANIFEST" \
  --seed run=$SEED --seed bootstrap=20260706 --seed position=20260601 \
  --output outputs/tables --output outputs/figures --output outputs/downstream \
  --output "${SUBMITTED:-none}"
echo "RUN_ALL_SUBMITTED — CPU deliverables in outputs/; GPU results land under $RESULTS + outputs/ as jobs finish."

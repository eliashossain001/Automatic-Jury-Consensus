#!/usr/bin/env bash
# Bucket 1 — regenerate the paper's core dependence + filtering tables and figures
# from CACHED VOTES only (no GPU). Idempotent; writes to outputs/tables and
# outputs/figures. Does NOT touch results/grpo_judges/ or experiments/.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
mkdir -p outputs/tables outputs/figures

CAL=experiments/h1_measurement/results/calibration_manifest.parquet
UF=outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv
DEP=outputs/tables/dependence.csv
GRPO_IDS=(--judge-id qwen-2.5-7b-grpo::pairwise --judge-id qwen-2.5-7b-grpo::likert
          --judge-id mistral-7b-grpo::pairwise --judge-id mistral-7b-grpo::likert
          --judge-id phi-3.5-mini-grpo::pairwise --judge-id phi-3.5-mini-grpo::likert)
BASE6=(--judge-id qwen-2.5-7b::pairwise --judge-id qwen-2.5-7b::likert
       --judge-id mistral-7b-v0.3::pairwise --judge-id mistral-7b-v0.3::likert
       --judge-id phi-3.5-mini::pairwise --judge-id phi-3.5-mini::likert)

echo "### [1/5] Dependence metrics (rho_bar, n_eff, eigen-rank, co-failure, contrasts) ###"
rm -f "$DEP"
python scripts/lib/compute_dependence.py --bank configs/judge_bank.yaml    --manifest "$CAL" --label rewardbench_base_10 --out "$DEP"
python scripts/lib/compute_dependence.py --bank configs/judge_bank_uf.yaml --manifest "$UF"  --label ultrafeedback_base_10 --out "$DEP"
python scripts/lib/compute_dependence.py --bank configs/judge_bank.yaml    --manifest "$CAL" --label rewardbench_base_matched6 "${BASE6[@]}" --out "$DEP"
python scripts/lib/compute_dependence.py --bank configs/grpo_judge_bank.yaml --manifest "$CAL" --label grpo_trained_6 "${GRPO_IDS[@]}" --out "$DEP"

echo "### [2/5] Cross-dataset replication + GRPO base-vs-trained tables ###"
python scripts/figures/make_repro_tables.py --dependence "$DEP" --out-dir outputs/tables

echo "### [3/5] Filter comparison + CorrFilter-vs-consensus bootstrap CIs ###"
python scripts/lib/run_filters.py --bank configs/grpo_judge_bank.yaml --manifest "$CAL" \
  --label grpo --out outputs/tables/filters.csv --gain-out outputs/tables/corrfilter_gain.csv "${GRPO_IDS[@]}"
python scripts/lib/run_filters.py --bank configs/judge_bank.yaml --manifest "$CAL" \
  --label base_matched6 --out outputs/tables/filters.csv --gain-out outputs/tables/corrfilter_gain.csv "${BASE6[@]}"

echo "### [4/5] Figures (base-vs-GRPO dependence; n_eff ceiling) ###"
python scripts/figures/make_base_vs_grpo_figure.py >/dev/null && \
  cp -f results/grpo_judges/figures/base_vs_grpo_dependence.pdf outputs/figures/ 2>/dev/null || true
python scripts/figures/make_repro_tables.py --dependence "$DEP" --figure outputs/figures/neff_vs_ceiling.pdf --out-dir outputs/tables

echo "### [5/5] OPTIONAL: bias-cluster / router (reproduced by their own scripts if present) ###"
for s in filters/run_bias_cluster_filter.py routing_selector/run_regime_router.py routing_selector/run_unified_router.py; do
  if [ -f "scripts/$s" ]; then echo "  available: scripts/$s (run manually to regenerate its outputs/ dir)"; fi
done

echo "REPRODUCE_ALL_DONE — see outputs/tables/*.csv and outputs/figures/*.pdf"

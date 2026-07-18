#!/usr/bin/env bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=normal
#SBATCH --account=nyousefi
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
# Fail-fast end-to-end sanity: CPU reproduce + a tiny GPU DPO train (verifies the stack).
set -euo pipefail
source jobs/newton/env.sh
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate" && export PY="$REPO/.venv/bin/python"

echo "### [1/3] CPU reproduce (numbers must match paper) ###"
bash scripts/reproduce_all.sh >/dev/null
$PY - <<'PY'
import pandas as pd
d=pd.read_csv("outputs/tables/dependence.csv").set_index("label")
assert abs(d.loc["rewardbench_base_10","rho_bar"]-0.2056)<0.01, "base rho_bar drift"
assert abs(d.loc["grpo_trained_6","rho_bar"]-0.2348)<0.01, "grpo rho_bar drift"
print("reproduce OK: base=%.4f grpo=%.4f"%(d.loc["rewardbench_base_10","rho_bar"],d.loc["grpo_trained_6","rho_bar"]))
PY

echo "### [2/3] GPU check: nvidia-smi + torch.cuda ###"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda', torch.cuda.get_device_name(0))"

echo "### [3/3] tiny DPO-judge train (32 pairs, verifies trl/peft/bnb) ###"
$PY scripts/train_judges_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct --base-id qwen-1.5b-smoke \
  --family qwen --scale 1.5B --seed 0 --limit 32 --epochs 1 --batch 2 --grad-accum 1 \
  --output "$RESULTS/.smoke/dpo_judge" $(prec_flag)
test -f "$RESULTS/.smoke/dpo_judge/adapter_model.safetensors" && echo "SMOKE_TEST_PASSED" || { echo "SMOKE_TEST_FAILED"; exit 1; }

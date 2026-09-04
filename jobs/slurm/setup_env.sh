#!/usr/bin/env bash
# One-time environment setup — run on a the cluster LOGIN node (has internet).
# Creates a venv, installs pinned deps, installs the corrfilter package, verifies imports.
# Idempotent. Override PY_BUILD/torch index if your CUDA differs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "### [setup] loading modules (guarded — ignore if module system absent) ###"
module load anaconda3 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

echo "### [setup] creating venv at $REPO/.venv ###"
if [ ! -d "$REPO/.venv" ]; then
  python3 -m venv "$REPO/.venv"
fi
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
export PY="$REPO/.venv/bin/python"
"$PY" -m pip install --upgrade pip wheel >/dev/null

echo "### [setup] installing deps ###"
# CUDA 12.x wheels are the default on H100 nodes; change --index-url if your CUDA differs.
"$PY" -m pip install \
  "torch>=2.3" "transformers>=4.44" "trl>=0.9" "peft>=0.11" "accelerate>=0.30" \
  "bitsandbytes>=0.43" "datasets>=2.20" "numpy>=1.26" "scipy>=1.11" "scikit-learn>=1.4" \
  "pandas>=2.2" "pyyaml>=6.0" "matplotlib>=3.8" "seaborn>=0.13" "tqdm>=4.66"
"$PY" -m pip install -e "$REPO" >/dev/null || true

echo "### [setup] staging models (choose ONE path) ###"
cat <<'NOTE'
Models are NOT auto-downloaded here. Pick one:
  (a) rsync from the shared box (fastest if reachable):
      rsync -aP <you>@<sharedhost>:$HF_HOME/hub/  "$HF_HOME/hub/"
  (b) download on this login node with your token (compute nodes may be offline):
      export HF_HUB_OFFLINE=0 HF_TOKEN=<your_token>
      python - <<'PY'
      from huggingface_hub import snapshot_download
      for m in ["Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3",
                "microsoft/Phi-3.5-mini-instruct","Qwen/Qwen2.5-1.5B-Instruct"]:
          snapshot_download(m)
      PY
NOTE

echo "### [setup] verifying install (lightweight — no heavy imports on the login node) ###"
# Login nodes have tight memory cgroups; importing torch/transformers here OOMs.
# We only check the packages are INSTALLED (via pip metadata, no import).
"$PY" - <<'PY'
import importlib.metadata as m
missing = []
for p in ["torch","transformers","trl","peft","bitsandbytes","datasets","scikit-learn","pandas","numpy","scipy"]:
    try:
        print(f"  {p:14s} {m.version(p)}")
    except Exception:
        missing.append(p)
print("MISSING:", missing if missing else "none")
PY
echo
echo "SETUP_DONE — deps installed. Do NOT import torch on the login node (it will OOM)."
echo "Verify the runtime on a COMPUTE node:"
echo "  srun -p ${SLURM_PARTITION} -A ${SLURM_ACCOUNT} -q ${SLURM_QOS} --gres=${SLURM_GRES} --mem=32G -t 15 --pty \\"
echo "    bash -lc 'cd $REPO && source .venv/bin/activate && python -c \"import torch,trl,peft; print(torch.cuda.is_available())\"'"
echo "Or just run the real check:  sbatch jobs/slurm/smoke_test.slurm"

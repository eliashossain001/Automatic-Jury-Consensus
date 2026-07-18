#!/usr/bin/env bash
# Shared Newton environment — sourced by every job. NO hardcoded local paths:
# everything derives from the repo location or is overridable via environment.
set -euo pipefail

# repo root = two levels up from this file (jobs/newton/env.sh)
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export RESULTS="${RESULTS:-$REPO/results/strengthening_phase}"
export LOGDIR="${LOGDIR:-$REPO/logs}"
mkdir -p "$RESULTS" "$LOGDIR"

# --- BLAS/OpenMP thread caps: login nodes OOM when OpenBLAS spawns one thread per
#     core ("OpenBLAS error: Memory allocation still failed"). 1 is safe everywhere;
#     GPU stages are GPU-bound so this costs nothing. Bump for CPU-heavy overrides. ---
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-1}"   # curb jemalloc/glibc arena thread spawn

# --- SLURM defaults (override by exporting before sbatch, or via sbatch CLI flags) ---
export SLURM_PARTITION="${SLURM_PARTITION:-normal}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nyousefi}"
export SLURM_QOS="${SLURM_QOS:-normal}"
export SLURM_GRES="${SLURM_GRES:-gpu:nvidia_h100_pcie:1}"
export SLURM_TIME="${SLURM_TIME:-12:00:00}"

# --- HuggingFace cache / offline ---
# Point HF_HOME at a shared cache if one exists on Newton; else a repo-local cache.
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"       # set 0 on a node with internet to download
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- precision: H100/A100 -> bf16; TITAN/V100 -> fp16 ---
export PRECISION="${PRECISION:-bf16}"              # bf16 | fp16
prec_flag() { [ "$PRECISION" = "bf16" ] && echo "--bf16" || echo "--fp16"; }
export -f prec_flag

# --- python: prefer an activated venv/conda; fall back to `python` ---
export PY="${PY:-python}"

# --- experiment knobs (all overridable) ---
export CAL_MANIFEST="${CAL_MANIFEST:-experiments/h1_measurement/results/calibration_manifest.parquet}"
export UF_MANIFEST="${UF_MANIFEST:-outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv}"
export POLICY_BASE="${POLICY_BASE:-Qwen/Qwen2.5-1.5B-Instruct}"   # small DPO policy for Bucket 3
export RETENTION="${RETENTION:-0.60}"
export SEED="${SEED:-20260707}"

cd "$REPO"
echo "[env] REPO=$REPO PRECISION=$PRECISION PART=$SLURM_PARTITION GRES=$SLURM_GRES PY=$($PY --version 2>&1)"

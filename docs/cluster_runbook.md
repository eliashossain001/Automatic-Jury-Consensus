# Strengthening-phase runbook (Newton)

One entrypoint runs every bucket: **`bash jobs/newton/run_all.sh`**. CPU stages run
inline; GPU stages are submitted as a SLURM DAG with dependencies and are resumable.

## Buckets
| # | What | Where results land |
|---|------|--------------------|
| 1 | Reproduce paper dependence/filter tables + figures (CPU) | `outputs/tables/`, `outputs/figures/` |
| 2 | DPO-trained judges vs base (GPU) | `outputs/dpo_judges/table_dpo_vs_base.csv` |
| 3 | Downstream DPO: majority vs CorrFilter policies (GPU) | `outputs/downstream/table_policy_comparison.csv` |
| 4 | Extended 15–20 judge bank, n_eff saturation (GPU) | `outputs/extended_bank/` |
| 5 | Qualitative failure examples (CPU) | `outputs/qualitative/` |
| 6 | Reviewer summary figure, parameterized (CPU) | `outputs/figures/summary_dependence_training.pdf` |

## First-time setup (login node, has internet)
```bash
cd <repo>
bash jobs/newton/setup_env.sh          # venv + deps + verify; then stage models (see below)
```
**Stage models** (compute nodes are usually offline). Either rsync the shared cache or
download on the login node with your HF token — both snippets are printed by `setup_env.sh`.
Set `HF_HOME` to wherever they land (default `<repo>/.hf_cache`).

**Stage GRPO artifacts** (Bucket 3 needs the existing GRPO judge votes; Bucket 2 trains fresh):
```bash
rsync -aP <you>@<sharedhost>:<repo>/results/grpo_judges/ results/grpo_judges/
```

## Smoke test (one GPU job, ~15 min) — do this before run_all
```bash
sbatch jobs/newton/smoke_test.slurm     # CPU reproduce assert + tiny DPO train
```

## Run everything
```bash
bash jobs/newton/run_all.sh             # CPU inline + submit GPU DAG
squeue -u $USER                         # watch; logs in logs/%x-%j.out
```
Options: `--cpu-only` (buckets 1/5/6 only), `--no-submit` (build but don't sbatch).

## Configuration (all overridable via env; no hardcoded paths)
`jobs/newton/env.sh` centralizes everything:
`SLURM_PARTITION` (normal) · `SLURM_ACCOUNT` (nyousefi) · `SLURM_QOS` (normal) ·
`SLURM_GRES` (gpu:nvidia_h100_pcie:1) · `SLURM_TIME` · `PRECISION` (bf16 on H100/A100,
fp16 on V100) · `HF_HOME` · `POLICY_BASE` (Qwen2.5-1.5B; bump to 3B for a stronger
downstream) · `RETENTION` (0.60) · `SEED`.
Example: `SLURM_GRES=gpu:nvidia_h100_80gb_hbm3:1 PRECISION=bf16 bash jobs/newton/run_all.sh`

## Resume / provenance
- Each GPU stage writes `results/strengthening_phase/.done/<stage>` and skips if its
  adapter/output already exists — re-running `run_all.sh` resumes.
- `run_all.sh` writes `results/strengthening_phase/manifest.json` (git commit, model IDs
  from configs, dataset sha1s, seeds, output paths).

## What is paper-grade vs appendix (be conservative)
- **Bucket 3** is the headline: report only if `table_policy_comparison.csv` shows the
  CorrFilter policy ahead with a paired-bootstrap CI **excluding 0**. If it overlaps,
  bump `POLICY_BASE` to 3B and re-run before claiming anything; else → appendix.
- **Bucket 2**: report if ΔρΜ>0 and CorrFilter gain CI excludes 0 (claim: "not GRPO-specific").
- **Bucket 4**: report the saturation figure if n_eff stays flat as n→20; else appendix.
- AlpacaEval/MT-Bench need an external judge (no API here) — not run; leave as future work.

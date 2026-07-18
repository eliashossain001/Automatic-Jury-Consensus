# CorrFilter

**Consensus Is Not Reliability: A Taxonomy of Dependence Failures in LLM Judge Banks**

Code for measuring and mitigating *structured co-failure* in LLM judge banks used for
alignment-data filtering.

LLM judges increasingly govern alignment pipelines through **k-of-n consensus** — a
preference pair is retained when enough judges agree. This assumes judges fail
*independently*. They don't. This repo provides the tooling to (1) **measure** that
dependence, (2) **classify** which of three dependence regimes a judge bank is in, and
(3) **mitigate** it with regime-appropriate filters.

## Key findings (from the paper)

- A ten-judge bank (5 open-weight models × 2 prompt styles) on **RewardBench v2** shows
  mean pairwise error correlation **ρ̄ = 0.21**, collapsing 10 nominal judges to an
  effective ensemble of only **n_eff ≈ 3.5**. Replicates on **UltraFeedback** (ρ̄ = 0.22,
  n_eff = 3.4).
- The usual diversification axes don't fix it: prompt-style diversity gives no
  decorrelation, model-family diversity is inconsistent, and all judges show position bias.
- Dependence splits into **three regimes**, each needing a different mitigation:
  - **Mild dependence** — consensus stays approximately reliable.
  - **Global co-failure** — shared blind spots → confident agreement on wrong labels.
    A *correlation-aware filter* cuts false retention.
  - **Vulnerable subgroup** — a biased subset drives wrong majorities while making the
    ensemble look *less* correlated. A *bias-cluster filter* improves precision by **+4.8 pts**.
- Preference optimization makes it worse: training the same judges with **GRPO/DPO raises**
  ρ̄ (to 0.24 / 0.33 from 0.18) and shrinks n_eff — yet the correlation-aware filter still
  beats majority/supermajority consensus at matched retention.

## Repository layout

```
corrfilter/
├── src/corrfilter/            Python package
│   ├── data/                  RewardBench v2 / UltraFeedback calibration loaders
│   ├── judges/                Judge abstraction: HF local-weights + API judges, prompts, registry
│   ├── voting/                Calibration + filtering runners with on-disk vote cache
│   ├── correlation/           Error-correlation R, n_eff, eigenspectrum, clustering
│   ├── analysis/              Hypothesis tests, agreement, eigen/failure-mode decomposition
│   ├── cfi/                   Correlated-Failure Index: banks, bias clusters, adaptive-r,
│   │                          consensus, corrfilter score, triggers, bootstrap, metrics
│   ├── viz/                   Heatmaps, dendrograms, eigenspectrum, n_eff-collapse figures
│   └── filtering.py           Core filtering entry points
├── configs/                   YAML judge-bank and calibration-set definitions
├── scripts/                   Numbered driver scripts (01_… → 32_…) + analysis/ helpers
├── jobs/newton/               SLURM job scripts for the strengthening-phase cluster run
├── tests/                     Unit tests (pytest)
├── pyproject.toml             Package + dev tooling (ruff, pytest)
├── README.md
└── README_NEWTON.md           Cluster (SLURM) runbook for the full strengthening phase
```

> **Note on data:** experiment outputs, model artifacts, vote caches, figures, and the
> paper source are intentionally **not** tracked in git (see `.gitignore`) — they are large
> and regenerable. Everything below reproduces them from scratch.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/eliashossain001/corrfilter.git
cd corrfilter
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for pytest + ruff
```

Judges load open-weight models via `transformers`. Set a Hugging Face token (for gated
models) in a `.env` file at the repo root — it is gitignored and never committed:

```bash
echo "HF_TOKEN=hf_xxx" > .env
```

## Quick start — H1 (measuring dependence)

```bash
python scripts/01_build_calibration_set.py                        # RewardBench v2 calibration set
python scripts/02_run_judges.py --config configs/judge_bank.yaml  # run the 10-judge bank (cached)
python scripts/03_compute_correlation.py                          # ρ̄, n_eff, eigenspectrum
python scripts/04_h1_analysis.py                                  # hypothesis tests + figures
```

Outputs land in `experiments/h1_measurement/`; the writeup is in `reports/h1_measurement.md`.

## Reproducibility

The `scripts/` directory is an **ordered pipeline** — file names are numbered in dependency
order. Broadly:

| Scripts | Stage |
|---------|-------|
| `01`–`04` | H1: build calibration set, run judges, compute correlation, analyze dependence |
| `05`–`09` | CFI: manifest, votes, correlated-failure analysis, CorrFilter run |
| `10`–`19` | Adaptive-r, poisoned-UltraFeedback attacks, position poisoning, cluster/bias filters, regime router |
| `20`–`27` | Robustness, confidence intervals, mainstream-alignment integration, dependence replication, simulations |
| `28`–`32` | GRPO/DPO-trained judges, base-vs-trained comparison, abstention robustness, seed contrast |

Helper analysis scripts live in `scripts/analysis/` (bootstrap CIs, dependence tables,
repro tables). Judge banks and calibration sets are fully specified in `configs/*.yaml`, so
a run is reproducible from `(config, script)` alone. Vote results are cached on disk, so
re-running a stage reuses prior judge calls instead of recomputing them.

### Full cluster run (strengthening phase)

The complete set of GPU experiments (DPO-trained judges, downstream DPO policies, extended
15–20 judge bank) is orchestrated for SLURM. See **[README_NEWTON.md](README_NEWTON.md)**:

```bash
bash jobs/newton/setup_env.sh     # venv + deps + stage models
sbatch jobs/newton/smoke_test.slurm
bash jobs/newton/run_all.sh       # CPU stages inline + submit GPU DAG (resumable)
```

Each stage writes a `manifest.json` recording git commit, model IDs, dataset hashes, and
seeds for provenance.

## Tests

```bash
pytest            # or: pip install -e ".[dev]" && pytest
```

Covers correlation estimation, effective sample size, filtering, consensus, adaptive-r,
CFI banks/triggers/retention, and prompt handling.

## Citation

If you use this code, please cite:

```bibtex
@article{hossain2026consensus,
  title   = {Consensus Is Not Reliability: A Taxonomy of Dependence Failures in LLM Judge Banks},
  author  = {Hossain, Elias},
  year    = {2026}
}
```

## License

See repository for license terms.

# CorrFilter

**Consensus Is Not Reliability: A Taxonomy of Dependence Failures in LLM Judge Banks**

Code for measuring and mitigating *structured co-failure* in LLM judge banks used for
alignment-data filtering.

LLM judges increasingly govern alignment pipelines through **k-of-n consensus** — a
preference pair is retained when enough judges agree. This assumes judges fail
*independently*. They don't. This repo provides the tooling to (1) **measure** that
dependence, (2) **classify** which of three dependence regimes a judge bank is in, and
(3) **mitigate** it with regime-appropriate filters.

## Key findings

- **Strong, structural dependence.** A ten-judge bank (5 open-weight models × 2 prompt
  styles) shows mean pairwise error correlation **ρ̄ = 0.21**, collapsing 10 nominal judges
  to an effective ensemble of only **n_eff ≈ 3.5** — a hard ceiling of `1/ρ̄ ≈ 4.9`. Roughly
  **27%** of items have the majority jointly wrong.
- **Replicates across three preference datasets** (same bank, same estimators), worst on
  safety data:

  | Dataset | ρ̄ | n_eff (of 10) | conditional co-failure |
  |---|---|---|---|
  | RewardBench v2 | 0.21 | 3.5 | 0.50 |
  | UltraFeedback | 0.22 | 3.4 | 0.51 |
  | **PKU-SafeRLHF** (human safety labels) | **0.27** | **2.9** | **0.53** |

- **Diversity doesn't fix it.** Prompt-style diversity gives zero decorrelation, model-family
  diversity is inconsistent, and all judges show position bias (H1 rejected).
- **Scaling the bank doesn't recover independence.** Growing 4→16 judges (5 families,
  1.5B–14B), n_eff reaches only **4.7 at n=16 (~29% of nominal, down from 71% at n=4)** —
  more coverage, not more independence.
- **Preference optimization amplifies it.** Training the same judges **raises** ρ̄ and shrinks
  n_eff on a matched 6-judge bank — DPO more than GRPO — yet the correlation-aware
  **CorrFilter** still cuts false-retention below majority/supermajority consensus at matched
  60% retention:

  | Training | ρ̄ | n_eff | CorrFilter FRR reduction @0.60 |
  |---|---|---|---|
  | Base (untrained) | 0.178 | 3.17 | — |
  | GRPO-trained | 0.235 | 2.76 | 4.2 pts [2.8, 5.6] |
  | DPO-trained | 0.326 | 2.28 | 6.0 pts [4.4, 7.7] |

- **Three dependence regimes**, each needing a different mitigation:
  - **Mild dependence** — consensus stays approximately reliable.
  - **Global co-failure** — shared blind spots → confident agreement on wrong labels; a
    *correlation-aware filter* cuts false retention.
  - **Vulnerable subgroup** — a biased subset drives wrong majorities while making the
    ensemble look *less* correlated; a *bias-cluster filter* improves precision.
  Vulnerable subgroups occur **naturally**, not only under synthetic poisoning: **150/319**
  consensus-wrong items carry a natural subgroup signature, and leave-cluster-out flips the
  majority to correct on 47% of wrong items.
- **Robust to gold-label noise.** Flipping 1/3/5/10% of gold labels (40 seeds) barely moves
  ρ̄ (0.21 → 0.23); the high-consensus subset is *more* dependent (ρ̄ = 0.34), not a
  label-noise artifact.
- **Cheap to deploy.** R is estimated once per bank (scoring is `O(|S|²)`, no per-item
  training, no matrix inversion); a **low-rank (rank-2/3) approximation of R recovers — even
  denoises — the full-R benefit**.

### Reported honestly (where mitigations do *not* yet win)

- **Regime router:** a supervised dev-set router matches ~78% regime accuracy but does not
  beat the best single filter (gain −0.6 pts, 95% CI [−1.5, +0.1] includes 0). It's a
  diagnostic, not a universal policy.
- **Downstream DPO:** CorrFilter cut training contamination 24.9% → 20.5%, but a 1.5B DPO
  policy's reward accuracy was unchanged within uncertainty — **no downstream win claimed**;
  contamination-scaling is future work.
- **No universal filter:** different regimes need different filters; regime identification is
  the prerequisite.

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

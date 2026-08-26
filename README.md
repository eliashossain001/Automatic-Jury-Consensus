# CorrFilter

**Agreement Is Not Evidence: Measuring Error Dependence and Effective Evidence in LLM Judge Consensus**

Consensus filtering treats judge agreement as accumulated evidence. That is only valid if
judges fail *independently*. This repository measures that assumption directly, and it
fails — worse as judges get better.

The code here does three things: it **measures** how much independent evidence a judge
bank actually carries, it **classifies** which dependence regime the bank is in, and it
**filters** with a rule matched to that regime.

## What the experiments show

**Consensus carries far less evidence than it counts.** A ten-judge bank (5 open-weight
models × 2 prompt styles) has mean pairwise error correlation ρ̄ = 0.206, so ten judges
supply the evidence of n_eff ≈ 3.5 independent ones — even a unanimous verdict is worth
about 3.5 votes, against a hard ceiling of 1/ρ̄ ≈ 4.86. The pooled-vote panel test that
practitioners actually run declares winners a dependence-aware test does not support in up
to 28% of 100-item evaluations.

**No diversification axis restores independence.** Prompt style gives zero decorrelation,
model family is inconsistent in sign, and provider mixing does not help either. Growing the
bank from 4 to 16 judges (5 families, 1.5B–14B) reaches only n_eff ≈ 4.7 — more coverage,
not more independence. The result replicates across three preference datasets:

| Dataset | ρ̄ | n_eff (of 10) | conditional co-failure | majority co-failure |
|---|---|---|---|---|
| RewardBench v2 | 0.206 | 3.51 | 0.50 | 0.27 |
| UltraFeedback | 0.219 | 3.36 | 0.51 | 0.26 |
| PKU-SafeRLHF (human safety labels) | 0.27 | 2.9 | 0.53 | — |

**Dependence rises with capability.** Preference-optimizing the same judges raises ρ̄ from
0.178 to 0.235 (GRPO) and 0.326 (DPO). Three flagship judges from three providers reach
91–93% individual accuracy while their residual errors correlate at ρ̄ = 0.56 and co-fail
at 7.7× the independent rate — cross-provider pairs as dependent as within-provider ones,
so the operative axis is capability tier, not provider.

| Bank | ρ̄ | n_eff | CorrFilter FRR reduction @ 0.60 |
|---|---|---|---|
| Base (untrained, matched 6) | 0.178 | 3.17 | — |
| GRPO-trained | 0.235 | 2.76 | 4.2 pts, 95% CI [2.8, 5.6] |
| DPO-trained | 0.326 | 2.28 | 6.0 pts, 95% CI [4.4, 7.7] |

**Dependence is regimes, not a severity scale.** A shared blind spot *raises* measured
correlation; a biased subgroup that drives wrong majorities *lowers* it. The bank looks
most independent exactly when it is least trustworthy, and the two need opposite
mitigations — a correlation-aware score for global co-failure, a bias-cluster filter
(+4.8 precision points) for a vulnerable subgroup. No fixed aggregation rule is optimal
across regimes; that is proved in the appendix and confirmed empirically.

**Downstream, contamination matters monotonically.** Consensus-retained contamination
degrades a DPO policy's reward margins monotonically from 20% upward.

### Where the mitigations do not win

Reported as such, deliberately:

- **Routing between filters** does not universally beat the best in-domain fixed filter.
  It is a diagnostic, not a policy.
- **Cross-task routing is unestablished.** A preregistered eleven-configuration screen
  found no successful cross-task router; what it did establish is a two-sided boundary
  outside which no aggregation rule has room to help.
- **No universal filter exists.** Regime identification is the prerequisite, and it is
  itself imperfect.

## Layout

```
corrfilter/
├── src/corrfilter/          the library — everything importable
│   ├── data/                RewardBench v2 / UltraFeedback / generic-task loaders
│   ├── judges/              judge abstraction: HF local weights, API judges (Gemini,
│   │                        OpenRouter), prompts, forced-choice, position handling
│   ├── voting/              calibration + filtering runners over an on-disk vote cache
│   ├── correlation/         error-correlation R, n_eff, eigenspectrum, clustering
│   ├── analysis/            dependence estimators, bootstrap CIs, filter comparison,
│   │                        H1 contrasts, agreement, eigen/failure-mode decomposition
│   ├── cfi/                 correlated-failure injection: bias banks, adaptive-R,
│   │                        consensus, CorrFilter score, triggers, metrics
│   ├── screening/           synthetic regime generators, deployment features, and the
│   │                        routing-feasibility gate
│   ├── viz/                 heatmaps, dendrograms, eigenspectra, n_eff-collapse figures
│   ├── routing.py           regime diagnostics, filter family, learned-router toolkit
│   ├── filtering.py         core filtering entry points
│   └── evaluation.py        consensus scoring, matched retention, evaluable pools
├── scripts/                 drivers, grouped by experimental stage
│   ├── calibration/         build the calibration set, run the judge bank
│   ├── dependence/          R, n_eff, eigen decomposition, replications, non-error
│   ├── cfi/                 bias banks, CorrFilter, adaptive-R, low-rank approximation
│   ├── attacks/             synthetic-poisoned UltraFeedback, position-aligned poisoning
│   ├── filters/             bias-cluster, learned-cluster, natural-subgroup filters
│   ├── routing/             regime routers and the crossed/mixed/extended benchmarks
│   ├── frontier/            Gemini, OpenRouter, multi-provider frontier banks
│   ├── crosstask/           the routing-feasibility screening campaign
│   ├── training/            GRPO/DPO-trained judges, forced-choice abstention control
│   ├── downstream/          downstream DPO policy validation and contamination scaling
│   ├── robustness/          bank composition, confidence intervals, gold sensitivity
│   └── figures/             every paper figure and table
├── configs/                 judge-bank and calibration-set definitions (YAML)
├── jobs/newton/             SLURM scripts for the cluster runs
├── docs/                    cluster runbook, experiment log, script map, phase-2 plan
├── tests/                   213 unit tests (pytest)
└── pyproject.toml           package + dev tooling (ruff, pytest)
```

Result artefacts live under `experiments/` (cached votes, manifests, checkpoints),
`outputs/` (one directory per experiment: CSVs, a `summary.md`, figures), and `results/`.
All three are gitignored — they are large and regenerable, and every number in the paper
traces to a file in one of them.

> Coming from an older log or notebook? The pipeline used to be a flat numbered
> `scripts/01_…`–`58_…`. [`docs/script_map.md`](docs/script_map.md) maps every old name to
> its current path.

## Install

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for pytest + ruff
```

The editable install is what puts `corrfilter` on the path; scripts import it directly and
do no `sys.path` manipulation. Put API credentials in a gitignored `.env` at the repo root:

```bash
echo "HF_TOKEN=hf_xxx"           >  .env    # gated open-weight models
echo "GEMNI_API_KEY=..."         >> .env    # Gemini frontier bank
echo "OPEN_ROUTER_API_KEY=..."   >> .env    # OpenAI / Anthropic / xAI via OpenRouter
```

## Quick start — measuring dependence

```bash
python scripts/calibration/build_calibration_set.py                                # RewardBench v2 calibration set
python scripts/calibration/run_judge_bank.py --config configs/judge_bank.yaml      # 10-judge bank (GPU, cached)
python scripts/dependence/compute_correlation.py                                   # ρ̄, n_eff, eigenspectrum
python scripts/dependence/h1_family_prompt_contrast.py                             # family/prompt contrasts + figures
```

Everything lands in `experiments/h1_measurement/`. The first step is the only one that
needs a GPU; once the vote cache exists, every analysis below runs on a laptop CPU.

## Reproducing the results

Vote collection happens once per bank; all analysis replays the cache.

| Want | Run |
|---|---|
| Core dependence tables and filter comparison | `bash scripts/reproduce_all.sh` |
| CFI experiment end to end (replay mode) | `bash scripts/cfi/run_experiment.sh` |
| CorrFilter vs. consensus at matched retention | `python -m corrfilter.analysis.filters --bank configs/judge_bank.yaml --manifest <manifest>` |
| Dependence metrics for any bank | `python -m corrfilter.analysis.dependence --bank <config.yaml> --manifest <manifest> --label <name>` |
| Frontier banks (spends money; dry-run by default) | `python scripts/frontier/run_gemini_judges.py --live`, `python scripts/frontier/run_openrouter_judges.py --live` |
| Routing benchmarks | `python scripts/routing/build_crossed_pool.py` then `run_crossed_router.py` / `build_mixed_pool.py` + `run_mixed_eval.py` |
| Cross-task feasibility screen | `python scripts/crosstask/run_screening_votes.py` then `analyze_screening.py` |
| Paper figures | `python scripts/figures/paper_figures.py`, `tier_figure.py`, `summary_figure.py` |

Banks and calibration sets are fully specified in `configs/*.yaml`, so a run is reproducible
from `(config, script)` alone. Each stage writes a `manifest.json` recording git commit,
model IDs, dataset hashes, and seeds.

Standing conventions, held across every experiment: seeds `run=20260707`,
`bootstrap=20260706`, `position=20260601`; paired item bootstrap with B=2000 and 95%
percentile intervals; retention matched across methods within a bank; Ledoit-Wolf shrunk
correlation with the listwise/pairwise selection rule in `corrfilter.analysis.dependence`.

### Cluster runs

The GPU set (DPO/GRPO-trained judges, downstream DPO policies, the extended 16-judge bank)
is orchestrated for SLURM — see [`docs/cluster_runbook.md`](docs/cluster_runbook.md):

```bash
bash jobs/newton/setup_env.sh     # venv + deps + stage models
sbatch jobs/newton/smoke_test.slurm
bash jobs/newton/run_all.sh       # CPU stages inline, GPU DAG submitted (resumable)
```

## Tests

```bash
pytest            # 213 tests
ruff check src scripts tests
```

Coverage includes correlation estimation, effective sample size, filtering and matched
retention, consensus and tie policy, adaptive-R, CFI banks/triggers/retention, the routing
module's parity with the original regime routers, the feasibility gates, the API-judge
adapters, and the cost meter.

## Citation

```bibtex
@article{hossain2026agreement,
  title   = {Agreement Is Not Evidence: Measuring Error Dependence and Effective
             Evidence in {LLM} Judge Consensus},
  author  = {Hossain, Elias},
  year    = {2026}
}
```

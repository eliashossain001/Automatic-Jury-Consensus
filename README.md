# CorrFilter

Code for **Agreement Overstates Evidence: Error Dependence in LLM Judge Consensus**.

This repository measures dependent errors in LLM judge banks and studies their
consequences for consensus filtering and evaluation. The central contribution is a
measurement diagnosis and an operational evaluation protocol—not a claim that one new
aggregation rule universally replaces majority voting.

## Main findings

- **Nominal votes overstate independent evidence.** In the primary ten-judge open-weight
  bank, mean error correlation is $\bar\rho=0.206$, corresponding to an effective ensemble
  size of only $n_{\mathrm{eff}}=3.51$.
- **Naively pooling votes changes conclusions.** On 100-item Focus panels, a pooled-vote
  test reports a significant winner unsupported by item-level inference in up to 28% of
  evaluations.
- **Capability does not guarantee diversity.** GPT, Claude, and Grok judges achieve
  91–93% individual accuracy on the shared RewardBench subset, while residual errors have
  $\bar\rho=0.56$ and co-occur at 7.7 times the independence expectation.
- **Preference optimization does not reproduce the frontier pattern.** In the matched
  forced-choice experiment, GRPO reduces mean error correlation from 0.174 to 0.153;
  DPO gives 0.168 and is statistically indistinguishable from the untrained bank.
- **Dependence structure matters.** Controlled global co-failure and vulnerable-subgroup
  interventions produce opposite diagnostic signatures and favor different filters.
  These experiments identify mechanisms; they do not estimate natural prevalence.
- **The measurement is robust to random label error.** Across 40 seeds, randomly flipping
  1–10% of calibration labels leaves the conclusion intact: $\bar\rho=0.208$–0.226 and
  $n_{\mathrm{eff}}=3.48$–3.30, versus 0.206 and 3.51 without flips.
- **The deployment selector is a bounded safeguard.** Under the strict disjoint-panel
  protocol its gain over the best fixed rule is +0.12 precision points with a 95% interval
  of [-0.04, +0.30]. When frozen and transferred to eight natural LLM-AggreFact
  components, it selects plain majority on all 160 splits, avoids transferring a harmful
  Bias-Cluster rule, and recovers 85.5% of per-deployment oracle headroom. It does not beat
  majority or the per-deployment oracle.
- **Retained contamination affects downstream margins.** DPO reward margins decline from
  20% retained contamination onward; downstream preference-accuracy changes at the tested
  scale remain within uncertainty.

## Recommended evaluation protocol

1. Label a representative calibration panel (approximately 100 items in our experiments).
2. Report each judge's accuracy, mean error correlation, and effective ensemble size.
3. Treat the item—not each judge vote—as the unit of statistical inference.
4. Inspect eigenstructure, subgroup disagreement, and leave-cluster-out flips in addition
   to mean correlation.
5. Compare any dependence-aware rule with majority at matched retention. If there is no
   measurable headroom, keep majority.
6. Recalibrate after changing the task or judge bank. Treat learned selection as protection
   against negative transfer, not proof that it improves on majority.

## Repository layout

```text
corrfilter/
├── src/corrfilter/
│   ├── analysis/       Agreement, hypothesis tests, and failure decomposition
│   ├── cfi/            Consensus, CorrFilter, adaptive-R, clusters, metrics
│   ├── correlation/    Error correlation, shrinkage, eigenstructure, n_eff
│   ├── data/           Preference and generic binary-task adapters
│   ├── judges/         HF, Gemini, OpenRouter, and forced-choice judges
│   ├── voting/         Cached vote collection
│   ├── evaluation.py   Tie-safe, retention-matched evaluation primitives
│   ├── filtering.py    Core filtering entry points
│   └── routing.py      Shared deployment-routing primitives
├── configs/            Pinned judge-bank and experiment configuration
├── scripts/            Numbered experiment and analysis drivers
├── experiments/router_upgrade/  Deployment-selector experiment source
├── jobs/newton/        Reproducible SLURM workflows
├── tests/              CPU unit and regression tests
├── pyproject.toml
└── README_NEWTON.md    Cluster setup and execution notes
```

Generated data, vote caches, checkpoints, figures, paper sources, and experiment outputs
are intentionally excluded from Git; the small router-upgrade source package is retained.
Root-level ignore rules do not hide package modules
such as `src/corrfilter/data` or plotting code under `scripts/figures`.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/eliashossain001/corrfilter.git
cd corrfilter
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional: legacy SDK-backed judges and preference-training scripts
pip install -e ".[api,training]"
```

Set provider credentials only when collecting new API votes. Never commit `.env`:

```bash
export HF_TOKEN=...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
```

All API runners default to dry-run or require an explicit live flag; cached-vote analyses
make no paid calls.

## Reproducing the analyses

The numbered scripts follow the experiment chronology:

| Scripts | Purpose |
|---|---|
| `01`–`04` | Build RewardBench calibration data and measure dependence |
| `05`–`09` | Controlled co-failure collection and baseline CorrFilter analysis |
| `10`–`19` | Adaptive correlation, poisoning, subgroup filters, and early routing |
| `20`–`27` | Robustness, confidence intervals, replication, and downstream simulations |
| `28`–`32` | Preference-trained judges and seed/abstention checks |
| `33`–`42` | Frontier banks, non-error dependence, mixed deployments, significance flips, and natural subgroup stability |
| `43`–`52` | Preregistered factuality/code actionability screen, including reported null configurations |
| `53`–`58` | Forced-choice controls, non-inferiority, expanded aggregation, and natural-mechanism analysis |
| `59`–`61` | Strict deployment selector, frozen AggreFact transfer, and final selector figure |

The null and pilot scripts in the cross-task screen are retained because they support the
paper's stated boundary conditions. The superseded selector plotter and internal slide
generators have been removed.

### Core measurement

```bash
python scripts/01_build_calibration_set.py
python scripts/02_run_judges.py --config configs/judge_bank.yaml
python scripts/03_compute_correlation.py
python scripts/04_h1_analysis.py
python scripts/gold_sensitivity.py
python scripts/41_significance_flip.py
```

### Frontier and preference-training analyses

```bash
python scripts/34_gemini_bank_analysis.py
python scripts/40_multiprovider_full_analysis.py
python scripts/53_forced_choice_bank.py
python scripts/54_abstention_control_analysis.py
```

These commands consume cached votes unless their corresponding collection runner is
invoked explicitly.

### Deployment selector and natural transfer

```bash
python scripts/59_deployment_filter_selector.py
python scripts/60_external_natural_selector.py
python scripts/61_make_selector_figure.py
```

Script 59 reconstructs 456 preference deployments with 100 trusted calibration items and
200 disjoint held-out items. Script 60 fits the selector on preference deployments and
freezes it before evaluation on eight LLM-AggreFact components. Repeated external splits
are summarized with uncertainty clustered by component.

### Full cluster workflow

See [README_NEWTON.md](README_NEWTON.md):

```bash
bash jobs/newton/setup_env.sh
sbatch jobs/newton/smoke_test.slurm
bash jobs/newton/run_all.sh
```

## Evaluation safeguards

`src/corrfilter/evaluation.py` is the single source of truth for scored consensus:

- even-bank ties abstain when evaluated against gold;
- fixed tie breaking with constant gold is rejected because it rewards ties by
  construction;
- matched retention is denominated over the evaluable pool;
- non-evaluable items cannot consume top-k retention slots.

Stable BLAKE2b position assignment replaces Python's process-randomized `hash()`. The
regression tests cover both corrections.

## Tests and static checks

```bash
pytest -q
ruff check src scripts experiments/router_upgrade tests
python -m compileall -q src scripts experiments/router_upgrade tests
```

Tests cover correlation estimation, effective size, filtering, consensus, tie handling,
matched retention, judge adapters, stable position assignment, generic factuality tasks,
and deployment routing. Tests that require generated experiment pools skip when those
ignored artifacts are absent.

## Reproducibility and artifacts

- Seeds and model identifiers are pinned in scripts and YAML configuration.
- Vote caches are resumable and separate by logical judge.
- API adapters record abstentions, failures, token use, and model/provider responses.
- Large artifacts remain local under `data/`, `experiments/`, `results/`, and `outputs/`.
- Do not interpret repeated splits over one dataset component as independent deployments;
  external selector uncertainty is clustered over the eight components.

## Citation

```bibtex
@article{hossain2026agreement,
  title  = {Agreement Overstates Evidence: Error Dependence in LLM Judge Consensus},
  author = {Hossain, Elias},
  year   = {2026}
}
```

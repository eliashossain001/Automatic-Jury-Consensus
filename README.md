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

## Folder structure

```text
<repo-root>/
├── README.md
├── README_CLUSTER.md                       # SLURM runbook for the GPU stages
├── pyproject.toml
├── .gitignore
├── .rsync-exclude                          # Excluded from cluster transfers
│
├── configs/                                # Pinned YAML configuration
│   ├── judge_bank.yaml                     # Primary ten-judge open-weight bank
│   ├── judge_bank_extended.yaml            # 16-judge bank, 1.5B-14B
│   ├── judge_bank_gemini.yaml              # Gemini frontier bank
│   ├── judge_bank_openrouter.yaml          # GPT / Claude / Grok bank
│   ├── judge_bank_uf.yaml, judge_bank_pku.yaml
│   ├── calibration.yaml, calibration_pku.yaml
│   ├── cfi.yaml, cfi_bias_prompts.yaml     # Co-failure injection settings
│   ├── grpo_judge_bank.yaml, grpo_judge_bank_hightok.yaml, grpo_seed_bank.yaml
│   ├── dpo_judge.yaml
│   └── mainstream_alignment_integration.yaml
│
├── src/
│   └── corrfilter/                         # Importable Python package
│       ├── analysis/                       # Agreement, hypothesis tests, decomposition
│       ├── cfi/                            # Consensus, CorrFilter, adaptive-R, clusters
│       ├── correlation/                    # Error correlation, shrinkage, n_eff
│       ├── data/                           # Preference and binary-task adapters
│       ├── judges/                         # HF, Gemini, OpenRouter, forced-choice
│       ├── voting/                         # Cached vote collection
│       ├── viz/                            # Shared plotting helpers
│       ├── evaluation.py                   # Tie-safe, retention-matched primitives
│       ├── filtering.py                    # Core filtering entry points
│       └── routing.py                      # Shared deployment-routing primitives
│
├── scripts/                                # Experiment drivers, grouped by stage
│   ├── calibration/                        # 1. Trusted panel and bank votes
│   │   ├── build_calibration_set.py        # Stratified RewardBench v2 panel
│   │   ├── run_judge_bank.py               # Run every judge, cache votes
│   │   ├── sweep_extended_bank.py          # Dependence vs nominal judge count
│   │   ├── evaluate_judge_bank.py          # Trained bank vs matched base bank
│   │   └── write_run_manifest.py           # Provenance manifest for a run
│   │
│   ├── dependence/                         # 2. Core measurement (Sec. 5.1)
│   │   ├── compute_error_correlation.py    # R, n_eff, eigenspectrum, heatmap
│   │   ├── test_diversification_contrasts.py  # Family / prompt-style contrasts
│   │   ├── decompose_eigen_failure_modes.py   # Eigenvectors to item-level modes
│   │   ├── estimate_position_aware_dependence.py  # Position-conditioned R
│   │   ├── replicate_dependence_ultrafeedback.py  # Second-dataset replication
│   │   ├── simulate_oracle_independence.py    # Matched independent-bank control
│   │   ├── analyze_nonerror_correlation.py    # Correct votes correlate too
│   │   ├── test_panel_significance_flip.py    # Pooled vs item-level verdicts
│   │   └── analyze_prompt_length_effect.py    # Length as a dependence covariate
│   │
│   ├── robustness/                         # 3. Does the measurement survive?
│   │   ├── analyze_bank_composition.py     # 73 subbanks of the cached votes
│   │   ├── compute_confidence_intervals.py # Bootstrap and paired-bootstrap CIs
│   │   ├── analyze_gold_label_sensitivity.py  # 1-10% label flips, 40 seeds
│   │   ├── analyze_abstention_robustness.py   # Dependence on committed votes
│   │   ├── compare_training_seeds.py       # Do same-model seeds co-fail?
│   │   ├── find_natural_subgroup_cases.py  # Unpoisoned subgroup signatures
│   │   ├── analyze_natural_subgroup_stability.py  # Stability of that estimate
│   │   ├── extract_failure_examples.py     # Qualitative consensus failures
│   │   └── diagnose_tie_breaking.py        # Even-bank tie-handling diagnostic
│   │
│   ├── regimes/                            # 4. Controlled regime interventions
│   │   ├── build_cofailure_manifest.py     # Per-item trigger annotations
│   │   ├── run_cofailure_votes_replay.py   # Deterministic CPU replay, no GPU
│   │   ├── run_cofailure_votes_gpu.py      # Real biased-prompt re-inference
│   │   ├── compute_cofailure_metrics.py    # Per-item consensus / filter scores
│   │   ├── analyze_cofailure_experiment.py # Agreement-vs-accuracy curves
│   │   ├── plot_cofailure_figures.py       # Figures from the analysis CSVs
│   │   ├── build_poisoned_ultrafeedback.py # Weak-dependence contamination
│   │   ├── run_ultrafeedback_judges.py     # Content-based votes on those pairs
│   │   ├── evaluate_poisoned_ultrafeedback.py  # Filters at matched retention
│   │   ├── run_position_aligned_poisoning.py   # Vulnerable-subgroup regime
│   │   └── discover_natural_mechanisms.py  # Non-position mechanisms, no inference
│   │
│   ├── filters/                            # 5. Aggregation rules as probes
│   │   ├── run_corrfilter_baselines.py     # CorrFilter vs naive baselines
│   │   ├── run_adaptive_r_experiment.py    # How R is estimated in deployment
│   │   ├── validate_direction_randomized_r.py  # Gold-free R under randomization
│   │   ├── run_bias_cluster_filter.py      # Support outside a vulnerable cluster
│   │   ├── learn_bias_cluster.py           # Learn that cluster from ~100 items
│   │   ├── run_corrfilter_plus.py          # Independence-reward variant
│   │   └── approximate_lowrank_corrfilter.py   # Factor approximation of R
│   │
│   ├── frontier/                           # 6. Frontier judge banks (Sec. 5.2)
│   │   ├── run_gemini_judges.py            # Gemini bank, dry-run by default
│   │   ├── analyze_gemini_bank.py          # Gemini vs the open-weight bank
│   │   ├── run_openrouter_judges.py        # GPT / Claude / Grok via OpenRouter
│   │   ├── analyze_openrouter_pilot.py     # Multi-provider pilot analysis
│   │   └── analyze_multiprovider_banks.py  # All banks on identical items
│   │
│   ├── preference_training/                # 7. Does DPO/GRPO cause it? (Sec. 5.3)
│   │   ├── train_judge_grpo.py             # GRPO-train one judge, LoRA
│   │   ├── train_judge_dpo.py              # DPO-train one judge, LoRA
│   │   ├── evaluate_grpo_corrfilter.py     # Co-failure metrics, trained banks
│   │   ├── simulate_grpo_reward_contamination.py  # Offline reward-signal pilot
│   │   ├── run_forced_choice_banks.py      # Matched base/GRPO/DPO forced choice
│   │   └── analyze_forced_choice_banks.py  # The matched dependence comparison
│   │
│   ├── routing_selector/                   # 8. Routing and selector (Sec. 5.5)
│   │   ├── run_regime_router.py            # Classify regime, map to a filter
│   │   ├── run_unified_router.py           # Soft, confidence-weighted routing
│   │   ├── evaluate_routing_accuracy.py    # Routing accuracy vs ground truth
│   │   ├── run_mixed_regime_benchmark.py   # Mixed-regime deployment pool
│   │   ├── test_router_noninferiority.py   # Preregistered non-inferiority test
│   │   ├── train_router_dev.py             # Small-sample dev-set router check
│   │   ├── select_deployment_filter.py     # Strict 456-deployment selector
│   │   └── transfer_selector_to_aggrefact.py   # Frozen transfer, 8 components
│   │
│   ├── cross_task_screen/                  # 9. Feasibility screen, incl. nulls
│   │   ├── run_factuality_judges.py        # Pointwise factuality collection
│   │   ├── report_factuality_pilot.py      # Judge-competence pilot gate
│   │   ├── run_factuality_frontier_judges.py   # Frontier factuality bank
│   │   ├── run_screening_votes.py          # GPU collection for the campaign
│   │   ├── analyze_screening_campaign.py   # Routing-feasibility table
│   │   ├── build_actionability_artifacts.py    # Figure and LaTeX fragments
│   │   ├── run_nei_mechanism_pilot.py      # Preregistered NEI pilot
│   │   ├── collect_bankd_votes.py          # Full bank-D collection, gated
│   │   ├── collect_aggrefact_votes.py      # AggreFact component-level votes
│   │   ├── run_frontier_code_pilot.py      # Pointwise code-judging pilot
│   │   └── run_expanded_aggregation.py     # Expanded aggregation family
│   │
│   ├── downstream/                         # 10. Does contamination reach training?
│   │   ├── build_filtered_dpo_data.py      # Preference files, matched retention
│   │   ├── train_policy_dpo.py             # DPO-train a small policy
│   │   ├── evaluate_policy.py              # Held-out reward accuracy, margins
│   │   ├── summarize_downstream.py         # Paired-CI policy comparison
│   │   └── integrate_reward_model_weights.py   # Filter signals as loss weights
│   │
│   ├── figures/                            # Paper figure and table generators
│   │   ├── make_paper_figures.py           # Shared style plus the main figures
│   │   ├── make_tier_figure.py             # Capability-tier block structure of R
│   │   ├── make_selector_figure.py         # Deployment-selector summary
│   │   ├── make_base_vs_grpo_figure.py     # Base vs GRPO dependence panels
│   │   ├── make_training_summary_figure.py # Base to preference-trained overview
│   │   ├── make_cross_dataset_figure.py    # RewardBench / UltraFeedback / PKU
│   │   ├── make_taxonomy_hero.py           # Dependence-regime taxonomy diagram
│   │   └── make_repro_tables.py            # Paper tables from dependence.csv
│   │
│   ├── lib/                                # Shared helpers, imported by stages
│   │   ├── compute_dependence.py           # Canonical dependence estimator
│   │   ├── bootstrap_ci.py                 # Bootstrap and paired-bootstrap CIs
│   │   └── run_filters.py                  # Matched-retention filter comparison
│   │
│   └── pipelines/                          # Shell drivers chaining the stages
│       ├── reproduce_all.sh                # End-to-end reproduction
│       ├── run_bank_parallel.sh            # Parallel bank vote collection
│       ├── run_cofailure_experiment.sh     # Co-failure stage chain
│       └── watch_and_eval_ultrafeedback.sh # Watch-and-evaluate helper
│
├── experiments/
│   └── router_upgrade/                     # Deployment-selector pool builders
│                                           # (the only experiment package in Git;
│                                           #  other experiment trees are outputs)
│
├── jobs/
│   └── slurm/                              # Reproducible SLURM workflows
│       ├── env.sh                          # Shared env; set SLURM_ACCOUNT here
│       ├── setup_env.sh                    # venv, deps, model staging
│       ├── run_all.sh                      # CPU inline plus GPU DAG submit
│       ├── smoke_test.slurm                # Fail-fast end-to-end sanity check
│       └── *.slurm                         # Per-stage batch jobs
│
└── tests/                                  # CPU unit and regression tests
```

Scripts are grouped by pipeline stage. Directory names are not numbered; the stage table
under [Reproducing the analyses](#reproducing-the-analyses) gives the execution order, and
the comments above repeat it. Every script resolves the repository root as
`Path(__file__).resolve().parents[2]` and is run from the repository root, for example
`python scripts/dependence/compute_error_correlation.py`.

Generated data, vote caches, checkpoints, figures, paper sources, and experiment outputs
are intentionally excluded from Git; the small router-upgrade source package is retained.
Root-level ignore rules do not hide package modules such as `src/corrfilter/data` or the
plotting code under `scripts/figures`.

## Installation

Python 3.10 or newer is required.

```bash
git clone <repository-url>
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

Stage directories are named, not numbered, so the column below gives the order. Later
stages consume the vote caches written by earlier ones, which means an analysis-only rerun
needs nothing but those caches.

| # | Stage | Purpose | Needs a GPU or API key? |
|---|---|---|---|
| 1 | `calibration/` | Build the trusted panel and collect the ten-judge bank's votes | GPU for vote collection |
| 2 | `dependence/` | Measure $\bar\rho$, $n_{\mathrm{eff}}$, eigenstructure; panel significance | No |
| 3 | `robustness/` | Subbanks, bootstrap CIs, label-noise and abstention sensitivity | No |
| 4 | `regimes/` | Controlled global co-failure and vulnerable-subgroup interventions | GPU only for `run_cofailure_votes_gpu.py` |
| 5 | `filters/` | CorrFilter and Bias-Cluster as diagnostic probes | No |
| 6 | `frontier/` | Gemini, GPT, Claude, and Grok judge banks | API key for collection |
| 7 | `preference_training/` | Matched untrained / GRPO / DPO forced-choice comparison | GPU for training |
| 8 | `routing_selector/` | Regime routing and the deployment-level selector | No |
| 9 | `cross_task_screen/` | Preregistered factuality and code screen, including nulls | GPU or API key for collection |
| 10 | `downstream/` | Whether retained contamination reaches DPO training | GPU for training |
| — | `figures/`, `lib/`, `pipelines/` | Figure generators, shared helpers, shell drivers | No |

The null and pilot scripts in the cross-task screen are retained because they support the
paper's stated boundary conditions.

### Core measurement

```bash
python scripts/calibration/build_calibration_set.py
python scripts/calibration/run_judge_bank.py --config configs/judge_bank.yaml
python scripts/dependence/compute_error_correlation.py
python scripts/dependence/test_diversification_contrasts.py
python scripts/robustness/analyze_gold_label_sensitivity.py
python scripts/dependence/test_panel_significance_flip.py
```

### Frontier and preference-training analyses

```bash
python scripts/frontier/analyze_gemini_bank.py
python scripts/frontier/analyze_multiprovider_banks.py
python scripts/preference_training/run_forced_choice_banks.py
python scripts/preference_training/analyze_forced_choice_banks.py
```

These commands consume cached votes unless their corresponding collection runner is
invoked explicitly.

### Deployment selector and natural transfer

```bash
python scripts/routing_selector/select_deployment_filter.py
python scripts/routing_selector/transfer_selector_to_aggrefact.py
python scripts/figures/make_selector_figure.py
```

`select_deployment_filter.py` reconstructs 456 preference deployments with 100 trusted
calibration items and 200 disjoint held-out items. `transfer_selector_to_aggrefact.py`
fits the selector on preference deployments and freezes it before evaluation on eight
LLM-AggreFact components. Repeated external splits
are summarized with uncertainty clustered by component.

### Full cluster workflow

See [README_CLUSTER.md](README_CLUSTER.md):

```bash
bash jobs/slurm/setup_env.sh
sbatch jobs/slurm/smoke_test.slurm
bash jobs/slurm/run_all.sh
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

This repository accompanies a paper currently under double-blind review, so author and
affiliation details are withheld. Please cite the anonymized preprint:

```bibtex
@misc{anon2026agreement,
  title        = {Agreement Overstates Evidence: Error Dependence in LLM Judge Consensus},
  author       = {Anonymous},
  year         = {2026},
  note         = {Under review}
}
```

The citation will be updated with author information once review concludes.

# Script map: old numbered names to the current layout

The pipeline used to be a flat, numbered `scripts/` directory (`01_…` through `58_…`).
It is now grouped by experimental stage, and the modules that several scripts imported
from each other have moved into the installed package. This file is the translation
table, kept so that older logs, notebooks, and the manuscript appendix can be followed.

## Driver scripts

### Calibration set and vote collection — `scripts/calibration/`

| was | is |
|---|---|
| `scripts/01_build_calibration_set.py` | `build_calibration_set.py` |
| `scripts/run_bank_parallel.sh` | `run_bank_parallel.sh` |
| `scripts/02_run_judges.py` | `run_judge_bank.py` |

### Dependence measurement (RQ1) — `scripts/dependence/`

| was | is |
|---|---|
| `scripts/03_compute_correlation.py` | `compute_correlation.py` |
| `scripts/06_eigen_failure_decomposition.py` | `eigen_failure_decomposition.py` |
| `scripts/run_extended_bank.py` | `extended_bank_scaling.py` |
| `scripts/04_h1_analysis.py` | `h1_family_prompt_contrast.py` |
| `scripts/26_independence_simulation.py` | `independence_simulation.py` |
| `scripts/55_length_effect_analysis.py` | `length_effect.py` |
| `scripts/35_nonerror_correlation.py` | `nonerror_correlation.py` |
| `scripts/24_position_aware_dependence.py` | `position_aware.py` |
| `scripts/25_replicate_dependence_uf.py` | `replicate_ultrafeedback.py` |
| `scripts/41_significance_flip.py` | `significance_flip.py` |

### Correlated-failure injection and CorrFilter — `scripts/cfi/`

| was | is |
|---|---|
| `scripts/10_adaptive_r_experiment.py` | `adaptive_r.py` |
| `scripts/11_direction_randomized_adaptive_r.py` | `adaptive_r_direction_randomized.py` |
| `scripts/08_analyze_cfi.py` | `analyze_gpu_bank.py` |
| `scripts/07_analyze_cfi.py` | `analyze_replay.py` |
| `scripts/05_build_cfi_manifest.py` | `build_manifest.py` |
| `scripts/23_corrfilter_plus.py` | `corrfilter_plus.py` |
| `scripts/approx_corrfilter.py` | `lowrank_approximation.py` |
| `scripts/08_plot_cfi.py` | `plot.py` |
| `scripts/07_run_cfi_bank.py` | `run_bank_gpu.py` |
| `scripts/06_run_cfi_votes.py` | `run_bank_replay.py` |
| `scripts/09_run_corrfilter.py` | `run_corrfilter.py` |
| `scripts/run_cfi_experiment.sh` | `run_experiment.sh` |

### Synthetic poisoning attacks — `scripts/attacks/`

| was | is |
|---|---|
| `scripts/12_build_poisoned_uf.py` | `build_poisoned_ultrafeedback.py` |
| `scripts/14_eval_poisoned_uf.py` | `eval_poisoned_uf.py` |
| `scripts/15_position_poisoning.py` | `position_aligned_poisoning.py` |
| `scripts/13_run_uf_judges.py` | `run_poisoned_uf_judges.py` |

### Bias-cluster and subgroup filters (RQ3) — `scripts/filters/`

| was | is |
|---|---|
| `scripts/16_cluster_filter.py` | `bias_cluster.py` |
| `scripts/57_expanded_aggregation_bankB.py` | `expanded_aggregation.py` |
| `scripts/17_learned_bias_cluster.py` | `learned_bias_cluster.py` |
| `scripts/58_natural_mechanism_discovery.py` | `natural_mechanism_discovery.py` |
| `scripts/find_natural_subgroup_cases.py` | `natural_subgroup_cases.py` |
| `scripts/42_natural_subgroup_stability.py` | `natural_subgroup_stability.py` |

### Regime identification and routing (RQ4) — `scripts/routing/`

| was | is |
|---|---|
| `experiments/router_upgrade/build_crossed_pool.py` | `build_crossed_pool.py` |
| `experiments/router_upgrade/build_mixed_pool.py` | `build_mixed_pool.py` |
| `experiments/router_upgrade/extend_pku.py` | `extend_pku.py` |
| `experiments/router_upgrade/frontier_bank_transfer.py` | `frontier_bank_transfer.py` |
| `scripts/56_router_noninferiority.py` | `noninferiority_test.py` |
| `scripts/18_regime_router.py` | `regime_router.py` |
| `experiments/router_upgrade/run_crossed_router.py` | `run_crossed_router.py` |
| `experiments/router_upgrade/run_extended_eval.py` | `run_extended_eval.py` |
| `experiments/router_upgrade/run_mixed_eval.py` | `run_mixed_eval.py` |
| `experiments/router_upgrade/sg_size_study.py` | `sg_size_study.py` |
| `scripts/train_router_dev.py` | `train_dev_router.py` |
| `scripts/19_unified_router_v2.py` | `unified_router.py` |

### Superseded routing evaluations (audit trail) — `scripts/routing/superseded/`

| was | is |
|---|---|
| `experiments/router_upgrade/build_pool.py` | `build_pool.py` |
| `experiments/router_upgrade/build_pool_eqsize.py` | `build_pool_eqsize.py` |
| `scripts/37_mixed_regime_benchmark.py` | `mixed_regime_benchmark.py` |
| `scripts/36_routing_accuracy.py` | `routing_accuracy.py` |
| `experiments/router_upgrade/run_learned_router.py` | `run_learned_router.py` |

### Frontier / multi-provider banks (RQ2) — `scripts/frontier/`

| was | is |
|---|---|
| `scripts/34_gemini_bank_analysis.py` | `analyze_gemini_bank.py` |
| `scripts/40_multiprovider_full_analysis.py` | `analyze_multiprovider.py` |
| `scripts/39_openrouter_pilot_analysis.py` | `analyze_openrouter_pilot.py` |
| `scripts/52_frontier_code_pilot.py` | `code_pilot.py` |
| `scripts/33_run_gemini_judges.py` | `run_gemini_judges.py` |
| `scripts/38_run_openrouter_judges.py` | `run_openrouter_judges.py` |

### Cross-task routing-feasibility campaign (RQ4 boundary) — `scripts/crosstask/`

| was | is |
|---|---|
| `scripts/48_actionability_artifacts.py` | `actionability_artifacts.py` |
| `scripts/51_aggrefact_screen_votes.py` | `aggrefact_screen_votes.py` |
| `scripts/47_analyze_screening.py` | `analyze_screening.py` |
| `experiments/router_upgrade/build_factuality_pools.py` | `build_factuality_pools.py` |
| `scripts/44_factuality_pilot_report.py` | `factuality_pilot_report.py` |
| `scripts/45_run_factuality_frontier.py` | `run_factuality_frontier.py` |
| `scripts/43_run_factuality_judges.py` | `run_factuality_judges.py` |
| `experiments/router_upgrade/run_factuality_transfer.py` | `run_factuality_transfer.py` |
| `scripts/46_run_screening_votes.py` | `run_screening_votes.py` |

### Preference-optimized judges (RQ2) — `scripts/training/`

| was | is |
|---|---|
| `scripts/31_abstention_robustness.py` | `abstention_robustness.py` |
| `scripts/30_base_vs_grpo_figure.py` | `base_vs_grpo_figure.py` |
| `scripts/29_grpo_corrfilter_eval.py` | `corrfilter_vs_consensus.py` |
| `scripts/eval_judge_bank.py` | `eval_judge_bank.py` |
| `scripts/54_abstention_control_analysis.py` | `forced_choice_analysis.py` |
| `scripts/53_forced_choice_bank.py` | `forced_choice_bank.py` |
| `scripts/27_grpo_reward_simulation.py` | `grpo_reward_simulation.py` |
| `scripts/32_seed_contrast.py` | `seed_contrast.py` |
| `scripts/train_judges_dpo.py` | `train_dpo_judge.py` |
| `scripts/28_train_grpo_judge.py` | `train_grpo_judge.py` |

### Downstream DPO policy validation — `scripts/downstream/`

| was | is |
|---|---|
| `scripts/build_filtered_dpo_data.py` | `build_filtered_dpo_data.py` |
| `experiments/downstream_dpo_validation/dpo_common.py` | `dpo_common.py` |
| `scripts/eval_policy.py` | `eval_policy.py` |
| `experiments/downstream_dpo_validation/eval_scaling_cis.py` | `eval_scaling_cis.py` |
| `experiments/downstream_dpo_validation/evaluate_dpo_models.py` | `evaluate_dpo_models.py` |
| `experiments/downstream_dpo_validation/prepare_dpo_datasets.py` | `prepare_dpo_datasets.py` |
| `experiments/downstream_dpo_validation/prepare_scaling_datasets.py` | `prepare_scaling_datasets.py` |
| `experiments/downstream_dpo_validation/summarize_downstream_results.py` | `summarize_dpo_validation.py` |
| `scripts/summarize_downstream.py` | `summarize_policy_comparison.py` |
| `experiments/downstream_dpo_validation/train_dpo_by_filter.py` | `train_dpo_by_filter.py` |
| `scripts/train_policy_dpo.py` | `train_policy_dpo.py` |

### Downstream: contamination scaling — `scripts/downstream/contamination_scaling/`

| was | is |
|---|---|
| `experiments/downstream_dpo_validation/contamination_scaling/aggregate_scaling.py` | `aggregate_scaling.py` |
| `experiments/downstream_dpo_validation/contamination_scaling/make_high_contamination_manifest.py` | `make_high_contamination_manifest.py` |
| `experiments/downstream_dpo_validation/contamination_scaling/run_contamination_scaling.py` | `run_contamination_scaling.py` |

### Downstream: no matched retention — `scripts/downstream/no_matched_retention/`

| was | is |
|---|---|
| `experiments/downstream_dpo_validation/no_matched_retention/aggregate_nmr.py` | `aggregate_nmr.py` |
| `experiments/downstream_dpo_validation/no_matched_retention/prepare_natural_datasets.py` | `prepare_natural_datasets.py` |

### Downstream: position-aligned poisoning — `scripts/downstream/position_aligned_poisoning/`

| was | is |
|---|---|
| `experiments/downstream_dpo_validation/position_aligned_poisoning/aggregate_position.py` | `aggregate_position.py` |
| `experiments/downstream_dpo_validation/position_aligned_poisoning/diagnose_pos10.py` | `diagnose_pos10.py` |
| `experiments/downstream_dpo_validation/position_aligned_poisoning/make_position_poisoned_manifest.py` | `make_position_poisoned_manifest.py` |
| `experiments/downstream_dpo_validation/position_aligned_poisoning/run_position_poisoning.py` | `run_position_poisoning.py` |
| `experiments/downstream_dpo_validation/position_aligned_poisoning/targeted_eval_probe.py` | `targeted_eval_probe.py` |

### Robustness and statistical stability — `scripts/robustness/`

| was | is |
|---|---|
| `scripts/20_bank_robustness.py` | `bank_composition.py` |
| `scripts/21_confidence_intervals.py` | `confidence_intervals.py` |
| `scripts/gold_sensitivity.py` | `gold_sensitivity.py` |
| `scripts/22_mainstream_alignment_integration.py` | `mainstream_alignment_integration.py` |
| `scripts/analysis/d5_tie_break_diagnostic.py` | `tie_break_diagnostic.py` |

### Figures and paper tables — `scripts/figures/`

| was | is |
|---|---|
| `scripts/make_paper_figures.py` | `paper_figures.py` |
| `scripts/analysis/pku_summary_fig.py` | `pku_summary.py` |
| `scripts/extract_failure_examples.py` | `qualitative_examples.py` |
| `scripts/analysis/make_repro_tables.py` | `repro_tables.py` |
| `scripts/figures/make_summary_figure.py` | `summary_figure.py` |
| `scripts/make_taxonomy_hero.py` | `taxonomy_hero.py` |
| `scripts/make_tier_figure.py` | `tier_figure.py` |

## Code promoted into the package

These were imported across script boundaries with `sys.path` manipulation. They are now
ordinary modules, imported as `from corrfilter... import ...` with no path juggling.

| was | is |
|---|---|
| `scripts/analysis/compute_dependence.py` | `src/corrfilter/analysis/dependence.py` |
| `scripts/analysis/bootstrap_ci.py` | `src/corrfilter/analysis/bootstrap_ci.py` |
| `scripts/analysis/run_filters.py` | `src/corrfilter/analysis/filters.py` |
| `experiments/router_upgrade/feasibility_screen.py` | `src/corrfilter/screening/feasibility.py` |
| `experiments/router_upgrade/screening_sources.py` | `src/corrfilter/screening/sources.py` |
| `experiments/router_upgrade/build_crossed_pool.py (generators)` | `src/corrfilter/screening/generators.py` |
| `experiments/router_upgrade/build_mixed_pool.py (generators)` | `src/corrfilter/screening/generators.py` |
| `experiments/router_upgrade/build_pool.py (extract_features)` | `src/corrfilter/screening/features.py` |
| `experiments/router_upgrade/run_learned_router.py (model/CV helpers)` | `src/corrfilter/routing.py` |

## Removed

| file | why |
|---|---|
| `scripts/make_lab_meeting_deck.py`, `scripts/make_slides.py` | presentation builders, no paper artefact |
| `scripts/49_nei_mechanism_pilot.py` | NEI subgroup mechanism, rejected at its preregistered gate |
| `scripts/50_bankd_full_collection.py` | bank-D full collection, code-gated off after that rejection |
| `scripts/uf_watch_and_eval.sh` | one-off watcher for a completed run |
| `ICLR27_EDITABLE_REVISION_SUMMARY.md`, `ROUTER_EXPERIMENT_AUDIT.md` | finished revision records |
| `outputs/P0-0_*.md`, `outputs/POSITION_BLAST_RADIUS.md`, `outputs/REVIEW_RESPONSE.md` | superseded status dumps |
| `reports/` deck sources and `h1_measurement.md` | superseded by the manuscript |

Everything above is recoverable from git history; nothing under `outputs/`,
`results/`, or `experiments/` (the result artefacts themselves) was touched.

## Manuscript appendix

The manuscript no longer names individual scripts: rather than tracking the new paths,
`app:repro-code`, `app:repro-structure`, and the scattered file references were rewritten
to describe stages instead. See [`paper_appendix_edits.md`](paper_appendix_edits.md) for
the exact find/replace text. This table therefore serves the logs and the artefact
directories, not the paper.

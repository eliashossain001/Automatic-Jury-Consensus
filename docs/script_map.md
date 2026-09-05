# Script map: old numbered names to the current layout

The pipeline used to be a flat, numbered `scripts/` directory (`01_…` through `61_…`),
which contained three colliding numbers (two `06_`, two `07_`, two `08_`). It is now
grouped by experimental stage, with descriptive filenames. This file is the translation
table, kept so that older logs, notebooks, and the manuscript appendix can be followed.

Directory names are not numbered; execution order is given by the stage table in the
README. Every script resolves the repository root as `Path(__file__).resolve().parents[2]`
and runs from the repository root.

## Driver scripts


### Calibration set and vote collection — `scripts/calibration/`

| was | is |
|---|---|
| `scripts/01_build_calibration_set.py` | `build_calibration_set.py` |
| `scripts/eval_judge_bank.py` | `evaluate_judge_bank.py` |
| `scripts/02_run_judges.py` | `run_judge_bank.py` |
| `scripts/run_extended_bank.py` | `sweep_extended_bank.py` |
| `scripts/write_manifest.py` | `write_run_manifest.py` |


### Dependence measurement — `scripts/dependence/`

| was | is |
|---|---|
| `scripts/35_nonerror_correlation.py` | `analyze_nonerror_correlation.py` |
| `scripts/55_length_effect_analysis.py` | `analyze_prompt_length_effect.py` |
| `scripts/03_compute_correlation.py` | `compute_error_correlation.py` |
| `scripts/06_eigen_failure_decomposition.py` | `decompose_eigen_failure_modes.py` |
| `scripts/24_position_aware_dependence.py` | `estimate_position_aware_dependence.py` |
| `scripts/25_replicate_dependence_uf.py` | `replicate_dependence_ultrafeedback.py` |
| `scripts/26_independence_simulation.py` | `simulate_oracle_independence.py` |
| `scripts/04_h1_analysis.py` | `test_diversification_contrasts.py` |
| `scripts/41_significance_flip.py` | `test_panel_significance_flip.py` |


### Robustness and statistical stability — `scripts/robustness/`

| was | is |
|---|---|
| `scripts/31_abstention_robustness.py` | `analyze_abstention_robustness.py` |
| `scripts/20_bank_robustness.py` | `analyze_bank_composition.py` |
| `scripts/gold_sensitivity.py` | `analyze_gold_label_sensitivity.py` |
| `scripts/42_natural_subgroup_stability.py` | `analyze_natural_subgroup_stability.py` |
| `scripts/32_seed_contrast.py` | `compare_training_seeds.py` |
| `scripts/21_confidence_intervals.py` | `compute_confidence_intervals.py` |
| `scripts/analysis/d5_tie_break_diagnostic.py` | `diagnose_tie_breaking.py` |
| `scripts/extract_failure_examples.py` | `extract_failure_examples.py` |
| `scripts/find_natural_subgroup_cases.py` | `find_natural_subgroup_cases.py` |


### Dependence-regime interventions — `scripts/regimes/`

| was | is |
|---|---|
| `scripts/08_analyze_cfi.py` | `analyze_cofailure_experiment.py` |
| `scripts/05_build_cfi_manifest.py` | `build_cofailure_manifest.py` |
| `scripts/12_build_poisoned_uf.py` | `build_poisoned_ultrafeedback.py` |
| `scripts/07_analyze_cfi.py` | `compute_cofailure_metrics.py` |
| `scripts/58_natural_mechanism_discovery.py` | `discover_natural_mechanisms.py` |
| `scripts/14_eval_poisoned_uf.py` | `evaluate_poisoned_ultrafeedback.py` |
| `scripts/08_plot_cfi.py` | `plot_cofailure_figures.py` |
| `scripts/07_run_cfi_bank.py` | `run_cofailure_votes_gpu.py` |
| `scripts/06_run_cfi_votes.py` | `run_cofailure_votes_replay.py` |
| `scripts/15_position_poisoning.py` | `run_position_aligned_poisoning.py` |
| `scripts/13_run_uf_judges.py` | `run_ultrafeedback_judges.py` |


### Aggregation rules and filters — `scripts/filters/`

| was | is |
|---|---|
| `scripts/approx_corrfilter.py` | `approximate_lowrank_corrfilter.py` |
| `scripts/17_learned_bias_cluster.py` | `learn_bias_cluster.py` |
| `scripts/10_adaptive_r_experiment.py` | `run_adaptive_r_experiment.py` |
| `scripts/16_cluster_filter.py` | `run_bias_cluster_filter.py` |
| `scripts/09_run_corrfilter.py` | `run_corrfilter_baselines.py` |
| `scripts/23_corrfilter_plus.py` | `run_corrfilter_plus.py` |
| `scripts/11_direction_randomized_adaptive_r.py` | `validate_direction_randomized_r.py` |


### Frontier judge banks — `scripts/frontier/`

| was | is |
|---|---|
| `scripts/34_gemini_bank_analysis.py` | `analyze_gemini_bank.py` |
| `scripts/40_multiprovider_full_analysis.py` | `analyze_multiprovider_banks.py` |
| `scripts/39_openrouter_pilot_analysis.py` | `analyze_openrouter_pilot.py` |
| `scripts/33_run_gemini_judges.py` | `run_gemini_judges.py` |
| `scripts/38_run_openrouter_judges.py` | `run_openrouter_judges.py` |


### Preference-trained judges — `scripts/preference_training/`

| was | is |
|---|---|
| `scripts/54_abstention_control_analysis.py` | `analyze_forced_choice_banks.py` |
| `scripts/29_grpo_corrfilter_eval.py` | `evaluate_grpo_corrfilter.py` |
| `scripts/53_forced_choice_bank.py` | `run_forced_choice_banks.py` |
| `scripts/27_grpo_reward_simulation.py` | `simulate_grpo_reward_contamination.py` |
| `scripts/train_judges_dpo.py` | `train_judge_dpo.py` |
| `scripts/28_train_grpo_judge.py` | `train_judge_grpo.py` |


### Regime routing and the deployment selector — `scripts/routing_selector/`

| was | is |
|---|---|
| `scripts/36_routing_accuracy.py` | `evaluate_routing_accuracy.py` |
| `scripts/37_mixed_regime_benchmark.py` | `run_mixed_regime_benchmark.py` |
| `scripts/18_regime_router.py` | `run_regime_router.py` |
| `scripts/19_unified_router_v2.py` | `run_unified_router.py` |
| `scripts/59_deployment_filter_selector.py` | `select_deployment_filter.py` |
| `scripts/56_router_noninferiority.py` | `test_router_noninferiority.py` |
| `scripts/train_router_dev.py` | `train_router_dev.py` |
| `scripts/60_external_natural_selector.py` | `transfer_selector_to_aggrefact.py` |


### Cross-task feasibility screen — `scripts/cross_task_screen/`

| was | is |
|---|---|
| `scripts/47_analyze_screening.py` | `analyze_screening_campaign.py` |
| `scripts/48_actionability_artifacts.py` | `build_actionability_artifacts.py` |
| `scripts/51_aggrefact_screen_votes.py` | `collect_aggrefact_votes.py` |
| `scripts/50_bankd_full_collection.py` | `collect_bankd_votes.py` |
| `scripts/44_factuality_pilot_report.py` | `report_factuality_pilot.py` |
| `scripts/57_expanded_aggregation_bankB.py` | `run_expanded_aggregation.py` |
| `scripts/45_run_factuality_frontier.py` | `run_factuality_frontier_judges.py` |
| `scripts/43_run_factuality_judges.py` | `run_factuality_judges.py` |
| `scripts/52_frontier_code_pilot.py` | `run_frontier_code_pilot.py` |
| `scripts/49_nei_mechanism_pilot.py` | `run_nei_mechanism_pilot.py` |
| `scripts/46_run_screening_votes.py` | `run_screening_votes.py` |


### Downstream effects on training — `scripts/downstream/`

| was | is |
|---|---|
| `scripts/build_filtered_dpo_data.py` | `build_filtered_dpo_data.py` |
| `scripts/eval_policy.py` | `evaluate_policy.py` |
| `scripts/22_mainstream_alignment_integration.py` | `integrate_reward_model_weights.py` |
| `scripts/summarize_downstream.py` | `summarize_downstream.py` |
| `scripts/train_policy_dpo.py` | `train_policy_dpo.py` |


### Figures and paper tables — `scripts/figures/`

| was | is |
|---|---|
| `scripts/30_base_vs_grpo_figure.py` | `make_base_vs_grpo_figure.py` |
| `scripts/analysis/pku_summary_fig.py` | `make_cross_dataset_figure.py` |
| `scripts/make_paper_figures.py` | `make_paper_figures.py` |
| `scripts/analysis/make_repro_tables.py` | `make_repro_tables.py` |
| `scripts/61_make_selector_figure.py` | `make_selector_figure.py` |
| `scripts/make_taxonomy_hero.py` | `make_taxonomy_hero.py` |
| `scripts/make_tier_figure.py` | `make_tier_figure.py` |
| `scripts/figures/make_summary_figure.py` | `make_training_summary_figure.py` |


### Shared helpers — `scripts/lib/`

| was | is |
|---|---|
| `scripts/analysis/bootstrap_ci.py` | `bootstrap_ci.py` |
| `scripts/analysis/compute_dependence.py` | `compute_dependence.py` |
| `scripts/analysis/run_filters.py` | `run_filters.py` |


### Shell drivers — `scripts/pipelines/`

| was | is |
|---|---|
| `scripts/reproduce_all.sh` | `reproduce_all.sh` |
| `scripts/run_bank_parallel.sh` | `run_bank_parallel.sh` |
| `scripts/run_cfi_experiment.sh` | `run_cofailure_experiment.sh` |
| `scripts/uf_watch_and_eval.sh` | `watch_and_eval_ultrafeedback.sh` |


## Notes

`scripts/analysis/` was renamed to `scripts/lib/`. Its modules are still imported by the
stage scripts through a `sys.path` insertion rather than as package modules, so their
import names are unchanged.

Nothing was deleted in this reorganization: every script that existed under the numbered
layout is present above. Files removed in earlier cleanups remain recoverable from git
history.

## Manuscript appendix

See [`paper_appendix_edits.md`](paper_appendix_edits.md) for the appendix text that
describes the repository structure. That file was written against a different
reorganization of this repository and its paths should be checked against the table above
before use.

# ICLR 2027 Phase 2 Implementation Plan

> **Note on paths.** Script paths in this file refer to a superseded `scripts/`
> layout. See [`script_map.md`](script_map.md) for the current locations.

Paper: *Agreement Is Not Evidence: Measuring Error Dependence and Effective Evidence in LLM Judge
Consensus* (`main.tex`).
Prepared: 2026-08-11. **Planning only. Nothing in this document has been run. No experiment starts
without explicit approval.**

Companion documents: `docs/reviewer_risk.md` (what each experiment retires) and
`outputs/dpo_judges/D5_RESOLUTION.md` (the artifact cleanup that produced P0-0).

> Written before the repository was reorganised. Script numbers below (`53`-`62` and the
> planned `55`-`60` names) refer to the old flat `scripts/NN_*.py` layout; see
> [`script_map.md`](script_map.md) for where each one now lives. The experiments,
> hypotheses, gates, and decision rules are unaffected.

---

## 0. Scope, resources, and standing conventions

**Hardware.** 2x NVIDIA TITAN RTX (24 GB each), 4-bit `nf4`, one 7-9B judge resident per card
(`app:repro-hw`). All measurement, filtering, routing, and CI stages are CPU-only over cached
parquet votes; GPU cost is incurred once per vote-generation pass.

**Paid inference.** OpenRouter (GPT-5.6-sol, Claude Opus 5, Grok 4.5) and Gemini (3.1-pro,
3.6-flash, 3.5-flash-lite), pinned IDs, verified from response metadata. Calibration anchor for
costing: the frontier code pilot was 60 items x 6 judges for \$2.22 (OpenRouter) plus an estimated
\$0.2-0.4 (Gemini), i.e. **~\$0.007 per judge-item**.

**Standing conventions** (do not vary across Phase 2 unless an experiment says so):
seeds `run=20260707`, `bootstrap=20260706`, `position=20260601`; paired item bootstrap with
B=2000 and 95% percentile intervals; retention matched across methods within a bank;
`retention_mode="affirmed"` (the convention already used by `feasibility_screen.py`, and the one
that avoids the degeneracy documented in D5); Ledoit-Wolf shrunk correlation with the
listwise/pairwise estimator-selection rule of `compute_dependence.estimate_R`.

**Pre-registration.** Each experiment below states its hypothesis, gates, and decision rule. Write
the pre-registration file (`experiments/<name>/PREREG.md`) and commit it **before** the first
result is inspected. Every experiment states what happens if the hypothesis fails, and in no case
is the answer "drop the experiment".

**New script numbering.** Existing scripts run `01`-`52`. Phase 2 claims `53`-`62`.

**Dependency and sequencing.**

```
P0-0 (code fix, CPU, days)  ──►  P0-1 (GPU, ~1 week)
        │                              │
        └──────────────►  P1-4 (CPU)   │
                              │        ▼
P0-2 Phase A (CPU) ──► P0-2 Phase B (GPU, conditional)
        │
        └──────────────►  P0-3 screen (paid, cheap) ──► P0-3 full (paid, conditional)
                              ▲
P1-5 (CPU) ───────────────────┘   (P1-5 supplies the extra filters P0-3's headroom gate tests)
```

P0-0 gates everything, because P0-1 and P1-5 both re-use the FRR-gain machinery it fixes.
P1-4 and P1-5 are CPU-only and can run concurrently with any GPU or API work.

---

## P0-0. Fix the tie-break / retention defect in the shared evaluation path

**Newly added**, arising from the D5 resolution. Not in the original priority list, but it is a
prerequisite: P0-1 and P1-5 both compute paired FRR gains through the same code.

**Hypothesis.** None. This is a correctness fix, not a measurement.

**Defect.** `majority_consensus(V, M)` defaults to `tie_break=1` while callers on the RewardBench
calibration manifest pass `gold = 1` for every item, so split-vote items are correct by
construction. Separately, `paired_frr_gain` computes `k = round(retention * n_manifest_items)`
rather than on the evaluable pool. Full trace and quantification in
`outputs/dpo_judges/D5_RESOLUTION.md`.

**Protocol.**
1. Add `tie_policy` to `majority_consensus` with values `{"abstain", "fixed"}`, defaulting to
   `"abstain"`. Keep `tie_break` for backward compatibility but raise if `tie_policy="fixed"` is
   combined with a constant `gold` vector, which is the exact circularity that caused D5.
2. Add `retention_mode` to `paired_frr_gain` matching `feasibility_screen.py`'s existing
   `"affirmed"` semantics; default it to the evaluable pool.
3. Re-run the affected analyses and diff every number: `scripts/training/eval_judge_bank.py`,
   `corrfilter.analysis.filters`, `results/grpo_judges/filtering/method_comparison.csv`,
   `results/grpo_judges/corrfilter_vs_consensus_paired_ci.csv`.
4. Audit the other 18 call sites of `majority_consensus` for the constant-gold pattern and record
   each as affected or unaffected. Known unaffected: `scripts/filters/bias_cluster.py` and
   `scripts/filters/learned_bias_cluster.py` (gold is `np.where(poisoned, 0, 1)`, not constant).

**Datasets / bank / inference.** RewardBench v2 calibration (1,195 items), existing caches.
**No inference, no training.**

**Metrics.** Every re-derived number, side by side with its published value, with the paired
bootstrap CI recomputed under the corrected protocol.

**Statistical tests.** Paired item bootstrap, B=2000, seed 20260706 (unchanged, so differences are
attributable to the protocol and not to resampling).

**Compute.** CPU only, under one hour.

**Files.** `src/corrfilter/cfi/consensus.py`, `corrfilter.analysis.bootstrap_ci`,
`tests/test_filtering.py` (add tie-policy and retention-mode cases),
`outputs/dpo_judges/`, `results/grpo_judges/filtering/`, `main.tex` Table `tab:dpo` and §7.2,
plus a new `outputs/P0-0_PROTOCOL_DIFF.md`.

**Success criteria.** Every published number is either reproduced within bootstrap noise or
explicitly re-reported with the corrected value and a note. The audit of all 18 call sites is
complete with a written verdict for each.

**How either result changes the paper.**
- *Most numbers unchanged* (expected): add a short appendix paragraph on tie handling and matched
  retention. Reviewer-facing risk D5 closes fully.
- *GRPO gain loses significance* (already measured: +4.20 [2.87, 5.64] becomes +3.17 [-0.19, 6.57]):
  take option (c) from the resolution document and drop the CorrFilter-gain column from Table 4,
  letting §7.2 rest on the dependence increase, which the Phase 1 revision already made
  load-bearing. Update the third contribution bullet, which currently cites "-4.2/-6.0
  false-retention points", to the DPO figure alone or to the tie-free pair.
- *A further number moves materially*: report it. This is a correctness fix; the corrected value
  is the value.

---

## P0-1. Forced-choice abstention control for the preference-training route

**Priority: highest.** Retires risk B7, the weakest evidential link in Discovery 2.

**Hypothesis.** H1: preference optimization (GRPO and DPO) increases inter-judge error dependence
relative to a matched untrained bank, and this increase is **not** an artifact of the trained
judges' elevated abstention rate. Formally, under a decoding protocol in which abstention is
structurally impossible, `rho_bar(GRPO) > rho_bar(base)` and `rho_bar(DPO) > rho_bar(base)`, with
paired bootstrap CIs on the differences excluding zero.

**Why the current evidence is insufficient.** The published comparison has the trained banks
abstaining on 43% of items against 2% for base. That is three confounds at once: (i) the
correlation is measured on a self-selected item subset; (ii) the estimator differs by construction
(`listwise_ledoit_wolf` for base, `pairwise_complete_shrunk` for the trained banks, per
`estimate_R`'s 0.70 listwise floor); (iii) committed-vote accuracy is selection-affected. The
existing mitigation, a 4x decode-budget re-run, only moves abstention from 43% to 40%
(`results/grpo_judges/abstention_robustness/abstention_robustness_summary.csv`). It never matches.

**Protocol.**

*Arm A (primary): forced-choice verdict scoring.* Replace free generation with a scored forward
pass. For each item and judge, render the existing prompt template unchanged, then read the
next-token logits and compare the summed log-probability of the verdict tokens for option A
against option B (for `likert`, the two score-comparison continuations). The vote is the argmax.
Abstention becomes structurally impossible, so every bank yields a complete 1,195 x n vote matrix,
`listwise_ledoit_wolf` applies to all banks, and (i), (ii), (iii) are eliminated together. This is
a strictly stronger control than rate-matching.

*Banks (all three re-run under Arm A, this is the point).*
- base matched 6: qwen-2.5-7b, mistral-7b-v0.3, phi-3.5-mini x {pairwise, likert},
  `configs/judge_bank.yaml`.
- GRPO 6: `configs/grpo_judge_bank.yaml`, adapters `results/grpo_judges/adapters/*_seed0`.
- DPO 6: `configs/dpo_judge.yaml`, adapters `results/strengthening_phase/dpo_judges/adapters/*_seed0`.
- Optional extension if Arm A is cheap in practice: the full open 10-judge bank, which would let
  Table 1 Panel B be reported on the same footing as Panel A.

*Controls.* Identical prompts, identical position assignment (`position_seed=20260601`, so the same
slot per item as the cached runs), identical `max_length=1024`, greedy/deterministic throughout.
Verify the forced-choice protocol against the cached votes on the base bank, where abstention is
only 1.9%: agreement between forced-choice and generated votes on committed items should be high,
and this agreement rate is itself a reported diagnostic. **If that agreement is below 0.95, the
forced-choice protocol is measuring something different from the published one and Arm A must be
reported as a separate protocol rather than as a correction.**

*Arm B (secondary, no inference, run first as a cheap preview).* Restrict all three banks to the
intersection of items on which every bank committed under the original protocol (roughly 600 items,
from the 815/817 evaluable counts) and recompute the dependence block there. This is a matched-item
but still selection-affected comparison; it is a preview, not a substitute for Arm A.

**Datasets.** RewardBench v2 calibration, 1,195 items, unchanged manifest.

**Required inference / training.** Inference only. **No training**; the existing LoRA adapters are
reused as-is. 3 banks x 3 base models x 2 prompt styles x 1,195 items = 21,510 scored forward
passes, plus 11,950 more if the 10-judge extension is approved.

**Metrics.** `rho_bar`, `n_eff`, `n_eff_eig`, mean per-judge accuracy (now on a complete matrix, so
finally like-for-like), conditional co-failure and lift, majority accuracy **and** majority tie
rate (which also fills the `--` cells in Table 1 Panel B), family and prompt contrasts.

**Statistical tests.** Paired item bootstrap (B=2000, seed 20260706) on `Delta rho_bar` and
`Delta n_eff` for GRPO-minus-base and DPO-minus-base. Because forced choice changes the marginal
error rates, also report **Yule's Q** per pair, the marginal-insensitive check already used in
§7.3, and a permutation test over judge-pair labels for the block difference. Pre-register the
primary endpoint as `Delta rho_bar` with a two-sided 95% CI.

**Compute budget.** GPU only, no API spend. A scored forward pass replaces an 8-token generation,
so per-item cost should be at or below the original calibration pass, which covered 1,195 items x
up to 10 logical judges on the same hardware. Budget **1-2 days of wall clock across both cards**
for the three 6-judge banks, plus one day if the 10-judge extension is approved. Adapter loading,
not compute, is likely to dominate; load each base model once and sweep its adapters and prompt
styles, as `scripts/cfi/run_bank_gpu.py` already does.

**Files to modify / create.**
- New: `src/corrfilter/judges/forced_choice.py` (verdict-token scoring, additive; do not change the
  existing generation path), `scripts/training/forced_choice_bank.py` (runner, resumable, caches to a
  separate `votes_dir` so the published caches are never overwritten),
  `scripts/training/forced_choice_analysis.py`, `experiments/abstention_control/PREREG.md`.
- Outputs: `outputs/abstention_control/{bank_metrics.csv, protocol_agreement.csv, summary.md}`.
- Paper: §7.2, Table `tab:dpo`, Table `tab:headline` Panel B, §1 Discovery 2, contribution bullet 2,
  App. D, App. U.
- Do **not** modify `configs/*.yaml` in place; add `configs/*_forced_choice.yaml` variants.

**Success / failure criteria.**
- **Confirmed**: both `Delta rho_bar` CIs exclude zero and are positive. Discovery 2's training
  route is established under the strongest available control.
- **Partially confirmed**: DPO holds, GRPO does not (plausible, given DPO's much larger published
  effect, 0.326 vs 0.235). Then the claim narrows to "DPO increases dependence; GRPO's effect is
  not resolvable at this sample size", and the abstract changes from "raises rho_bar from 0.18 to
  0.24 (GRPO) and 0.33 (DPO)" accordingly.
- **Refuted**: neither CI excludes zero, or the sign reverses. Then the published increase was an
  abstention artifact.

**How either result changes the paper.** Confirmation retires B7 and lets Discovery 2 keep both
routes, and it additionally removes the estimator confound and fills Table 1's missing cells, so it
is worth running even under the confirmed branch. Refutation is a substantial revision: Discovery 2
becomes the frontier-capability result alone, the "two routes" framing in the intro and contribution
bullet 2 collapses to one, and the "new, to our knowledge" novelty claim for the training route is
withdrawn. **We commit in advance to reporting the refuted branch in the main body, not the
appendix.** The paper survives it: Discoveries 1 and 3 are untouched, and a negative
preference-training result is itself informative given how natural the intervention is.

---

## P0-2. Natural, non-position vulnerable subgroup

**Priority: highest.** Retires risk C4 and a substantial part of C1, the "your regimes are
synthetic and your one real vulnerability is position bias" objection.

**Hypothesis.** H2: at least one shared judge weakness other than position bias, measurable from
natural model outputs without any injection, produces the vulnerable-subgroup signature, namely
(a) a judge subgroup C whose errors concentrate on trigger-firing items, (b) majority verdicts on
those items that flip to correct when C is removed, and (c) the **diagnostic inversion**: mean error
correlation measured on trigger-firing items is *lower* than on non-firing items, while precision on
them is worse.

The inversion, (c), is the discriminating prediction. Any shared weakness produces (a) and (b) to
some degree; only the vulnerable-subgroup regime produces (c). Pre-register (c) as the primary
endpoint.

**Candidate mechanisms**, in the order we will test them, all with detectors that already exist in
`src/corrfilter/cfi/triggers.py`: verbosity (`detect_verbosity_trigger`, length ratio >= 0.5),
sycophancy (`detect_sycophancy_trigger`), refusal/safety (`detect_refusal_trigger`, Safety subset
and refusal lexicon), surface polish (`detect_polish_trigger`). Testing four mechanisms requires a
multiplicity correction; see below.

**Protocol.**

*Phase A: discovery from cached votes. No inference at all.*
1. For each mechanism m and judge j, estimate a signed bias
   `b_j(m) = P(j affirms the trigger-favoured side | trigger fires) - P(j affirms it | trigger absent)`,
   with a per-judge item bootstrap CI. This is the direct analogue of the per-judge slot gap that
   made position bias the paper's "strongest measured" weakness, so the mechanisms become
   **comparable on one scale**, which is itself a paper-worthy table.
2. Rank mechanisms by `max_j |b_j(m)|` and by the spread of `b_j(m)` across judges. A vulnerable
   subgroup requires *spread*, not a uniformly shared bias: a bias every judge has equally raises
   correlation (Regime I), whereas a bias only some judges have splits the bank (Regime III).
3. For the top-ranked mechanism, form C as the k=5 judges with the largest `b_j(m)`, exactly
   mirroring the H1-imported construction of App. `app:cluster`, and test (a), (b), (c).
4. Replicate on all three datasets with cached votes: RewardBench v2 (1,195), UltraFeedback
   (2,000), PKU-SafeRLHF (800). Refusal/safety is expected to be strongest on PKU.
5. Stability sweep over cluster size k in {3,4,5,6} and over clustering method (threshold on
   `b_j(m)` vs hierarchical on the trigger-conditional error correlation), reusing
   `scripts/filters/natural_subgroup_stability.py`'s sweep design.

*Phase B: validation. Conditional on Phase A finding a candidate.* Build a trigger-stratified
evaluation pool (roughly 1,000 items, balanced trigger-firing vs matched non-firing on the same
prompts where possible) and run the standard 10-judge bank on it, then test whether the
**bias-cluster filter** recovers precision on the natural subgroup as it does on the synthetic one
(+4.8 points at 10% contamination). This is the result that would convert C4 from PARTIAL to
ANSWERED.

**Datasets.** Phase A: existing caches only. Phase B: a new stratified pool drawn from the same
three sources, so no new data collection, only new judging.

**Required inference / training.** Phase A: **none**. Phase B: one 10-judge bank pass over ~1,000
items, no training.

**Metrics.** Per-judge bias `b_j(m)` with CIs; cluster spread; trigger-conditional `rho_bar` and
`n_eff` versus non-trigger `rho_bar` and `n_eff` (the inversion); majority precision on
trigger-firing items; leave-cluster-out flip rate; bias-cluster filter precision and false retention
at matched retention against naive majority, supermajority, and CorrFilter.

**Statistical tests.** Paired item bootstrap (B=2000) for the trigger-vs-non-trigger `rho_bar`
difference and for the filter precision difference. Because four mechanisms x three datasets are
screened, apply **Holm-Bonferroni across the 12 primary tests** and report both raw and adjusted
intervals. Pre-register the family.

**Compute budget.** Phase A: CPU, a few hours. Phase B: one bank pass, comparable to a fraction of
the original calibration pass; budget **1-2 days wall clock**. No API spend.

**Files to modify / create.**
- New: `scripts/55_natural_bias_discovery.py` (Phase A),
  `scripts/56_natural_subgroup_validation.py` (Phase B),
  `experiments/natural_subgroup_v2/PREREG.md`.
- Reuse without modification: `src/corrfilter/cfi/triggers.py`, `src/corrfilter/filtering.py`,
  `scripts/filters/natural_subgroup_stability.py`.
- Outputs: `outputs/natural_bias/{per_judge_bias.csv, mechanism_ranking.csv, inversion_tests.csv,
  summary.md}`.
- Paper: §7.7 (Regime III), App. `app:threats-position` and §8 Limitations (ii), which currently
  concede exactly this gap; App. `app:cluster`.

**Success / failure criteria.**
- **Success**: at least one non-position mechanism shows the inversion (c) with a Holm-adjusted CI
  excluding zero on at least two of three datasets, and the bias-cluster filter recovers precision
  in Phase B.
- **Partial**: (a) and (b) hold but the inversion (c) does not. The mechanism is then a shared
  weakness in the **global co-failure** regime, not a vulnerable subgroup. This is a real finding
  and is reportable: it would show that position bias is unusual in *splitting* the bank, which
  sharpens rather than weakens the taxonomy.
- **Failure**: no mechanism reaches significance on any dataset. Then position bias remains the sole
  validated instance and Limitation (ii) stands as written, but strengthened by a **measured
  negative**: we can then say we searched four candidate mechanisms across three datasets on one
  common scale and found none, which is far better than the current silence.

**How either result changes the paper.** Success promotes Regime III from "constructed, with
existence cases" to "observed in natural data", which is the single largest available improvement
to the paper's ecological validity, and would move the natural-subgroup material from App. D into
§7.7. Partial or failure converts §8 Limitation (ii) from an admission into a bounded, evidenced
claim. Note that the per-judge bias table from Phase A is worth publishing **in every branch**,
because it is the first like-for-like comparison of shared judge weaknesses on a single scale.

---

## P0-3. One successful external / cross-task validation setting

**Priority: highest.** Converts the paper's largest negative result into a bounded positive one,
or, failing that, hardens the boundary claim.

**Hypothesis.** H3: there exists a judge task outside preference filtering that simultaneously
satisfies the four preregistered gates (competence, error support, valid regimes, aggregation-rule
headroom), and on which a regime router is not significantly worse than a regime oracle while
beating the best transferred fixed filter.

**What the existing screen already tells us.** Eleven candidates were rejected, and the two
near-misses are informative rather than random (`tab:screenall`):
- `vitc-bankD-only` (procedure-diverse factuality bank): competence yes, **global regime yes**,
  headroom 0.90 points yes, and it fails **only** the subgroup gate. It is one valid subgroup regime
  away from passing.
- `vitc-bankB-mix` (capability-tier-mixed bank): headroom 1.11 points, the highest in the campaign,
  yet neither CorrFilter (-0.00) nor the bias-cluster filter (-1.49) captures any of it. The binding
  constraint there is the **filter family**, which is precisely what P1-5 addresses.

The search is therefore not blind. It has two specific leads and one structural hypothesis: the
subgroup gate is the binding constraint on factuality banks, and the filter family is the binding
constraint on tier-mixed banks.

**Protocol.**

*Stage 1: targeted re-screen of the two near-misses (cheap, cached votes).*
Apply P0-2's Phase A discovery method to `vitc-bankD-only` and `vitc-bankB-mix`: search for a
**natural** procedure-cluster or tier-cluster co-failure, rather than requiring the synthetic
subgroup generator to produce one. If a natural subgroup exists in the procedure-diverse bank, the
subgroup gate passes and the candidate proceeds. Feed P1-5's stronger aggregation rules into the
headroom computation for the tier-mixed bank.

*Stage 2: screen new task families (paid, cheap).* Screen with the existing
`corrfilter.screening.feasibility` on 200-item samples. Candidate families, chosen
for plausibly landing in the intermediate dependence region rather than at either boundary:
1. **Math verification** with a frontier bank (step-level correctness on contested solutions).
   Rationale: the cross-task screen found code errors near-independent at the open tier and
   high-dependence at the frontier tier; math may sit between.
2. **Summarisation faithfulness** (AggreFact was gated at campaign time; retry access, else
   substitute a comparable faithfulness source).
3. **Instruction-constraint checking** (verifiable multi-constraint compliance). Rationale: RewardBench
   `Precise IF` is the subset where the paper's own panel test found essentially no margin, which
   suggests genuine judge disagreement, and disagreement is what headroom requires.
4. **Multilingual preference**, as a controlled shift that keeps the task but changes the judges'
   shared blind spots.
5. **Safety/harmlessness pairwise on a second source**, given PKU already passes competence and
   headroom (0.63) and fails only on regime validity.

*Stage 3: full experiment on any gate-passing candidate.* Router versus regime-oracle versus best
transferred fixed filter, the identical comparison run in §7.7, with leave-one-source-out transfer.

**Preregistered gates.** Unchanged from the campaign, so results are directly comparable and the
screen cannot be accused of moving goalposts: competence (majority accuracy > chance + 0.05), error
support (>= 15 majority-wrong items), valid global regime, valid subgroup regime, headroom (mean
H >= 0.5 points, or >= 20% of instances nonzero, or a regime-matched filter >= 0.5 points in its
intended regime), and filter variation. **One documented change**: the filter family `F` is extended
with P1-5's rules, which makes the headroom gate *easier* to pass and must therefore be disclosed
explicitly, with the original 4-filter headroom reported alongside.

**Datasets.** VitaminC and HaluEval pools already built (`outputs/factuality_bank/`,
`outputs/screening_bank/`); new pools for Stage 2 candidates, ~200 items each for screening and
~1,000 for a full run.

**Judge bank.** Frontier six (three OpenRouter, three Gemini) for competence, plus the open 10 for
tier mixing, matching the campaign's bank definitions.

**Required inference / training.** API inference only; no training. The router is logistic
regression on unsupervised diagnostics, CPU, as in §7.7.

**Metrics.** The screening block (competence, coverage, `rho_err`, `n_eff/n`, vote-pattern
diversity, majority-wrong count, per-instance headroom `H_d`) and, at Stage 3, router versus
regime-oracle versus fixed-filter precision at matched retention.

**Statistical tests.** Per-instance paired bootstrap over deployment instances (B=2000) for the
router-minus-oracle and router-minus-fixed differences, as in Tables `tab:rvsro` and `tab:lodo`. The
router-versus-regime-oracle comparison is an **equivalence** question, so pre-register a TOST-style
reading with the equivalence margin set to the campaign's 0.5-point practical threshold, rather than
reading a non-significant difference as equivalence.

**Compute / API budget.** Stage 1: CPU, free. Stage 2: 5 candidates x 200 items x 6 frontier judges
x ~\$0.007 = **~\$42**; budget **\$75** with retries and dev iteration. Stage 3 (conditional):
1,000 items x 6 judges = **~\$42**; budget **\$60**. **Total ceiling \$150**, and Stage 3 spends
nothing unless a candidate passes.

**Files to modify / create.**
- New: `scripts/57_screen_new_task_families.py`, `scripts/58_natural_subgroup_in_screened_banks.py`,
  `experiments/crosstask_v2/PREREG.md`.
- Reuse: `corrfilter.screening.feasibility`, `corrfilter.screening.sources`,
  `build_factuality_pools.py`, `scripts/crosstask/actionability_artifacts.py` (regenerates
  `fig_actionability.png` and `tab:screenall`).
- Paper: §7.9, §8, App. P, Figure `fig:actionability`, Table `tab:screenall`, and the abstract's
  closing sentence.

**Success / failure criteria.**
- **Success**: >= 1 candidate passes all four gates and Stage 3 shows the router within the
  equivalence margin of the regime oracle while beating the best transferred fixed filter with a CI
  excluding zero.
- **Partial**: a candidate passes the gates but Stage 3 reproduces the preference-setting result
  (router matches the oracle, does not beat the fixed filter). This still establishes that the gates
  are satisfiable outside preference filtering, which is the harder half of the claim.
- **Failure**: no candidate passes. The screen grows from 14 to ~20 configurations, the boundary
  claim is made on a larger and more diverse base, and §7.9's negative result strengthens.

**How either result changes the paper.** Success replaces "successful general cross-task routing is
not established by this paper" with a scoped positive claim and changes both the abstract's last
substantive sentence and §8's third bound. Partial keeps the routing claim scoped as-is but removes
the "no task family even qualifies" weakness. Failure is publishable as written; the Phase 1 framing
already presents the campaign as a boundary study, so nothing needs rewriting, only extending. This
is the experiment with the **highest variance in outcome and the lowest cost**, which is why it is
P0 despite being the least likely to succeed.

---

## P1-4. Dependence-estimation sample efficiency and calibration stability

Retires risks D4 and App. U.3 (calibration dependence), and produces a practitioner-facing result
the paper currently lacks.

**Hypothesis.** H4: the dependence quantities differ sharply in how many trusted labels they need.
Specifically, `rho_bar` and `n_eff` are stable at small m (tens of labels), whereas the eigenspace
and the bias-cluster identity require substantially more, so the paper's *measurement* claims are
cheap to reproduce while its *adaptive filtering* claims are not. If true, this is a cleaner and
more useful statement than the current "gains degrade gracefully".

**Protocol.**
1. Subsample the calibration set at m in {25, 50, 100, 200, 400, 800, 1133}, with 50 seeds per m,
   stratified by RewardBench subset. Re-estimate R at each draw.
2. Estimand stability: bias and variance of `rho_bar`, `n_eff`, `n_eff_eig` against the full-data
   values.
3. Eigenspace stability: principal angles between the top-k eigenspace of `R_m` and of `R_full`,
   k in {1,2,3}, plus the eigenvalue share of the top component.
4. Cluster stability: adjusted Rand index between the k=5 bias cluster from `R_m` and from
   `R_full`, and the overlap with the H1-imported cluster, extending the existing
   `outputs/learned_bias_cluster/cluster_overlap_analysis.csv` (which already sweeps
   calibration sizes 25/50/100/200 but only reports Jaccard against H1).
5. Downstream conclusion stability: at each m, does the regime call flip, does the filter ranking
   flip, and does the sign of the CorrFilter gain flip? This is the endpoint practitioners care
   about, and it is the one the paper currently does not report.
6. Estimator comparison at small m: Ledoit-Wolf shrinkage versus the raw sample correlation,
   reporting the shrinkage intensity as a function of m.
7. Label-noise interaction: repeat at 5% label noise, reusing `scripts/robustness/gold_sensitivity.py`'s
   flipping procedure, to test whether small m and noisy labels compound.

**Datasets / bank.** All three cached datasets, existing 10-judge bank.

**Required inference / training.** **None.** Pure resampling over cached votes.

**Metrics.** As listed. Primary deliverable is a **minimum-m recommendation per claim type**, with a
stated tolerance (for example, m such that `n_eff` is within 10% of its full-data value in 95% of
draws).

**Statistical tests.** Percentile intervals over the 50 seeds per cell; no null-hypothesis test is
appropriate here, and none will be reported. Report intervals, not p-values.

**Compute.** CPU, a few hours, embarrassingly parallel over seeds.

**Files.** New `scripts/59_sample_efficiency.py`, `experiments/sample_efficiency/PREREG.md`;
outputs `outputs/sample_efficiency/{estimand_stability.csv, eigenspace_stability.csv,
cluster_stability.csv, conclusion_flips.csv, summary.md}` and one figure. Paper: App.
`app:threats-calibration`, §8 Limitation (iii), and a new appendix subsection; promote the headline
number to §4 if it is crisp.

**Success / failure criteria.** Success is a defensible minimum-m recommendation for each claim
type. There is no failure branch that wastes the work: if even `rho_bar` needs many labels, that is
a stronger limitation than currently stated and must be reported; if everything is stable at m=50,
that materially strengthens the paper's practical story.

**How either result changes the paper.** Either way this converts Limitation (iii) from a hedge into
a quantified statement, and gives reviewers a concrete answer to "how much trusted data does this
need". Low risk, low cost, high reviewer value.

---

## P1-5. Stronger aggregation baselines, where conceptually appropriate

Retires risk D2 and directly addresses the open question §8 already names: whether the headroom the
tier-mixed bank shows (1.11 points) is unreachable in principle or merely unreachable by the two
filters tested.

**Hypothesis.** H5: the paper's filter family is expressively limited, and part of the measured
aggregation-rule headroom is captured by rules that model the dependence structure explicitly rather
than scoring agreement. Corollary H5b: **conditional-independence** aggregators (Dawid-Skene) will
*underperform* exactly where dependence is highest, which is a positive test of the paper's central
thesis rather than a competing method.

**Which baselines are conceptually appropriate, and which are not.**

*In scope* (same inputs as the paper's filters: votes, optional small calibration set, no per-item
features):
1. **Dawid-Skene** EM with per-judge confusion matrices. The canonical aggregation baseline, cited
   in the paper's own related work, and it assumes conditional independence given the truth. Include
   it primarily as a **diagnostic**: its degradation as `rho_bar` rises is a direct prediction of the
   paper's thesis.
2. **Accuracy-weighted majority.** Already implemented as `independent_accuracy_weighted` in
   `src/corrfilter/filtering.py`; report it consistently everywhere rather than only in some tables.
3. **Correlation-corrected weighted vote**: weights `w = R^{-1} 1` (the GLS / minimum-variance
   weights implied by the paper's own variance-inflation lemma). This is the natural strong
   baseline the theory implies but the paper never actually evaluates, and it is the single most
   likely reviewer suggestion.
4. **Vote-pattern classifier**: logistic regression or a shallow tree over the n-dimensional vote
   pattern, trained on the small calibration set. This is the expressiveness ceiling for
   pattern-based rules and directly probes whether the tier-mixed bank's headroom is reachable.
5. **Per-instance filter selection oracle** (already computed as the per-instance oracle) kept as
   the upper bound.

*Out of scope, with reasons stated in the paper.* Methods that change the problem setting rather
than the aggregation rule: per-item feature models (require item representations the filtering
setting does not assume), judge fine-tuning (changes the bank, not the rule), and large-labelled-set
learned aggregators (violate the small-calibration premise that motivates the whole setup). Say this
explicitly in the appendix so the omission reads as scoping, not evasion.

**Protocol.** Evaluate all rules on every existing benchmark at matched retention under the P0-0
corrected protocol: the clean calibration bank, the three regimes (CFI global, synthetic-poisoned UF
weak, position-aligned subgroup), the 304 mixed-regime deployments, and every screened cross-task
configuration. Report each rule's gain over naive majority per configuration, and regress (rank
correlation only, given the sample size) each rule's gain against `rho_bar` to test H5b.

**Datasets / bank.** All existing cached benchmarks. No new data.

**Required inference / training.** **None.** All rules are fit on cached votes and small
calibration splits. Dawid-Skene EM and the classifiers are CPU-seconds.

**Metrics.** Precision and false retention at matched retention; gain over naive majority; for
Dawid-Skene, the gain as a function of `rho_bar` across configurations.

**Statistical tests.** Paired item bootstrap per configuration (B=2000). Across configurations, use
Spearman rank correlation with an explicit note that the configuration count is small and no model
is fitted, matching the existing exploratory-headroom analysis's discipline.

**Compute.** CPU, hours. No API spend.

**Files.** Extend `src/corrfilter/filtering.py` with the new rules (additive, keeping the existing
`FilterResult` interface so nothing downstream changes), new `scripts/60_strong_baselines.py`,
`tests/test_filtering.py` additions; outputs `outputs/strong_baselines/`. Paper: §7.6, App. J
(`app:algorithms`), the §8 open-questions sentence about "aggregation rules able to reach the
headroom", and Theorem 1's discussion, since a GLS-weighted rule is a natural member of the very
class the theorem quantifies over.

**Success / failure criteria.**
- **H5 supported**: at least one new rule captures a significant share of the tier-mixed bank's
  headroom. This is a genuine positive result and would change §8's open question into a partial
  answer.
- **H5 not supported**: no rule beats the existing family. This **strengthens** the paper, because
  the headroom then looks structural rather than a limitation of two hand-designed filters, and the
  non-dominance theorem gains empirical weight.
- **H5b**: Dawid-Skene's gain should correlate negatively with `rho_bar`. If it does not, the
  conditional-independence story needs qualification and we say so.

**How either result changes the paper.** This experiment cannot fail uninformatively, which is why
it is worth running despite being P1. The main risk is scope creep: it must stay a **baseline study
inside the measurement paper**, not become a method contribution. Guard: the new rules appear in the
appendix and in one main-body sentence, and no new rule is named or branded.

---

## 6. Consolidated budget and sequencing

| Experiment | Inference | Training | GPU wall clock | API spend | Blocking |
| --- | --- | --- | --- | --- | --- |
| P0-0 code fix | none | none | none | \$0 | blocks P0-1, P1-5 |
| P0-1 forced choice | 21.5k-33.5k forward passes | none | 1-3 days | \$0 | after P0-0 |
| P0-2 Phase A | none | none | none | \$0 | none |
| P0-2 Phase B | ~10k votes | none | 1-2 days | \$0 | after Phase A |
| P0-3 Stage 1 | none | none | none | \$0 | benefits from P1-5 |
| P0-3 Stage 2 | ~6k API judge-items | none | none | ~\$42 (cap \$75) | after Stage 1 |
| P0-3 Stage 3 | ~6k API judge-items | none | none | ~\$42 (cap \$60) | conditional on a pass |
| P1-4 sample efficiency | none | none | none | \$0 | none |
| P1-5 baselines | none | none | none | \$0 | after P0-0 |

**Totals: no training runs, 3-5 days of GPU wall clock, and an API ceiling of \$150** (of which
\$60 is spent only if a cross-task candidate passes its gates). Five of the nine work items require
no inference of any kind.

**Suggested order.** Week 1: P0-0, then P0-2 Phase A and P1-4 in parallel (all CPU, all
zero-inference, and P0-2 Phase A determines whether Phase B is needed at all). Week 2: P0-1 on GPU
while P1-5 runs on CPU. Week 3: P0-3 Stages 1-2, and P0-2 Phase B if triggered. Week 4: P0-3 Stage 3
if triggered, then paper integration.

**Decision gates.**
- After P0-0: if any published number moves materially, pause and agree the disclosure before
  continuing.
- After P0-2 Phase A: Phase B runs only if a mechanism clears the pre-registered inversion test.
- After P0-3 Stage 2: Stage 3 runs only on a gate-passing candidate. No "least bad" selection, in
  keeping with the campaign's existing rule.

---

## 7. What this plan deliberately does not do

- **No larger or closed-weights judge bank** (risk B4/B5). It is the most expensive way to buy the
  least incremental credibility, and App. U.1 already scopes the claim correctly.
- **No downstream validation above 1.5B** (risk E1). The dose-response result is already measured
  and honestly scoped; scaling it is a separate paper's worth of compute.
- **No natural regime prevalence study** (the remainder of C1). P0-2 makes the mechanism natural,
  which is the part that matters for the taxonomy; measuring how often each regime occurs in the
  wild needs a labelled corpus we do not have.
- **No new filters presented as contributions.** P1-5 is a baseline study. The paper is a
  measurement paper and the Phase 1 revision was largely about making that unmistakable; Phase 2
  must not undo it.

---

## 8. Traceability to the risk register

| Risk | Severity | Retired by | Expected outcome |
| --- | --- | --- | --- |
| D5 (tie-break artifact) | HIGH | P0-0 | fully retired |
| B7 (abstention confound) | HIGH | P0-1 | fully retired, in either direction |
| C4 (position bias is the only vulnerability) | HIGH | P0-2 | retired on success, bounded on failure |
| C1 (synthetic regimes) | HIGH | P0-2 | partially retired |
| D1 (routing does not work) | HIGH | P0-3 | already answered in Phase 1; P0-3 can upgrade it |
| D2 (small filter gains, limited family) | MEDIUM | P1-5 | retired either way |
| D4 (adaptive variants need labels) | HIGH | P1-4 | quantified rather than hedged |
| B4, B5 (bank size, frontier CIs) | MEDIUM | not addressed | remains scoped in App. U.1 |
| E1 (downstream scale) | HIGH | not addressed | remains scoped in App. U.5 |

Three of the four HIGH risks that Phase 1 could not close are addressed here, at no training cost
and under \$150 of paid inference.

**Awaiting approval. Nothing will be executed until it is given.**

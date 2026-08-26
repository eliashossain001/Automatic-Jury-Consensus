# Experiment log

Running record of every experiment bucket, newest sections appended at the end.
Paths in older entries follow the pre-reorganisation layout; see
[`script_map.md`](script_map.md).

---

## Buckets 2 & 3

Date: 2026-06-03. Machine: local TITAN RTX workstation. GPU used: **GPU 0**
(`CUDA_VISIBLE_DEVICES=0`); GPU 1 was occupied by an unrelated geneformer job
throughout.

This file is the single source of truth for what was built, what ran, and what
to run next. All new work is additive: no existing H1 scripts were modified.

---

## Bucket 2 — Eigenvector Failure-Mode Decomposition  ✅ COMPLETE

Converts the H1 error-correlation eigenspectrum into interpretable, item-level
failure modes. Analysis-only, deterministic, no GPU, no inference.

**New code**
- `src/corrfilter/analysis/eigen_failure.py` — item-loading projection, per-item
  heuristic signals (verbosity / polish / sycophancy / safety / factuality /
  position-swap / prompt-divergence / cross-family), single-tag assignment,
  eigenvector→failure-family alignment scoring, judge-loading interpretation.
  Reuses `analysis.top_eigen_directions` and the `cfi.replay` lexicons.
- `scripts/dependence/eigen_failure_decomposition.py` — orchestration + outputs.

**Command**
```
python scripts/dependence/eigen_failure_decomposition.py \
    --project-root /home/elias/elias_projects/corrfilter
```

**Inputs reused** — `experiments/h1_measurement/results/correlation.npz`
(R, E[1133×10], eigvals, logical_ids); H1 vote cache (to map E rows → item_ids
and read per-judge position-swap); `calibration_manifest.parquet`.

**Items processed** — 1133 listwise-complete calibration items × 10 judges;
top 50 items extracted per eigenvector (v1, v2, v3).

**Outputs** (`outputs/eigen_failure_modes/`)
- `eigen_top_items_for_manual_labeling.csv` — 150 rows (50 × 3), exact columns
  requested (item_id, eigenvector, rank, loading_score, prompt, candidate_a,
  candidate_b, gold_label, judge_votes, heuristic_tag, manual_failure_label, notes).
- `eigen_v{1,2,3}_top_items.csv` — full per-item records (subset, per-judge
  vote+error, consensus ratio, position-swap metadata, signals, tag).
- `eigen_failure_summary.md` — eigenvalues/variance, v1–v3 interpretation,
  failure-family alignment table, top-10 items per eigenvector.
- `eigen_item_loading_v{1,2,3}.png`, `failure_tag_by_eigenvector.png`.

**Key findings**
- v1 (29.2% var): near-uniform judge sign → **common difficult-item** direction
  (item-loading vs error-count r=+0.99; cross-family co-failure r=+0.93).
- v2 (14.5%): within-model pairwise-vs-likert sign flip (qwen) → **prompt
  divergence** (matches the prior report).
- v3 (9.9%): prompt-method contrast on llama/mistral; top items skew **safety/
  refusal** (alignment r=+0.23, dominant tag safety_refusal 21/50).
- Determinism verified: identical CSV checksums across reruns.

---

## Bucket 3 — CFI + CorrFilter v1  ⏳ CODE COMPLETE, REAL-GPU RUN IN PROGRESS

### Discovery
A substantial CFI/CorrFilter scaffold already existed (`src/corrfilter/cfi/*`,
scripts 05–08) but had never been run (`experiments/h2_cfi/` was empty), and the
α_subset score was already implemented. Per the reuse/additive constraints, the
new work **extends and runs** that infrastructure rather than duplicating it.
Decisions confirmed with the user: **real GPU inference**, **all 8 conditions**,
**triggered-only on a 400-item stratified subset** (the multi-day full run was
declined; see "Cost" below).

### New code (additive)
- `configs/cfi_bias_prompts.yaml` — 8 conditions (clean + 7 biases:
  verbosity_bias, sycophancy_bias, jailbreak_shell_tolerance,
  polite_hallucination_preference, length_target_preference,
  style_template_preference, position_bias_stress_test), each with a mild
  system-prompt instruction + heuristic trigger; bank-ratio conditions
  (all_clean / mixed_25 / mixed_50 / mixed_75 / all_biased).
- `src/corrfilter/cfi/bias_bank.py` — config-driven `BiasCondition`/`BiasBank`
  loader + per-item trigger evaluation (reuses `cfi.triggers` lexicons).
- `src/corrfilter/cfi/cfi_votes.py` — assembles a `(mechanism, ratio)` bank
  variant from cached biased + clean votes; deterministic nested biased-judge
  selection by mechanism name.
- `src/corrfilter/cfi/prompts.py` — added `BiasedPrompt.from_instruction(...)`
  so the config-driven bank can inject biases without an enum (no-op for clean).
- `src/corrfilter/filtering.py` — CorrFilter v1: the 5 filtering methods
  (naive_majority, naive_supermajority_75, independent_accuracy_weighted,
  corrfilter_subset, corrfilter_subset_risk_tau) + unified evaluation
  (retention, false-retention, precision, recall, F1, avg α, ρ_S). Thin layer
  over `cfi.consensus` + `cfi.corrfilter_score`.
- `scripts/cfi/run_bank_gpu.py` — efficient real-GPU runner: loads each of the 5
  base models **once**, sweeps both prompt methods × all biased mechanisms,
  caches per `(mechanism, logical_judge)`, resumable, supports
  `--triggered-only` / `--items-file` / `--max-items` / `--mechanism` / `--base-id`.
- `scripts/cfi/analyze_gpu_bank.py` — agreement-vs-accuracy (triggered vs
  non-triggered), false-retention curves, biased-ratio curves (majority vs
  CorrFilter), judge×mechanism failure heatmap; CSVs + report + figures.
- `scripts/cfi/run_corrfilter.py` — CorrFilter comparison on clean H1 bank
  (default) or any mechanism's all-biased variant.
- `tests/test_filtering.py` — 8 tests (α=√|S| at R=I; α=1 at R=11ᵀ; positive
  correlation reduces α; empty subset safe; deterministic exact retention
  matching; tie handling; evaluation fields).

### Fix
- `src/corrfilter/cfi/corrfilter_score.py`: `retention_match_threshold` no longer
  retains NaN-scored items in the `n_keep ≥ n` branch (one pre-existing failing
  test, `test_cfi_retention.py::test_retention_match_drops_nan_scores`, now passes).

### Tests
```
python -m pytest tests/         # 99 passed (was 98 passed / 1 failed pre-fix)
```

### Validation already run (no waiting on the GPU job)
- **CorrFilter on real clean H1 bank** (1195 items):
  ```
  python scripts/cfi/run_corrfilter.py --project-root /home/elias/elias_projects/corrfilter
  ```
  Sane sanity check — on the clean bank CorrFilter ≈ majority (FRR 0.277 vs
  0.271) and keeps slightly lower within-subset ρ_S (0.208 vs 0.216); the
  headline benefit is expected to appear on biased banks. Wrote
  `outputs/corrfilter/{filtering_comparison.csv, filtered_items_corrfilter.csv,
  filtered_items_naive_majority.csv, corrfilter_report.md}`.
- **Real-GPU runner smoke** (phi-3.5-mini, 2 mechanisms, 16 items): 4 passes
  cached, position bias flipped 5/16 votes vs clean → bias injection confirmed.
- **Analysis (08) on a votes snapshot**: produced all 4 CSVs + report + 5
  figures; correctly skips mechanisms without votes.

### Real-GPU run IN PROGRESS
```
export CUDA_VISIBLE_DEVICES=0
python scripts/cfi/run_bank_gpu.py \
    --items-file outputs/cfi/cfi_subset_items.txt --triggered-only \
    2>&1 | tee outputs/cfi/cfi_full_run.log
```
- Scope: 5 base models × 2 prompt methods × 7 biased mechanisms, triggered items
  only, within a deterministic stratified **400-item** subset
  (`outputs/cfi/cfi_subset_items.txt`; ~409 manifest rows). ~10–11k generations.
- Measured throughput ~3 s/item (phi) → est. **~9–12 h** wall clock.
- Resumable: re-running the same command skips already-cached votes.
- Votes land under `outputs/cfi/votes/<mechanism>/<logical_id>.parquet`; run
  summary at `outputs/cfi/votes/cfi_run_summary.json`.

**Items processed (target):** 400-item stratified subset (Factuality 79, Focus
85, Safety 84, Math 63, Precise IF 55, Ties 34), biased inference on the
triggered subset per mechanism; non-triggered / non-sampled items reuse the
clean H1 vote at analysis time.

### Failures / errors
- None blocking. One pre-existing unit-test failure was fixed (above).
- Cost note: the **full faithful** run (all 1195 items, every mechanism on every
  item) was measured at **~3–5 days** and intentionally not launched.

---

## RESULTS — first real run (completed 2026-06-03, 9.87h, GPU 0)

Run finished cleanly: 70/70 passes, 10,932 biased votes, all 7 mechanisms × 10
judges cached. Analysis (08) + CorrFilter (09) ran on the 409-row 400-item
subset. Outputs in `outputs/cfi/` and `outputs/corrfilter/<mechanism>/`.

**Finding 1 — CFI mechanism partially fires (C2).** Majority false-retention
rises with the biased fraction for 3 of 7 mechanisms (peaks near 75% biased):
polite_hallucination +0.059, position_bias +0.056, verbosity +0.027 (all_biased
vs all_clean). The other 4 are null: sycophancy (only 9 items trigger),
jailbreak, length_target, style_template barely move majority error. Judges are
most swayed (conditional error on triggered items) by polite_hallucination
(0.46) and verbosity (0.45); least by jailbreak (0.27).

**Finding 2 — CorrFilter v1 with clean R does NOT beat naive supermajority
(negative result).** At matched retention (supermajority-0.75's operating
point), filtering clearly helps over the unfiltered majority (FRR ~0.27–0.33 →
~0.17–0.30), but CorrFilter's top-α ranking ties or slightly loses to plain
top-consensus supermajority: Δ(super−CF) is +0.006 on 3 mechanisms and −0.005 to
−0.030 on the 4 where it matters most (position −0.030, polite −0.017). It does
keep marginally lower within-subset ρ_S, but that does not convert to fewer
wrong retentions here. Interpretation: scoring uses the **clean H1 R**
(deployment-realistic), which cannot see the *injected* correlation, so α_subset
can't preferentially discount biased-judge agreement. This is exactly the
setting the proposal's `cfi_variant_R` (oracle/adaptive) ablation targets.

**Finding 3 — with variant R, CorrFilter wins (the clean-R result is a R-source
limitation, not an α_subset flaw).** Re-running CorrFilter with R re-estimated
from the biased bank's own votes (`--r-mode variant`, oracle/adaptive) at matched
retention 0.5623:

| mechanism | clean-R FRR | variant-R FRR | supermajority FRR |
|---|---|---|---|
| position_bias_stress_test | 0.304 | **0.183** | 0.274 |
| polite_hallucination_preference | 0.296 | **0.252** | 0.278 |

Variant-R CorrFilter beats both supermajority and clean-R CorrFilter, and the
within-subset ρ_S it acts on rises from ~0.20 to ~0.32 — i.e. it is correctly
identifying and discounting the *injected* correlated agreement. So α_subset
works as designed; the open problem is estimating the deployment-time R that
reflects the bank's true (possibly biased) correlation, not the clean
calibration R.

**Honest read:** the pipeline and the CFI mechanism are validated. CorrFilter v1
needs adaptive/variant R to beat supermajority; with clean R it merely ties. The
most valuable next experiment is a *realistic* (non-oracle) adaptive-R estimator,
since `cfi_variant_R` peeks at the biased votes' gold-derived errors.

## Bucket 4 — Adaptive-R estimation for CorrFilter  ✅ COMPLETE (analysis-only, no new inference)

Reframes the clean-R shortfall as **correlation drift** (not a CorrFilter
failure) and tests whether the oracle variant-R gain is recoverable without full
gold-derived biased errors.

**New code**
- `src/corrfilter/cfi/adaptive_r.py` — R estimators (`oracle_variant_R`,
  `disagreement_R` gold-free, `small_gold_R`, `hybrid_R`) + diagnostics
  (`frobenius_distance`, `eigenvector_overlap`). Reuses the H1 estimator
  functions (Ledoit-Wolf, pairwise-complete shrinkage).
- `scripts/cfi/adaptive_r.py` — full sweep over 7 mechanisms × 5
  biased fractions × R modes (clean / disagreement / small_gold[25,50,100,200]×10
  seeds / hybrid[λ=0,.25,.5,.75,1]×{disagreement,small_gold} / oracle), all at the
  supermajority-0.75 matched retention.
- `scripts/cfi/run_corrfilter.py` — `--r-mode` extended:
  `clean | oracle_variant (=variant) | disagreement | identity`.
- `tests/test_adaptive_r.py` — 6 tests (the gold-free≡oracle identity, hybrid
  endpoints, valid correlation matrices, Frobenius/overlap, small-gold→oracle).

**Commands**
```
python scripts/cfi/adaptive_r.py \
    --project-root /home/elias/elias_projects/corrfilter \
    --items-file outputs/cfi/cfi_subset_items.txt
python -m pytest tests/          # 105 passed
# single-mechanism, gold-free CorrFilter via script 09:
python scripts/cfi/run_corrfilter.py --mechanism position_bias_stress_test \
    --r-mode disagreement --items-file outputs/cfi/cfi_subset_items.txt \
    --target-retention 0.5623 --out-dir outputs/corrfilter/position_disagreement
```

**Outputs** (`outputs/adaptive_r/`) — `adaptive_r_results.csv` (1925 rows),
`adaptive_r_by_mechanism.csv`, `adaptive_r_by_calibration_size.csv`,
`adaptive_r_lambda_sweep.csv`, `adaptive_r_summary.md`, and 5 figures.

**Key structural fact:** gold is fixed-direction (chosen ≻ rejected ⇒ gold ≡ 1),
so per-item error = 1 − vote and the **vote-correlation equals the
error-correlation exactly** (verified to 1e-15). `disagreement_R` (gold-free) is
therefore the algebraic twin of `oracle_variant_R`; `small_gold_R` is a
sample-size ablation. `oracle_variant_R` stays labeled as an UPPER BOUND.

**Results — mean FRR by R mode (all_biased, matched retention):**

| R mode | mean FRR | gap from oracle |
|---|---|---|
| naive_majority | 0.286 | +0.080 |
| naive_supermajority_75 | 0.223 | +0.016 |
| clean_R | 0.232 | +0.026 |
| **disagreement_R (gold-free)** | **0.205** | **−0.002** |
| small_gold_R (avg) | 0.210 | +0.003 |
| hybrid_R (avg) | 0.215 | +0.008 |
| oracle_variant_R (UB) | 0.206 | +0.000 |

**Answers**
1. **disagreement-only R recovers the full oracle gain** (gap −0.002; even edges
   the listwise oracle by using all items pairwise). No labels needed.
2. **Labeled examples needed:** small_gold Frobenius-to-oracle falls 1.67→0.44 as
   size 25→200; FRR gap +0.008→−0.002. ~25 labels already gets most of it, but
   moot here since disagreement matches at zero labels.
3. **Hybrid beats clean_R**, monotonically improving as λ→0 (pure adaptive best:
   0.232 → 0.205). No benefit to retaining clean weight under drift.
4. **Most R-drift-sensitive mechanisms:** position_bias (‖R_clean−R_oracle‖_F =
   2.73, eig-overlap 0.57, clean gap +0.122) and polite_hallucination (2.52,
   0.61, +0.044); all others have small drift (<0.75) and small gap.
5. **position_bias is the strongest case** for adaptive CorrFilter: disagreement_R
   cuts its FRR by 0.122 vs clean_R, by far the largest gain.

**Interpretation:** the clean-R result was correlation drift, exactly as the
reframing says. R accuracy predicts CorrFilter performance (Frobenius-to-oracle
vs oracle-gap, `figures/frobenius_vs_performance.png`), and a fully gold-free
deployment estimator (vote correlation on the live bank) closes the gap. The
caveat: this clean recovery relies on the fixed-direction gold; with
varying-direction gold, vote-correlation only approximates error-correlation and
small_gold / hybrid would matter more. That generalization is the next study.

## Bucket 5 — Direction-randomized adaptive-R validation  ✅ COMPLETE (analysis-only)

Last cheap check before DPO: does gold-free `disagreement_R` survive breaking the
fixed-direction gold assumption that made it equal the oracle in Bucket 4?

**New code**
- `scripts/cfi/adaptive_r_direction_randomized.py` — two settings on the same CFI
  votes: `fixed_direction` and `randomized_direction` (frame flip on ~50% of
  items, seeds 0–4). Reuses the Bucket-4 estimators.
- `tests/test_adaptive_r.py` — +1 test (frame-flip error/oracle-R invariance). **106 pass.**

**Construct (important).** Randomization flips the *presentation frame* — invert
the observable vote AND the gold on flipped items — so the per-item error
`1[O≠gold] = 1 − v` stays invariant. This preserves oracle R and the achievable
filtering benefit (non-vacuous), while scrambling only the gold-free observable
`disagreement_R = corr(O)`. (A first version flipped gold *alone*; that decouples
gold from quality and collapses every method to FRR≈0.5 — vacuous — so it was
corrected to the frame-flip construct that matches the task wording.)

**Command**
```
python scripts/cfi/adaptive_r_direction_randomized.py \
    --project-root /home/elias/elias_projects/corrfilter \
    --items-file outputs/cfi/cfi_subset_items.txt
```
**Outputs** (`outputs/direction_randomized/`): `direction_randomized_results.csv`
(7350 rows), `direction_randomized_summary.md`, 4 figures.

**Results — mean FRR / oracle gap by R mode (all_biased):**

| R mode | FRR fixed | FRR rand | gap fixed | gap rand |
|---|---|---|---|---|
| naive_majority | 0.286 | 0.344 | +0.080 | +0.127 |
| naive_supermajority_75 | 0.223 | 0.223 | +0.016 | +0.005 |
| clean_R | 0.232 | 0.232 | +0.026 | +0.015 |
| disagreement_R (gold-free) | 0.205 | 0.237 | −0.002 | **+0.020** |
| small_gold_R (avg) | 0.211 | 0.218 | +0.005 | +0.001 |
| hybrid_R (avg) | 0.214 | 0.224 | +0.008 | +0.007 |
| oracle_variant_R (UB) | 0.206 | 0.217 | +0.000 | +0.000 |

`disagreement_R` vs oracle R-quality: Frobenius 0.11→0.45, top-3 eigenvector
overlap 0.998→0.974.

**Answers**
1. **disagreement_R does NOT fully survive randomization.** Its oracle gap goes
   −0.002 → **+0.020 FRR** (now worse than clean_R's +0.015). The eigenstructure
   is largely preserved (overlap 0.97), which is why it still beats naive
   majority, but the residual Pearson-centering drift costs ~2 FRR points.
2. **Small calibration recovers it:** `small_gold_R` gap is +0.004 at 25 labels
   and ≈0 (−0.001 to −0.002) by 100–200 labels — it matches the oracle.
3. **Hybrid beats both** clean_R (0.232) and pure disagreement_R (0.237): best
   FRR 0.228 at λ=0.5 (disagreement base).
4. **Largest disagreement_R residual gap under randomization:** style_template
   (+0.042), length_target (+0.035), sycophancy (+0.030) — the *weak-signal*
   mechanisms, not the high-drift ones. position_bias / polite_hallucination have
   huge clean-R drift yet disagreement_R handles them (gap +0.007 / +0.003)
   because their correlation signal is strong and eigenstructure-dominant.

**VERDICT (decision for the paper):** the gold-free claim holds only under
fixed-direction gold (a benchmark artifact). Under direction randomization
gold-free `disagreement_R` loses ~2 FRR points (gap +0.020 > 0.01), while a small
labeled set (~25–100 items) or a λ≈0.5 hybrid recovers the oracle. **The paper
should claim "small-calibration Adaptive CorrFilter," not gold-free** — with the
honest nuance that gold-free works when the preference direction is fixed and
degrades gracefully (not catastrophically) when it is not.

## Bucket 6 — Synthetic-Poisoned UltraFeedback (H2b)  ⏳ CODE COMPLETE, JUDGE RUN IN PROGRESS

Tests whether Small-Calibrated Adaptive CorrFilter improves corrupted-label
filtering over naive k-of-n on a real preference dataset. No DPO.

**New code**
- `scripts/attacks/build_poisoned_ultrafeedback.py` — loads UltraFeedback binarized, keeps the
  content-canonical pair (A = truly-better), and plants *structured* label
  corruption: flips toward the response a weak heuristic prefers (verbosity /
  length / polite / style), only where the heuristic disagrees with truth.
  Rates 5/10/20% nested. Built: 2000 pairs, 100/200/400 poisoned, 4 heuristics
  evenly (100 each at 20%).
- `scripts/attacks/run_poisoned_uf_judges.py` — clean bank over the UF pairs, real inference,
  load-each-model-once, single VoteCache, resumable. Votes are content-based
  (vote = 1 iff judge picked the truly-better response) and label-independent, so
  one cache serves all corruption rates.
- `scripts/attacks/eval_poisoned_uf.py` — filtering comparison at matched retention
  (supermajority-0.75): no_filter, random_same_retention, naive_majority,
  naive_supermajority_75, CorrFilter {clean_R, disagreement_R, small_gold_R
  (25/50/100/200 × 10 seeds), hybrid_R}, oracle_filter, oracle_variant_R.
  Reuses `cfi.adaptive_r` + `cfi.corrfilter_score` + `cfi.consensus`.

**Construct.** Identical algebra to the direction-randomized test: corruption
flips the label on poisoned items, so the affirm-stated-label vote is `O = V` on
clean and `1−V` on poisoned, with reliability gold `g = 1`(clean)/`0`(poisoned).
A filter keeps items it judges clean; precision = clean fraction among kept.
Structured (heuristic-aligned) corruption is designed to make biased judges
co-affirm bad labels, the regime where CorrFilter should help.

**Commands**
```
python scripts/attacks/build_poisoned_ultrafeedback.py --n 2000 --project-root .
CUDA_VISIBLE_DEVICES=0 python scripts/attacks/run_poisoned_uf_judges.py --project-root .   # ~18-20h, resumable
python scripts/attacks/eval_poisoned_uf.py --project-root .                       # after votes land
```

**Status.** Manifest + both scripts done; evaluator validated end-to-end on the
partial smoke cache (no errors). Full judge run launched on GPU 0 (2000 pairs ×
10 logical judges ≈ 20k votes; measured ~2 s/item on phi, blended ~3.5 s →
est. ~18–20 h). Outputs land under `outputs/synthetic_poisoned_ultrafeedback/`.

### RESULTS (run completed 2026-06-10, 20k votes, both GPUs ~5h; eval auto-fired)

Matched retention = supermajority-0.75 operating point (969/2000 = 48.5%).
Precision / false-retention / F1 by method:

| corruption | naive majority | naive super-75 | small_gold Adaptive | oracle |
|---|---|---|---|---|
| 5%  | 0.980 (FRR .020, F1 .662) | 0.980 | 0.978 (FRR .022, F1 .661) | 1.000 |
| 10% | 0.956 (FRR .044, F1 .669) | 0.956 | 0.949 (FRR .051, F1 .664) | 1.000 |
| 20% | 0.901 (FRR .099, F1 .680) | 0.901 | 0.895 (FRR .105, F1 .675) | 1.000 |

All CorrFilter variants (clean_R / disagreement_R / small_gold_R / hybrid_R /
oracle_variant_R) tie or trail naive majority within ±0.6 pt. small_gold by
calibration size (25/50/100/200 labels) is flat and equals clean_R — it does
**not** reach near-oracle (gap 2 / 5 / 11 pts at 5 / 10 / 20%).

**VERDICT — H2b NOT CONFIRMED.** Small-Calibrated Adaptive CorrFilter does not
gain ≥5 precision points over naive k-of-n (it is ~flat / −0.2 to −0.6 pt). Why:
on the *clean* open-weights bank, structured verbosity/length/polite/style
poisoning did not induce correlated co-failure (R_dis↔oracle eigenvector overlap
0.99 — almost no correlation drift), so naive consensus already detects bad
labels well (precision 0.90 at 20%) and CorrFilter has no redundant agreement to
discount. Even oracle_variant_R (knows deployment R) only matches naive — the
residual oracle gap is *individual* judge error on hard items, not co-failure,
which correlation-aware filtering cannot fix.

**Caveat / honest next step.** The bank's strongest shared bias is position
(universal, from H1), but position was excluded from the content-label
corruption (it is presentation-based). The poisoning therefore under-exercised
the bank's actual correlated weakness. A position-aligned poisoning attack, or a
bank with stronger shared content biases (as in the CFI prompt-injected runs
where adaptive R gave 0.30→0.18), is where CorrFilter gains would be expected.

**DPO remains gated and should NOT proceed** on the H2b premise: on this
benchmark CorrFilter-filtered data would not be expected to beat naive-filtered.

## Bucket 7 — Position-Aligned Poisoning (H2b follow-up)  ✅ COMPLETE (analysis-only, no new inference)

H2b's content poisoning induced no correlated co-failure, so we attacked the
bank's empirically dominant shared bias (position), reusing the 20k-vote cache.

**New code:** `scripts/attacks/position_aligned_poisoning.py`. Reconstructs each judge's slot
pick from the cached `position_swapped` flag (`slotA_pick = vote XOR swap`),
weights judges by their **H1-measured** position-bias magnitude (`|gap|/100`),
and poisons items by the position-weighted mass of judges that picked the truly
worse response (`score_i = Σ_j w_j(1−V_ij)`). Same 7-method comparison + matched
retention as H2b, plus correlation diagnostics.

**Command:** `python scripts/attacks/position_aligned_poisoning.py --project-root .`
**Outputs:** `outputs/position_poisoning/{position_poisoning_results.csv,
_diagnostics.csv, _calibration.csv, _judge_contrib.csv, summary.md, figures/}`.

**The attack worked as a co-failure attack** (unlike H2b): 100% of poisoned items
are majority-affirmed, so naive consensus crashes (precision 0.90/0.82/0.79 at
5/10/20%, vs H2b's 0.98/0.96/0.90), and it induced **real eigenstructure drift**
— top-3 eigenvector overlap vs clean fell to 0.72–0.78 (H2b: 0.99).

**But CorrFilter still gives no benefit:**

| rate | naive maj | naive super | small_gold adaptive | clean_R | oracle_variant | oracle_filter |
|---|---|---|---|---|---|---|
| 5%  | 0.897 | 0.897 | 0.897 | 0.897 | 0.897 | 1.000 |
| 10% | 0.824 | 0.824 | 0.835 (+1.1) | 0.817 | 0.825 | 1.000 |
| 20% | 0.791 | 0.791 | 0.773 (−1.8) | 0.758 | 0.770 | 1.000 |

Best adaptive gain over naive k-of-n = **+1.1 pts** (≥5 target NOT met); clean_R
and even oracle_variant_R fall *below* naive at 20%.

**VERDICT — essentially (B): no benefit even when attacking the strongest shared
bias.** The decisive, honest mechanism: position-aligned label corruption *splits*
the bank (the position-biased subset affirms the flip; the rest dissent), which
**lowers** mean off-diagonal correlation (0.217 → 0.075) and **raises** effective
independence (n_eff 3.39 → 5.98). So the high-consensus errors are NOT backed by
high within-set error-correlation — they look comparatively independent. α_subset
rewards apparent independence, so it cannot flag these items and even mis-ranks
them as reliable (hence clean_R/oracle < naive at 20%).

**Interpretation for the thesis.** "Consensus is not reliability under dependence"
is empirically upheld (high consensus, wrong labels), but CorrFilter's specific
remedy — discount *correlated* agreement — does not cover this failure mode: a
shared bias held by a *minority-to-bare-majority* subset produces high-consensus,
*low-correlation* errors. CorrFilter helps only when the co-failing agreement is
also internally high-correlation (the CFI prompt-injected regime, where adaptive R
gave 0.30→0.18). This bounds the method's claim and is a paper-worthy negative.
**DPO remains gated** — no evidence CorrFilter improves real-data label filtering.

## Bucket 8 — Cluster-aware filtering prototype  ✅ COMPLETE (analysis-only, no new inference)

Tests whether a cluster-aware filter fixes the position-aligned failure where
consensus is driven by a vulnerable *subgroup* (low global correlation, so
CorrFilter's α_subset rewards the apparent independence).

**New code:** `scripts/filters/bias_cluster.py`. Four cluster-aware scores, all
using calibration-time cluster definitions only (no test-label tuning):
`bias_cluster` (independent support = #agreeing judges outside the H1 top-5
position-biased cluster), `spectral_cluster` (entropy of the agreeing set over
clean-R spectral clusters), `leave_cluster_out` (vote fraction after removing the
dominant agreeing cluster), `hybrid_cluster` (α_subset × (1 − max cluster
fraction)). Compared vs naive maj/super, CorrFilter {clean/small_gold/oracle},
oracle, on position + h2b + CFI(position/polite/verbosity).
**Outputs:** `outputs/cluster_filter/{cluster_filter_results.csv, summary.md, figures/}`.

**Position-aligned poisoning — precision (the failure case):**

| method | 5% | 10% | 20% |
|---|---|---|---|
| naive majority / super | 0.897 | 0.824 | 0.791 |
| corrfilter_clean_R | 0.897 | 0.817 | **0.758** |
| corrfilter_small_gold_R | 0.897 | 0.820 | 0.772 |
| **bias_cluster** | **0.928** | **0.872** | **0.814** |
| spectral / leave-out / hybrid | ~0.90 | 0.83/0.82/0.84 | 0.75/0.76/0.75 |
| oracle | 1.000 | 1.000 | 1.000 |

bias_cluster gain over naive majority: **+3.1 / +4.8 / +2.3 pts** (peak +4.8 at 10%).

**Mean precision gain over naive, by method × attack:**

| method | position | h2b | cfi |
|---|---|---|---|
| bias_cluster | **+3.4** | −1.2 | −1.4 |
| spectral_cluster | −1.3 | −2.5 | −12.9 |
| leave_cluster_out | −1.2 | −1.2 | −13.0 |
| hybrid_cluster | −0.7 | −2.8 | −11.1 |
| corrfilter_small_gold (ref) | −1.0 | ~0 | +2.8 |

**VERDICT — partial / promising, ≥5 pt target NOT cleanly met (peak +4.8).**
- The **a-priori bias-cluster penalty is the only method that helps** on position
  poisoning (+2 to +5 pts), *reversing* CorrFilter's −3 to −4 pt loss. It nearly
  hits the +5 bar (+4.8 at 10%) but does not clear it, and gives no benefit on
  h2b/CFI.
- **Data-driven cluster methods fail** (spectral/leave-out/hybrid), badly on CFI
  (−11 to −13). Reason: clean-R spectral clustering does not recover the
  position-polarity vulnerability (the split was near-degenerate, 7/10 judges in
  one cluster). The cluster must match the *actual* vulnerability, which requires
  measuring the bias — unsupervised clustering of R is insufficient.
- CorrFilter still owns its home turf (CFI, internally-correlated co-failure:
  small_gold/oracle ≈ +2.8 over naive); cluster methods don't help there.

**Recommendation (per the task's own framing):** lean toward the
**measurement/diagnostic paper**, with **bias-cluster-aware filtering as the most
promising mitigation** but not yet a clean ≥5-pt win. The defensible narrative:
(1) consensus is not reliability under dependence — measured + causally shown;
(2) the dependence has two regimes — globally-correlated co-failure (CorrFilter
helps) and dominant-vulnerable-subgroup co-failure (CorrFilter fails; a
bias-informed cluster penalty *partially* recovers it); (3) repairing the second
regime needs the vulnerable cluster to be identified, which pure R-clustering
does not deliver. **DPO remains gated.**

## Bucket 9 — Small-Calibration Learned Bias Cluster  ✅ COMPLETE (analysis-only, no new inference)

Deployability test: can the vulnerable cluster be LEARNED in-domain from a small
labeled calibration set, instead of imported from H1?

**New code:** `scripts/filters/learned_bias_cluster.py`. Per (size K, seed): sample K
labeled items, learn the cluster from their **poison-affirmation rate** (top-5
judges that most affirm the bad labels), evaluate ALL methods on the **held-out
remainder** (no leakage). Also reports a **label-free** position-sensitivity
cluster (top-5 by |slot-A rate − 0.5| from votes+swap, 0 labels). Sizes
25/50/100/200 × 10 seeds; position (primary) + h2b + CFI.
**Outputs:** `outputs/learned_bias_cluster/{learned_bias_cluster_results.csv,
cluster_overlap_analysis.csv, summary.md, figures/calibration_size_curve.png}`.

**Position attack — learned_bias_cluster gain over naive majority (held-out):**

| poisoning | 25L | 50L | 100L | 200L | H1-imported | label-free pos-sens |
|---|---|---|---|---|---|---|
| 5%  | +1.2 | +1.5 | +1.6 | +2.1 | +3.1 | +3.1 |
| 10% | +3.8 | +4.6 | **+4.9** | +4.6 | +4.8 | +4.8 |
| 20% | +1.5 | +0.6 | +2.4 | +2.0 | +2.3 | +2.3 |

**Cluster overlap with H1 (position):** 3.0 / 3.3 / 3.6 / 3.6 shared judges (of 5)
at 25/50/100/200 labels; Jaccard 0.45 → 0.59. The label-free position-sensitivity
cluster matches H1's benefit **exactly** (+3.1/+4.8/+2.3).

**Answers**
- **Q1 (learnable?)** Yes. ~50–100 in-domain labels recover the H1 benefit; the
  **label-free** position-sensitivity cluster recovers it with **zero labels**
  (the vulnerability is already in the votes+swap structure).
- **Q2 (recover +4.8?)** Yes — learned +4.9 at 10%/100L ≈ H1 +4.8.
- **Q3 (>+5?)** No. Peak +4.9 — the same ceiling H1 hits; bias-cluster filtering
  does not cleanly clear +5 by any route.
- **Q4 (labels needed?)** ~100 for full recovery at 10% (+4.9); 50 close (+4.6);
  25 partial (+3.8); 0 via position-sensitivity.
- **Q5 (resembles H1?)** Yes — majority overlap (3–3.6 of 5), rising with K; the
  label-free pos-sens cluster is essentially identical.

**Regime-specificity (honest caveat):** the bias-cluster filter helps only in the
dominant-vulnerable-subgroup regime (position). On h2b it is neutral (~−0.1); on
**CFI it HURTS** (learned 0.64–0.68 vs naive 0.72; there CorrFilter small_gold is
the right tool, 0.74). A deployed system would need to know/route by regime.

**CONCLUSION: bias-aware filtering IS deployable with a small (or zero) calibration
budget** — the in-domain learned cluster matches the H1-imported benefit within
~1 pt, and a label-free position-sensitivity cluster matches it exactly. But the
benefit is **modest and regime-specific** (peak ~+4.9, never clears +5; can hurt
off-regime). Net: this supports a **measurement/diagnostic paper** whose
mitigation contribution is "bias-cluster-aware filtering, deployable from
in-domain signals, partially repairs subgroup-driven consensus failure" — not a
universal CorrFilter v2. **DPO still gated.**

## Bucket 10 — Regime Router  ✅ COMPLETE (analysis-only, no new inference)

Can a router pick the right filter per dataset/item from diagnostics, without
being told the regime, and match the per-regime best?

**New code:** `scripts/routing/regime_router.py`. Diagnostics per dataset (ρ̄, n_eff,
eig-drift vs clean, within-subset ρ, leave-cluster-out flip rate using the
label-free position cluster). Three routers (dataset / item / hybrid) with
**clean-bank-referenced thresholds** (no test-tuning; ablation sweeps them);
small_gold R + held-out eval on disjoint splits. Baselines incl. an oracle router
(best filter per *true* regime).
**Outputs:** `outputs/regime_router/{regime_router_results.csv, regime_diagnostics.csv,
regime_confusion_matrix.csv, router_ablation_results.csv, summary.md, figures/}`.

**Regimes ARE diagnostically separable** (unsupervised): global = ρ̄ rises above
clean (0.27–0.33 vs 0.22); subgroup = ρ̄ falls + eig-drift (0.08, overlap 0.73 at
20%); weak ≈ clean. **Dataset-router classification accuracy = 78%** (global 3/3,
weak 3/3, subgroup 1/3 — misses position 5%/10% where the signal is genuinely weak).

**Precision by method × regime (held-out):**

| method | weak | global | subgroup | overall |
|---|---|---|---|---|
| naive_majority / super-75 | 0.944 | 0.732 | 0.837 | 0.838 |
| corrfilter_small_gold_R | 0.940 | **0.746** | 0.841 | 0.842 |
| bias_cluster | 0.933 | 0.720 | **0.871** | 0.841 |
| router_hybrid | 0.942 | 0.747 | 0.847 | **0.845** |
| oracle_router (knows regime) | 0.944 | 0.746 | 0.871 | 0.854 |

Router gains (overall): vs naive **+0.7 pt**, vs best fixed single filter
(CorrFilter) **+0.3 pt** (within noise), vs oracle router **−0.9 pt** (recovers
~44% of the oracle's edge over naive). The router correctly routes global→CorrFilter
and weak→supermajority every time, and subgroup→bias_cluster only at strong signal
(20%).

**VERDICT — diagnostic taxonomy, not a unified filter.** The router does **not
decisively beat the best fixed single filter** (+0.3 pt, within noise), though it
beats naive and recovers part of the oracle-router gap. The honest conclusion for
the paper: this is a **measurement/diagnostic contribution** — a taxonomy of
consensus-failure regimes that ARE identifiable from unsupervised diagnostics
(78%), each with a characterised best filter — rather than a single unified
filtering method. Per-regime gains are real but modest and dilute in aggregate;
routing is feasible and promising but its net benefit over "just use CorrFilter"
is within noise here. **DPO stays gated.**

---

## Programme synthesis (Buckets 1–10)

1. **H1** — LLM-judge errors are strongly, structurally correlated (ρ̄=0.206,
   n_eff=3.51); universal position bias; cross-family > within-family co-failure.
2. **Eigen/CFI** — failure modes decompose into common-difficulty / prompt-
   divergence / safety directions; injected biases causally inflate naive
   false-retention.
3. **CorrFilter** — helps only under *globally-correlated* co-failure (CFI),
   and only with adaptive/in-domain R (clean-R ties supermajority — correlation drift).
4. **H2b + position poisoning** — on real poisoned data CorrFilter ≈ naive; under
   a *vulnerable-subgroup* attack consensus fails badly yet CorrFilter can't help
   (high-consensus, *low*-correlation errors).
5. **Cluster-aware** — a bias-cluster filter partially repairs the subgroup regime
   (+~4.8 peak, never clears +5), learnable from ~100 in-domain labels or label-free
   position-sensitivity, but regime-specific.
6. **Router** — regimes are detectable (78%) and routable, but routing's aggregate
   benefit doesn't decisively beat the best single filter.

**Net thesis:** *Consensus is not reliability under dependence* — upheld and
sharpened into a two-regime taxonomy with characterised (partial) mitigations.
The deliverable is a rigorous diagnostic/measurement paper; a universal
dependence-aware filter remains open. **No DPO run** (correctly gated throughout).

## Bucket 11 — Unified Regime-Aware Filter v2 (soft routing)  ✅ COMPLETE (analysis-only)

Adds confidence-weighted SOFT routing over Bucket 10's hard router: per-item
softmax regime probabilities blend percentile-ranked filter scores
(`p_weak·super + p_global·corrfilter + p_subgroup·bias_cluster`). Constants fixed
a-priori from clean reference; held-out eval; ablations over each signal and
temperature.
**New code:** `scripts/routing/unified_router.py`.
**Outputs:** `outputs/unified_router_v2/{unified_router_v2_results.csv,
routing_confidence_analysis.csv, ablation_results.csv, summary.md, figures/}`.

**Overall precision (held-out):** naive 0.838 · best single (CorrFilter) 0.842 ·
**router_soft 0.840** · router_hard 0.834 · router_dataset 0.834 · oracle_router 0.854.

- **Soft > hard** (0.840 vs 0.834): confidence-weighting removes the
  hard-misclassification cliff (temperature ablation confirms: moderate T 0.5–1.0
  beats near-hard T 0.2). 
- **Soft does NOT beat the best single filter** (0.840 vs CorrFilter 0.842;
  within noise) and trails the oracle router (−1.4 pt).
- Ablations: dropping `rho_bar` or within-subset `rho_S` hurts most (the
  global-regime detectors); `lco`/`eig` matter least in aggregate.

**Answer to the key question:** soft routing is *more robust* than hard routing
but is **not** more reliable at beating the best single filter — its aggregate
gain over "just use CorrFilter" is within noise.

## FINAL CONCLUSION (Buckets 1–11)

The unified-filter program is **negative-but-rigorous**: across hard, soft,
dataset- and item-level routing, no router decisively beats the best single
filter (all within ~1 pt; oracle-router ceiling only +1.6 over naive). The
per-regime gains are real (CorrFilter +1.4 on global; bias-cluster +3.4 on
subgroup) but small and dilute in aggregate, and low-signal cases (5–10%
poisoning) are hard to classify.

**The paper should be a measurement/diagnostic paper:** (1) LLM-judge banks have
strong structured dependence; (2) consensus failure splits into a globally-
correlated regime (CorrFilter helps, needs adaptive/in-domain R) and a
dominant-vulnerable-subgroup regime (CorrFilter fails; a label-free bias-cluster
filter partially helps); (3) regimes are diagnostically identifiable (~78%) but a
unified router's net benefit is within noise. *Consensus is not reliability under
dependence* — upheld, taxonomized, with partial and characterized mitigations; a
universal dependence-aware filter remains open. **DPO never run (correctly gated).**

## Bucket 12 — Reviewer-strengthening package (robustness + CIs)  ✅ COMPLETE (analysis-only)

Addresses two reviewer concerns: external validity (single bank) and uncertainty
on small gains. No new inference/methods/attacks/DPO/tuning.

**Exp A — Bank composition robustness** (`scripts/robustness/bank_composition.py` →
`outputs/robustness/`): 73 subbanks (pairwise/Likert-only, leave-one-family-out,
random 6/8 × 20 seeds, same/cross-family). Findings are composition-invariant:
ρ̄∈[0.13,0.28] and n_eff/n∈[0.35,0.66] on all 68 multi-judge subbanks; position
bias (max slot gap >0.10) on every subbank; H1 stays rejected; all 3 regime
signatures recur in 12/13 structured subbanks.

**Exp B — Confidence intervals** (`scripts/robustness/confidence_intervals.py` →
`outputs/confidence/`): 2000-sample paired bootstrap on Tables H2b/position/
bias-cluster/router. **bias-cluster gain over naive is significant at every rate**
(10%: +0.049 [+0.036,+0.061] → the +4.8 survives, also beats CorrFilter at 5/10%);
CorrFilter gains not significant (sig. negative at 20% on both poisonings); **soft
router vs best-single +0.000 [−0.003,+0.004] = within noise** (only oracle router
significant). Seed std 0.003–0.04; random-subbank ρ̄ 0.204±0.021.

**Synthesis:** `reviewer_response_analysis.md` answers the 5 reviewer questions.
Net: bank composition does not change conclusions; regimes are stable; the
bias-cluster mitigation and the (negative) CorrFilter results both strengthen
under CIs; the router is confirmed within-noise. **Central thesis unchanged and
strengthened: consensus is not reliability under dependence.**

## Next recommended run

1. Confirm completion: `cat outputs/cfi/votes/cfi_run_summary.json` (expect
   ~70 passes) and check `outputs/cfi/cfi_full_run.log`.
2. Analyze on the same subset:
   ```
   python scripts/cfi/analyze_gpu_bank.py \
       --project-root /home/elias/elias_projects/corrfilter \
       --items-file outputs/cfi/cfi_subset_items.txt
   ```
   → `outputs/cfi/{cfi_agreement_accuracy,cfi_false_retention,cfi_biased_ratio,
   cfi_failure_heatmap}.csv`, `cfi_report.md`, `figures/`.
3. CorrFilter on the most correlated biased banks (where it should help most):
   ```
   for M in position_bias_stress_test verbosity_bias polite_hallucination_preference; do
     python scripts/cfi/run_corrfilter.py --mechanism $M \
       --items-file outputs/cfi/cfi_subset_items.txt --target-retention 0.8 \
       --out-dir outputs/corrfilter/$M
   done
   ```
4. Optional scale-up: drop `--items-file` (and/or `--triggered-only`) for a
   fuller pass; budget per the cost table above.
5. Optional: add `corrfilter_subset_risk_tau` calibration (sweep `--alpha-floor`)
   once the biased-bank α distributions are known.

## Bucket 13 — Prof. Lim / external-review package (frontier pilot, non-errors, routing accuracy, mixed regimes)  ✅ COMPLETE (2026-07-25)

Four additions, all additive; 127 tests pass (21 new). Full details and paper edits below.

**A — Gemini frontier-judge pilot (single-provider).** New `src/corrfilter/judges/gemini_judge.py`
(REST v1beta, retries, failure logging, dry-run-safe), `configs/judge_bank_gemini.yaml`,
`scripts/frontier/run_gemini_judges.py` (resumable, checkpointed, `--live` gated),
`scripts/frontier/analyze_gemini_bank.py`. Judges: gemini-3.1-pro-preview (thinking low, the
minimum), gemini-3.6-flash, gemini-3.5-flash-lite (thinking minimal); pairwise prompt,
temp 0, position seed 20260601. Items: 400-item stratified RB subset + 300-item stratified
PKU subsample. Actual cost $3.12 (2,100 live calls). RESULT: individually far stronger on RB
(92.5/85.5/82.1 vs best open 73.3) but errors twice as correlated (rho_bar 0.42 vs 0.20,
n_eff 1.64/3, co-failure lift 4.0x); PKU erases the accuracy advantage (Gemini majority 65.3
< open pairwise 69.0). Cross-provider error correlation is low (0.09 RB / 0.17 PKU) →
provider mixing decorrelates where family/prompt did not (combined-8 n_eff 3.75).
Outputs: `outputs/gemini_bank/` (votes, run logs, analysis_{rewardbench,pku}).

**B — Correlated non-errors (`scripts/dependence/nonerror_correlation.py`).** With fixed-direction
gold, corr(correctness) == corr(errors) exactly (verified to 1e-15), so the new content is
asymmetric: co-success 0.71 (lift 1.12) vs co-failure 0.50 (lift 1.36) on RB; mean 6.35
nominal correct votes = 2.85 effective (2.2x over-count; 2.6x PKU); unanimous 10-of-10 =
3.51 effective (= n_eff). Leave-pair-out difficulty stratification: residual rho 0.111 on
both datasets (~half of raw dependence is shared easiness). `outputs/nonerror_correlation/`.

**C — Routing accuracy (`scripts/routing/superseded/routing_accuracy.py`).** Persisted explicitly: n=9
instances (labels synthetic-by-construction), hard 7/9 [Wilson 0.45,0.94], supervised LOO
7/9, soft dataset-argmax 3/9; per-regime PRF + confusion matrices; script-18 confusion
reproduced exactly. `outputs/routing_accuracy/`.

**D — Mixed-regime benchmark (`scripts/routing/superseded/mixed_regime_benchmark.py` + new
`src/corrfilter/routing.py` with parity tests vs scripts 18/19).** 54 instances (18/regime),
4 mixtures, matched retention, instance-stratified bootstrap. Hard router accuracy 36/54
(0.67 [0.53,0.78]); beats naive on balanced (+0.7 [0.0,1.3]) and subgroup-dominant (+1.4
[0.7,2.1]) but NEVER significantly beats the best fixed filter; oracle router beats best
fixed on all four mixtures (+0.5..+1.1, significant). Gap decomposition: misroute rate
17-37% x cost 2.2-5.0 pts consumes the whole headroom. VERDICT unchanged and sharpened:
do not claim routing is practically superior. `outputs/mixed_regime/`.

**Paper (`main.tex`).** Added §Frontier-Judge Pilot (tab:gemini), correlated-non-errors
paragraph + tab:nonerror in RQ1, rewrote §Regime Identification and Routing (explicit
7/9, CIs, mixed-regime results), appendix tab:routeracc + tab:mixed + routing-accuracy
and mixed-benchmark details, updated threats (single-provider scope, preview-ID caveat,
mixture-prevalence caveat), reproducibility subsection app:repro-frontier, and replaced
every bare "~78%" with the 7/9 (and 36/54) accounting. Braces balanced, no dangling refs,
no em-dashes. NOT compiled locally (iclr2026_conference.{sty,bst,bib} live on Overleaf).

### Bucket 13 audit addendum (2026-07-25)

Full scientific audit performed. (1) 86 independent recomputations from raw parquets
(own V/M assembly, own metric code) match every paper number; max diff 2e-3 on a rounded
lift. (2) Gemini correlation verified genuine: plain Pearson rho_bar 0.44 > shrunk 0.42;
raw 2x2 joint-error counts 3-5x the independence expectation; swap-decode sanity clean.
Heatmaps + co-failure matrices now in outputs/gemini_bank/analysis_*/figures/.
(3) Cross-provider gaps significant and seed-stable; CIs persisted to cross_provider.csv
and quoted in the paper; wording made observational. (4) Routing: router/oracle keep-set
identities exact (0 violations over 54 instances x 5 fields); gap == misroute-mass
identity holds to 3 decimals on all 4 mixtures; cal/eval splits verified disjoint;
one instance re-derived end-to-end from raw votes matches exactly. (5) Fresh-tree
reproduction (configs+scripts+src+caches only): all 41 regenerated CSVs byte-identical.
(6) Bootstrap CIs reproduce under canonical seed and are stable under independent seeds.
(7) CORRECTIONS: pre-existing stale conditional co-failure 0.52/0.52 in tab:replication,
L656 text, and the contributions bullet fixed to canonical 0.50/0.51
(outputs/tables/dependence.csv: 0.4968/0.5101); stale "106 unit tests" in app:impl fixed;
the correctness-correlation identity is now correctly stated as unconditional for binary
indicators (the fixed-gold assumption matters only for vote-vs-error correlation).
Audit script: scratchpad audit_recompute.py (session-local).

## Bucket 14 — Multi-provider frontier evaluation (OpenRouter)  ✅ COMPLETE (2026-07-28)

Full 400-RB + 300-PKU runs for openai/gpt-5.6-sol, anthropic/claude-opus-5, x-ai/grok-4.5
(pinned IDs, reasoning effort minimal, temp 0 where supported, identical protocol/seed as
Gemini/open banks; provider echoes logged — no model substitution; Claude PKU: 32/300
content-filter refusals logged as abstains). Total OpenRouter spend $9.39 / 2,100 calls
(smoke $0.37 + 200-item pilot $3.09 + full $5.93); balance left ~$10.38.

HEADLINE (full sample, scripts 38-40, outputs/openrouter_bank/analysis_full_*):
- GPT+Claude+Grok bank is the MOST correlated bank in the project: rho 0.56 RB / 0.60 PKU,
  significantly above Gemini-only (paired diff +0.15/+0.26) and the open bank (+0.36/+0.29).
- Block decomposition (mixed 11-judge R): cross-provider frontier 0.42 ~= within-provider
  frontier 0.40 (RB diff -0.02 ns); frontier-open 0.07/0.17; open-open 0.20/0.29. Provider
  identity adds nothing at the frontier; pattern consistent with capability tier + shared
  difficulty. NOT causal.
- Difficulty conditioning: open-bank difficulty leaves frontier corr unchanged (0.465->0.467)
  — frontier misses are NOT the open bank's hard items; leave-pair-out frontier difficulty
  halves it (->0.28 RB, ->0.19 PKU) with a large residual. 14 RB items missed by >=5/6
  frontier judges, spread across subsets, only moderately hard for open bank (4.4/10 wrong).
- PKU: frontier accuracy partially recovers (Claude 73.9, Grok 71.7 committed vs 68.0 open
  best) but frontier majority 0.680 < open pairwise 0.690; mixed-11 best (0.713). Truncation
  concern resolved: longest rendered PKU prompt ~669 est tokens < 1024 limit.

Paper: sec:frontier retitled + multi-provider paragraphs, tab:multiprov, tab:blocks;
abstract + discussion + threats + app:repro-frontier updated; earlier "provider mixing
decorrelates" claim revised to tier interpretation everywhere (tab:gemini caption,
tab:setup caption). AUDIT: 68/68 independent recomputations from raw parquets pass;
braces/refs clean; 131 tests pass; compile still Overleaf-side.

## Bucket 15 — Narrative restructure of main.tex (2026-07-28, no new experiments)

Editorial rewrite implementing the AC-review blueprint. New title "Agreement Is Not
Evidence: Measuring Correlated Failure in LLM Judge Consensus". Abstract rewritten as
one arc (evidence -> capability -> regimes -> consequences). Intro rewritten around
three Discoveries; boxed CorrFilter equation removed from intro (formula remains in
methods eq:alpha); RQs redefined (Evidence/Capability/Structure/Consequences);
contributions cut 5 -> 3. Results reorganised: Finding 1 (measurement+non-errors),
Finding 2 (GRPO/DPO) + Finding 2 at the Frontier (moved after GRPO), Finding 3
Regimes I-III, Implications (filtering, routing). New headline Figure 1
figures/fig_tier_dependence.pdf (scripts/figures/tier_figure.py; tier-block R heatmap +
nominal-vs-effective bars); fig:taxonomy moved before Implications; fig:neffsat
promoted to main text. Related work: new "Label aggregation and ensemble diversity"
paragraph citing dawid1979observer/kish1965survey/kuncheva2003measures/breiman2001random
(BibTeX entries provided in a comment above \bibliography — ADD TO OVERLEAF .bib).
Discussion opens with the thesis, surfaces both null results (routing; downstream DPO)
and the scalable-oversight implication; new Conclusion paragraph. 136 legacy comment
lines deleted. All numbers unchanged; braces/refs/figure envs verified; 131 tests pass.
Compile on Overleaf (new citations require the .bib additions).

### Bucket 15 completion pass (2026-07-28)

Deferred blueprint items applied: theory section retitled "Why Dependence Caps
Consensus: Theoretical Analysis" with classical design-effect credit (kish1965survey)
and a structure-vs-magnitude closing sentence; fig:taxonomy moved to open Finding 3
(figure order: tier -> neffsat -> taxonomy -> mainresults); frontier setup prose
compressed to protocol-identical + App pointer (model IDs retained in text);
Implications subsection opens with a Findings->consequences bridge. Verified: braces,
13/13 figure envs, 0 dangling refs, 0 em-dashes, 131 tests pass.

### Bucket 15 final framing pass (2026-07-28)

Hidden-assumption elevation applied: abstract opens "Modern LLM evaluation quietly
rests on an unexamined assumption... first to measure that assumption systematically
in the data-filtering setting"; intro gap sharpened ("a hidden assumption that no one
has checked. This paper checks it."); contributions rewritten to the approved triad
(measure the failure / capability + regimes / practical implications); Conclusion
aligned to the canonical takeaway sentence. All "first" claims scoped with "to our
knowledge... in the data-filtering setting" consistently (abstract, contribution 1,
related work). Verified: braces, refs, tests 131/131.

### Bucket 15 review-hardening pass (2026-07-28)

External-review fixes applied to main.tex (no new inference): (1) All 5 qualitative
placeholders replaced with REAL traces reconstructed from caches — items 1401/1299
per-judge vote vectors (source run identified as outputs/corrfilter/
position_bias_stress_test, values match paper exactly) and canonical subgroup trace
item uf40540 (6/9 affirm, 5/6 in cluster, beta_C=1, LCO=1). (2) Novelty rescoped:
"first" claims removed from abstract/contribution; gap restated as
evidence/capability/structure not previously quantified inside consensus filtering;
Goel et al. 2025 (arXiv:2502.04313) cited and differentiated in related work + added
to filecontents bib. (3) Base-rate critique preempted: phi caveat added; conditional
co-failure lift (7.7x vs 1.3x) now leads in intro Discovery 2 and frontier section;
mean pairwise Yule's Q computed from caches (RB 0.96 frontier vs 0.38 open; PKU 0.91
vs 0.50) -> outputs/openrouter_bank/yule_q_by_bank.csv, quoted in text. (4)
Ten-judges objection preempted: 5-model pairwise bank on full calibration set
computed (rho 0.223, n_eff 2.64/5 = 53% of nominal) and stated in Finding 1. (5)
Mixed-bank caveat added (n_eff a means; majority accuracy 0.915/0.713 the payoff).
(6) tab:measure moved from appendix into Finding 1 main text. Verified: 0
placeholders, braces/refs clean, no duplicate labels, 131 tests pass.

## Bucket 16 — Contamination-scaling DPO (B(iii))  ✅ COMPLETE (2026-07-28)

Position-poisoned dose-response: DPO policies (Qwen-2.5-0.5B+LoRA, identical hparams,
800 pairs, nested clean core) at controlled contamination {0,20,30,40}% + 3 filtered
arms from the 40%-poisoned pool at matched retention. Per-item implicit-reward margins
+ paired bootstrap (eval_scaling_cis.py). RESULT: clean-eval margin degrades
monotonically and SIGNIFICANTLY: -0.42 [-0.63,-0.21] @20%, -0.48 @30%, -0.65
[-0.96,-0.34] @40% (clean-trained margin 1.06); RewardBench margins -0.26..-0.41 all
sig. Accuracy within noise at this scale (-1.75 pts @40%, CI [-5.25,+1.75]) ->
explains the earlier filter-comparison accuracy null (flat accuracy region at 21-25%
contamination). Filtered arms at 40% raw: naive retains least contamination (19.9%)
and degrades least; corrfilter/bias-cluster retain more at this extreme rate (23-25%)
— beyond their characterized 5-20% range; reported as measured. Paper: app:threats-dpo
rewritten (dose-response + boundary of the null), discussion nulls sentence upgraded.
Artifacts: experiments/downstream_dpo_validation/contamination_scaling/{data,models,
results}/ incl. per_item_margins.parquet, scaling_cis.csv. 7 policies trained (~2h,
both TITANs). Also: Gemini full-1178 extension DONE ($3.87; accs 93.4/88.5/83.3);
OpenRouter watcher relaunched DETACHED (setsid; survives session exit), polling for
top-up >= $14 to launch the 1178-item extension (~$12.8).

### Bucket 15 page-limit pass 2 (2026-07-29)

Main body 11pp -> target 8pp: abstract halved; intro (opening, 3 Discoveries, RQs
inline, contributions) compressed ~40%; tab:measure AND tab:multiprov moved to
app:trainfig (all numbers retained in prose); Finding-1 lead/robustness/replication/
PKU merged+halved; frontier core paragraph tightened (phi caveat + Yule Q + blocks
retained); Regime II/III halved; mitigation paragraphs halved; routing paragraph
tightened; discussion p2 + conclusion tightened; condensed related-work trimmed;
results roadmap shortened. Main text now: 1 figure (fig:tier), 0 tables, ~40k prose
chars (~8pp est). All refs/braces clean; 131 tests.

## Bucket 17 — Learned-router investigation  ✅ COMPLETE (2026-07-30, research only; paper untouched)

See outputs/router_upgrade/REPORT.md. Headline (uniform-size control pool, grouped
CV, no leakage): learned logistic regression routes at ~0.90-0.94 vs heuristic 0.54;
downstream +1.24 pts over best fixed filter (95% CI [+0.83,+1.72], the first
significant routing win in the project) and statistically at the oracle (-0.04
[-0.11,+0.02]). Size confound in the first pool detected and removed via equal-size
control. Confidence fallback: no benefit. Cross-mechanism extrapolation imperfect
with base features (0.62), repaired by extended features (>=0.91). Caveats: same
three synthetic generators; supervision needs labeled instances.

### Bucket 17 adversarial verification (2026-07-30)

See outputs/router_upgrade/verify/VERIFICATION_REPORT.md. Reproducibility PASS
(bit-identical rebuild; independent-seed bootstrap matches). MAJOR CONFOUND FOUND:
clean-UF vs clean-CFI instances are 100% separable on the routing features ->
global-class routing partly = dataset-source detection; pooled 92-94% inflated.
Confound-free results survive: within-UF weak-vs-subgroup 83-85% vs heuristic 50%;
within-CFI clean-vs-biased 87%; UF-subset downstream gain over best fixed +1.81 pts
with 95% of oracle gap closed. Required before paper use: source-balanced pool +
pre-registered logreg + both confound experiments disclosed. Paper NOT modified.

## Bucket 18 (2026-07-30): Crossed routing benchmark (confound fix; paper untouched)
- Redesigned routing benchmark: 3 regimes x 2 sources fully crossed, 204 instances
  (`scripts/routing/build_crossed_pool.py`). Judge-level error generators with
  matched 0.45 wrong-vote marginal on uniform-random bad items; regimes differ ONLY in error
  dependence (weak=independent, global=bank-wide shock, subgroup=cpos-cluster shock).
- First attempt (item-level uniform flips) FAILED its own signature gate (global rho fell with
  rate) -> rewritten at judge level; gate now passes on both sources (global rho 0.23->0.33
  monotone, matches CFI 0.26-0.32; subgroup lco 0.16->0.34 with rho inversion; weak ~ clean).
- Controls: source->regime ceiling 37.3% (was 100%); config-level label-shuffle -> 25.7% (chance);
  source still identifiable from features (95.6%) but now orthogonal to regime.
- Results (`run_crossed_router.py`): heuristic(per-source refs) 73.0%; logreg(pre-reg) 83.7/85.2%
  base/all; lightgbm 88.4%. Downstream (logreg/all): +0.24 [0.11,0.38] vs heuristic, +0.41
  [0.25,0.59] vs best-fixed; oracle +0.49; ~84% of oracle gap closed in-pool.
- LOSO: classification transfers (80-88% vs 37% ceiling) but downstream gain does NOT survive
  source transfer (+0.18 [-0.04,0.41] uf; -0.24 [-0.66,0.15] rb). Leave-rate-out fine except
  extrapolation to 5% (60-67%). Synthetic->CFI transfer = 33% chance: stylized global does not
  teach behavioural-bias global (different eigenstructure). Alt objective (per-filter precision
  regression + argmax, no regime labels) matches classification routing downstream.
- Verdict: paper's routing-as-paragraph stance stands; no paper edits made. Full report:
  `outputs/router_upgrade/CROSSED_BENCHMARK_REPORT.md`. Cost $0 (cached votes only).

## Bucket 19 (2026-07-31): Mixed-regime deployment evaluation (advisor question; paper untouched)
- Built mixed-regime benchmark (`build_mixed_pool.py`): 304 instances, 76 configs; pairwise
  {wg,ws,gs} at shares 10/25/50/75/90 + four 3-way mixtures, rates {10,20}%, both sources,
  4 seeds; bad-item set partitioned among regime error generators (matched 0.45 marginals);
  all crossed-pool confound controls preserved. Routers trained ONLY on the 204 pure instances,
  deployed frozen. Refactored build_crossed_pool (load_sources/wrong_probs/marginal_params);
  rebuild regression check passed (diffs <= 2e-13 float noise).
- ANSWER to advisor: regime-classification routing does NOT significantly beat the best fixed
  filter on mixtures (logreg pre-reg +0.08 pts [-0.09,0.26]); direct per-filter performance
  prediction is the only significant win (+0.24 [0.08,0.40] ridge), ~29% of oracle headroom
  (+0.83 [0.69,0.98]). Best fixed on training pool = naive_majority.
- Mechanism: regime->filter mapping breaks under mixing — per-instance oracle matches the
  majority-regime filter only 14.2%; even TRUE majority-regime filter gains just +0.11;
  supermajority_75 never oracle-optimal on mixtures. Router confidently right about regime,
  wrong about filter (calibration: majority-match monotone 0.47->1.00, picked-optimal flat/
  declining). Mixture entropy correlates with router entropy (0.35) and errors (-0.47).
- Heterogeneity: helps subgroup-dominated gs blends (+1.1..+2.2, significant), HURTS wg blends
  (-0.5..-0.8, significantly negative), null on 3-way. Routing accuracy vs majority regime
  (secondary): logreg 0.81, heuristic 0.62.
- Verdict: routing stays exploratory; practical recs = naive majority strong default, direct
  performance prediction if anything, taxonomy is a diagnostic not a policy. Report:
  `outputs/router_upgrade/MIXED_REGIME_REPORT.md`; plots in `outputs/router_upgrade/mixed_plots/`.
  Cost $0.

## Bucket 20 (2026-07-31): Paper updated with completed routing evidence (explicit instruction)
- main.tex updated conservatively; routing remains secondary, central narrative unchanged.
- Abstract: one clause ("identifying the regime is not sufficient to select the right filter
  under mixed regimes"). Contribution 3: mapping instability named as the binding limitation.
- sec:router rewritten: corrected benchmark (85% logreg vs 37% ceiling / 73% heuristic, LOSO
  0.69-0.88), mixed deployment (+0.08 [-0.09,0.26] ns; direct prediction +0.24 [0.08,0.40];
  oracle +0.83 [0.69,0.98]; 14% mapping match; true-majority filter +0.09 [-0.11,0.30] ns).
- Discussion: open problem reframed from regime detection to regime->filter mapping; CFI
  transfer failure noted as scope limit.
- app:router: old 9-instance and 54-instance benchmarks marked superseded (confound
  documented); new content = crossed benchmark (tab:routeracc reused), mixed deployment
  downstream (tab:mixed reused), by-mixture CIs (tab:mixedmix new), mechanism + failure
  analysis, figures fig:mixshare (mixed_precision_by_share.png) + fig:mixcal
  (mixed_calibration.png, added to run_mixed_eval.py); graphicspath += outputs/router_upgrade/
  mixed_plots/. Repro appendix updated (experiments/router_upgrade/ scripts, seeds, $0 cost).
- Number corrections found during edit: true-majority filter is +0.09 [-0.11,+0.30] paired on
  its 240 instances (report's +0.11 was pool-mismatched; report fixed); heuristic per-source
  accuracy 0.719 uf / 0.741 rb in LOSO columns; supermajority==naive at matched retention on
  all 304 (tie identity, not a finding).
- Checks: refs/labels/braces/environments balanced; no undefined refs; no em-dashes added;
  cannot compile locally (Overleaf-side .sty). USER MUST: upload mixed_precision_by_share.png
  and mixed_calibration.png to Overleaf (outputs/router_upgrade/mixed_plots/) and recompile;
  verify 8-page main-body budget (sec:router grew ~5 lines, contributions +2, abstract +1,
  discussion +6).

## Bucket 21 (2026-08-04): Routing extension: third dataset, cross-dataset deployment, decomposition
- Audit + terminology fixes (3 oracle constructs defined; benchmark heuristic renamed
  "unsupervised rule" vs framework softmax router). Dataset search: PKU-SafeRLHF open-bank
  votes found (results/strengthening_phase/pku_saferlhf/votes/, same 10 logical judges, 800
  items, swaps, oriented gold) -> third source; frontier banks noted as future work.
- New scripts: extend_pku.py (96 pure + 152 mixed PKU instances; signature gate PASSES:
  global rho .29->.37, subgroup lco .11->.25 inversion), run_extended_eval.py (ext_*.csv incl.
  per-instance artifact with predictions/probs/filters), sg_size_study.py; 11 new unit tests
  (142 total pass).
- Identification (300 pure, 3 sources): logreg-all .882+-.020 CV, LOSO .68-.83; lgbm .885,
  LOSO .80-.91; heuristic .717.
- IN-DOMAIN (456 mixed): best fixed on 3-source training pool = bias_cluster; NOTHING beats it
  (logreg -0.04 ns, direct +0.02 ns, regime-oracle -0.11 ns; per-instance oracle +0.61).
- CROSS-DATASET (train uf+rb, deploy 152 pku mixed): optimal fixed filter changes identity
  (naive -> bias_cluster); every router beats transferred fixed significantly: heuristic +0.56,
  logreg-all +0.56 [0.27,0.87], direct-lgbm +0.74 [0.45,1.04] = 56% headroom; regime-oracle
  +0.84; oracle +1.32. KEY PRACTICAL RESULT: routing pays exactly where manual choice is
  impossible (unseen deployments).
- Decomposition (456): 57% of logreg regret = right regime/wrong mapping, 21% = wrong regime,
  22% = no-majority mixtures. Improvements (conf fallback, direct margin; thresholds from pure
  CV only): null in-domain. Budgets: SG labels 25==200; router saturates ~100 train instances.
- Paper: sec:router + discussion updated (two-halves story); app:router += terminology,
  three-dataset extension + tab:routedeploy, decomposition, budgets; repro appendix updated;
  tests 131->142. Abstract/contributions UNCHANGED. Reports:
  EXTENDED_ROUTING_REPORT.md + ADVISOR_SUMMARY.md in outputs/router_upgrade/.

## Bucket 22 (2026-08-04): Independent audit of the routing extension (read-only)
- All 8 audit items PASS. Every headline number independently recomputed from raw CSVs and
  matches (ident 0.882+-0.020; cross-dataset +0.56 [0.27,0.87] seed-stable; headroom 0.43;
  oracle +1.32; decomposition 21.0/56.9/22.1 sums to 1.0). No leakage: all fits on pure pools,
  scaler in-pipeline, thresholds from pure OOF, train/deploy config-disjoint.
- Transferred-fixed comparison verified honest: chosen on uf+rb only (naive), best-on-pku is
  bias_cluster, routers trained on uf+rb only.
- NEW stress evidence (not yet in paper): all 3 LODO directions — deploy pku +0.56*, deploy uf
  +0.58* [0.29,0.89] (transferred=corrfilter, best=bias_cluster), deploy rb +0.09 ns
  (transferred=bias_cluster stays optimal). Routing wins iff fixed-filter identity flips;
  never significantly worse. vs lucky transferred-bias-cluster: -0.15 [-0.34,+0.04] ns.
- Minor manuscript issues found (NOT fixed, read-only): no-majority bucket wording omits the
  three 50/50 ties; legacy "oracle router" naming in tab:router/app:ci vs new "regime-oracle
  router"; recommend stating the clean-reference-from-target-source assumption explicitly and
  adding the 2 extra LODO directions.
- Verdict: paper fairly supports routing claims; submittable without more routing experiments.
  Full report: outputs/router_upgrade/AUDIT_EXTENDED_ROUTING.md.

## Bucket 23 (2026-08-04): Manuscript revision per audit findings
- Added tab:lodo (all 3 LODO transfer directions: pku +0.56* / uf +0.58* [0.29,0.89] / rb +0.09
  ns) + prescribed mechanism interpretation; sec:router + Discussion updated to
  "aggregation-rule shift" framing (wins iff filter identity changes; never significantly worse).
- Added transfer-protocol paragraph incl. clean-reference deployment assumption (label-free,
  disclosed as assumption not leakage) + identical-training-information statement.
- Added conservative robustness paragraph (no universal advantage; gains fractions of a point)
  + lucky-operator sensitivity (-0.15 [-0.34,+0.04] ns, explicitly retrospective/favorable).
- Fixed no-majority decomposition wording (equal 3-way + three 50/50 ties; buckets MECE, sum 1.0;
  values 21.0/56.9/22.1/0 unchanged).
- Standardized "regime-oracle router" (5 replacements: app:ci row, app:router mode list w/
  definition, tab:router caption + row, crossed paragraph "oracle regime router").
- Consistency check: all verified numbers intact; refs/braces/envs clean; no em-dashes.
  NOTE: revision prompt mislabeled +1.32 as "regime-oracle gain"; kept correctly attached to
  per-instance oracle (+0.84 remains regime-oracle). No new figures; Overleaf recompile +
  8-page check remain user-side.

## Bucket 24 (2026-08-04): Prof. Lim response — router-vs-regime-oracle + frontier-bank transfer
- Budget: $0.00 of $10.38 OpenRouter ceiling spent (cached votes only). Baseline reproduced
  first (142 tests; artifact checksums saved and re-verified after new work — unchanged).
- PRIMARY (Lim Q1): automatic logreg router minus regime-oracle router, paired CIs: in-domain
  +0.06 [-0.05,0.17]; pku -0.07; uf +0.13; rb +0.12 [0.00,0.23]; frontier bank -0.08 — ALL ns.
  Router is on par with being told the correct regime, everywhere. Regime-oracle sits -0.59..
  -0.83 below per-instance oracle (gap unreachable via regime knowledge).
- NEW 4th source fb (frontier_bank_transfer.py): 6-judge frontier bank (3 Gemini + GPT-5.6-sol/
  Opus-5/Grok-4.5), 400-item RB intersection, clean rho 0.458, cluster k=3; signature gate
  PASSES (global 0.60->0.66; subgroup lco inversion). Zero-shot deploy of open-bank-trained
  routers on 152 fb mixed: transferred fixed (bias_cluster) remains optimal; logreg -0.30*
  vs fixed but -0.08 ns vs regime-oracle; regime-oracle itself -0.19* below fixed; headroom
  +0.22. Caveat disclosed: 400-item pool -> instance overlap.
- Completed conditions matrix: logreg {-0.04, +0.56*, +0.58*, +0.09, -0.30*}; direct_ridge
  {+0.02, +0.71*, +0.51*, -0.74*, 0.00}. No variant uniformly never-worse; operative variable
  = aggregation-rule shift. direct_ridge rb failure -0.74 [-1.03,-0.45] is the honest boundary.
- Manuscript NOT modified this round. Report: outputs/router_upgrade/LIM_RESPONSE_REPORT.md
  (incl. paper-ready numbers). New CSVs: lim_router_vs_regime_oracle, fb_*, fb_transfer_results.

## Bucket 25 (2026-08-04): Final pre-submission manuscript revision (Lim-response results integrated)
- Abstract: router "statistically indistinguishable from one told the true regime; regime no
  longer determines the right filter". Contribution 3: router matches regime-informed choice ->
  limitation structural. sec:router rewritten: tab:rvsro result (-0.08..+0.13 all ns; regime-
  oracle trails per-instance oracle -0.59..-0.83 -> structural), judge-bank transfer sentence,
  no-variant-dominates statement; dropped redundant inline true-majority sentence (number
  retained in tab:mixed). Discussion: perfect-regime-knowledge-would-not-help + structural gap +
  judge-bank boundary + future work (fundamentally different judge-based tasks).
- Appendix: robustness paragraph corrected ("logistic router" never-worse; direct-ridge rb
  failure -0.74 [-1.03,-0.45] added); tab:lodo extended with direct-ridge column; NEW tab:rvsro
  (5 deployments, router-regime-oracle + structural column); NEW "Judge-bank transfer" paragraph
  + tab:fbtransfer (explicitly a judge-bank transfer, NOT a different task; 400-item overlap +
  k=3 + own clean-R disclosed; future-work sentence); roadmap + repro updated
  (frontier_bank_transfer.py, lim_router_vs_regime_oracle.csv, offsets 1e5-5e5).
- Checks: refs/labels/braces/envs clean; no em-dashes; all preserved numbers intact; all new
  table values verified verbatim; 142 tests pass. Main body grew ~10 lines across abstract/
  contributions/sec:router/discussion — USER: recompile on Overleaf and re-check the 8-page
  budget; no new figures to upload.

## Bucket 26 (2026-08-05): Task-generalization audit + plan (NO code/paper changes)
- Phase 1-3 deliverable for Lim's remaining concern (generalize routing beyond preference
  filtering to different LLM-judge TASKS). Full repo audit: pairwise assumptions localized to
  data/rewardbench.py, judges/prompts.py+base.py+*_judge.py, position_sensitivity_cluster,
  H1GAP, Source len fields, position-derived extended features. Everything downstream of
  (V,M,gold) is already pair-agnostic; 7 pre-registered BASE router features are position-free.
- Benchmarks verified via web: LLM-AggreFact (~59.7k human-labelled (doc,claim,binary) tuples,
  POINTWISE Supported/Unsupported — top pick), CodeJudgeBench (5,352 unit-test-verified
  chosen/rejected code pairs from LiveCodeBench-v6 — drop-in pairwise domain shift; 50-pair
  pilot gate required, 7-9B judges may be weak), JudgeBench (350 pairs — deployment-only probe).
- Plan: minimum experiment = pointwise factuality (1k stratified short-doc AggreFact items,
  existing bank + 2 NEW pointwise prompt styles, $0 API, ~8-16 TITAN GPU-h), signature gate
  BLOCKING, zero-shot task transfer from preference pools, router-vs-regime-oracle primary.
  Target design = leave-one-TASK-FAMILY-out over {preference, factuality, code} + optional
  $3-5 frontier factuality run (within $10.38) to separate task/bank/tier transfer.
- Effort: 400-600 LOC adapters (JudgeTaskBatch loader, pointwise prompts, swap-free
  correlation-cluster) + ~10 tests; no changes to verified pipeline. Full document:
  outputs/router_upgrade/GENERALIZATION_AUDIT_PLAN.md.

## Bucket 27 (2026-08-05, in progress): Pointwise factuality generalization (Stages 1-3)
- Stage 1 DONE: src/corrfilter/data/generic_task.py (JudgeTaskBatch + validate, FactualityItem,
  AggreFact loader [BLOCKED: gated on HF, Elias must accept terms at
  huggingface.co/datasets/lytang/LLM-AggreFact], VitaminC loader = active primary: 47,411
  binary items after NEI exclusion [7,268] + dedup [518], median doc 136 chars),
  judges/prompts_pointwise.py (pw_direct + pw_analysis, NOT-SUPPORTED-safe parser, malformed ->
  abstain), routing.correlation_cluster (swap-free frozen cluster), scripts/43 (staged runner,
  shuffle_position=False) + scripts/44 (competence gate). 28 new tests; suite 170 passing.
- Extended router features audited: all 35 are swap-free (pdis_* = pairwise-judge disagreement,
  not position); cluster definition is the only pairwise-specific component swapped out.
- Stage 2 smoke (15 items x 10 judges) PASSED: coverage 1.0, 0 abstains, majority 0.667 vs 0.60
  chance, clean FINAL formatting; ~1s/item direct, ~5s/item analysis.
- Stage 3 pilot (150 items) RUNNING in background; then Stage 4 main (1,000) ~7 GPU-h;
  Stage 5 gate (build_factuality_pools.py: controlled + real-mechanism regimes, blocking gate,
  cluster stability) and Stage 6 (run_factuality_transfer.py: zero-shot from preference pools,
  Tables C/D/E, +-0.5pt equivalence margin) scripts are written and ready.

## Bucket 27 COMPLETE (2026-08-05): Factuality generalization — Outcome C for routing, two headline findings
- Stage 4 main run DONE overnight (screen 'factuality', 8.5 GPU-h, $0): 1,000 VitaminC items x
  10 judges, coverage 0.999, majority 0.827 vs 0.574 chance. HEADLINE 1: evidence collapse
  TRANSFERS to pointwise factuality: error rho 0.354, n_eff 2.39/10 (frontier-tier ceiling on
  7-9B judges).
- Stage 5 construct-validity gate FAILED and the failure is robust: weak passes; global fails
  (vote-rho ceiling: drift +0.004 despite naive precision 0.95->0.80; CorrFilter delta<=0);
  subgroup PARTIALLY detectable (HEADLINE 2: inversion transfers, rho 0.537->0.439, LCO
  0.09->0.19) but not actionable (bias-cluster delta<0). Root cause: per-instance fixed-menu
  headroom = 0.16 pts mean, p90 = 0 — nothing to route between at ceiling dependence.
  Robust to k=5/k=3 cluster sensitivity AND to the matched-retention fix (protocol degeneracy
  found: non-abstain count > affirmed pool at this agreement level; added additive
  retention_mode="affirmed" param, default preserves verified behavior; 170 tests pass).
- Stage 6 routing correctly BLOCKED per protocol (run_factuality_transfer refuses without gate
  marker). Frontier API spend correctly NOT triggered ($0 total; OpenRouter $10 + Gemini key
  untouched). Report: outputs/router_upgrade/FACTUALITY_GENERALIZATION_RESULTS.md (Outcome C
  for regime-valid routing; recommends keeping the paper's future-work sentence; the two
  positive findings are appendix-note candidates pending Elias's approval).
- Screen session closed. AggreFact still gated (user action pending, optional).

## Bucket 28 (2026-08-05, in progress): Routing-feasibility screening campaign
- Appendix summary written (APPENDIX_FACTUALITY_SUMMARY.md); all factuality artifacts preserved.
- feasibility_screen.py: full Sec-3 metric block + Sec-4 gate with rejection reasons; validated
  free on VitaminC strata (all/real/synth): ALL REJECT (invalid regimes / zero headroom),
  consistent with the 1,000-item result — the screen reproduces the boundary case at 200 items.
- New candidates built: HaluEval-QA pointwise (200 balanced, model-generated hallucinations),
  CodeJudgeBench codegen pointwise (50 pairs -> 100 items, unit-test gold) + pairwise (50);
  4 new judge procedures (pw_decompose, pw_contradict, pwc_direct, pwc_trace); Bank B = +
  qwen-2.5-14b/1.5b (tier diversity). Pairwise screening supported via chance_override=0.5 +
  flip-mode injection (additive params).
- GPU campaign RUNNING in screen 'screening' (order: halueval -> bankD -> bankB -> cjb_point ->
  cjb_pair; ~6-7 GPU-h, $0 API); watcher armed; scripts/47 analysis ready. 178 tests pass.
- AggreFact component screening still blocked on user's HF access grant.

## Bucket 28 COMPLETE (2026-08-06): Screening campaign — all 11 candidates REJECT; decision D
- Campaign finished (~7 GPU-h, $0 API). Unified table (screening/screen_summary_all.csv):
  4 factuality sources, 5 bank compositions, 2 code protocols — none passes the full gate.
- Pre-registered hypotheses (INTERIM Sec 6-8) scored: Bank D largely CORRECT (A+D rho .287,
  headroom .11, rejects; surprise: D-only = best factuality bank, 5/6 gates, CorrFilter +0.61);
  Bank B directionally correct, headroom UNDERestimated (1.11 pts crossed the gate) but both
  regime gates fail (menu can't capture it); code competence-risk materialized exactly
  (pointwise .510 vs .500 chance; trace judges parse-dead 52-100% abstain) and pairwise gave
  the LOW-dependence anchor (rho .04, n_eff 7.3, zero headroom).
- Middle-band theory now anchored on BOTH sides + preference in-band. Deliverables A-E written
  (ROUTING_FEASIBILITY_SCREEN_IMPLEMENTATION, CROSS_TASK_CANDIDATE_SCREENING_RESULTS,
  FACTUALITY_SOURCE_BANK_SCREENING, CODE_JUDGE_PILOT_RESULTS, SELECTED_CROSS_TASK_EXPERIMENT
  = recommendation D with paths that could change it). 178 tests pass. Paper untouched.

## Bucket 29 (2026-08-06): Manuscript revision — cross-task campaign integrated top-to-bottom
- Followed the 10-step workflow: audit (MANUSCRIPT_CROSS_TASK_REVISION_AUDIT.md), generated
  artifacts (scripts/48 -> fig_actionability.png + table fragments from verified_master.csv,
  no hand-typed numbers), appendix-first edits, then main, then framing.
- NEW: app:crosstask (boundary case, retention-degeneracy documentation, gates incl. H_d
  formula as implemented, 14-row tab:screenall, near-misses, filter differentiation,
  exploratory Spearman analysis, 7 threats, repro); main sec:boundary + tab:boundary (6 rows) +
  fig:actionability; Discussion "necessary but not sufficient" block; conclusion +4 sentences;
  abstract +1 sentence; contributions bullet 3 extended; RQ4 extended (RQ5 NOT added —
  approval-flagged); stale future-work claims replaced (2 sites); scripts 01-48 / 178 tests.
- Checks: refs/braces/envs clean; no em-dashes; automated prose-number verification ALL OK vs
  raw outputs; 178 tests pass. AUTHOR ACTIONS: upload fig_actionability.png to Overleaf,
  recompile, check 8-page budget (est. +0.9-1.1 pages — cuts approval-flagged), review 7
  approval flags in MANUSCRIPT_CROSS_TASK_REVISION_REPORT.md.

## Bucket 30 (2026-08-06, in progress): Cross-task routing final experiment (bankD factuality)
- Target per instruction: procedure-diverse bankD (verified from raw: majority .835/.545, rho
  .234, n_eff 3.22, headroom .895, CorrFilter +.611, 5/6 gates). Missing piece = procedure-
  targeted subgroup regime.
- PREREGISTERED (SUBGROUP_MECHANISM_PREREG.md, written before any NEI vote): mechanism =
  natural VitaminC NOT-ENOUGH-INFO items (unsupported without contradiction; gold=0 per the
  prompts' own stated semantics; NEI stays excluded from the main two-class pool); frozen
  subgroup = the 5 pw_contradict judges (metadata; label-free clustering as sensitivity);
  H-M1 false-affirm gap >=15 pts, H-M2 within-group wrong-agreement coherence, H-M3 >=25%
  split-majority items; abstention-clean check; STOP RULE -> Fallback A/B if failed.
  Supportive prior: contradict < decompose accuracy in all 5 model pairs on clean screen.
- NEI pilot RUNNING (screen 'neipilot': 250 NEI items x 10 bankD judges, ~2-2.5 GPU-h, $0);
  watcher armed. scripts/49 (pilot+analysis+stop rule) and scripts/50 (gated full collection:
  bankD on 1,000-item main manifest + 400-item NEI pool, est. 8-11 GPU-h) ready.
- Paper NOT to be edited during this experiment (per instruction Sec 14).

## Bucket 30 COMPLETE (2026-08-06): bankD subgroup mechanism REJECTED by preregistered stop rule
- NEI pilot (250 items x 10 bankD judges, 2.5 GPU-h, $0): H-M1 FAIL (false-affirm gap +2.3 pts,
  needed >=15: contradict .208 vs decompose .185); H-M2 FAIL (within-group wrong-agreement .225
  < cross .301 — procedure identity forms no error cluster); H-M3 FAIL (split-majority 3.2% vs
  >=25%); abstention-clean TRUE (genuine failure, not parsing; power adequate: Wilson +-2.2 pts).
- Interpretation: both procedures reject unverifiable content ~80%; contradict judges' clean-
  pool deficit = over-REJECTING supported claims (opposite vulnerability). Composite-claim
  variants inherit the failure (cross-corr > within); not serially tested per stop-rule spirit.
- Full collection correctly never ran (scripts/50 code-gated on VALIDATED). No routing run.
  Paper untouched; its "cross-task routing remains unestablished" statement stays accurate.
- Deliverable: CROSS_TASK_ROUTING_FINAL_EXPERIMENT.md (15 sections; recommendation = fallbacks,
  both user-gated: A = AggreFact access click ($0, preferred); B = frontier code bank ($40-80,
  needs budget approval)). Repeated preregistered subgroup failures across 5 bank compositions
  + 1 natural mechanism reinforce the manuscript's actionability-boundary claim.

## Bucket 31 (2026-08-06, in progress): AggreFact component screening (access GRANTED)
- Elias granted Hub access: LLM-AggreFact dev = 30,420 items, 0 non-binary, 51 dupes dropped.
  Component audit: 8 eligible at <=2,000-char control (RAGTruth, ExpertQA, Lfqa, Reveal,
  FactCheck-GPT, AggreFact-XSum, AggreFact-CNN@139, ClaimVerify@147); TofuEval x2 + Wice
  excluded (context length, documented). Screening manifest: 1,186 items, label-stratified
  (six components exactly 50/50; CNN .91, ClaimVerify .71 by availability), seed 20270501.
- Vote collection RUNNING in screen 'aggrefact' (1,186 x 10 open-bank pointwise judges,
  ~8-9 GPU-h, $0, checkpointed; scripts/51); watcher armed. On completion: per-component
  feasibility screens (native_zeros regimes) -> gates -> selection table -> full routing
  experiment iff a component passes all gates.
- Priority-3 prepared (FRONTIER_CODE_PROPOSAL.md): 6 frontier judges pointwise code, pilot
  $<=2.5 gate then $15-25 full (OpenRouter share exceeds $10.38 balance -> top-up or 400-item
  design); NOT launched, approval checklist included. Paper untouched per instruction.

## Bucket 31 COMPLETE (2026-08-07): AggreFact component screening — all 8 REJECT; factuality work stops
- Collection: 1,186 items x 10 judges done (one disk-full crash on the final judge, recovered;
  disk freed 460MB -> 107GB by deleting 6 unreferenced Qwen3-family models, 106G).
- Preregistered gates applied unchanged. Verdict per component:
  AggreFact-CNN: LOW COMPETENCE (maj .914 == .914 majority-class chance; imbalance);
  XSum ZERO HEADROOM (0.00; CorrFilter -1.7); ClaimVerify ZERO HEADROOM (0.00);
  FactCheck-GPT ZERO HEADROOM (0.00); Lfqa ZERO HEADROOM (0.21); RAGTruth ZERO HEADROOM
  (0.22; CorrFilter -2.6, bias-cl -4.0); ExpertQA INVALID REGIMES (headroom 0.60 crosses the
  gate but CorrFilter -0.25 / bias-cluster -4.7 — inaccessible-headroom pattern again);
  Reveal TOO FEW ERRORS (13 majority-wrong; judges 0.913 vs 0.5).
- rho_err 0.32-0.52, n_eff 1.8-2.6 across ALL components incl. model-generated-claim sources:
  the high-dependence boundary now replicates across 12 pointwise-factuality sources/strata +
  5 bank compositions. No component selected; NO routing run; paper untouched.
- Per directive: factuality work STOPS. Decision point for Elias: approve/decline the
  frontier-code pilot (FRONTIER_CODE_PROPOSAL.md: Stage P $<=2.5 gate, Stage F $15-25 needs
  top-up or 400-item design). Full table: screening/screen_summary_aggrefact.csv.

## Bucket 32 (2026-08-07): Stage-P frontier code pilot — GATE FAIL, STOP (per directive)
- 60 pointwise code items x 6 pinned frontier judges. Per-judge acc: opus-5 .855, gpt-5.6-sol
  .817, gemini-3.1-pro .750, grok-4.5 .733, 3.6-flash .683, 3.5-flash-lite .529. Majority
  0.800 vs 0.50. GATE: majority>=.65 PASS; majority-wrong 12 < 15 FAIL; abstention flash-lite
  15% / opus 8.3% >= 5% FAIL; both classes PASS -> STOP, no Stage F.
- Substantive signals: rho_err 0.329, n_eff 2.27/6 (ratio .38), vote patterns 20/60, ALL FOUR
  filters identical precision .8947, preliminary headroom 0.00 pts. Frontier code judging lands
  at the HIGH-dependence boundary — competence excellent, dependence collapses, exactly the
  paper's tier finding (Discovery 2). The code task now anchors BOTH boundary modes by tier:
  local 7-9B = competence failure; frontier = dependence/headroom failure.
- Cost: OpenRouter EXACT $2.22 (balance 10.379 -> 8.162; reasoning tokens dominated: grok 36.2k
  out on 60 items); Gemini est. $0.20-0.35 (73k in-tokens/model, billed on Elias's key). NOTE:
  per-judge cost meter read $0 (PRICES table lacks these model ids) — ceiling held by luck;
  fix before any future paid run. Paper untouched.

## Bucket 33 (2026-08-07): Wrap-up — cost-meter fix, artifact preservation, appendix summary
- ALL new experiments STOPPED per Elias. No screens running; no GPU/API jobs.
- Cost-meter bug FIXED: PRICES now carries the six pinned frontier models (OpenRouter rates
  from the live /api/v1/models endpoint, validated: token reconstruction $2.224 vs billed
  $2.217; Gemini rates from the project's own billed run_log_*.json); live judges with unknown
  price now WARN and expose cost_known=False. 4 new regression tests incl. billing
  reconstruction; suite 182 passing.
- Frontier pilot artifacts write-protected (manifest, votes, runlog, result, log; chmod a-w).
- APPENDIX_FRONTIER_CODE_SUMMARY.md prepared (insertion PENDING AUTHOR APPROVAL; framed strictly
  as additional actionability-boundary evidence; main cross-task claim unchanged: routing works
  on preference tasks, cross-task routing remains unestablished). Manuscript untouched.

## Bucket 34 (2026-08-07): Frontier pilot inserted in appendix (author-approved); revision closed
- app:crosstask += frontier-pilot paragraph; tab:screenall regenerated to 15 rows; fig:action-
  ability regenerated (solid frontier marker); captions updated; costs $16.38 -> ~$18.9; tests
  178 -> 182 in paper. Constraints verified: RQ4 framing kept, ONE inverted-U statement, no
  frontier detail in main text, central conclusion grep-verified intact. All numbers from
  verified_master.csv (15 rows) via scripts/48. 182 tests pass. Report:
  FRONTIER_INSERTION_REVISION_REPORT.md. AUTHOR: re-upload fig_actionability.png, recompile,
  8-page check, Bucket-29 flags.

## Bucket 35 (2026-08-07): 8-page main body achieved; final compression complete
- LOCAL COMPILE ESTABLISHED (iclr2026 style found in tasfia/ repo; ref.bib+additions cover all
  26 citations; 4 remaining latex errors = 4 missing APPENDIX-only figure PDFs, draft-boxed).
- Main body: 11 pages -> EXACTLY 8 (References begin at top of p9). Semantic compression only:
  no font/margin/spacing changes; fig_actionability canvas 3.9->3.35in (content identical,
  regenerated from data). Sections compressed: Intro (opening+3 discoveries+RQs+contributions),
  Related Work, Theory exposition (statements kept verbatim), Findings 7.1-7.5 prose
  (numbers->table refs), 7.7, 7.8 (~40%), 7.9 -> ONE paragraph + figure (tab:boundary removed;
  all rows live in appendix tab:screenall), Discussion (recap+routing paragraphs merged, open
  questions cut), Conclusion tightened but CARRIES the central claim explicitly, Results
  roadmap + Practical-deployment paragraph removed (refs preserved elsewhere).
- Verified: all validated numbers present/unchanged (50-value check); ONE inverted-U statement;
  frontier appendix insertion intact; no undefined refs; 182 tests pass. Backup:
  main_pre8p_backup.tex. Local build files staged: iclr2026_conference.{sty,bst},
  iclr2026_conference.bib (copy of ref.bib) — Overleaf remains source of truth for styles.

# ICLR 2027 Reviewer Risk Register

Paper: *Agreement Is Not Evidence: Measuring Error Dependence and Effective Evidence in LLM Judge Consensus*
Prepared: 2026-08-11 (Phase 1, manuscript-only revision; no new experiments).

Each entry gives the likely criticism, its severity, and the status:

- **ANSWERED**: existing evidence in the paper already rebuts it; the rebuttal text can be lifted from the cited section.
- **PARTIAL**: existing evidence limits the damage but does not close the objection; the manuscript now concedes the gap explicitly.
- **OPEN**: requires new experiments (Phase 2) or cannot be closed at all.

---

## A. Novelty and positioning

### A1. "Correlated LLM errors are already known (Goel et al. 2025); this is a replication." — HIGH severity — ANSWERED
Goel et al. establish that error *similarity* rises with capability. The paper's contribution is
what follows from dependence inside consensus *filtering*, and the intro now enumerates eight
results that do not follow from similarity alone (effective evidence per verdict including
unanimity; dependence-aware panel testing; the preference-training route; distinct regimes; the
diagnostic inversion; aggregation-rule non-dominance; downstream DPO cost; actionability
boundary). Evidence: §1 ("What is new relative to the closest work"), §2, App. C.
**Residual risk**: a reviewer may still consider the frontier result (§7.3) alone incremental. The
defence is the preference-training route (§7.2), which is not in prior work.

### A2. "The design effect / n_eff is textbook survey statistics (Kish 1965)." — MEDIUM — ANSWERED
The paper says so itself (§5.1, "the algebra is the classical design effect ... instantiated for
judge banks"). The claimed novelty is the measurement in this setting and Theorem 1, which is not
classical. Do not defend the algebra; defend the measurement and the regime result.

### A3. "CorrFilter / bias-cluster filter / router are the real contribution and they are weak." — HIGH — ANSWERED (post-revision)
This was a genuine framing risk before the revision. The methods section now states that the two
filters are constructed as probes sensitive to one dependence structure and insensitive to the
other, so their inverted ranking is itself the measurement (§4, §7.6). The router is explicitly
framed as a probe of how much structure is exploitable, not a proposed method (§7.7).

---

## B. Measurement validity

### B1. "phi/Pearson correlation on binary error indicators has marginal-dependent range, so rho-bar is not comparable across banks with different accuracy." — HIGH — ANSWERED
This is the single most technically credible attack, because the frontier banks have far lower
marginal error rates than the open bank. The paper verifies with marginal-insensitive statistics
that agree (Yule's Q 0.96 vs 0.38; §7.3), and reports conditional co-failure with lift over the
independent rate as a second, scale-free view (Table 1). Cite both in a rebuttal.

### B2. "Dependence is just item difficulty: hard items are missed by everyone." — HIGH — PARTIAL
Existing evidence: difficulty stratification leaves within-stratum correlation 0.111, roughly half
the raw value (§7.1, Table 5); conditioning frontier pairs on open-bank difficulty leaves them
unchanged (0.465 to 0.467) while within-tier leave-pair-out conditioning roughly halves them
(§7.3, App. D). The paper concedes that about half the frontier dependence is tier-specific shared
difficulty. **Concession to make explicitly in rebuttal**: shared difficulty is not a competing
explanation but a mechanism; the evidence consequence (n_eff collapse) holds either way.

### B3. "Gold labels are noisy, and label noise manufactures correlated errors." — MEDIUM — ANSWERED
App. U.5: flipping 1/3/5/10% of gold labels moves rho-bar only 0.206 to 0.226 and n_eff 3.51 to
3.30 (40 seeds each); restricting to a pseudo-clean high-consensus subset yields *higher*
dependence (rho-bar 0.34). This is a clean answer, and it should be cited early in the response.

### B4. "n = 10 judges, 5 base models, one size class (7-9B): the spectrum is too small to talk about eigenstructure or regimes." — MEDIUM — PARTIAL
Existing evidence: 73 subbanks (App. I) with composition-invariant conclusions, a 16-judge
1.5B-14B scaling bank (n_eff 4.71, 29% of nominal), and a six-judge frontier bank. The paper
already declines to claim that numerical thresholds transfer (App. U.1, and now §8 Limitations
(iv)). **Not closable without a larger bank.**

### B5. "The frontier CIs rest on 30-75 residual errors per judge on 400 items." — MEDIUM — PARTIAL
Conceded in App. U.1 and reflected in the reported CIs, which are wide ([0.42, 0.68] for the
three-provider bank). The block comparison that carries the argument (cross-provider vs
frontier-open) is significant with margin (Table 8). Do not over-defend the point estimate 0.56.

### B6. "Panel A and Panel B of Table 1 are not comparable; the training rows are on different items with 43% abstention." — MEDIUM — ANSWERED (post-revision)
The revision states this in the caption, does not report majority accuracy for Panel B, and §7.2
now flags that the committed-vote accuracy gain for GRPO/DPO is selection-affected and is not what
the finding rests on. Previously the table implied a like-for-like accuracy comparison.

### B7. "GRPO/DPO judges abstain on 43% of items; the dependence increase could be a selection artifact of which items they commit on." — HIGH — PARTIAL
Existing evidence: the 4x-decode-budget re-run reduces abstention to 40% and still gives
rho-bar 0.227 vs base 0.178 (App. D, `abstention_robustness_summary.csv`). This is a mitigation,
not a control: abstention is never matched to the base bank's 2%. **This is the weakest link in
Discovery 2 and should be conceded in the rebuttal.** A Phase-2 experiment with forced-choice
decoding (no abstention option) would close it.

### B8. "Correctness correlation equals error correlation 'by construction', so the correlated-non-errors result is a tautology." — LOW — ANSWERED
It is stated as an identity in the paper (Pearson invariance under joint complementation, §7.1).
The non-tautological content is the *effective count*: 6.35 nominal correct votes are worth 2.85,
and unanimity is worth 3.51. Frame the identity as the reason the result is unavoidable, not as
the result.

---

## C. Regimes and the taxonomy

### C1. "Two of the three regimes are synthetic. You built the phenomenon you then detect." — HIGH — PARTIAL
Conceded in App. U.2 and §8 Limitations (i). Partial mitigations that exist: the position attack
targets the bank's *measured* strongest vulnerability rather than an arbitrary one (slot gaps up
to 70 points, App. K); the subgroup signature appears without poisoning on 47% of majority-wrong
RewardBench items and 41% on UltraFeedback (§7.6); the weak-dependence regime uses
heuristic-preferred flips approximating reward hacking. **Not closable in Phase 1.** The honest
position, now in the paper, is that the regimes are empirically separable mechanisms, not a claim
about natural prevalence.

### C2. "Why three regimes? The taxonomy is not shown to be complete." — MEDIUM — ANSWERED
App. U.6 explicitly declines completeness and names the missing cases (multi-cluster, drift, mixed
regimes). The claim in the paper is separability, not exhaustiveness.

### C3. "The 47%/41% natural-subgroup result is cherry-picked; you only look at majority-wrong items." — MEDIUM — PARTIAL
The paper already labels these "existence cases, not a population prevalence" (§7.6). A reviewer
may still object that the denominator is chosen post hoc. The stability check across clustering
granularities and the UltraFeedback replication are the available defence.

### C4. "Position bias is the only validated subgroup vulnerability, so the 'regime' is really 'position bias'." — HIGH — PARTIAL
Conceded in App. U.4 and §8 Limitations (ii). The framework keys on cluster membership rather than
on position, but only position bias is empirically validated. **Phase-2 candidate**: instantiate
the subgroup regime with a verbosity or sycophancy cluster.

---

## D. Aggregation, filtering, and routing

### D1. "Routing does not work; why is it in the paper?" — HIGH — ANSWERED (post-revision)
The revision leads §7.7 with the negative result, states that classification routing does not
universally beat the best in-domain fixed filter, and states in §7.9 that successful general
cross-task routing is not established by this paper. Routing is now presented as a probe and a
boundary finding. The positive result is scoped to one condition (aggregation-rule shift across
preference datasets, +0.56 and +0.58 points).

### D2. "The filter gains are small (a few points) and evaluated at one matched retention." — MEDIUM — PARTIAL
Effect sizes with CIs are reported (-4.2 [2.8, 5.6] and -6.0 [4.4, 7.7] false-retention points;
+4.8 [+3.6, +6.1] precision points). Retention is matched to the 0.75 supermajority throughout and
retention curves exist (App. J). A reviewer wanting a full retention sweep in the main body has a
fair point; the material is in the appendix.

### D3. "The theorem is trivial: of course no fixed rule is optimal if you construct two opposite regimes." — MEDIUM — PARTIAL
Theorem 1 is a formalisation of an empirically observed inversion, not a surprise in isolation.
The defence is that the inversion is measured (rho-bar falls 0.22 to 0.08 while precision
collapses) before it is proved, and that the practical consequence (a diagnostic that inverts) is
not obvious a priori. Do not oversell the theorem.

### D4. "Adaptive CorrFilter needs in-domain gold labels, so the comparison to label-free naive majority is unfair." — HIGH — ANSWERED
Conceded and quantified: App. U.3 reports graceful degradation as the calibration set shrinks
(25-200 labels), and a fully label-free position-sensitivity cluster achieves the same +4.8
precision points as the learned and H1-imported constructions
(`outputs/learned_bias_cluster/learned_bias_cluster_results.csv`). The label-free result is the
correct rebuttal and should be stated first.

### D5. "Internal inconsistency: `outputs/dpo_judges/table_dpo_vs_base.csv` reports a 20.27-point CorrFilter gain for the base matched bank, but the paper reports 4.2 (GRPO) and 6.0 (DPO)." — **RAISED TO HIGH** — RESOLVED for the artifact, **OPEN for one paper number**
Traced and resolved 2026-08-11; see `outputs/dpo_judges/D5_RESOLUTION.md`. Root cause:
`majority_consensus` defaults to `tie_break=1` while the caller passes `gold = 1` for every item,
so split-vote items are scored correct by construction; `consensus_level` ranks those items last,
so the baseline discards them and CorrFilter is credited for retaining them. Diagnostic signature:
the baseline's false-retention rate at 60% retention (0.4226) exceeds the bank's unfiltered
majority error rate (0.2666). A second defect: retention is computed on all manifest items rather
than the evaluable pool, so "0.60 retention" is 61% of the base bank's evaluable pool but 88% of
the trained banks'. The artifact CSV now reports `n/a` for that cell, matching the paper.

**The part that is still open.** The same defect reaches the two gains the paper does report.
Under the corrected tie-free protocol, DPO survives (+6.01 becomes +6.52 [2.76, 10.33]) but
**GRPO does not: +4.20 [2.87, 5.64] becomes +3.17 [-0.19, +6.57]**, so its CI now covers zero.
A reviewer who reruns the released analysis would find this. Three options are laid out in the
resolution document; the smallest is to drop the CorrFilter-gain column from Table 4 and let §7.2
rest on the dependence increase, which the Phase 1 revision already made the load-bearing claim.
Scope is bounded: the dependence measurements use a different code path and are unaffected, and
the +4.8 bias-cluster result in the abstract is not implicated because its gold vector is not
constant. **Authors' decision required before submission.**

---

## E. Downstream and impact

### E1. "The downstream DPO validation is 0.5B-1.5B scale and the accuracy effect is null." — HIGH — PARTIAL
The paper states this itself (App. U.5): margins degrade monotonically and significantly from 20%
contamination (-0.42 to -0.65), while preference accuracy stays within noise, and it explains the
earlier filter-comparison null retrospectively. **Concede**: policy-level validation at larger
scale is open. The claim in the paper is a reward-margin effect, not a capability effect, and §8
now says so in those words.

### E2. "The evaluation-side claim (28% of panel verdicts flip) is the most actionable result and is buried." — MEDIUM — PARTIAL
It is now in the abstract and in Discovery 1. It remains a subsection paragraph in §7.1 with the
figure in App. D. If space allows in a later revision, promoting `fig:sigflip` to the main body is
the highest-value figure swap available.

### E3. "Frontier models are proprietary and will drift; the numbers are not reproducible." — MEDIUM — ANSWERED
Conceded in App. U.1 with pinned model IDs, response-metadata verification, and a note that one
Gemini ID is a preview. The open-bank results, which carry Discovery 1 and Discovery 3, are fully
reproducible.

---

## F. Presentation

### F1. "53 pages of appendix for a 9-page paper." — MEDIUM — no action needed
Not a rejection reason at ICLR, but the main body must be self-contained. Post-revision, every
main-body claim carries an explicit appendix pointer.

### F2. "Too many acronyms and constructs (CorrFilter, bias-cluster filter, regime router, headroom, n_eff, alpha_subset)." — LOW — ANSWERED (post-revision)
Terminology was normalised in this pass: three regimes are named once in §1 (global co-failure,
weak dependence, vulnerable subgroup) with the same names used in the §7 headings, and
"effective ensemble size (n_eff)" is the single formal term with "effective evidence" reserved as
its gloss.

---

## Priority ranking for a rebuttal

1. B1 (phi range) and B2 (difficulty) - the two attacks most likely to come from a statistically
   sophisticated reviewer, and both have prepared answers.
2. B7 (abstention confound in GRPO/DPO) - the weakest evidential link; concede and scope.
3. C1/C4 (synthetic regimes, position bias) - concede, point to the natural-subgroup existence
   cases and the label-free cluster result.
4. A1 (novelty vs Goel et al.) - answer with the eight-item list, not with a general claim.
5. D1 (routing) - the revision has already removed the overclaim; do not re-argue it.

# Paper edits: removing script-level detail

`main.tex` names individual scripts in fourteen places. After the repository
reorganisation those names no longer resolve, and rather than track the new ones the
manuscript drops the file-level detail entirely: it now describes *stages*, and every
substantive fact — seeds, costs, item counts, table references, output paths, test
counts — is preserved exactly.

`main.tex` is not in this repository (`.gitignore` excludes `*.tex`, and the working copy
lives Overleaf-side). Apply the edits below there. Line numbers are from the 2026-08-11
revision and are approximate; the find-text is exact.

---

## 1. Vulnerable-subgroup appendix (~L1395, ~L1402)

**Find**

```latex
A granularity sweep ($k\in\{2,3,4,5\}$;
\texttt{42\_natural\_subgroup\_stability.py}) bounds the sensitivity: prevalence
```

**Replace**

```latex
A granularity sweep ($k\in\{2,3,4,5\}$) bounds the sensitivity: prevalence
```

**Find**

```latex
We report these as existence cases within our banks
(\texttt{find\_natural\_subgroup\_cases.py}); they do not establish prevalence in
```

**Replace**

```latex
We report these as existence cases within our banks; they do not establish prevalence in
```

## 2. Bank-composition robustness (~L1799)

**Find**

```latex
signatures on $73$ subbanks of the cached votes (scripts \texttt{20}), with no new
```

**Replace**

```latex
signatures on $73$ subbanks of the cached votes, with no new
```

## 3. Confidence intervals (~L1856)

**Find**

```latex
majority and the best single filter (scripts \texttt{21}). A gain is
```

**Replace**

```latex
majority and the best single filter. A gain is
```

## 4. Cross-task campaign reproducibility (~L2771–2788)

Replace the whole `\paragraph{Reproducibility.}` block:

```latex
\paragraph{Reproducibility.}
Deterministic manifests (seeds \texttt{20260810-12}, \texttt{20270201},
\texttt{20270210}, \texttt{20270215}, \texttt{20270220}, \texttt{20270301}); votes,
gate records, and per-candidate JSON under \texttt{outputs/factuality\_bank/},
\texttt{outputs/screening\_bank/}, and \texttt{outputs/router\_upgrade/screening/};
the unified table and the actionability figure regenerate from those records.
Campaign compute: $\approx 15$ GPU-hours on one TITAN RTX (factuality bank $+$
screening) at \$$0$ external API spend; the follow-up frontier pilot adds
$\approx$\$$2.5$; $182$ unit tests pass, including screening-gate, leakage,
determinism, generator-marginal, and cost-meter regression tests.
```

## 5. `app:repro-code` (~L3081–3103)

Replace the whole subsection:

```latex
\subsection{Code Organization and Reproduction Workflow}
\label{app:repro-code}
The release is an ordered pipeline over cached judge votes, grouped by experimental
stage. Its stages build the stratified calibration set and run the judge bank;
estimate $R$, $\neff$, and the eigen decomposition; run the correlated-failure-injection
banks and CorrFilter, including adaptive-$R$ estimation; build and evaluate
synthetic-poisoned UltraFeedback and its position-aligned variant; apply the
bias-cluster and learned-cluster filters; train and evaluate the regime routers;
recompute bank-composition robustness and bootstrap confidence intervals; run the
preference-optimization (GRPO/DPO), cross-dataset replication, and frontier-bank
experiments; and produce the correlated non-error analysis, the panel-verdict
significance analysis, and the natural-subgroup stability sweep. Library modules hold
the correlation math, the $\asub$ and $\beta_C$ scorers, the adaptive-$R$ estimators,
the shared dependence and bootstrap estimators, the routing diagnostics, and the
API-judge adapters; $178$ unit tests cover these paths. The workflow is: (1)~populate
the open-bank vote caches once (GPU) and the API caches once (paid, opt-in, and
dry-run by default); (2)~regenerate any table or figure from those caches (CPU).
Together the stages reproduce every quantitative result in the paper.
```

## 6. `app:repro-structure` (~L3105–3120)

Replace the whole subsection:

```latex
\subsection{Repository Structure}
\label{app:repro-structure}
The release is organised as: \texttt{configs/} (judge-bank and calibration-set
definitions in YAML, including the frontier banks); \texttt{scripts/} (the ordered
pipeline, grouped by experimental stage, plus shell launchers);
\texttt{src/corrfilter/} (library: correlation math, the $\asub$ and $\beta_C$
scorers, adaptive-$R$ estimators, CFI bias registry, the shared routing module, the
screening gates, and the judge package with the HF judge and the Gemini and OpenRouter
API-judge adapters); \texttt{data/cache/} (downloaded datasets); \texttt{experiments/}
(per-experiment artefacts: the calibration manifest, cached votes,
\texttt{correlation.npz}, \texttt{h1\_result.json}, the routing pools, and the
downstream DPO checkpoints); \texttt{outputs/} (one subdirectory per experiment, each
holding result CSVs, a \texttt{summary.md}, and a \texttt{figures/} folder, including
the frontier caches under \texttt{outputs/gemini\_bank/} and
\texttt{outputs/openrouter\_bank/}); and \texttt{tests/} ($178$ unit tests). Every
paper number traces to a CSV/JSON under \texttt{experiments/} or \texttt{outputs/}.
```

## 7. `app:repro-frontier` — Gemini paragraph (~L3141–3143)

**Find**

```latex
extension of all three Gemini judges to the full $1{,}178$-item calibration set. Reproduce with
\texttt{scripts/33\_run\_gemini\_judges.py} (dry-run by default; \texttt{--live} to
spend) and \texttt{scripts/34\_gemini\_bank\_analysis.py}.
```

**Replace**

```latex
extension of all three Gemini judges to the full $1{,}178$-item calibration set. The
runner is dry-run by default; live calls are opt-in.
```

## 8. `app:repro-frontier` — OpenRouter paragraph (~L3153–3158)

**Find**

```latex
under \texttt{outputs/openrouter\_bank/} (git-ignored); reproduce with
\texttt{scripts/38\_run\_openrouter\_judges.py},
\texttt{39\_openrouter\_pilot\_analysis.py}, and
\texttt{40\_multiprovider\_full\_analysis.py} (block decomposition,
difficulty conditioning, and the PKU truncation check). Total measured external
```

**Replace**

```latex
under \texttt{outputs/openrouter\_bank/} (git-ignored), together with the block
decomposition, difficulty conditioning, and the PKU truncation check. Total measured external
```

## 9. `app:repro-frontier` — routing paragraph (~L3163–3189)

**Find** (from "The non-error analysis" to the end of the subsection)

```latex
pilot, i.e.\ $\approx\$18.9$ in total. The non-error analysis
(Table~\ref{tab:nonerror}) is \texttt{scripts/35\_nonerror\_correlation.py}. The
original routing evaluations (\texttt{scripts/36\_routing\_accuracy.py},
...
screen adds $36$ more for the generic adapter, pointwise parsers,
feasibility gates, and injection modes, $178$ total), no API
calls, and CPU-only runtime of a few minutes over the cached votes.
```

**Replace**

```latex
pilot, i.e.\ $\approx\$18.9$ in total. The original routing evaluations are retained
for the audit trail but superseded (App.~\ref{app:router}). The corrected study
comprises the crossed regime-identification benchmark (Table~\ref{tab:routeracc};
instance seeds from base \texttt{20261001}), the mixed-regime deployment benchmark
(Tables~\ref{tab:mixed}--\ref{tab:mixedmix}; base \texttt{20261101}), the
three-dataset extension with its cross-dataset deployment, failure decomposition, and
calibration-budget studies, and the judge-bank transfer
(Tables~\ref{tab:routedeploy}--\ref{tab:fbtransfer}; counter offsets
\texttt{1e5}--\texttt{5e5} keep RNG streams disjoint; per-instance predictions, regime
probabilities, and selected/oracle filters saved to \texttt{ext\_perinstance.csv}; the
router-versus-regime-oracle comparison to
\texttt{lim\_router\_vs\_regime\_oracle.csv}). All use bootstrap seed
\texttt{20260706} and shared routing primitives factored into a single library module
(parity-tested against the original regime routers; the routing study adds $11$ unit
tests covering oracle computation, pooled metrics, generator marginal matching,
determinism, and train/deploy leakage, and the cross-task screen adds $36$ more for
the generic adapter, pointwise parsers, feasibility gates, and injection modes, $178$
total), no API calls, and CPU-only runtime of a few minutes over the cached votes.
```

## 10. `app:repro-runtime` (~L3192–3196)

**Find**

```latex
The one-time GPU stage (\texttt{02} and the CFI/UF bank runs) dominates wall-clock;
it serves one $7$--$9$B judge at a time under \texttt{nf4} on a single TITAN RTX and
caches votes to parquet. Every analysis script (\texttt{03--04}, \texttt{06--21}) is
CPU-only over those caches and completes in seconds to a few minutes each; the
$2000$-resample confidence-interval pass (\texttt{21}) is the slowest.
```

**Replace**

```latex
The one-time GPU stage (vote generation for the open bank and the CFI/UF banks)
dominates wall-clock; it serves one $7$--$9$B judge at a time under \texttt{nf4} on a
single TITAN RTX and caches votes to parquet. Every analysis stage is CPU-only over
those caches and completes in seconds to a few minutes each; the $2000$-resample
confidence-interval pass is the slowest.
```

## 11. Reproduction pitfalls, bootstrap seeds (~L3216)

**Find**

```latex
\item \textbf{Bootstrap seeds.} Use \texttt{20260601} (precision CIs, \texttt{21})
```

**Replace**

```latex
\item \textbf{Bootstrap seeds.} Use \texttt{20260601} (precision CIs)
```

## 12. Downstream dose–response (~L3495)

**Find**

```latex
scale (e.g.\ on PKU-SafeRLHF) remains open. Pipeline:
\texttt{experiments/downstream\_dpo\_validation/} (\texttt{config\_scaling.yaml},
\texttt{prepare\_scaling\_datasets.py}, \texttt{eval\_scaling\_cis.py}).
```

**Replace**

```latex
scale (e.g.\ on PKU-SafeRLHF) remains open. Pipeline artefacts and configuration:
\texttt{experiments/downstream\_dpo\_validation/} (\texttt{config\_scaling.yaml}).
```

---

## Notes

- Test counts are left exactly as the manuscript has them (`$178$`, `$182$`). They were
  already stale relative to this repository before the reorganisation, which is a
  separate decision from the one made here.
- After applying, `grep -n "\.py\|texttt{[0-9][0-9]" main.tex` should return only the
  judge-model names on the experimental-setup line and the seed values — no filenames
  and no script numbers.
- Nothing else in the manuscript refers to the code layout, so no other section needs
  touching. Numbers, tables, figures, and claims are unchanged throughout.

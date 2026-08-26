"""D5 diagnostic — trace and quantify the tie-break artifact in paired FRR-gain numbers.

Additive, analysis-only, deterministic. Reads cached votes; runs no inference and
writes nothing outside outputs/dpo_judges/. Does NOT modify any existing pipeline.

Background
----------
`bootstrap_ci.paired_frr_gain` scores the naive-consensus baseline and CorrFilter with
`majority_consensus(V, M)`, whose default `tie_break=1`. On the RewardBench calibration
manifest the caller passes `gold = np.ones(n)` (the decoded convention: vote 1 == affirm
the true winner). Tie items (equal committed votes for 0 and 1) are therefore predicted
1 and scored *correct by construction*, so any filter that retains tie items is credited
for free. `consensus_level` ranks those same items last (level 0.5), so the baseline
systematically discards the free-credit items while CorrFilter retains some of them.

A second, independent comparability problem: the retained count is
`k = round(retention * n_items_in_manifest)`, not `retention * n_evaluable`. With the
trained banks abstaining on ~43% of items, `k = 717` is 61% of the base bank's evaluable
pool but 88% of the trained banks', so a nominally "matched 0.60 retention" is not
matched across banks.

Usage
-----
  python scripts/robustness/tie_break_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.analysis.dependence import estimate_R, item_ids_from_manifest, load_votes
from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus, vote_fraction
from corrfilter.cfi.corrfilter_score import corrfilter_score

ROOT = Path(__file__).resolve().parents[2]


SEED = 20260706          # same bootstrap seed as bootstrap_ci.DEFAULT_SEED
B = 2000
MANIFEST = "experiments/h1_measurement/results/calibration_manifest.parquet"
BASE_IDS = ["qwen-2.5-7b::pairwise", "qwen-2.5-7b::likert",
            "mistral-7b-v0.3::pairwise", "mistral-7b-v0.3::likert",
            "phi-3.5-mini::pairwise", "phi-3.5-mini::likert"]
BANKS = {
    "base_matched": ("configs/judge_bank.yaml", BASE_IDS),
    "grpo_trained": ("configs/grpo_judge_bank.yaml", None),
    "dpo_trained": ("configs/dpo_judge.yaml", None),
}


def _topk(score, k):
    n = score.shape[0]
    keep = np.zeros(n, dtype=bool)
    if k > 0:
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
        keep[order[:min(k, n)]] = True
    return keep


def _frr(keep, correct, evaluable, idx):
    kk = keep[idx] & evaluable[idx]
    nk = int(kk.sum())
    return np.nan if nk == 0 else 1.0 - float((kk & correct[idx]).sum()) / nk


def paired_gain(V, M, R, gold, retention, *, drop_ties, k_on_evaluable):
    """Paired consensus-minus-CorrFilter FRR gain in points, with 95% bootstrap CI.

    drop_ties=False, k_on_evaluable=False reproduces the published protocol exactly.
    """
    n = V.shape[0]
    p1 = vote_fraction(V, M)
    level = consensus_level(V, M)
    maj = majority_consensus(V, M)
    evaluable = maj != ABSTAIN
    if drop_ties:
        evaluable = evaluable & (p1 != 0.5)
    correct = evaluable & (maj == gold)
    sc_cons = np.where(evaluable, level, -np.inf)
    sc_cf = np.where(evaluable, corrfilter_score(V, M, R, maj).score, -np.inf)
    k = int(round(retention * (evaluable.sum() if k_on_evaluable else n)))
    keep_c, keep_f = _topk(sc_cons, k), _topk(sc_cf, k)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = _frr(keep_c, correct, evaluable, idx) - _frr(keep_f, correct, evaluable, idx)
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    a = np.arange(n)
    return {
        "retention": retention, "k": k, "n_evaluable": int(evaluable.sum()),
        "consensus_frr": round(float(_frr(keep_c, correct, evaluable, a)), 4),
        "corrfilter_frr": round(float(_frr(keep_f, correct, evaluable, a)), 4),
        "gain_pts": round(float(diffs.mean()) * 100, 2),
        "ci_low_pts": round(float(lo) * 100, 2),
        "ci_high_pts": round(float(hi) * 100, 2),
    }


def main():
    item_ids = item_ids_from_manifest(ROOT / MANIFEST)
    gold = np.ones(len(item_ids), dtype=np.int8)
    rows = []
    for label, (cfg, ids) in BANKS.items():
        V, M, lids = load_votes(ROOT / cfg, item_ids, ids)
        R, estimator, _, _ = estimate_R(V, M, gold)
        p1 = vote_fraction(V, M)
        n_tie = int(((p1 == 0.5) & (M.astype(bool).sum(1) > 0)).sum())
        for retention in (0.50, 0.60, 0.70):
            for proto, kw in (("published", dict(drop_ties=False, k_on_evaluable=False)),
                              ("tie_free", dict(drop_ties=True, k_on_evaluable=True))):
                r = paired_gain(V, M, R, gold, retention, **kw)
                rows.append({"bank": label, "n_judges": len(lids), "estimator": estimator,
                             "n_tie_items": n_tie, "protocol": proto, **r})
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/dpo_judges/d5_tie_break_diagnostic.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Step 2: expanded aggregation family on vitc_bankB_mix, cached votes only.

CPU-only, no judge inference. Rebuilds the SAME screening instances the campaign used
(same seeds, same calibration split, same retention_mode="affirmed") and scores both the
original four-filter family and an expanded family on each instance.

Question: does a richer aggregation family capture the ~1.111 points of per-instance-oracle
headroom that the original four filters left on the table?

Expanded methods, all using ONLY information the original protocol already grants (the
same SG_SIZE small-gold calibration split that corrfilter_small_gold_R uses, or nothing):
  accuracy_weighted   weights proportional to calibrated judge accuracy
  reliability_weighted log-odds (Naive-Bayes optimal) weights under independence
  dawid_skene         EM over per-judge confusion matrices, unsupervised
  gls_corr_weighted   w = R^-1 1, the minimum-variance weights implied by the paper's own
                      variance-inflation lemma, using the same small-gold R as CorrFilter

No method sees an evaluation-split label. Nothing is tuned on the test split.

Outputs -> outputs/expanded_aggregation/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "router_upgrade"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_crossed_pool as bcp  # noqa: E402
import feasibility_screen as fs  # noqa: E402
from corrfilter.cfi.adaptive_r import small_gold_R  # noqa: E402
from corrfilter.cfi.consensus import (ABSTAIN, majority_consensus,  # noqa: E402
                                      supermajority_consensus)
from corrfilter.data.generic_task import batch_from_cache  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402


class MultiCache:
    """First-hit-wins union of several VoteCaches (mirrors scripts/cross_task_screen/analyze_screening_campaign.py)."""

    def __init__(self, *dirs):
        self.caches = [VoteCache(d) for d in dirs]

    def load(self, lid):
        for c in self.caches:
            df = c.load(lid)
            if df is not None:
                return df
        return None

OUT = ROOT / "outputs/expanded_aggregation"
SB = ROOT / "outputs/screening_bank"
FBV = ROOT / "outputs/factuality_bank/votes"
ORIGINAL = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
EXPANDED = ["accuracy_weighted", "reliability_weighted", "dawid_skene", "gls_corr_weighted"]
SEED = 20260706
B = 20000


# ----------------------------------------------------------------- new aggregators
def _weighted_margin(O, M, w):
    """Signed weighted vote margin per item; positive favours label 1."""
    Mb = M.astype(bool)
    W = w[None, :] * Mb
    tot = W.sum(1)
    ones = (O * W).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = np.where(tot > 0, ones / np.maximum(tot, 1e-12), 0.5)
    return p1 - 0.5


def accuracy_weights(O, M, gold, cal_idx):
    Mb = M[cal_idx].astype(bool)
    acc = []
    for j in range(O.shape[1]):
        m = Mb[:, j]
        acc.append(float((O[cal_idx][m, j] == gold[cal_idx][m]).mean()) if m.any() else 0.5)
    return np.clip(np.array(acc), 0.01, 0.99)


def reliability_weights(acc):
    """Naive-Bayes optimal log-odds weights; the classic independence-assuming rule."""
    return np.log(acc / (1.0 - acc))


def gls_weights(R):
    """w = R^-1 1, minimum-variance weights under the paper's own variance model."""
    n = R.shape[0]
    try:
        w = np.linalg.solve(R + 1e-6 * np.eye(n), np.ones(n))
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(R) @ np.ones(n)
    if w.sum() < 0:
        w = -w
    return np.clip(w, 0.0, None)          # keep non-negative so it stays a voting rule


def dawid_skene(O, M, n_iter=60, tol=1e-6):
    """Unsupervised EM with per-judge 2x2 confusion matrices. Returns P(label=1)."""
    Mb = M.astype(bool)
    n, m = O.shape
    p = np.clip(np.where(Mb, O, 0).sum(1) / np.maximum(Mb.sum(1), 1), 1e-3, 1 - 1e-3)
    prev = None
    for _ in range(n_iter):
        prior = np.clip(p.mean(), 1e-3, 1 - 1e-3)
        cm = np.zeros((m, 2, 2))
        for j in range(m):
            s = Mb[:, j]
            if not s.any():
                cm[j] = 0.5
                continue
            for t, wt in ((1, p[s]), (0, 1 - p[s])):
                d = wt.sum()
                v1 = (wt * O[s, j]).sum()
                cm[j, t, 1] = (v1 + 1.0) / (d + 2.0)      # Laplace smoothing
                cm[j, t, 0] = 1.0 - cm[j, t, 1]
        l1 = np.full(n, np.log(prior))
        l0 = np.full(n, np.log(1 - prior))
        for j in range(m):
            s = Mb[:, j]
            v = O[s, j].astype(int)
            l1[s] += np.log(np.clip(cm[j, 1, v], 1e-9, 1))
            l0[s] += np.log(np.clip(cm[j, 0, v], 1e-9, 1))
        mx = np.maximum(l1, l0)
        p = np.exp(l1 - mx) / (np.exp(l1 - mx) + np.exp(l0 - mx))
        if prev is not None and np.max(np.abs(p - prev)) < tol:
            break
        prev = p.copy()
    return p


# ----------------------------------------------------------------- per-instance eval
def evaluate_all(src, O, Md, gold, counter, R_h1, retention_mode="affirmed"):
    """Original family via the campaign's own evaluate(), plus the expanded family,
    scored on the identical evaluation split at the identical matched retention."""
    metrics, fvec, pred_hard, diag = bcp.evaluate(src, O, Md, gold, counter, R_h1,
                                                  retention_mode=retention_mode)
    rng = np.random.default_rng(bcp.BASE + 104729 * (counter + 1))
    cal_idx = rng.choice(len(gold), size=bcp.SG_SIZE, replace=False)
    ev = np.ones(len(gold), bool)
    ev[cal_idx] = False
    R_sg = small_gold_R(O, Md, gold, cal_idx)
    Oe, Me, ge = O[ev], Md[ev], gold[ev]
    label = majority_consensus(Oe, Me)
    sm = supermajority_consensus(Oe, Me, 0.75)
    n_match = max(int((sm == 1).sum() if retention_mode == "affirmed"
                      else (sm != ABSTAIN).sum()), 1)
    pool = label == 1

    acc = accuracy_weights(O, Md, gold, cal_idx)
    scores = {
        "accuracy_weighted": _weighted_margin(Oe, Me, acc),
        "reliability_weighted": _weighted_margin(Oe, Me, reliability_weights(acc)),
        "gls_corr_weighted": _weighted_margin(Oe, Me, gls_weights(R_sg)),
        "dawid_skene": dawid_skene(Oe, Me) - 0.5,
    }
    for name, sc in scores.items():
        keep = bcp.keep_at_matched_retention(np.where(pool, sc, -np.inf), pool, n_match)
        metrics[name] = bcp.filter_metrics(keep, ge)
    return metrics


def paired_ci(a, b, seed=SEED, B=B):
    d = (np.asarray(a, float) - np.asarray(b, float)) * 100.0
    d = d[~np.isnan(d)]
    if d.size < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(B)])
    return float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(SB / "manifest_vitc_bankscreen.parquet")
    cache = MultiCache(FBV, SB / "votes/vitc_extra")
    MODELS = ["llama-3.1-8b", "qwen-2.5-7b", "gemma-2-9b", "mistral-7b-v0.3", "phi-3.5-mini"]
    lids = ([f"{m}::{s}" for m in MODELS for s in ("pw_direct", "pw_analysis")]
            + [f"{m}::{s}" for m in ("qwen-2.5-14b", "qwen-2.5-1.5b")
               for s in ("pw_direct", "pw_analysis")])

    batch = batch_from_cache(man, cache, lids)
    V, M, gold = batch.judge_votes, batch.availability, batch.gold_labels
    keep = M.sum(1) > 0
    V, M, gold = V[keep], M[keep], gold[keep]
    src, R = fs._mk_source("vitc_bankB_mix", V, M, gold, lids, cluster_k=5)

    inst_n = min(fs.INST_SIZE, src.n)
    sg_orig = bcp.SG_SIZE
    bcp.SG_SIZE = min(100, max(30, inst_n // 3))
    rows, counter = [], 0
    for regime in ("weak", "global", "subgroup"):
        for c, O, Md, g in fs._instances(src, gold, R, regime, 0.20, counter,
                                         inject_mode="native_zeros"):
            metrics = evaluate_all(src, O, Md, g, c, R)
            for f, m in metrics.items():
                rows.append({"instance": f"vitc_bankB_mix_{regime}_{c}", "regime": regime,
                             "method": f, **m})
        counter += fs.N_INST
    bcp.SG_SIZE = sg_orig

    md = pd.DataFrame(rows)
    md.to_csv(OUT / "bankB_mix_methods.csv", index=False)
    piv = md.pivot_table(index="instance", columns="method", values="precision")
    regime_of = md.drop_duplicates("instance").set_index("instance")["regime"]

    # ---- headroom under each family ----
    orig_best = piv[ORIGINAL].max(1)
    exp_best = piv[ORIGINAL + EXPANDED].max(1)
    oracle = piv["oracle_filter"]
    naive = piv["naive_majority"]
    head = pd.DataFrame({
        "instance": piv.index, "regime": [regime_of[i] for i in piv.index],
        "headroom_original_pts": (orig_best - naive) * 100,
        "headroom_expanded_pts": (exp_best - naive) * 100,
        "oracle_gap_original_pts": (oracle - orig_best) * 100,
        "oracle_gap_expanded_pts": (oracle - exp_best) * 100,
        "best_original": piv[ORIGINAL].idxmax(1), "best_expanded": piv[ORIGINAL + EXPANDED].idxmax(1),
    })
    head.to_csv(OUT / "headroom_comparison.csv", index=False)

    # ---- per-method summary ----
    summ = []
    for m in ORIGINAL + EXPANDED + ["oracle_filter"]:
        if m not in piv.columns:
            continue
        gm, lo, hi = paired_ci(piv[m], naive)
        gb, blo, bhi = paired_ci(piv[m], orig_best)
        summ.append({"method": m, "precision": round(float(piv[m].mean()), 4),
                     "gain_vs_majority_pts": round(gm, 3), "vs_maj_ci": f"[{lo:+.3f},{hi:+.3f}]",
                     "sig_vs_majority": bool(lo > 0 or hi < 0),
                     "gain_vs_best_original_pts": round(gb, 3),
                     "vs_best_orig_ci": f"[{blo:+.3f},{bhi:+.3f}]",
                     "sig_vs_best_original": bool(blo > 0 or bhi < 0)})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "method_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== per-method performance (24 instances, vitc_bankB_mix) ===")
    print(s.to_string(index=False))
    print("\n=== headroom: original vs expanded family ===")
    print(f"  mean headroom, original family : {head.headroom_original_pts.mean():.3f} pts")
    print(f"  mean headroom, expanded family : {head.headroom_expanded_pts.mean():.3f} pts")
    print(f"  mean oracle gap, original      : {head.oracle_gap_original_pts.mean():.3f} pts")
    print(f"  mean oracle gap, expanded      : {head.oracle_gap_expanded_pts.mean():.3f} pts")
    hd, hlo, hhi = paired_ci(piv[ORIGINAL + EXPANDED].max(1), orig_best)
    print(f"  expanded-best minus original-best: {hd:+.3f} [{hlo:+.3f},{hhi:+.3f}] pts")
    print("\n=== which method wins, by regime ===")
    print(pd.crosstab(head.regime, head.best_expanded).to_string())
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

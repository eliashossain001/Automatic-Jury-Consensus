"""Shared regime-routing primitives (canonical versions of the helpers that
the filter and router scripts in scripts/filters/ and scripts/routing_selector/ each carry as private copies; behaviour matches scripts/routing_selector/run_regime_router.py and
scripts/routing_selector/run_unified_router.py exactly -- verified by tests/test_routing_module.py).

Used by scripts/routing_selector/run_mixed_regime_benchmark.py; the older scripts keep their own
copies untouched for backward compatibility.
"""

from __future__ import annotations

import numpy as np

from corrfilter.cfi.consensus import consensus_level, vote_fraction
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold

ABSTAIN = -1
CLUSTER_K = 5
SG_SIZE = 100
TEMP = 0.5
WEAK_PRIOR = 0.6
ALL_SIGNALS = {"rho_bar", "eig", "rho_S", "lco"}
REGIME_FILTER = {"weak": "supermajority_75", "global": "corrfilter_small_gold_R", "subgroup": "bias_cluster"}
REGIMES = ["weak", "global", "subgroup"]

# H1-measured per-judge position-bias gaps (accuracy points), scripts/regimes/ and scripts/routing_selector/.
H1GAP = {"gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8, "llama-3.1-8b::pairwise": 63.7,
         "llama-3.1-8b::likert": 17.4, "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
         "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9, "qwen-2.5-7b::pairwise": 46.7,
         "qwen-2.5-7b::likert": 51.7}

# Clean-referenced hard-router margins (scripts/routing_selector/run_regime_router.py).
HARD_THR = {"rho": 0.03, "lco": 0.05, "eig": 0.85}


def load_votes_swaps(cache, logical_ids, item_ids):
    """(V, M, S) int8 matrices from a VoteCache; S is the position-swap flag."""
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), np.int8)
    M = np.zeros((n, m), np.int8)
    S = np.zeros((n, m), np.int8)
    idx = {it: i for i, it in enumerate(item_ids)}
    for j, lid in enumerate(logical_ids):
        dfj = cache.load(lid)
        if dfj is None:
            continue
        for r in dfj.itertuples(index=False):
            i = idx.get(str(getattr(r, "item_id")))
            if i is None:
                continue
            v = int(getattr(r, "vote"))
            if v == ABSTAIN:
                continue
            V[i, j] = v
            M[i, j] = 1
            S[i, j] = 1 if bool(getattr(r, "position_swapped")) else 0
    return V, M, S


def filter_metrics(keep, gold):
    """Retention / precision / recall / F1 / FRR of a boolean keep mask."""
    keep = keep.astype(bool)
    clean = gold == 1
    n = len(gold)
    nk = int(keep.sum())
    kc = int((keep & clean).sum())
    prec = kc / nk if nk else 0.0
    rec = kc / int(clean.sum()) if clean.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"retention_rate": round(nk / n, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "false_retention_rate": round(1 - prec if nk else 0.0, 4),
            "n_kept": nk, "n_kept_clean": kc, "n_eval": n, "n_clean": int(clean.sum())}


def position_sensitivity_cluster(V, M, S, k=CLUSTER_K):
    """Label-free vulnerable cluster: top-k judges by |slot-A pick rate - 0.5|."""
    slotA = np.where(S == 1, 1 - V, V)
    possens = np.abs(slotA.sum(0) / np.maximum(M.sum(0), 1) - 0.5)
    top = set(np.argsort(-possens)[:k].tolist())
    return np.array([j in top for j in range(V.shape[1])])


def agree_features(O, M, label, cpos_mask, R_clean):
    """Per-item agreeing-set features (matches scripts/routing_selector/run_regime_router.py `_agree_features`)."""
    n, m = O.shape
    Mb = M.astype(bool)
    Rs = (R_clean + R_clean.T) / 2.0
    conc = np.zeros(n); rho = np.zeros(n); disagree = np.zeros(n)
    indep = np.full(n, -np.inf); lco_flip = np.zeros(n)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        s = int(agree.sum())
        if s == 0:
            continue
        inC = int((agree & cpos_mask).sum())
        conc[i] = inC / s
        indep[i] = s - inC
        if s >= 2:
            sub = Rs[np.ix_(agree, agree)]
            rho[i] = float(sub[~np.eye(s, dtype=bool)].mean())
        cl = Mb[i] & cpos_mask
        ncl = Mb[i] & ~cpos_mask
        ar_c = (O[i][cl] == label[i]).mean() if cl.any() else 0.0
        ar_n = (O[i][ncl] == label[i]).mean() if ncl.any() else 0.0
        disagree[i] = abs(ar_c - ar_n)
        if ncl.any():
            lco_flip[i] = 1.0 if ar_n < 0.5 else 0.0
    return {"conc": conc, "rho": rho, "disagree": disagree, "indep": indep, "lco_flip": lco_flip}


def filter_scores(O, M, label, R_h1, R_sg, cpos_mask, feats=None):
    """Per-item keep-scores for the four base filters (scripts/routing_selector/run_regime_router.py `_filter_scores`)."""
    if feats is None:
        feats = agree_features(O, M, label, cpos_mask, R_h1)
    return {
        "naive_majority": np.abs(vote_fraction(O, M) - 0.5),
        "supermajority_75": consensus_level(O, M),
        "corrfilter_small_gold_R": np.nan_to_num(corrfilter_score(O, M, R_sg, label).score, nan=-np.inf),
        "bias_cluster": feats["indep"],
    }


def pct_rank(x):
    """Percentile-rank to [0,1] (NaN/-inf -> 0); matches scripts/routing_selector/run_regime_router.py `_pct`."""
    x = np.asarray(x, float)
    valid = np.isfinite(x)
    r = np.zeros(len(x))
    if valid.sum() == 0:
        return r
    r = np.argsort(np.argsort(np.where(valid, x, -np.inf))) / max(len(x) - 1, 1)
    r[~valid] = 0.0
    return r


def classify_regime_hard(d, ref, thr=HARD_THR):
    """Dataset-level hard rule (scripts/routing_selector/run_regime_router.py `classify_regime`)."""
    if d["rho_bar"] - ref["rho_bar"] > thr["rho"]:
        return "global"
    if (d["lco_flip"] - ref["lco_flip"] > thr["lco"]) or (d["eig_overlap"] < thr["eig"]):
        return "subgroup"
    return "weak"


def regime_probs(rho_bar_dev, eig_drift, rho_S_dev, lco, temp=TEMP, use=ALL_SIGNALS,
                 weak_prior=WEAK_PRIOR):
    """Per-item (p_weak, p_global, p_subgroup) softmax (scripts/routing_selector/run_unified_router.py `regime_probs`)."""
    n = len(lco)
    g = np.zeros(n)
    s = np.zeros(n)
    if "rho_bar" in use:
        g += max(0.0, rho_bar_dev)
    if "rho_S" in use:
        g += np.clip(rho_S_dev, 0, None)
    if "eig" in use:
        s += max(0.0, eig_drift)
    if "lco" in use:
        s += lco
    w = np.full(n, weak_prior)
    E = np.stack([w, g, s], axis=1) / max(temp, 1e-6)
    E -= E.max(axis=1, keepdims=True)
    P = np.exp(E)
    P /= P.sum(axis=1, keepdims=True)
    return P[:, 0], P[:, 1], P[:, 2]


def keep_at_matched_retention(score, pool, n_match):
    """Keep top-`n_match` scored items inside `pool` (scripts/routing_selector/run_regime_router.py `keep_by`)."""
    s = np.where(pool, score, -np.inf)
    k, _ = retention_match_threshold(s, n_match)
    return k & pool


def position_poison_mask(V, M, logical_ids, rate, subset_idx=None):
    """Position-aligned poison mask: top-`rate` items by H1-weighted wrong-vote
    mass (scripts/regimes/run_position_aligned_poisoning.py and scripts/routing_selector/run_regime_router.py generator), optionally restricted to `subset_idx`."""
    w = np.array([H1GAP.get(l, 0.0) for l in logical_ids]) / 100.0
    pos_score = ((1 - V) * M * w[None, :]).sum(1)
    if subset_idx is None:
        subset_idx = np.arange(V.shape[0])
    order = subset_idx[np.argsort(-pos_score[subset_idx], kind="stable")]
    n_poison = int(round(rate * len(subset_idx)))
    pm = np.zeros(V.shape[0], bool)
    pm[order[:n_poison]] = True
    return pm


def correlation_cluster(R, k=CLUSTER_K):
    """Label-free vulnerable cluster for tasks WITHOUT position swaps.

    Pre-registered rule (frozen before any deployment evaluation): spectrally
    bipartition the clean vote-correlation matrix R by the sign of the second
    eigenvector; the candidate vulnerable side is the one with the higher
    internal mean correlation (the most co-varying group). If that side has
    more than ``k`` judges, keep its ``k`` most internally correlated members;
    if fewer than 2, fall back to the top-``k`` judges by mean off-diagonal
    correlation. Deterministic; uses no gold labels and no deployment data.
    """
    Rs = (np.asarray(R) + np.asarray(R).T) / 2.0
    nj = Rs.shape[0]
    off = ~np.eye(nj, dtype=bool)
    vals, vecs = np.linalg.eigh(Rs)
    v2 = vecs[:, -2] if nj >= 2 else vecs[:, -1]
    side = v2 >= 0
    if side.sum() in (0, nj):
        side = v2 >= np.median(v2)

    def internal_mean(mask):
        sub = Rs[np.ix_(mask, mask)]
        o = ~np.eye(int(mask.sum()), dtype=bool)
        return float(sub[o].mean()) if mask.sum() > 1 else -np.inf

    cand = side if internal_mean(side) >= internal_mean(~side) else ~side
    idx = np.where(cand)[0]
    if len(idx) < 2:
        order = np.argsort(-(Rs * off).sum(1) / (nj - 1))
        idx = order[:k]
    elif len(idx) > k:
        internal = Rs[np.ix_(idx, idx)].sum(1) - 1.0
        idx = idx[np.argsort(-internal)][:k]
    mask = np.zeros(nj, dtype=bool)
    mask[idx] = True
    return mask

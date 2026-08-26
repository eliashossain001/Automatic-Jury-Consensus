"""Shared regime-routing primitives.

Canonical versions of the helpers that the filter and router scripts each used
to carry as private copies. Behaviour is parity-tested against the original
regime routers in ``tests/test_routing_module.py``.

Covers: vote/swap loading, position-sensitivity clustering, agreement features,
the fixed filter family and its scores, hard and soft regime classification,
matched-retention keep sets, and pooled deployment metrics.
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
FIXED = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
# Pre-registered router feature set (deployment-computable, no position fields).
BASE_FEATS = ["rho_bar", "n_eff", "eig_overlap", "lco_mean", "conc_mean", "rho_S_mean", "disagree_mean"]
N_REPS = 10
N_BOOT = 2000
BOOT_SEED = 20260706

# H1-measured per-judge position-bias gaps (accuracy points).
H1GAP = {"gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8, "llama-3.1-8b::pairwise": 63.7,
         "llama-3.1-8b::likert": 17.4, "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
         "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9, "qwen-2.5-7b::pairwise": 46.7,
         "qwen-2.5-7b::likert": 51.7}

# Clean-referenced hard-router margins.
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
            i = idx.get(str(r.item_id))
            if i is None:
                continue
            v = int(r.vote)
            if v == ABSTAIN:
                continue
            V[i, j] = v
            M[i, j] = 1
            S[i, j] = 1 if bool(r.position_swapped) else 0
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
    """Per-item agreeing-set features (matches scripts/routing/regime_router.py `_agree_features`)."""
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
    """Per-item keep-scores for the four base filters (scripts/routing/regime_router.py `_filter_scores`)."""
    if feats is None:
        feats = agree_features(O, M, label, cpos_mask, R_h1)
    return {
        "naive_majority": np.abs(vote_fraction(O, M) - 0.5),
        "supermajority_75": consensus_level(O, M),
        "corrfilter_small_gold_R": np.nan_to_num(corrfilter_score(O, M, R_sg, label).score, nan=-np.inf),
        "bias_cluster": feats["indep"],
    }


def pct_rank(x):
    """Percentile-rank to [0,1] (NaN/-inf -> 0); matches scripts/routing/regime_router.py `_pct`."""
    x = np.asarray(x, float)
    valid = np.isfinite(x)
    r = np.zeros(len(x))
    if valid.sum() == 0:
        return r
    r = np.argsort(np.argsort(np.where(valid, x, -np.inf))) / max(len(x) - 1, 1)
    r[~valid] = 0.0
    return r


def classify_regime_hard(d, ref, thr=HARD_THR):
    """Dataset-level hard rule (scripts/routing/regime_router.py `classify_regime`)."""
    if d["rho_bar"] - ref["rho_bar"] > thr["rho"]:
        return "global"
    if (d["lco_flip"] - ref["lco_flip"] > thr["lco"]) or (d["eig_overlap"] < thr["eig"]):
        return "subgroup"
    return "weak"


def regime_probs(rho_bar_dev, eig_drift, rho_S_dev, lco, temp=TEMP, use=ALL_SIGNALS,
                 weak_prior=WEAK_PRIOR):
    """Per-item (p_weak, p_global, p_subgroup) softmax (scripts/routing/unified_router.py `regime_probs`)."""
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
    """Keep top-`n_match` scored items inside `pool` (scripts/routing/regime_router.py `keep_by`)."""
    s = np.where(pool, score, -np.inf)
    k, _ = retention_match_threshold(s, n_match)
    return k & pool


def position_poison_mask(V, M, logical_ids, rate, subset_idx=None):
    """Position-aligned poison mask: top-`rate` items by H1-weighted wrong-vote
    mass (scripts/attacks/position_aligned_poisoning.py/18 generator), optionally restricted to `subset_idx`."""
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


def pooled_precision(methods_df, decisions: dict[str, str]) -> float:
    """Pooled precision when each instance uses ``decisions[instance]`` as its filter."""
    kept = clean = 0
    idx = methods_df.set_index(["instance", "method"])
    for inst, m in decisions.items():
        r = idx.loc[(inst, m)]
        kept += int(r["n_kept"])
        clean += int(r["n_kept_clean"])
    return clean / max(kept, 1)


def pooled_confusion(methods_df, decisions: dict[str, str]) -> dict[str, float]:
    """Pooled precision/accuracy/F1/retention over the decided filters."""
    idx = methods_df.set_index(["instance", "method"])
    tp = fp = fn = tn = 0
    for inst, m in decisions.items():
        r = idx.loc[(inst, m)]
        kc, nk, nc, n = int(r.n_kept_clean), int(r.n_kept), int(r.n_clean), int(r.n_eval)
        tp += kc
        fp += nk - kc
        fn += nc - kc
        tn += n - nc - (nk - kc)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "retention": (tp + fp) / max(tp + fp + fn + tn, 1)}


class _XGBWrap:
    """XGBClassifier with string-label encoding."""

    def __init__(self, **kw):
        from xgboost import XGBClassifier
        self.m = XGBClassifier(**kw)
        self.classes_ = None

    def fit(self, X, y):
        import numpy as np
        self.classes_ = np.unique(y)
        self._map = {c: k for k, c in enumerate(self.classes_)}
        self.m.fit(X, np.array([self._map[v] for v in y]))
        return self

    def predict(self, X):
        return self.classes_[self.m.predict(X)]

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def make_models(seed):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=4),
        "xgboost": _XGBWrap(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                            colsample_bytree=0.9, random_state=seed, verbosity=0, n_jobs=4),
        "lightgbm": LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                                   colsample_bytree=0.9, random_state=seed, verbose=-1, n_jobs=4),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000,
                                                             random_state=seed)),
    }


def best_fixed_on(methods_df, instances) -> str:
    sub = methods_df[methods_df["instance"].isin(instances)]
    best, best_p = None, -1
    for m in FIXED:
        s = sub[sub["method"] == m]
        p = s["n_kept_clean"].sum() / max(s["n_kept"].sum(), 1)
        if p > best_p:
            best, best_p = m, p
    return best


def paired_boot_diff(methods_df, dec_a, dec_b, seed=BOOT_SEED):
    """Instance-bootstrap CI on pooled-precision(dec_a) - pooled-precision(dec_b)."""
    insts = sorted(dec_a)
    idx = methods_df.set_index(["instance", "method"])
    ka = np.array([[idx.loc[(i, dec_a[i])]["n_kept"], idx.loc[(i, dec_a[i])]["n_kept_clean"]] for i in insts])
    kb = np.array([[idx.loc[(i, dec_b[i])]["n_kept"], idx.loc[(i, dec_b[i])]["n_kept_clean"]] for i in insts])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(insts), len(insts))
        pa = ka[b, 1].sum() / max(ka[b, 0].sum(), 1)
        pb = kb[b, 1].sum() / max(kb[b, 0].sum(), 1)
        diffs.append(pa - pb)
    point = ka[:, 1].sum() / max(ka[:, 0].sum(), 1) - kb[:, 1].sum() / max(kb[:, 0].sum(), 1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(point), float(lo), float(hi)

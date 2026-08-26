"""Synthetic dependence-regime generators and the shared evaluation protocol.

Routing and screening benchmarks need instances whose *dependence structure* is
known while everything else is held fixed. These generators sample items and
bad-item locations identically across regimes, and vary only the joint error
structure of the bank:

* ``weak``     -- independent idiosyncratic errors at the common marginal rate;
* ``global``   -- one shared shock flips the whole bank together;
* ``subgroup`` -- a position-sensitive cluster carries the shock alone.

``marginal_params`` restores the same expected wrong-vote rate
(``WRONG_MARGINAL``) under all three, so a filter comparison across regimes is
not confounded by error volume. ``evaluate`` scores one instance under every
filter at matched retention and returns the deployment feature vector.

Used by the routing benchmarks (``scripts/routing/``) and by the cross-task
feasibility screen (``corrfilter.screening.feasibility``).
"""

from __future__ import annotations

import numpy as np

from corrfilter.cfi.adaptive_r import disagreement_R, eigenvector_overlap, small_gold_R
from corrfilter.cfi.consensus import ABSTAIN, majority_consensus, supermajority_consensus
from corrfilter.routing import (
    H1GAP,
    SG_SIZE,
    agree_features,
    classify_regime_hard,
    filter_metrics,
    filter_scores,
    keep_at_matched_retention,
    position_sensitivity_cluster,
)
from corrfilter.screening.features import extract_features

# --- crossed (single-regime) generator -------------------------------------
BASE = 20261001
INSTANCE_SIZE = 300


class Source:
    """A vote source: clean votes, swaps, per-source references and pools."""

    def __init__(self, name, V, M, Sw, len_true, len_worse, lids, R_h1):
        self.name, self.V, self.M, self.Sw = name, V, M, Sw
        self.n = V.shape[0]
        self.cpos = position_sensitivity_cluster(V, M, Sw)
        self.R_clean = disagreement_R(V, M)
        lab = majority_consensus(V, M)
        fc = agree_features(V, M, lab, self.cpos, R_h1)
        nj = V.shape[1]
        self.ref = {"rho_bar": float(np.mean(self.R_clean[~np.eye(nj, dtype=bool)])),
                    "rho_S": float(np.nanmean(fc["rho"])),
                    "lco_flip": float(np.nanmean(fc["lco_flip"]))}
        # weak pool: items where the LONGER response is the truly worse one
        self.weak_pool = np.where(len_worse > len_true)[0]
        w = np.array([H1GAP.get(l, 0.0) for l in lids]) / 100.0
        self.pos_score = ((1 - V) * M * w[None, :]).sum(1)                 # subgroup ranking
        self.glob_score = ((1 - V) * M).sum(1) / np.maximum(M.sum(1), 1)   # global ranking


WRONG_MARGINAL = 0.45  # expected wrong-vote fraction on bad items, ALL regimes
P_HI, P_LO = 0.95, 0.05  # co-failure error prob given shared shock z=1 / z=0
Z_SUBGROUP = 0.85        # subgroup-cluster shock rate


def marginal_params(nj: int, cl: np.ndarray):
    """(q_rest, p_z_global) that restore WRONG_MARGINAL under each generator."""
    n_cl = int(cl.sum())
    cl_marg = P_HI * Z_SUBGROUP + P_LO * (1 - Z_SUBGROUP)
    q_rest = (WRONG_MARGINAL * nj - cl_marg * n_cl) / max(nj - n_cl, 1)
    return float(np.clip(q_rest, 0.0, 1.0)), (WRONG_MARGINAL - P_LO) / (P_HI - P_LO)


def wrong_probs(regime: str, nj: int, cl: np.ndarray, rng, q_rest: float, p_z_global: float):
    """Per-judge wrong-vote probabilities for ONE bad item under a regime."""
    if regime == "global":
        z = rng.random() < p_z_global
        return np.full(nj, P_HI if z else P_LO)
    if regime == "subgroup":
        z = rng.random() < Z_SUBGROUP
        p = np.full(nj, q_rest)
        p[cl] = P_HI if z else P_LO
        return p
    return np.full(nj, WRONG_MARGINAL)  # weak: independent idiosyncratic errors


def make_instance(src: Source, regime: str, rate: float, seed_counter: int):
    """Sample items uniformly; inject regime-specific JOINT error structure.

    Item selection and the marginal wrong-vote rate on bad items are identical
    across regimes; only the dependence of the bank's errors differs.
    """
    rng = np.random.default_rng(BASE + 7919 * seed_counter)
    idx = np.sort(rng.choice(src.n, INSTANCE_SIZE, replace=False))
    O = src.V[idx].copy()
    Md = src.M[idx]
    nj = O.shape[1]
    n_bad = int(round(rate * INSTANCE_SIZE))
    bad = rng.choice(INSTANCE_SIZE, n_bad, replace=False)
    gold = np.ones(INSTANCE_SIZE, dtype=np.int8)
    gold[bad] = 0
    q_rest, p_z_global = marginal_params(nj, src.cpos)
    for t in bad:
        p = wrong_probs(regime, nj, src.cpos, rng, q_rest, p_z_global)
        wrong = rng.random(nj) < p
        # wrong vote on a truly-bad (gold=0) item = affirming it (vote 1)
        O[t] = np.where(Md[t] > 0, np.where(wrong, 1, 0), O[t])
    return O, Md, gold


def evaluate(src: Source, O, Md, gold, counter, R_h1, affirmed_only=True,
             retention_mode="nonabstain"):
    """Score one instance under every filter, with PER-SOURCE references.

    retention_mode: "nonabstain" (default; the original protocol, matching the
    supermajority-0.75 non-abstain count) or "affirmed" (match the
    supermajority-0.75 AFFIRMED count). The default is degenerate on banks
    whose agreement is so concentrated that nearly every item clears the
    supermajority bar on one side (then n_match >= |affirmed pool| and every
    filter keeps everything); "affirmed" is the non-degenerate reading for
    such banks, applied identically to every method."""
    rng = np.random.default_rng(BASE + 104729 * (counter + 1))
    cal_idx = rng.choice(len(gold), size=SG_SIZE, replace=False)
    ev = np.ones(len(gold), bool)
    ev[cal_idx] = False
    R_sg = small_gold_R(O, Md, gold, cal_idx)
    Oe, Me, ge = O[ev], Md[ev], gold[ev]
    label = majority_consensus(Oe, Me)
    sm = supermajority_consensus(Oe, Me, 0.75)
    n_match = max(int((sm == 1).sum() if retention_mode == "affirmed"
                      else (sm != ABSTAIN).sum()), 1)
    pool = (label == 1) if affirmed_only else np.ones(len(ge), bool)
    feats = agree_features(Oe, Me, label, src.cpos, R_h1)
    scores = filter_scores(Oe, Me, label, R_h1, R_sg, src.cpos, feats)
    R_O = disagreement_R(Oe, Me)
    nj = Oe.shape[1]
    d = {"rho_bar": float(np.mean(R_O[~np.eye(nj, dtype=bool)])),
         "eig_overlap": float(eigenvector_overlap(R_O, src.R_clean, 3)),
         "lco_flip": float(np.nanmean(feats["lco_flip"][pool]))}
    pred_hard = classify_regime_hard(d, src.ref)
    keeps = {f: keep_at_matched_retention(s, pool, n_match) for f, s in scores.items()}
    keeps["oracle_filter"] = keep_at_matched_retention(np.where(ge == 1, 1.0, -np.inf), pool, n_match)
    metrics = {f: filter_metrics(k, ge) for f, k in keeps.items()}
    fvec = extract_features(Oe, Me, label, feats, R_O, src.R_clean, R_h1, src.ref, src.cpos)
    return metrics, fvec, pred_hard, d


# --- mixed-regime generator ------------------------------------------------
MIX_BASE = 20261101
PAIR_SHARES = [0.10, 0.25, 0.50, 0.75, 0.90]
PAIRS = [("weak", "global"), ("weak", "subgroup"), ("global", "subgroup")]
THREEWAY = [
    {"weak": 1 / 3, "global": 1 / 3, "subgroup": 1 / 3},
    {"weak": 0.50, "global": 0.25, "subgroup": 0.25},
    {"weak": 0.25, "global": 0.50, "subgroup": 0.25},
    {"weak": 0.25, "global": 0.25, "subgroup": 0.50},
]


def mixtures():
    """Yield (mix_name, {regime: weight}) for the full grid."""
    for a, b in PAIRS:
        for s in PAIR_SHARES:
            yield f"{a[0]}{b[0]}_{int(round(100 * s))}", {a: 1 - s, b: s}
    for w in THREEWAY:
        tag = "".join(f"{r[0]}{int(round(100 * p))}" for r, p in sorted(w.items()))
        yield f"3way_{tag}", w


def majority_regime(weights: dict) -> str:
    mx = max(weights.values())
    top = [r for r, p in weights.items() if p == mx]
    return top[0] if len(top) == 1 else "none"


def make_mixed_instance(src: Source, weights: dict, rate: float, seed_counter: int):
    """Uniform item/bad-item sampling; bad items partitioned among regimes."""
    rng = np.random.default_rng(MIX_BASE + 7919 * seed_counter)
    idx = np.sort(rng.choice(src.n, INSTANCE_SIZE, replace=False))
    O = src.V[idx].copy()
    Md = src.M[idx]
    nj = O.shape[1]
    n_bad = int(round(rate * INSTANCE_SIZE))
    bad = rng.choice(INSTANCE_SIZE, n_bad, replace=False)
    gold = np.ones(INSTANCE_SIZE, dtype=np.int8)
    gold[bad] = 0
    # deterministic proportional partition (largest remainder), then shuffle
    regimes = sorted(weights)
    counts = {r: int(np.floor(weights[r] * n_bad)) for r in regimes}
    rem = n_bad - sum(counts.values())
    fracs = sorted(regimes, key=lambda r: -(weights[r] * n_bad - np.floor(weights[r] * n_bad)))
    for r in fracs[:rem]:
        counts[r] += 1
    perm = rng.permutation(bad)
    assign = {}
    pos = 0
    for r in regimes:
        for t in perm[pos:pos + counts[r]]:
            assign[t] = r
        pos += counts[r]
    q_rest, p_z_global = marginal_params(nj, src.cpos)
    for t in perm:
        p = wrong_probs(assign[t], nj, src.cpos, rng, q_rest, p_z_global)
        wrong = rng.random(nj) < p
        O[t] = np.where(Md[t] > 0, np.where(wrong, 1, 0), O[t])
    return O, Md, gold, counts

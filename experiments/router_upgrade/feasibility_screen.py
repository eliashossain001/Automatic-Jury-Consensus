#!/usr/bin/env python
"""Routing-feasibility screening pipeline (cheap gate before full experiments).

Screens a candidate combination (task, source/subset, judge bank, prompt bank,
~100-200-item sample) using cached votes only, and applies the routing-
feasibility gate. Reuses the verified primitives; nothing downstream of
(V, M, gold) is reimplemented.

Per candidate it computes: the clean-bank metric block (competence, coverage,
dependence, vote-pattern diversity, family/style contrasts), preliminary
filter metrics with per-instance oracle headroom (bootstrap deployment
instances from the sample), small controlled pilot regimes
(weak/global/subgroup via the verified generators, injection on native
gold=0 items), and the gate verdict with an explicit rejection reason.

Screening instances: N_INST subsamples of INST_SIZE items from the sample
(overlapping; disclosed). Matched retention uses retention_mode="affirmed"
throughout (the non-degenerate reading for high-agreement banks; applied
identically to every method).

Library use:
    from feasibility_screen import screen_candidate
    row = screen_candidate(name, manifest, cache_dir, lids, out_dir,
                           cluster_k=5)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP))

import build_crossed_pool as bcp  # noqa: E402
from build_crossed_pool import Source, marginal_params, wrong_probs  # noqa: E402

from corrfilter.cfi.adaptive_r import disagreement_R  # noqa: E402
from corrfilter.cfi.consensus import majority_consensus  # noqa: E402
from corrfilter.correlation import correlation_shrunk  # noqa: E402
from corrfilter.data.generic_task import batch_from_cache  # noqa: E402
from corrfilter.routing import agree_features, correlation_cluster  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

FIXED = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
N_INST = 8
INST_SIZE = 150
SEED = 20270201
HEADROOM_GATE_PTS = 0.5       # Sec 4.5 primary screening condition (points)
NONZERO_RATE_GATE = 0.20


def _clean_metrics(V, M, gold, lids):
    n, nj = V.shape
    Mb = M.astype(bool)
    maj = (np.where(Mb, V, 0).sum(1) > (M.sum(1) / 2)).astype(int)
    sm_thr = int(np.ceil(0.75 * nj))
    aff = np.where(Mb, V, 0).sum(1)
    E = (V != gold[:, None]).astype(float)
    R_err, _ = correlation_shrunk(np.where(Mb, E, np.nan)[~np.isnan(np.where(Mb, E, np.nan)).any(1)])
    off = ~np.eye(nj, dtype=bool)
    rho_err = float(np.nanmean(R_err[off]))
    R_vote = disagreement_R(V, M)
    rho_vote = float(np.nanmean(R_vote[off]))
    neff = nj / (1 + (nj - 1) * rho_err) if rho_err > -1 / (nj - 1) else float(nj)
    eig = np.linalg.eigvalsh((R_vote + R_vote.T) / 2)
    eig_rank = float(eig.sum() ** 2 / (eig ** 2).sum())
    patterns = {tuple(r) for r in np.where(Mb, V, -9)}
    p1 = V[Mb].mean()
    ent = -(np.clip(p1, 1e-9, 1) * np.log2(np.clip(p1, 1e-9, 1))
            + np.clip(1 - p1, 1e-9, 1) * np.log2(np.clip(1 - p1, 1e-9, 1)))
    pdis = []
    for i in range(nj):
        for j in range(i + 1, nj):
            both = Mb[:, i] & Mb[:, j]
            if both.any():
                pdis.append(float((V[both, i] != V[both, j]).mean()))
    wrong = np.where(Mb, E, 0)
    marg_err = wrong.sum() / M.sum()
    joint_half = int((wrong.sum(1) >= nj / 2).sum())
    cond_cofail = float((wrong.sum(1)[wrong.sum(1) > 0] >= 2).mean()) if (wrong.sum(1) > 0).any() else 0.0
    fam = [l.split("-")[0] for l in lids]
    style = [l.split("::")[1] if "::" in l else "?" for l in lids]

    def contrast(groups):
        same, cross = [], []
        for i in range(nj):
            for j in range(i + 1, nj):
                (same if groups[i] == groups[j] else cross).append(R_err[i, j])
        return float(np.mean(same) - np.mean(cross)) if same and cross else np.nan

    return {
        "n_items": n, "class_balance": round(float(gold.mean()), 3),
        "coverage": round(float(M.mean()), 3),
        "parse_failure": round(float(1 - M.mean()), 3),
        "majority_acc": round(float((maj == gold).mean()), 3),
        "chance": round(float(max(gold.mean(), 1 - gold.mean())), 3),
        "majority_wrong": int((maj != gold).sum()),
        "supermajority_wrong": int(((aff >= sm_thr) & (gold == 0)).sum()
                                   + (((M.sum(1) - aff) >= sm_thr) & (gold == 1)).sum()),
        "rho_err": round(rho_err, 3), "rho_vote": round(rho_vote, 3),
        "n_eff": round(float(neff), 2), "eig_rank": round(eig_rank, 2),
        "vote_patterns": len(patterns),
        "pattern_diversity": round(len(patterns) / n, 3),
        "vote_entropy": round(float(ent), 3),
        "pair_disagree": round(float(np.mean(pdis)), 3),
        "joint_half_wrong": joint_half,
        "cond_cofailure": round(cond_cofail, 3),
        "family_contrast": round(contrast(fam), 3),
        "style_contrast": round(contrast(style), 3),
        "per_judge_acc": [round(float((V[Mb[:, j], j] == gold[Mb[:, j]]).mean()), 3)
                          for j in range(nj)],
    }


def _mk_source(name, V, M, gold, lids, cluster_k):
    R = disagreement_R(V, M)
    src = Source(name, V, M, np.zeros_like(V), np.ones(len(gold)), np.zeros(len(gold)),
                 lids, R)
    src.cpos = correlation_cluster(R, k=cluster_k)
    lab = majority_consensus(V, M)
    fc = agree_features(V, M, lab, src.cpos, R)
    nj = V.shape[1]
    src.ref = {"rho_bar": float(np.mean(R[~np.eye(nj, dtype=bool)])),
               "rho_S": float(np.nanmean(fc["rho"])),
               "lco_flip": float(np.nanmean(fc["lco_flip"]))}
    return src, R


def _instances(src, gold_all, R, regime, rate, counter0, inject_mode="native_zeros"):
    """N_INST screening instances.

    inject_mode "native_zeros": correlated errors injected into a
    rate-fraction of NATIVE gold=0 items (pointwise tasks with two-class
    gold). "flip": preference-style injection, a uniform rate-fraction of
    items becomes gold=0 with synthesised votes (for oriented pairwise tasks
    whose clean gold is all 1)."""
    out = []
    for s in range(N_INST):
        rng = np.random.default_rng(SEED + 7919 * (counter0 + s))
        idx = np.sort(rng.choice(src.n, min(INST_SIZE, src.n), replace=False))
        O = src.V[idx].copy()
        Md = src.M[idx]
        gold = gold_all[idx].copy()
        if regime != "weak":
            nj = O.shape[1]
            if inject_mode == "flip":
                n_bad = int(round(rate * len(idx)))
                bad = rng.choice(len(idx), n_bad, replace=False)
                gold[bad] = 0
            else:
                zeros = np.where(gold == 0)[0]
                n_bad = min(int(round(rate * len(idx))), len(zeros))
                bad = rng.choice(zeros, n_bad, replace=False) if n_bad else []
            q_rest, p_z = marginal_params(nj, src.cpos)
            for t in bad:
                p = wrong_probs(regime, nj, src.cpos, rng, q_rest, p_z)
                w = rng.random(nj) < p
                O[t] = np.where(Md[t] > 0, np.where(w, 1, 0), O[t])
        elif inject_mode == "flip" and regime == "weak":
            n_bad = int(round(rate * len(idx)))
            bad = rng.choice(len(idx), n_bad, replace=False)
            gold[bad] = 0
            for t in bad:
                w = rng.random(O.shape[1]) < 0.45
                O[t] = np.where(Md[t] > 0, np.where(w, 1, 0), O[t])
        out.append((counter0 + s, O, Md, gold))
    return out


def screen_candidate(name, manifest, cache_dir, lids, out_dir, cluster_k=5,
                     rate=0.20, task_name="", extra=None, chance_override=None,
                     inject_mode="native_zeros", cache=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = batch_from_cache(manifest, cache if cache is not None else VoteCache(cache_dir), lids)
    V, M, gold = batch.judge_votes, batch.availability, batch.gold_labels
    keep = M.sum(1) > 0
    V, M, gold = V[keep], M[keep], gold[keep]
    man = manifest[keep.tolist()].reset_index(drop=True) if len(manifest) == len(keep) else manifest
    clean = _clean_metrics(V, M, gold, lids)
    clean["avg_ctx_chars"] = int(man.get("doc_chars", pd.Series([0])).mean()
                                 + man.get("claim_chars", pd.Series([0])).mean())
    clean["max_ctx_chars"] = int(man.get("doc_chars", pd.Series([0])).max()
                                 + man.get("claim_chars", pd.Series([0])).max())
    src, R = _mk_source(name, V, M, gold, lids, cluster_k)

    # regimes + filter metrics; calibration split scaled to the sample so
    # small screens keep a non-empty eval set (restored afterwards)
    rows = []
    counter = 0
    inst_n = min(INST_SIZE, src.n)
    sg_orig = bcp.SG_SIZE
    bcp.SG_SIZE = min(100, max(30, inst_n // 3))
    for regime in ("weak", "global", "subgroup"):
        for c, O, Md, g in _instances(src, gold, R, regime, rate, counter,
                                      inject_mode=inject_mode):
            metrics, fvec, pred_hard, d = bcp.evaluate(src, O, Md, g, c, R,
                                                       retention_mode="affirmed")
            for f, m in metrics.items():
                rows.append({"instance": f"{name}_{regime}_{c}", "regime": regime,
                             "method": f, **m})
        counter += N_INST
    bcp.SG_SIZE = sg_orig
    md = pd.DataFrame(rows)
    md.to_csv(out_dir / f"screen_methods_{name}.csv", index=False)
    piv = md[md.method.isin(FIXED)].pivot_table(index="instance", columns="method",
                                                values="precision")
    reg_of = {i: i.rsplit("_", 2)[-2] for i in piv.index}
    naive = piv["naive_majority"]
    best = piv[FIXED].max(1)
    head = (best - naive) * 100
    kn = md[md.method.isin(FIXED)].pivot_table(index="instance", columns="method",
                                               values="n_kept")
    overlap = float((kn.nunique(1) == 1).mean())
    weak_i = [i for i in piv.index if reg_of[i] == "weak"]
    glob_i = [i for i in piv.index if reg_of[i] == "global"]
    sub_i = [i for i in piv.index if reg_of[i] == "subgroup"]
    d_cf = float((piv.loc[glob_i, "corrfilter_small_gold_R"] - piv.loc[glob_i, "naive_majority"]).mean()) * 100
    d_bc = float((piv.loc[sub_i, "bias_cluster"] - piv.loc[sub_i, "naive_majority"]).mean()) * 100
    d_cf_sub = float((piv.loc[sub_i, "corrfilter_small_gold_R"] - piv.loc[sub_i, "naive_majority"]).mean()) * 100
    d_weak = float((piv.loc[weak_i, FIXED[1:]].max(1) - piv.loc[weak_i, "naive_majority"]).mean()) * 100
    filt = {
        "naive_prec": round(float(naive.mean()), 4),
        "supermaj_prec": round(float(piv["supermajority_75"].mean()), 4),
        "corrfilter_prec": round(float(piv["corrfilter_small_gold_R"].mean()), 4),
        "biascluster_prec": round(float(piv["bias_cluster"].mean()), 4),
        "keepset_identical_rate": round(overlap, 3),
        "instances_filters_differ": int((~(piv[FIXED].nunique(1) == 1)).sum()),
        "headroom_mean_pts": round(float(head.mean()), 3),
        "headroom_median_pts": round(float(head.median()), 3),
        "headroom_p90_pts": round(float(head.quantile(0.9)), 3),
        "headroom_max_pts": round(float(head.max()), 3),
        "nonzero_headroom_rate": round(float((head > 1e-9).mean()), 3),
        "headroom_ge_025_rate": round(float((head >= 0.25).mean()), 3),
        "headroom_ge_050_rate": round(float((head >= 0.50).mean()), 3),
        "headroom_ge_100_rate": round(float((head >= 1.00).mean()), 3),
        "corrfilter_delta_global_pts": round(d_cf, 3),
        "biascluster_delta_subgroup_pts": round(d_bc, 3),
        "corrfilter_delta_subgroup_pts": round(d_cf_sub, 3),
        "weak_best_delta_pts": round(d_weak, 3),
    }
    # regime signatures on the screening instances (from rho recomputation)
    # (evaluate() already returned d per instance; approximate via method rows)
    # gate
    if chance_override is not None:
        clean["chance"] = round(float(chance_override), 3)
    err_support = clean["majority_wrong"] >= 15
    competent = clean["majority_acc"] > clean["chance"] + 0.05
    global_ok = d_cf > 0.0
    subgroup_ok = (d_bc > 0.0) and (d_bc >= d_cf_sub)
    headroom_ok = (head.mean() >= HEADROOM_GATE_PTS
                   or float((head > 1e-9).mean()) >= NONZERO_RATE_GATE
                   or max(d_cf, d_bc) >= HEADROOM_GATE_PTS)
    variation_ok = filt["instances_filters_differ"] >= max(2, int(0.1 * len(piv)))
    gates = {"competence": bool(competent), "error_support": bool(err_support),
             "global_regime": bool(global_ok), "subgroup_regime": bool(subgroup_ok),
             "headroom": bool(headroom_ok), "filter_variation": bool(variation_ok)}
    if all(gates.values()):
        decision = "PROCEED"
    elif not competent:
        decision = "REJECT - LOW COMPETENCE"
    elif not err_support:
        decision = "REJECT - TOO FEW ERRORS"
    elif not headroom_ok or not variation_ok:
        decision = "REJECT - ZERO HEADROOM"
    else:
        decision = "REJECT - INVALID REGIMES"
    row = {"candidate": name, "task": task_name, **clean, **filt,
           **{f"gate_{k}": v for k, v in gates.items()}, "decision": decision,
           **(extra or {})}
    (out_dir / f"screen_{name}.json").write_text(json.dumps(row, indent=2, default=str))
    return row

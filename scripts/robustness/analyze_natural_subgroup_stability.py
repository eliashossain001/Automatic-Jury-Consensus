#!/usr/bin/env python
"""Stability of the natural vulnerable-subgroup estimate.

The natural-subgroup analysis (robustness/find_natural_subgroup_cases.py) reports how many
consensus-wrong items flip to correct when one error-correlation cluster of judges
is removed. Two robustness questions from review: (i) how sensitive is that
fraction to the clustering granularity k, and (ii) does it replicate on a second
dataset? This script sweeps k in {2,3,4,5} on RewardBench and on clean
UltraFeedback votes, reporting per-k prevalence, the cross-k overlap (Jaccard) of
flagged item sets, and the share of flagged items CorrFilter still retains.

Deterministic, CPU-only, no new inference. Usage:
  python scripts/robustness/analyze_natural_subgroup_stability.py
Outputs -> outputs/natural_subgroup/stability_sweep.csv, stability_summary.md
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from compute_dependence import estimate_R, item_ids_from_manifest, load_votes  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from corrfilter.cfi.consensus import ABSTAIN, majority_consensus  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402

RETENTION = 0.60
KS = (2, 3, 4, 5)


def flagged_items(V, M, R, gold, k_clusters):
    """Item indices whose wrong majority flips to correct under leave-cluster-out."""
    n, m = V.shape
    D = np.clip(1.0 - R, 0, 2)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    clusters = fcluster(Z, t=k_clusters, criterion="maxclust")
    M_b = M.astype(bool)
    maj = majority_consensus(V, M)
    wrong = (maj != ABSTAIN) & (maj != gold)
    flagged = []
    for i in np.where(wrong)[0]:
        committed = M_b[i]
        for c in sorted(set(clusters)):
            keep = committed & np.array([clusters[j] != c for j in range(m)])
            if keep.sum() < 2:
                continue
            sel = V[i][keep]
            ones, zeros = int((sel == 1).sum()), int((sel == 0).sum())
            new_maj = 1 if ones >= zeros else 0
            if new_maj == gold[i] and new_maj != maj[i]:
                flagged.append(i)
                break
    return set(flagged), int(wrong.sum()), maj


def cf_retains(V, M, R, maj, idx_set):
    n = V.shape[0]
    cf = corrfilter_score(V, M, R, maj)
    k = int(round(RETENTION * n))
    order = np.argsort(-np.nan_to_num(cf.score, nan=-np.inf), kind="stable")
    keep = np.zeros(n, bool)
    keep[order[:k]] = True
    return sum(1 for i in idx_set if keep[i])


def analyse(name, V, M, gold, R):
    rows, sets = [], {}
    for k in KS:
        fl, n_wrong, maj = flagged_items(V, M, R, gold, k)
        sets[k] = fl
        rows.append(dict(dataset=name, k_clusters=k, consensus_wrong=n_wrong,
                         flagged=len(fl), prevalence=round(len(fl) / max(n_wrong, 1), 4),
                         corrfilter_retains=cf_retains(V, M, R, maj, fl)))
    jac = {f"J(k{a},k{b})": round(len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1), 3)
           for a, b in itertools.combinations(KS, 2)}
    return pd.DataFrame(rows), jac


def main() -> None:
    out = ROOT / "outputs/natural_subgroup"
    out.mkdir(exist_ok=True)
    frames, jacs = [], {}

    ids = item_ids_from_manifest(ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet")
    V, M, _ = load_votes(ROOT / "configs/judge_bank.yaml", ids, None)
    gold = np.ones(len(ids), np.int8)
    R = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)["R"]
    df, jac = analyse("rewardbench", V, M, gold, R)
    frames.append(df)
    jacs["rewardbench"] = jac

    man = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    uf_ids = man.item_id.astype(str).tolist()
    import yaml
    from corrfilter.judges import load_bank_config
    from corrfilter.voting import VoteCache, load_vote_matrix
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    lids = [s.logical_id for s in bank.specs]
    Vu, Mu = load_vote_matrix(VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"),
                              lids, uf_ids)
    gold_u = np.ones(len(uf_ids), np.int8)
    Ru = estimate_R(Vu, Mu, gold_u)[0]
    dfu, jacu = analyse("ultrafeedback", Vu, Mu, gold_u, Ru)
    frames.append(dfu)
    jacs["ultrafeedback"] = jacu

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "stability_sweep.csv", index=False)
    with open(out / "stability_summary.md", "w") as f:
        f.write("# Natural vulnerable-subgroup stability sweep\n\n")
        f.write(all_df.to_markdown(index=False))
        f.write("\n\nCross-k Jaccard overlap of flagged item sets:\n\n")
        for ds, j in jacs.items():
            f.write(f"- {ds}: {j}\n")
    print(all_df.to_string(index=False))
    for ds, j in jacs.items():
        print(ds, j)


if __name__ == "__main__":
    main()

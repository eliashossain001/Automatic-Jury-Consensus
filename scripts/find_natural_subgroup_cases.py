"""Bucket B — find NATURALLY occurring vulnerable-subgroup signatures (no synthetic poison).

On the cached base judge bank, cluster the judges by their error-correlation, then look for
held-out items where (i) the majority is WRONG, and (ii) removing one correlated cluster of
judges flips the majority to CORRECT (leave-cluster-out). These are natural analogues of the
paper's position-aligned poisoning: an incorrect consensus driven primarily by a correlated
subset. For each case we record whether CorrFilter's alpha_subset would still retain it (a
failure) and whether a bias-cluster filter (drop the vulnerable cluster) fixes it.

Deterministic, CPU-only. Prevalence is reported but NOT claimed as representative.

Usage:
  python scripts/find_natural_subgroup_cases.py --bank configs/judge_bank.yaml \
      --manifest experiments/h1_measurement/results/calibration_manifest.parquet \
      --corr experiments/h1_measurement/results/correlation.npz --k-clusters 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus, vote_fraction  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402
from compute_dependence import estimate_R, item_ids_from_manifest, load_votes  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402


def _clip(s, n=200):
    return " ".join(str(s).split())[:n]


def _majority_of(V_row, mask_judges, gold_i):
    """Majority label over the judges selected by mask_judges (committed only)."""
    sel = V_row[mask_judges]
    if sel.size == 0:
        return ABSTAIN
    ones = int((sel == 1).sum()); zeros = int((sel == 0).sum())
    if ones == zeros:
        return 1  # tie-break to 1 (matches consensus module)
    return 1 if ones > zeros else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--manifest", default="experiments/h1_measurement/results/calibration_manifest.parquet")
    ap.add_argument("--corr", default="experiments/h1_measurement/results/correlation.npz")
    ap.add_argument("--k-clusters", type=int, default=3)
    ap.add_argument("--retention", type=float, default=0.60)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/natural_subgroup")
    args = ap.parse_args()

    item_ids = item_ids_from_manifest(args.manifest)
    man = (pd.read_parquet(args.manifest) if args.manifest.endswith(".parquet") else pd.read_csv(args.manifest))
    man["item_id"] = man["item_id"].astype(str); man = man.set_index("item_id")
    V, M, lids = load_votes(args.bank, item_ids, None)
    gold = np.ones(len(item_ids), dtype=np.int8)
    R = np.load(args.corr, allow_pickle=True)["R"].astype(np.float64) if Path(args.corr).exists() \
        else estimate_R(V, M, gold)[0]
    n, m = V.shape

    # cluster judges by 1 - R (correlated judges land together)
    D = np.clip(1.0 - R, 0, 2); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    clusters = fcluster(Z, t=args.k_clusters, criterion="maxclust")  # (m,) cluster id per judge
    cluster_ids = sorted(set(clusters))

    M_b = M.astype(bool)
    maj = majority_consensus(V, M)
    level = consensus_level(V, M)
    labellable = maj != ABSTAIN
    wrong = labellable & (maj != gold)
    cf = corrfilter_score(V, M, R, maj)
    # corrfilter keep-set at target retention
    k = int(round(args.retention * n))
    order = np.argsort(-np.nan_to_num(cf.score, nan=-np.inf), kind="stable")
    cf_keep = np.zeros(n, bool); cf_keep[order[:k]] = True

    cases = []
    n_wrong = int(wrong.sum())
    for i in np.where(wrong)[0]:
        committed = M_b[i]
        for c in cluster_ids:
            drop = np.array([clusters[j] == c for j in range(m)])
            keep_judges = committed & (~drop)
            if keep_judges.sum() < 2:
                continue
            new_maj = _majority_of(V[i], keep_judges, gold[i])
            if new_maj == gold[i] and new_maj != maj[i]:
                # leave-cluster-out flips wrong->correct: natural vulnerable subgroup
                row = man.loc[item_ids[i]]
                cases.append({
                    "item_id": item_ids[i], "subset": str(row.get("subset", row.get("category", "?"))),
                    "prompt": _clip(row.get("prompt", ""), 300),
                    "chosen_gold": _clip(row.get("chosen", "")), "rejected_gold": _clip(row.get("rejected", "")),
                    "consensus_level": round(float(level[i]), 3),
                    "majority_decision": "chosen" if maj[i] == 1 else "rejected", "majority_correct": False,
                    "vulnerable_cluster": int(c),
                    "cluster_judges": [lids[j] for j in range(m) if clusters[j] == c and committed[j]],
                    "flips_to_correct_when_cluster_removed": True,
                    "corrfilter_alpha_subset": round(float(cf.score[i]), 3) if not np.isnan(cf.score[i]) else None,
                    "corrfilter_retains_item": bool(cf_keep[i]),
                    "bias_cluster_would_fix": True,
                })
                break  # one cluster is enough to flag the item
    df = pd.DataFrame(cases)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    prevalence = round(len(cases) / max(n_wrong, 1), 4)
    # summary table (compact)
    keep_cols = ["item_id", "subset", "consensus_level", "vulnerable_cluster",
                 "corrfilter_alpha_subset", "corrfilter_retains_item", "bias_cluster_would_fix"]
    (df[keep_cols] if len(df) else pd.DataFrame(columns=keep_cols)).to_csv(out / "table_cases.csv", index=False)

    n_cf_retains = int(df["corrfilter_retains_item"].sum()) if len(df) else 0
    print(f"[natural_subgroup] clusters={args.k_clusters} judge->cluster: "
          f"{dict(zip([l.split('::')[0]+'.'+l.split('::')[1][0] for l in lids], clusters.tolist()))}")
    print(f"[natural_subgroup] consensus-wrong items: {n_wrong}; with leave-cluster-out flip "
          f"(natural subgroup signature): {len(cases)}  (prevalence {prevalence:.1%} of wrong items)")
    print(f"[natural_subgroup] of those, CorrFilter still RETAINS: {n_cf_retains} "
          f"(cases where a bias-cluster filter is needed, not alpha_subset)")
    print(f"-> {out}/cases.jsonl, table_cases.csv")
    # LaTeX table of up to n_examples
    ex = df.head(args.n_examples) if len(df) else df
    lines = [r"\begin{tabular}{p{0.34\linewidth} c c c c}", r"\toprule",
             r"Prompt (truncated) & subset & consensus & vuln.\ cluster & CorrFilter retains? \\", r"\midrule"]
    for _, r in ex.iterrows():
        lines.append(f"{str(r['prompt'])[:52].replace('&','\\&')}\\ldots & {r['subset']} & "
                     f"{r['consensus_level']:.2f} (wrong) & \\#{r['vulnerable_cluster']} & "
                     f"{'yes (fails)' if r['corrfilter_retains_item'] else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_cases.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

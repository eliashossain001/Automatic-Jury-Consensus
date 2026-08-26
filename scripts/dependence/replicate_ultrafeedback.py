"""Replicate the H1 dependence measurement on a SECOND open dataset (UltraFeedback).

The headline dependence numbers (mean error correlation, n_eff, eigen-effective rank,
co-failure, H1 diversity rejection) were measured on RewardBench v2. This script
reproduces all of them on UltraFeedback using the SAME ten open-weight judges and the
SAME estimators (Ledoit-Wolf shrinkage; bootstrap; family/prompt contrast). It reuses the
already-cached UF votes, so it requires NO new model inference and NO paid APIs.

Vote encoding for the cached UF run: V[i,j]=1 iff judge j picked the truly-better
response, so the gold label is the constant 1 and the error matrix is E=(V!=1).

Outputs: results/dependence_replication_uf/{correlation_uf.npz, dependence_summary.json,
comparison_vs_rewardbench.csv, heatmap_R_uf.png}.

Usage: python scripts/dependence/replicate_ultrafeedback.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.analysis import family_contrast, prompt_contrast
from corrfilter.correlation import compute_error_matrix, correlation_shrunk
from corrfilter.correlation.effective_size import (
    effective_eig_rank,
    effective_size,
    mean_off_diagonal,
)
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache, load_vote_matrix

ROOT = Path(__file__).resolve().parents[2]


OUT = ROOT / "results" / "dependence_replication_uf"
MANIFEST = ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv"
VOTES = ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"
RB_NPZ = ROOT / "experiments/h1_measurement/results/correlation.npz"
RB_H1 = ROOT / "experiments/h1_measurement/results/h1_result.json"


def cofailure_stats(E):
    """Per-judge error, pairwise both-wrong, conditional co-failure, majority co-failure."""
    n, m = E.shape
    pj = E.mean(0)
    both, cond = [], []
    for i in range(m):
        for j in range(i + 1, m):
            ei, ej = E[:, i], E[:, j]
            pb = float((ei * ej).mean())
            both.append(pb)
            if ei.mean() > 0:
                cond.append(pb / ei.mean())
    maj = float((E.mean(1) > 0.5).mean())
    return {"per_judge_error_mean": round(float(pj.mean()), 4),
            "per_judge_error_range": [round(float(pj.min()), 4), round(float(pj.max()), 4)],
            "pairwise_both_wrong": round(float(np.mean(both)), 4),
            "conditional_cofailure": round(float(np.mean(cond)), 4),
            "majority_cofailure": round(float(maj), 4)}


def boot_rho_ci(E, n_boot=2000, seed=20260601):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, E.shape[0], E.shape[0])
        R_b, _ = correlation_shrunk(E[idx])
        vals.append(mean_off_diagonal(R_b))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bank = load_bank_config(str(ROOT / "configs/judge_bank.yaml"))
    specs = bank.specs
    logical_ids = [s.logical_id for s in specs]
    man = pd.read_csv(MANIFEST); man["item_id"] = man["item_id"].astype(str)
    item_ids = man["item_id"].tolist()

    cache = VoteCache(VOTES)
    V, M = load_vote_matrix(cache, logical_ids, item_ids)
    gold = np.ones(V.shape[0], dtype=np.int64)              # UF encoding: V==1 is correct
    E, complete = compute_error_matrix(V, M, gold)
    retention = float(complete.mean())
    print(f"UF: {V.shape[0]} items, {len(logical_ids)} judges, listwise retention {retention:.1%} "
          f"({E.shape[0]} complete items)")

    R, shrinkage = correlation_shrunk(E)
    rho = float(mean_off_diagonal(R)); neff = float(effective_size(R)); eig = float(effective_eig_rank(R))
    rho_lo, rho_hi = boot_rho_ci(E)
    cof = cofailure_stats(E)
    fam = family_contrast(E, specs, n_boot=2000, threshold=0.10, seed=20260601)
    pmt = prompt_contrast(E, specs, n_boot=2000, threshold=0.10, seed=20260602)

    # vulnerable cluster (top-5 by mean off-diagonal correlation)
    off = (R.sum(1) - np.diag(R)) / (R.shape[0] - 1)
    top5 = [logical_ids[j] for j in np.argsort(-off)[:5]]

    uf = {
        "dataset": "UltraFeedback (binarized)", "n_items": int(V.shape[0]),
        "n_complete_items": int(E.shape[0]), "listwise_retention": round(retention, 4),
        "estimator": "listwise_ledoit_wolf", "shrinkage": round(float(shrinkage), 4),
        "rho_bar": round(rho, 4), "rho_bar_boot95": [rho_lo, rho_hi],
        "n_eff": round(neff, 3), "eigen_effective_rank": round(eig, 3),
        "cofailure": cof, "top5_cluster": top5,
        "family_contrast": {"rho_intra": round(fam.rho_intra, 4), "rho_cross": round(fam.rho_cross, 4),
                            "delta": round(fam.delta, 4), "accepts_h1": bool(fam.accepts_h1)},
        "prompt_contrast": {"rho_intra": round(pmt.rho_intra, 4), "rho_cross": round(pmt.rho_cross, 4),
                            "delta": round(pmt.delta, 4), "accepts_h1": bool(pmt.accepts_h1)},
    }

    # ---- RewardBench reference (already computed) ----
    z = np.load(RB_NPZ, allow_pickle=True); R_rb = z["R"]; E_rb = z["E"]
    h1 = json.loads(RB_H1.read_text())
    rb = {
        "dataset": "RewardBench v2", "n_complete_items": int(E_rb.shape[0]),
        "rho_bar": round(float(mean_off_diagonal(R_rb)), 4),
        "n_eff": round(float(effective_size(R_rb)), 3),
        "eigen_effective_rank": round(float(effective_eig_rank(R_rb)), 3),
        "cofailure": cofailure_stats(E_rb),
        "family_contrast": {"rho_intra": round(h1["family_contrast"]["rho_intra"], 4),
                            "rho_cross": round(h1["family_contrast"]["rho_cross"], 4),
                            "delta": round(h1["family_contrast"]["delta"], 4),
                            "accepts_h1": bool(h1["family_contrast"]["accepts_h1"])},
        "prompt_contrast": {"rho_intra": round(h1["prompt_contrast"]["rho_intra"], 4),
                            "rho_cross": round(h1["prompt_contrast"]["rho_cross"], 4),
                            "delta": round(h1["prompt_contrast"]["delta"], 4),
                            "accepts_h1": bool(h1["prompt_contrast"]["accepts_h1"])},
    }

    # ---- comparison table ----
    rows = [
        ("mean error correlation (rho_bar)", rb["rho_bar"], uf["rho_bar"]),
        ("n_eff (effective ensemble size)", rb["n_eff"], uf["n_eff"]),
        ("eigen-effective rank", rb["eigen_effective_rank"], uf["eigen_effective_rank"]),
        ("per-judge error rate", rb["cofailure"]["per_judge_error_mean"], uf["cofailure"]["per_judge_error_mean"]),
        ("pairwise both-wrong", rb["cofailure"]["pairwise_both_wrong"], uf["cofailure"]["pairwise_both_wrong"]),
        ("conditional co-failure P(B wrong|A wrong)", rb["cofailure"]["conditional_cofailure"], uf["cofailure"]["conditional_cofailure"]),
        ("majority co-failure (frac items)", rb["cofailure"]["majority_cofailure"], uf["cofailure"]["majority_cofailure"]),
        ("family contrast delta (cross-intra)", rb["family_contrast"]["delta"], uf["family_contrast"]["delta"]),
        ("family H1 accepted (diversity helps)", rb["family_contrast"]["accepts_h1"], uf["family_contrast"]["accepts_h1"]),
        ("prompt contrast delta", rb["prompt_contrast"]["delta"], uf["prompt_contrast"]["delta"]),
        ("prompt H1 accepted", rb["prompt_contrast"]["accepts_h1"], uf["prompt_contrast"]["accepts_h1"]),
    ]
    comp = pd.DataFrame(rows, columns=["metric", "RewardBench_v2", "UltraFeedback"])
    comp.to_csv(OUT / "comparison_vs_rewardbench.csv", index=False)
    np.savez(OUT / "correlation_uf.npz", R=R, E=E, logical_ids=np.array(logical_ids), shrinkage=shrinkage)

    summary = {"replication": "dependence measurement on a second open dataset",
               "no_new_inference": True, "no_paid_api": True,
               "ultrafeedback": uf, "rewardbench_v2": rb,
               "verdict": {
                   "dependence_reproduces": bool(uf["rho_bar"] > 0.10 and uf["n_eff"] < 0.7 * len(logical_ids)),
                   "cofailure_reproduces": bool(uf["cofailure"]["majority_cofailure"] > 0.10),
                   "H1_diversity_rejection_holds": bool(not uf["family_contrast"]["accepts_h1"]
                                                        and not uf["prompt_contrast"]["accepts_h1"]),
               }}
    (OUT / "dependence_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(R, vmin=-0.1, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(len(logical_ids))); ax.set_yticks(range(len(logical_ids)))
        sh = [l.replace("::", "\n") for l in logical_ids]
        ax.set_xticklabels(sh, rotation=90, fontsize=6); ax.set_yticklabels(sh, fontsize=6)
        ax.set_title(f"UltraFeedback error correlation R\n(rho_bar={rho:.3f}, n_eff={neff:.2f})", fontsize=10)
        fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(OUT / "heatmap_R_uf.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("heatmap skipped:", e)

    print("\n=== UltraFeedback vs RewardBench v2 ===")
    print(comp.to_string(index=False))
    print("\nverdict:", json.dumps(summary["verdict"], indent=2))
    print(f"\nwrote outputs under {OUT}")


if __name__ == "__main__":
    main()

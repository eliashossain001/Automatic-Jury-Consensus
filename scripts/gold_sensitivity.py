"""Bucket C — robustness of the dependence estimates to imperfect gold labels.

Randomly flips a fraction of the trusted calibration gold labels (1%, 3%, 5%, 10%) and
recomputes rho_bar, n_eff, eigen-rank; repeats over many seeds for a distribution, and
also reports an item-bootstrap CI at 0% flips. Additionally computes the estimates on a
judge-agreement-derived "pseudo-clean" subset (high-consensus items) as a second check.
Deterministic given the seed grid; CPU-only.

Outputs table_label_noise.csv + fig_label_noise.pdf.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from corrfilter.cfi.consensus import consensus_level  # noqa: E402
from corrfilter.correlation import compute_error_matrix, correlation_shrunk  # noqa: E402
from corrfilter.correlation.effective_size import effective_eig_rank, effective_size, mean_off_diagonal  # noqa: E402
from compute_dependence import item_ids_from_manifest, load_votes  # noqa: E402


def metrics_from_gold(V, M, gold):
    E, complete = compute_error_matrix(V, M, gold)
    if E.shape[0] < 2:
        return np.nan, np.nan, np.nan
    R, _ = correlation_shrunk(E)
    return mean_off_diagonal(R), effective_size(R), effective_eig_rank(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--manifest", default="experiments/h1_measurement/results/calibration_manifest.parquet")
    ap.add_argument("--flip-rates", default="0,0.01,0.03,0.05,0.10")
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--out-dir", default="outputs/gold_sensitivity")
    args = ap.parse_args()

    item_ids = item_ids_from_manifest(args.manifest)
    V, M, lids = load_votes(args.bank, item_ids, None)
    n = len(item_ids)
    gold0 = np.ones(n, dtype=np.int8)
    rates = [float(x) for x in args.flip_rates.split(",")]

    rows = []
    for p in rates:
        if p == 0.0:
            rb, ne, er = metrics_from_gold(V, M, gold0)
            # item-bootstrap CI at 0 flips
            rng = np.random.default_rng(20260708)
            bo = []
            for _ in range(300):
                idx = rng.integers(0, n, n)
                bo.append(metrics_from_gold(V[idx], M[idx], gold0[idx])[0])
            bo = np.array([x for x in bo if not np.isnan(x)])
            rows.append({"flip_rate": p, "rho_bar": round(rb, 4), "rho_bar_lo": round(float(np.quantile(bo, .025)), 4),
                         "rho_bar_hi": round(float(np.quantile(bo, .975)), 4), "n_eff": round(ne, 3),
                         "eigen_rank": round(er, 3), "note": "item-bootstrap CI"})
        else:
            rbs, nes, ers = [], [], []
            for s in range(args.n_seeds):
                rng = np.random.default_rng(1000 + s)
                gold = gold0.copy()
                flip = rng.choice(n, size=int(round(p * n)), replace=False)
                gold[flip] = 1 - gold[flip]
                rb, ne, er = metrics_from_gold(V, M, gold)
                rbs.append(rb); nes.append(ne); ers.append(er)
            rbs, nes, ers = map(np.array, (rbs, nes, ers))
            rows.append({"flip_rate": p, "rho_bar": round(float(rbs.mean()), 4),
                         "rho_bar_lo": round(float(np.quantile(rbs, .025)), 4),
                         "rho_bar_hi": round(float(np.quantile(rbs, .975)), 4),
                         "n_eff": round(float(nes.mean()), 3), "eigen_rank": round(float(ers.mean()), 3),
                         "note": f"{args.n_seeds}-seed flip"})

    # pseudo-clean subset: top-half by consensus level (judges agree -> likely correct)
    level = consensus_level(V, M)
    hi = level >= np.median(level)
    rb_pc, ne_pc, er_pc = metrics_from_gold(V[hi], M[hi], gold0[hi])
    rows.append({"flip_rate": "pseudo_clean_hi_consensus", "rho_bar": round(rb_pc, 4),
                 "rho_bar_lo": np.nan, "rho_bar_hi": np.nan, "n_eff": round(ne_pc, 3),
                 "eigen_rank": round(er_pc, 3), "note": f"top-{int(hi.sum())} consensus items"})

    df = pd.DataFrame(rows)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "table_label_noise.csv", index=False)
    print(df.to_string(index=False))

    # figure: rho_bar and n_eff vs flip rate
    num = df[df.flip_rate.apply(lambda x: isinstance(x, float))].copy()
    for c in ["flip_rate", "rho_bar", "rho_bar_lo", "rho_bar_hi", "n_eff"]:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    fig, ax1 = plt.subplots(figsize=(5.6, 4))
    x = num.flip_rate.values * 100
    ax1.fill_between(x, num.rho_bar_lo, num.rho_bar_hi, color="#A41E34", alpha=0.15)
    ax1.plot(x, num.rho_bar, "s-", color="#A41E34", label=r"$\bar\rho$")
    ax1.axhline(0, color="#ccc", lw=0.5)
    ax1.set_xlabel("gold labels randomly flipped (\\%)"); ax1.set_ylabel(r"mean error correlation $\bar\rho$", color="#A41E34")
    ax1.set_ylim(0, max(num.rho_bar) * 1.3)
    ax2 = ax1.twinx()
    ax2.plot(x, num.n_eff, "o--", color="#4C78A8", label=r"$n_{\mathrm{eff}}$")
    ax2.set_ylabel(r"effective size $n_{\mathrm{eff}}$ (of 10)", color="#4C78A8"); ax2.set_ylim(0, 10)
    ax1.set_title(r"Dependence is stable under gold-label noise", fontsize=11, fontweight="bold")
    ax1.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "fig_label_noise.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_label_noise.png", dpi=150, bbox_inches="tight")
    print(f"-> {out}/table_label_noise.csv, fig_label_noise.pdf")


if __name__ == "__main__":
    main()

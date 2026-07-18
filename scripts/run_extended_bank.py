"""Bucket 4 — dependence vs nominal judge count on the extended bank (CPU, after inference).

Given the extended-bank votes (produced by scripts/02 on configs/judge_bank_extended.yaml),
sweeps the nominal judge count n and records rho_bar, n_eff, eigen-rank at each n, then
plots the n_eff-vs-n saturation curve against the 1/rho_bar ceiling. Shows n_eff << n as n
grows. Also writes the family/prompt contrast at the full bank.

Usage:
  python scripts/run_extended_bank.py --bank configs/judge_bank_extended.yaml \
      --manifest experiments/h1_measurement/results/calibration_manifest.parquet \
      --out-dir outputs/extended_bank
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
from compute_dependence import (contrast, dependence_metrics, estimate_R,  # noqa: E402
                                item_ids_from_manifest, load_votes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/judge_bank_extended.yaml")
    ap.add_argument("--manifest", default="experiments/h1_measurement/results/calibration_manifest.parquet")
    ap.add_argument("--out-dir", default="outputs/extended_bank")
    ap.add_argument("--steps", default=None, help="comma-list of nominal n; default even sweep")
    args = ap.parse_args()

    item_ids = item_ids_from_manifest(args.manifest)
    V, M, lids = load_votes(args.bank, item_ids, None)
    gold = np.ones(len(item_ids), dtype=np.int8)
    # keep only judges that actually have votes (inference may be partial)
    have = M.sum(0) > 0
    V, M = V[:, have], M[:, have]
    lids = [l for l, h in zip(lids, have) if h]
    n_total = len(lids)
    steps = ([int(x) for x in args.steps.split(",")] if args.steps
             else sorted(set(list(range(4, n_total, 2)) + [n_total])))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in steps:
        n = min(n, n_total)
        row, R = dependence_metrics(V[:, :n], M[:, :n], gold, lids[:n], f"n={n}")
        rows.append({"nominal_n": n, "rho_bar": row["rho_bar"], "n_eff": row["n_eff"],
                     "eigen_rank": row["eigen_rank"], "ceiling_1_over_rho": row["neff_ceiling_1_over_rho"],
                     "majority_cofailure": row["majority_cofailure"]})
    df = pd.DataFrame(rows).drop_duplicates("nominal_n").sort_values("nominal_n")
    df.to_csv(out / "table_extended_dependence.csv", index=False)
    print(df.to_string(index=False))

    # full-bank contrasts
    _, Rfull = dependence_metrics(V, M, gold, lids, "full")
    fi, fc, fd = contrast(Rfull, lids, "family"); pi, pc, pd_ = contrast(Rfull, lids, "prompt")
    pd.DataFrame([{"grouping": "family", "intra": fi, "cross": fc, "delta": fd},
                  {"grouping": "prompt", "intra": pi, "cross": pc, "delta": pd_}]
                 ).to_csv(out / "extended_contrasts.csv", index=False)

    # saturation figure
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(df.nominal_n, df.nominal_n, "o--", color="#4C78A8", label="nominal $n$ (if independent)")
    ax.plot(df.nominal_n, df.n_eff, "s-", color="#A41E34", lw=2, label=r"measured $n_{\mathrm{eff}}$")
    ax.plot(df.nominal_n, df.ceiling_1_over_rho, ":", color="#666", label=r"ceiling $1/\bar\rho$")
    ax.fill_between(df.nominal_n, df.n_eff, df.nominal_n, color="#A41E34", alpha=0.06)
    ax.set_xlabel("nominal judge count $n$")
    ax.set_ylabel("effective ensemble size")
    ax.set_title(r"$n_{\mathrm{eff}}$ saturates far below $n$ as the bank grows",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_neff_saturation.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_neff_saturation.png", dpi=150, bbox_inches="tight")
    print(f"-> {out}/table_extended_dependence.csv, fig_neff_saturation.pdf  "
          f"(bank has {n_total} logical judges with votes)")


if __name__ == "__main__":
    main()

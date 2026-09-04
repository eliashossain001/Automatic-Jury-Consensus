"""Bucket A — cross-dataset dependence summary figure (RewardBench / UltraFeedback / PKU-SafeRLHF)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", default="outputs/tables/dependence.csv")
    ap.add_argument("--pku", default="outputs/pku_saferlhf/table_dependence.csv")
    ap.add_argument("--out", default="outputs/pku_saferlhf/fig_dependence_summary.pdf")
    args = ap.parse_args()

    rep = pd.read_csv(args.repro).set_index("label")
    pku = pd.read_csv(args.pku).set_index("label")
    data = [
        ("RewardBench v2", rep.loc["rewardbench_base_10"]),
        ("UltraFeedback", rep.loc["ultrafeedback_base_10"]),
        ("PKU-SafeRLHF", pku.iloc[0]),
    ]
    names = [d[0] for d in data]
    rho = [float(d[1]["rho_bar"]) for d in data]
    neff = [float(d[1]["n_eff"]) for d in data]
    cof = [float(d[1]["conditional_cofailure"]) for d in data]
    C = "#4C78A8"

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, vals, ttl, ylab in [
        (ax[0], rho, r"A. mean error correlation $\bar\rho$", r"$\bar\rho$"),
        (ax[1], neff, r"B. effective size $n_{\mathrm{eff}}$ (of 10)", r"$n_{\mathrm{eff}}$"),
        (ax[2], cof, r"C. conditional co-failure", r"$\Pr[B\ \mathrm{wrong}\mid A\ \mathrm{wrong}]$"),
    ]:
        bars = a.bar(names, vals, color=C, width=0.6)
        a.set_title(ttl, fontsize=11, fontweight="bold"); a.set_ylabel(ylab)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.2f}" if v < 1 else f"{v:.1f}",
                   ha="center", va="bottom", fontweight="bold", fontsize=9)
        a.tick_params(axis="x", labelrotation=20)
        a.spines[["top", "right"]].set_visible(False)
    ax[1].axhline(10, ls=":", color="#aaa", lw=1)
    fig.suptitle("Inter-judge dependence across three preference datasets, same 10-judge bank",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[pku_summary_fig] rho={rho} neff={neff} cond_cofailure={cof} -> {out}")


if __name__ == "__main__":
    main()

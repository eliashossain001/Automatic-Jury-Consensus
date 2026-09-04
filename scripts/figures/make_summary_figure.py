"""Bucket 6 — one parameterized reviewer-summary figure: base -> preference-trained judges.

Four panels on a matched bank:
  A  mean error correlation rho_bar   (base < trained)
  B  effective ensemble size n_eff    (base > trained)
  C  consensus false retention stays high (supermajority-0.75, trained bank)
  D  CorrFilter reduces false retention  (paired gain at matched retention, trained bank)

Parameterized so it works for GRPO now and DPO/ORPO later: pass --method-label and the
row labels to read from the Bucket-1 tables. Reads outputs/tables/{dependence,filters,
corrfilter_gain}.csv by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BASE_C, TRN_C, CF_C = "#4C78A8", "#A41E34", "#1F7A3D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dependence", default="outputs/tables/dependence.csv")
    ap.add_argument("--filters", default="outputs/tables/filters.csv")
    ap.add_argument("--gain", default="outputs/tables/corrfilter_gain.csv")
    ap.add_argument("--base-label", default="rewardbench_base_matched6")
    ap.add_argument("--trained-label", default="grpo_trained_6")
    ap.add_argument("--filters-trained-label", default="grpo")
    ap.add_argument("--method-label", default="GRPO")
    ap.add_argument("--retention", type=float, default=0.60)
    ap.add_argument("--out", default="outputs/figures/summary_dependence_training.pdf")
    args = ap.parse_args()

    dep = pd.read_csv(args.dependence).set_index("label")
    b, t = dep.loc[args.base_label], dep.loc[args.trained_label]
    filt = pd.read_csv(args.filters)
    sm = filt[(filt.label == args.filters_trained_label) & (filt.method == "supermajority_0.75")]
    sm_frr = float(sm["false_retention"].iloc[0]) if len(sm) else float("nan")
    gain = pd.read_csv(args.gain)
    g = gain[(gain.label == args.filters_trained_label) & (abs(gain.retention - args.retention) < 1e-6)]
    cons_frr = float(g["consensus_frr"].iloc[0]); cf_frr = float(g["corrfilter_frr"].iloc[0])
    gain_pts = float(g["gain_pts"].iloc[0]); glo = float(g["ci_low_pts"].iloc[0]); ghi = float(g["ci_high_pts"].iloc[0])

    ML = args.method_label
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.4))
    lab = ["Base", ML]

    ax[0].bar(lab, [b.rho_bar, t.rho_bar], color=[BASE_C, TRN_C], width=0.6)
    ax[0].set_title(r"A. $\bar\rho$ increases", fontweight="bold", fontsize=11)
    ax[0].set_ylabel("mean error correlation")
    for i, v in enumerate([b.rho_bar, t.rho_bar]):
        ax[0].text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax[1].bar(lab, [b.n_eff, t.n_eff], color=[BASE_C, TRN_C], width=0.6)
    ax[1].set_title(r"B. $n_{\mathrm{eff}}$ decreases", fontweight="bold", fontsize=11)
    ax[1].set_ylabel("effective ensemble size")
    for i, v in enumerate([b.n_eff, t.n_eff]):
        ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax[2].bar(["supermajority\n(k-of-n)"], [sm_frr], color=TRN_C, width=0.5)
    ax[2].set_ylim(0, max(0.3, sm_frr * 1.4))
    ax[2].set_title("C. Consensus still\nfalse-retains", fontweight="bold", fontsize=11)
    ax[2].set_ylabel("false-retention rate")
    ax[2].text(0, sm_frr + 0.006, f"{sm_frr:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax[3].bar(["consensus", "CorrFilter"], [cons_frr, cf_frr], color=[TRN_C, CF_C], width=0.6)
    ax[3].set_title(f"D. CorrFilter cuts FRR\n({gain_pts:+.1f} pts, 95% CI [{glo:.1f},{ghi:.1f}])",
                    fontweight="bold", fontsize=10)
    ax[3].set_ylabel(f"false retention @ {int(args.retention*100)}% ret.")
    for i, v in enumerate([cons_frr, cf_frr]):
        ax[3].text(i, v + 0.004, f"{v:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Preference optimization ({ML}) increases inter-judge dependence; "
                 f"consensus stays unreliable; CorrFilter mitigates it",
                 fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[make_summary_figure] {ML}: rho {b.rho_bar}->{t.rho_bar}, n_eff {b.n_eff}->{t.n_eff}, "
          f"supermaj FRR {sm_frr:.3f}, CorrFilter gain {gain_pts:+.1f} pts -> {out}")


if __name__ == "__main__":
    main()

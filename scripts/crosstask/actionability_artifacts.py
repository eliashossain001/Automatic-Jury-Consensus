#!/usr/bin/env python
"""Generate the manuscript artifacts for the cross-task boundary study.

Inputs: outputs/router_upgrade/screening/verified_master.csv (rebuilt from
per-candidate raw JSON). Outputs: the actionability scatter figure
(mixed_plots/fig_actionability.png) and LaTeX fragments for the main
boundary table and the appendix full table. No numbers are hand-typed:
everything renders from the verified CSV.

Usage: python scripts/crosstask/actionability_artifacts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "outputs/router_upgrade/screening"
PLOTS = ROOT / "outputs/router_upgrade/mixed_plots"

FAM_MARK = {"preference": "o", "factuality": "s", "factuality-bank": "D",
            "code-pointwise": "^", "code-pairwise": "v"}
FAM_COLOR = {"preference": "#2a78d6", "factuality": "#eb6834",
             "factuality-bank": "#eda100", "code-pointwise": "#e34948",
             "code-pairwise": "#1baf7a"}
LABELS = {"pref_uf": "pref-UF", "pref_rb": "pref-RB", "pref_pku": "pref-PKU",
          "vitc_bankD_only": "bank D", "vitc_bankB_mix": "bank B",
          "halueval": "HaluEval", "cjb_point": "code-pt", "cjb_pair": "code-pw",
          "cjb_point_frontier": "code-pt-frontier",
          "vitc_all": "VitC", "vitc_synth": "VitC-syn"}


def main() -> None:
    df = pd.read_csv(SCR / "verified_master.csv")
    # cjb_point headroom is competence-invalid; plot it hollow with a note
    nj_map = {"vitc_bankD_only": 10, "vitc_bankAD": 20, "vitc_bankB_mix": 14,
              "vitc_bankABD": 24, "cjb_point": 5, "cjb_point_frontier": 6}
    df["nj"] = df.candidate.map(nj_map).fillna(10)
    df["neff_ratio"] = df.n_eff / df.nj

    # per-(panel, candidate) label offsets and alignment, tuned to avoid
    # marker/label collisions in both panels
    OFF = {
        ("rho_err", "code-pw"): (6, -3, "left"),
        ("rho_err", "pref-RB"): (4, -11, "left"),
        ("rho_err", "pref-UF"): (-7, 6, "right"),
        ("rho_err", "bank D"): (-7, 5, "right"),
        ("rho_err", "pref-PKU"): (8, -2, "left"),
        ("rho_err", "bank B"): (2, 9, "left"),
        ("rho_err", "VitC"): (7, 1, "left"),
        ("rho_err", "HaluEval"): (0, 9, "center"),
        ("rho_err", "VitC-syn"): (-2, 9, "right"),
        ("rho_err", "code-pt"): (-8, 2, "right"),
        ("rho_err", "code-pt-frontier"): (-6, 7, "right"),
        ("neff_ratio", "code-pw"): (-7, 6, "right"),
        ("neff_ratio", "pref-RB"): (5, -4, "left"),
        ("neff_ratio", "pref-UF"): (8, -2, "left"),
        ("neff_ratio", "pref-PKU"): (4, 9, "left"),
        ("neff_ratio", "bank D"): (5, 7, "left"),
        ("neff_ratio", "bank B"): (2, 9, "left"),
        ("neff_ratio", "VitC"): (-7, 4, "right"),
        ("neff_ratio", "HaluEval"): (7, 3, "left"),
        ("neff_ratio", "VitC-syn"): (-3, 11, "right"),
        ("neff_ratio", "code-pt"): (8, -3, "left"),
        ("neff_ratio", "code-pt-frontier"): (-6, 7, "right"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.15), sharey=True)
    for ax, xcol, xlab in ((axes[0], "rho_err", "mean error correlation"),
                           (axes[1], "neff_ratio",
                            "effective-size ratio $n_{\\mathrm{eff}}/n$")):
        if xcol == "rho_err":
            ax.axvspan(0.15, 0.35, color="#f0efec", zorder=0)
            ax.text(0.25, 2.55, "interpretive\nregion", ha="center", fontsize=7.5,
                    color="#52514e", style="italic")
        for _, r in df.iterrows():
            invalid = r.candidate == "cjb_point"
            passed = r.decision == "PROCEED"
            y = min(r.headroom_mean_pts, 3.05)
            ax.scatter(r[xcol], y, marker=FAM_MARK[r.task], s=62,
                       facecolors="none" if invalid else FAM_COLOR[r.task],
                       edgecolors=FAM_COLOR[r.task],
                       linewidths=1.8 if passed else 1.0, zorder=3)
            lbl = LABELS.get(r.candidate)
            if lbl:
                dx, dy, ha = OFF.get((xcol, lbl), (4, 4, "left"))
                ax.annotate(lbl, (r[xcol], y), textcoords="offset points",
                            xytext=(dx, dy), fontsize=7, ha=ha, zorder=4)
        ax.set_xlabel(xlab, fontsize=9.5)
        ax.set_ylim(-0.22, 3.4)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("aggregation-rule headroom (pts)", fontsize=9.5)
    axes[0].set_xlim(0.0, 0.545)
    axes[1].set_xlim(0.08, 0.80)
    handles = [plt.Line2D([], [], marker=m, ls="", color=FAM_COLOR[f],
                          label=f, markersize=6)
               for f, m in FAM_MARK.items()]
    axes[1].legend(handles=handles, fontsize=7, frameon=False, loc="upper right",
                   bbox_to_anchor=(1.0, 0.88))
    fig.tight_layout()
    fig.savefig(PLOTS / "fig_actionability.png", dpi=200)

    # main boundary table (6 representative rows)
    MAIN = ["pref_uf", "vitc_bankD_only", "vitc_bankB_mix", "halueval",
            "cjb_point", "cjb_pair"]
    NAMES = {"pref_uf": "preference (UltraFeedback, ref.)",
             "vitc_bankD_only": "factuality, procedure-diverse bank",
             "vitc_bankB_mix": "factuality, tier-mixed bank",
             "halueval": "factuality (HaluEval)",
             "cjb_point": "code, pointwise",
             "cjb_pair": "code, pairwise"}
    rows = []
    for c in MAIN:
        r = df[df.candidate == c].iloc[0]
        head = "--" if c == "cjb_point" else f"${r.headroom_mean_pts:.2f}$"
        gate = ("PASS" if r.decision == "PROCEED"
                else "fail (" + {"REJECT - ZERO HEADROOM": "headroom",
                                 "REJECT - INVALID REGIMES": "regimes",
                                 "REJECT - LOW COMPETENCE": "competence"}[r.decision] + ")")
        rows.append(f"{NAMES[c]} & ${r.majority_acc:.3f}$ / ${r.chance:.2f}$ & "
                    f"${r.rho_err:.3f}$ & ${r.n_eff:.2f}$/{int(r.nj)} & {head} & {gate} \\\\")
    (SCR / "main_table_rows.tex").write_text("\n".join(rows) + "\n")

    # appendix full table (14 rows)
    arows = []
    for _, r in df.iterrows():
        head = "n/a" if r.candidate == "cjb_point" else f"${r.headroom_mean_pts:.2f}$"
        arows.append(
            f"{r.candidate.replace('_','-')} & {r.task} & ${r.majority_acc:.3f}$/${r.chance:.2f}$ & "
            f"${r.majority_wrong}$ & ${r.rho_err:.3f}$ & ${r.n_eff:.2f}$/{int(r.nj)} & "
            f"${r.pattern_diversity:.2f}$ & {head} & "
            f"{'y' if r.gate_competence else 'n'}/"
            f"{'y' if r.gate_global_regime else 'n'}/"
            f"{'y' if r.gate_subgroup_regime else 'n'}/"
            f"{'y' if r.gate_headroom else 'n'} & "
            f"{r.decision.replace('REJECT - ','').replace('_',' ').lower()} \\\\")
    (SCR / "appendix_table_rows.tex").write_text("\n".join(arows) + "\n")
    print("artifacts written:", PLOTS / "fig_actionability.png",
          SCR / "main_table_rows.tex", SCR / "appendix_table_rows.tex")


if __name__ == "__main__":
    main()

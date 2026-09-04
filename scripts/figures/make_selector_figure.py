#!/usr/bin/env python
"""Paper-ready figure for deployment-level direct filter selection."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RO = ROOT / "outputs" / "router_upgrade"
DS = ROOT / "outputs" / "deployment_selector"
FIG = ROOT / "figures"


def row(df, name):
    return df.loc[df.comparison == name].iloc[0]


def main():
    # Panel A: selector gain and total oracle headroom under three increasingly
    # strict/general protocols. All values and intervals are read from artifacts.
    old = pd.read_csv(RO / "mixed_significance.csv")
    strict = pd.read_csv(DS / "primary_significance.csv")
    ext = pd.read_csv(DS / "external_natural_significance.csv")
    protocols = [
        ("Mixed preference\n(deployment votes)",
         row(old, "direct_ridge - fixed_best"), row(old, "oracle_choice - fixed_best")),
        ("Mixed preference\n(strict calibration)",
         row(strict, "ridge_core - best_fixed"), row(strict, "perdeployment_oracle - best_fixed")),
        ("Natural AggreFact\n(frozen transfer)",
         row(ext, "frozen_selector - transferred_fixed"),
         row(ext, "perdeployment_oracle - transferred_fixed")),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 10, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    blue, navy, grey, dark = "#2878B5", "#174A6E", "#C6CDD3", "#263238"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.25, 3.15),
                                 gridspec_kw={"width_ratios": [1.35, 1.0]})

    y = np.arange(len(protocols))[::-1]
    for yi, (label, sel, oracle) in zip(y, protocols):
        head = float(oracle.gain_pts)
        gain = float(sel.gain_pts)
        ax.barh(yi, head, height=0.42, color=grey, edgecolor="none", zorder=1)
        ax.barh(yi, gain, height=0.42, color=blue, edgecolor="none", zorder=2)
        lo, hi = float(sel.ci95_low if "ci95_low" in sel else sel.ci_lo), \
                 float(sel.ci95_high if "ci95_high" in sel else sel.ci_hi)
        ax.errorbar(gain, yi, xerr=[[gain - lo], [hi - gain]], fmt="o", color=navy,
                    ecolor=navy, capsize=2.5, markersize=3.5, lw=1.1, zorder=3)
        recovered = 100 * gain / head if head > 0 else 0
        ax.text(head + 0.07, yi, f"{recovered:.0f}%", va="center", ha="left",
                fontsize=7.5, color=dark)
    ax.axvline(0, color="#808890", lw=0.8, ls="--")
    ax.set_yticks(y, [p[0] for p in protocols])
    ax.set_xlabel("precision gain over fixed baseline (points)")
    ax.set_title("A  Recovered oracle headroom", loc="left", fontweight="bold")
    ax.set_xlim(-0.22, 3.95)
    ax.grid(axis="x", color="#E6EAED", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=blue, label="selector gain"),
                       plt.Rectangle((0, 0), 1, 1, color=grey, label="oracle headroom")],
              loc="upper right", frameon=False, fontsize=7.5)

    # Panel B: natural external distribution. Dots show the eight component
    # means; large diamonds show component-balanced means.
    per = pd.read_csv(DS / "external_natural_per_split.csv")
    methods = ["transferred_fixed", "frozen_selector", "perdeployment_oracle"]
    labels = ["Transferred\nfixed", "Frozen direct\nselector", "Per-deployment\noracle"]
    colors = ["#8B98A1", blue, dark]
    component = per.groupby("component")[methods].mean()
    means = component.mean()
    xs = np.arange(3)
    for j, method in enumerate(methods):
        jitter = np.linspace(-0.075, 0.075, len(component))
        bx.scatter(np.full(len(component), j) + jitter, component[method] * 100,
                   s=13, color=colors[j], alpha=0.45, edgecolors="none", zorder=2)
        bx.scatter(j, means[method] * 100, s=55, marker="D", color=colors[j],
                   edgecolor="white", linewidth=0.8, zorder=4)
    bx.plot(xs, means[methods].to_numpy() * 100, color="#8C969D", lw=1.1, zorder=1)
    bx.annotate("+2.00 pts", xy=(1, means.frozen_selector * 100),
                xytext=(0.55, 83.0), ha="center", color=navy, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=navy, lw=0.8))
    bx.annotate("85.5% of headroom", xy=(1, means.frozen_selector * 100),
                xytext=(1.62, 81.6), ha="center", color=dark, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#68747C", lw=0.7))
    bx.set_xticks(xs, labels)
    bx.set_ylabel("component-balanced precision (%)")
    bx.set_title("B  Natural external transfer", loc="left", fontweight="bold")
    bx.set_ylim(55, 94)
    bx.grid(axis="y", color="#E6EAED", lw=0.6)
    bx.set_axisbelow(True)
    bx.spines[["top", "right"]].set_visible(False)
    bx.tick_params(axis="x", length=0)

    fig.suptitle("Direct filter prediction adapts when the optimal aggregation rule shifts",
                 x=0.5, y=1.015, fontsize=11, fontweight="bold")
    fig.tight_layout(pad=0.8, w_pad=1.4)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_selector_actionability.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_selector_actionability.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "**Figure: Direct filter prediction under aggregation-rule shift.** "
        "(A) Precision improvement of the direct selector (blue) and total per-deployment "
        "oracle headroom (gray). Error bars are 95% paired bootstrap intervals; the strict "
        "and external analyses cluster by mixture configuration and natural dataset component, "
        "respectively. (B) A selector trained only on preference deployments is frozen and "
        "evaluated on eight natural human-labeled AggreFact components. It selects majority, "
        "the externally optimal fixed rule, on every split; small points show component means "
        "and diamonds show component-balanced means. The selector improves by 2.00 points "
        "[0.71, 3.40] over the preference-selected fixed filter and recovers 85.5% of oracle "
        "headroom. No factuality labels are used for selector fitting."
    )
    (DS / "FIGURE_CAPTION.md").write_text(caption + "\n")
    print(FIG / "fig_selector_actionability.pdf")
    print(FIG / "fig_selector_actionability.png")


if __name__ == "__main__":
    main()

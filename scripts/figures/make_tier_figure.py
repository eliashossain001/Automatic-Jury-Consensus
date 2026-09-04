#!/usr/bin/env python
"""Headline figure: capability-tier block structure of judge error correlation.

Panel A: error-correlation matrix R of the mixed 11-judge bank (400 RewardBench
items), judges ordered open-weight then Gemini then GPT/Claude/Grok, with the
two capability-tier blocks outlined and block means annotated.
Panel B: nominal versus effective ensemble size for four banks on the same items.

Data: outputs/openrouter_bank/analysis_full_rewardbench/{R_mixed_11.csv,bank_metrics.csv}.
Output: figures/fig_tier_dependence.{pdf,png}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INK = "#333333"
ACCENT = "#2f6fb3"
NOMINAL = "#c9d4e0"

SHORT = {
    "llama-3.1-8b::pairwise": "Llama-8B", "qwen-2.5-7b::pairwise": "Qwen-7B",
    "gemma-2-9b::pairwise": "Gemma-9B", "mistral-7b-v0.3::pairwise": "Mistral-7B",
    "phi-3.5-mini::pairwise": "Phi-3.5", "gemini-3.1-pro::pairwise": "Gemini Pro",
    "gemini-3.6-flash::pairwise": "Gemini Flash", "gemini-3.5-flash-lite::pairwise": "Gemini Lite",
    "gpt-5.6-sol::pairwise": "GPT-5.6", "claude-opus-5::pairwise": "Claude Opus 5",
    "grok-4.5::pairwise": "Grok 4.5",
}


def main() -> None:
    adir = ROOT / "outputs/openrouter_bank/analysis_full_rewardbench"
    R = pd.read_csv(adir / "R_mixed_11.csv", index_col=0)
    labels = [SHORT[l] for l in R.index]
    M = R.to_numpy()
    n = len(labels)

    bm = pd.read_csv(adir / "bank_metrics.csv").set_index("label")
    banks = [("open-weight (10)", 10, bm.loc["open_all_10", "n_eff"]),
             ("Gemini (3)", 3, bm.loc["gemini_3", "n_eff"]),
             ("GPT+Claude+Grok (3)", 3, bm.loc["openrouter_3", "n_eff"]),
             ("mixed (11)", 11, bm.loc["mixed_11", "n_eff"])]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.6, 4.1), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- Panel A: heatmap ----
    Mplot = M.copy()
    np.fill_diagonal(Mplot, np.nan)
    im = ax.imshow(Mplot, cmap="Blues", vmin=0.0, vmax=0.65)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5, color=INK)
    ax.set_yticklabels(labels, fontsize=6.5, color=INK)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0") or "0", ha="center", va="center",
                    fontsize=5.2, color="white" if v > 0.36 else INK)
    for (a, b), lab, va in (((0, 5), "open tier", "bottom"), ((5, 11), "frontier tier", "top")):
        ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False,
                                   edgecolor=INK, linewidth=1.4))
    ax.text(2.0, -0.85, "open tier", fontsize=7.5, color=INK, ha="center", style="italic")
    ax.text(7.9, -0.85, "frontier tier", fontsize=7.5, color=INK, ha="center", style="italic")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A. Error correlation, mixed 11-judge bank", fontsize=9, color=INK, pad=16)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.ax.tick_params(labelsize=6.5, colors=INK)
    cb.outline.set_visible(False)

    # ---- Panel B: nominal vs effective ----
    x = np.arange(len(banks))
    w = 0.38
    ax2.bar(x - w / 2, [b[1] for b in banks], w, color=NOMINAL, label="nominal judges")
    ax2.bar(x + w / 2, [b[2] for b in banks], w, color=ACCENT, label="effective ($n_{\\mathrm{eff}}$)")
    for xi, (_, nom, eff) in zip(x, banks):
        ax2.text(xi - w / 2, nom + 0.15, str(nom), ha="center", fontsize=7.5, color=INK)
        ax2.text(xi + w / 2, eff + 0.15, f"{eff:.1f}", ha="center", fontsize=7.5, color=ACCENT)
    ax2.set_xticks(x)
    ax2.set_xticklabels([b[0] for b in banks], fontsize=7, color=INK)
    ax2.set_ylabel("judges", fontsize=8, color=INK)
    ax2.set_ylim(0, 12.2)
    ax2.tick_params(labelsize=7, colors=INK, length=0)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color("#bbbbbb")
    ax2.legend(fontsize=7, frameon=False, loc="upper left")
    ax2.set_title("B. Nominal vs. effective ensemble size", fontsize=9, color=INK, pad=10)
    ax2.grid(axis="y", color="#eeeeee", linewidth=0.7)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    out = ROOT / "figures"
    fig.savefig(out / "fig_tier_dependence.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_tier_dependence.png", dpi=200, bbox_inches="tight")
    print("wrote", out / "fig_tier_dependence.pdf")


if __name__ == "__main__":
    main()

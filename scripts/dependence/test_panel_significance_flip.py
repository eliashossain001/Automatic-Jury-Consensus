#!/usr/bin/env python
"""Dependence makes panel verdicts overconfident: significant at nominal n,
not at n_eff.

A standard practitioner test compares two systems by pooling all judge votes
over an evaluation set and applying a binomial test, implicitly treating the
n votes per item as independent. Under measured dependence they are not: the
within-item design effect is 1+(n-1)*rho_bar (~2.85 for the open bank), so the
correct unit is the item, not the vote. This script quantifies how often the
naive test declares a significant winner when the item-level (dependence-aware)
test does not, at deployment-typical evaluation sizes, and renders the figure.

Analysis-only; replays the H1 vote caches. Usage:
  python scripts/dependence/test_panel_significance_flip.py
Outputs:
  outputs/nonerror_correlation/significance_flip_rates.csv
  outputs/nonerror_correlation/significance_flip_example.json
  figures/fig_significance_flip.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INK = "#333333"
ACCENT = "#2f6fb3"
NAIVE = "#c9553d"

OPEN = [f"{b}::{s}" for b in ["llama-3.1-8b", "qwen-2.5-7b", "gemma-2-9b",
                              "mistral-7b-v0.3", "phi-3.5-mini"] for s in ["pairwise", "likert"]]
SUBSETS = ["Focus", "Factuality", "Math", "Precise IF"]
SIZES = (100, 150, 250)
N_DRAWS = 5000
SEED = 20260706


def load_bank():
    man = pd.read_parquet(ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet")
    man = man.drop_duplicates("item_id", keep="last")
    ids = man.item_id.astype(str).tolist()
    subset = pd.Series(man.subset.values, index=ids)
    idx = {i: k for k, i in enumerate(ids)}
    V = np.zeros((len(ids), len(OPEN)), np.int8)
    M = np.zeros_like(V)
    for j, lid in enumerate(OPEN):
        df = pd.read_parquet(ROOT / f"experiments/h1_measurement/votes/{lid.replace('::', '__')}.parquet")
        df = df[df.item_id.astype(str).isin(idx)].drop_duplicates("item_id", keep="last")
        for iid, v in zip(df.item_id.astype(str), df.vote.astype(int)):
            if v != -1:
                V[idx[iid], j] = v
                M[idx[iid], j] = 1
    return ids, subset, V, M.astype(bool)


def tests_for_draw(item_mean, item_T, b):
    T = item_T[b].sum()
    p = (item_mean[b] * item_T[b]).sum() / T
    z_naive = (p - 0.5) / np.sqrt(max(p * (1 - p), 1e-9) / T)
    se_item = item_mean[b].std(ddof=1) / np.sqrt(len(b))
    z_item = (item_mean[b].mean() - 0.5) / se_item
    return p, z_naive, z_item, se_item, T


def main() -> None:
    ids, subset, V, Mb = load_bank()
    item_mean = np.where(Mb.sum(1) > 0, (V * Mb).sum(1) / np.maximum(Mb.sum(1), 1), np.nan)
    item_T = Mb.sum(1)

    rng = np.random.default_rng(SEED)
    rows = []
    for s in SUBSETS:
        sel = np.where(np.array([subset[i] == s for i in ids]) & (item_T > 0))[0]
        for N in SIZES:
            if N > len(sel):
                continue
            naive_sig = item_sig = flip = 0
            for _ in range(N_DRAWS):
                b = rng.choice(sel, N, replace=False)
                _, zn, zc, _, _ = tests_for_draw(item_mean, item_T, b)
                naive_sig += abs(zn) > 1.96
                item_sig += abs(zc) > 1.96
                flip += (abs(zn) > 1.96) and (abs(zc) <= 1.96)
            rows.append(dict(subset=s, N=N, naive_sig=naive_sig / N_DRAWS,
                             item_level_sig=item_sig / N_DRAWS, flip_rate=flip / N_DRAWS))
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/nonerror_correlation"
    df.to_csv(out / "significance_flip_rates.csv", index=False)

    # deterministic illustrative flip instance: first seed in a fixed scan
    sel = np.where(np.array([subset[i] == "Focus" for i in ids]) & (item_T > 0))[0]
    example = None
    for seed in range(1000):
        b = np.random.default_rng(seed).choice(sel, 100, replace=False)
        p, zn, zc, se_item, T = tests_for_draw(item_mean, item_T, b)
        if abs(zn) > 1.96 and abs(zc) <= 1.96:
            pm = item_mean[b].mean()
            example = dict(subset="Focus", N=100, scan_seed=seed, votes=int(T),
                           pooled_pref=round(float(p), 4),
                           z_naive=round(float(zn), 2), z_item=round(float(zc), 2),
                           ci_naive=[round(float(p - 1.96 * np.sqrt(p * (1 - p) / T)), 4),
                                     round(float(p + 1.96 * np.sqrt(p * (1 - p) / T)), 4)],
                           ci_item=[round(float(pm - 1.96 * se_item), 4),
                                    round(float(pm + 1.96 * se_item), 4)])
            break
    (out / "significance_flip_example.json").write_text(json.dumps(example, indent=2))

    # ---- figure ----
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
    plot = df[df.N.isin([100, 150])]
    labels = [f"{r.subset}\nN={r.N}" for r in plot.itertuples()]
    x = np.arange(len(labels))
    ax.bar(x, plot.flip_rate, 0.6, color=ACCENT)
    for xi, v in zip(x, plot.flip_rate):
        ax.text(xi, v + 0.008, f"{v:.0%}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, color=INK)
    ax.set_ylabel("verdict-flip rate", fontsize=8, color=INK)
    ax.set_title("A. Naive test significant, item-level test not\n(share of panel evaluations)",
                 fontsize=9, color=INK)
    ax.set_ylim(0, max(plot.flip_rate) * 1.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=7, colors=INK)

    e = example
    ax2.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
    for k, (ci, z, name, col) in enumerate((
            (e["ci_naive"], e["z_naive"], "naive\n(pooled votes)", NAIVE),
            (e["ci_item"], e["z_item"], "item-level\n(dependence-aware)", ACCENT))):
        mid = (ci[0] + ci[1]) / 2
        ax2.errorbar([k], [mid], yerr=[[mid - ci[0]], [ci[1] - mid]], fmt="o",
                     color=col, capsize=6, markersize=6, linewidth=2)
        ax2.text(k + 0.14, mid, f"z={z:.2f}", ha="left", va="center", fontsize=7.5, color=col)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["naive\n(pooled votes)", "item-level\n(dependence-aware)"],
                        fontsize=7.5, color=INK)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylabel("panel preference for system A", fontsize=8, color=INK)
    ax2.set_title(f"B. One {e['N']}-item Focus evaluation\n(same votes, two conclusions)",
                  fontsize=9, color=INK)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.grid(axis="y", color="#eeeeee", linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.tick_params(length=0, labelsize=7, colors=INK)

    fig.tight_layout()
    fig.savefig(ROOT / "figures/fig_significance_flip.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures/fig_significance_flip.png", dpi=200, bbox_inches="tight")
    print(df.to_string(index=False))
    print("example:", example)


if __name__ == "__main__":
    main()

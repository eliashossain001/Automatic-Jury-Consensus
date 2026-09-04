"""Task 3: reviewer-friendly figure — GRPO training increases judge dependence.

Reads results/grpo_judges/base_vs_grpo_dependence.csv and renders a two-panel
grouped-bar figure (rho_bar and n_eff, Base vs GRPO on the matched 6-judge bank),
to PNG + PDF, plus a caption. Deterministic; no randomness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/grpo_judges/figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "results/grpo_judges/base_vs_grpo_dependence.csv")
df = df.set_index("bank")
base_key = "base_matched(6 judges,qwen/mistral/phi)"
grpo_key = "GRPO(6 judges,qwen/mistral/phi)"

rho_base, rho_grpo = df.loc[base_key, "rho_bar"], df.loc[grpo_key, "rho_bar"]
neff_base, neff_grpo = df.loc[base_key, "n_eff"], df.loc[grpo_key, "n_eff"]

BASE_C, GRPO_C = "#4C78A8", "#A41E34"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 4.3))
labels = ["Base\n(untrained)", "GRPO\n(trained)"]

# Panel A: rho_bar (higher = more dependent)
barsA = axA.bar(labels, [rho_base, rho_grpo], color=[BASE_C, GRPO_C], width=0.6)
axA.set_ylabel(r"mean error correlation  $\bar{\rho}$")
axA.set_title("A. Dependence increases", fontweight="bold", fontsize=12)
axA.set_ylim(0, max(rho_base, rho_grpo) * 1.35)
for b, v in zip(barsA, [rho_base, rho_grpo]):
    axA.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
axA.annotate("", xy=(0.78, rho_grpo + 0.022), xytext=(0.3, rho_base + 0.02),
             arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.0, mutation_scale=20,
                             connectionstyle="arc3,rad=-0.32"))
axA.text(0.5, max(rho_base, rho_grpo) * 1.27, f"+{(rho_grpo - rho_base):.3f}",
         ha="center", color="#333", fontsize=12, fontweight="bold")

# Panel B: n_eff (lower = fewer independent judges)
barsB = axB.bar(labels, [neff_base, neff_grpo], color=[BASE_C, GRPO_C], width=0.6)
axB.set_ylabel(r"effective ensemble size  $n_{\mathrm{eff}}$")
axB.set_title("B. Effective ensemble shrinks", fontweight="bold", fontsize=12)
axB.set_ylim(0, 6.8)
axB.axhline(6, ls=":", color="#888", lw=1.2)
axB.text(1.44, 6.08, "6 nominal judges", color="#888", fontsize=9, ha="right", va="bottom")
for b, v in zip(barsB, [neff_base, neff_grpo]):
    axB.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
axB.annotate("", xy=(0.9, neff_grpo + 0.22), xytext=(0.3, neff_base + 0.28),
             arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.0, mutation_scale=20,
                             connectionstyle="arc3,rad=-0.32"))
axB.text(0.5, max(neff_base, neff_grpo) + 1.2, f"{(neff_grpo - neff_base):+.2f}",
         ha="center", color="#333", fontsize=12, fontweight="bold")

fig.tight_layout()
fig.savefig(FIG / "base_vs_grpo_dependence.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "base_vs_grpo_dependence.pdf", bbox_inches="tight")
plt.close(fig)

caption = (
    "Figure: GRPO training increases judge interdependence. On a matched bank of 6 logical "
    "judges (3 base models — Qwen2.5-7B, Mistral-7B-Instruct-v0.3, Phi-3.5-mini — each crossed "
    "with two prompt styles), we compare the untrained base judges against the same models after "
    "GRPO training on a preference-label reward. (A) Mean pairwise error correlation rises from "
    f"rho_bar = {rho_base:.3f} to {rho_grpo:.3f}. (B) The effective ensemble size falls from "
    f"n_eff = {neff_base:.2f} to {neff_grpo:.2f} (out of 6 nominal judges). Dependence is estimated "
    "on committed votes using the pairwise-complete Pearson estimator with Ledoit-Wolf-style "
    "shrinkage, so the increase is not an artifact of the higher GRPO abstention rate. "
    "GRPO training reduces, rather than increases, judge diversity."
)
(FIG / "base_vs_grpo_dependence_caption.txt").write_text(caption + "\n")
print("wrote:")
for p in ["base_vs_grpo_dependence.png", "base_vs_grpo_dependence.pdf", "base_vs_grpo_dependence_caption.txt"]:
    print("  ", FIG / p)
print("\n" + caption)

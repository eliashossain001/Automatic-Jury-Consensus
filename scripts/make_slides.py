"""Comprehensive presentation deck (16:9) of the latest CorrFilter results & experiments.

Pure matplotlib -> multi-page PDF (no LaTeX). Covers: cross-dataset incl. PKU-SafeRLHF,
GRPO & DPO preference-trained judges (Tables 2 & 3), bank scaling, gold-label corruption
test (Fig. 12), natural subgroups, low-rank deployment (Table 16), router & downstream.
Output: outputs/slides/corrfilter_update.pdf
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAVY, RED, BLUE, GREEN, GRAY, INK = "#1F2D3D", "#A41E34", "#4C78A8", "#1F7A3D", "#8A97A6", "#222831"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def _chrome(fig, title, kicker=None, page=None, total=None):
    fig.add_artist(Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure, color=NAVY, zorder=0))
    fig.add_artist(Rectangle((0, 0.881), 1, 0.004, transform=fig.transFigure, color=RED, zorder=1))
    if kicker:
        fig.text(0.045, 0.958, kicker.upper(), color=RED, fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.045, 0.918, title, color="white", fontsize=21, fontweight="bold", va="center")
    fig.add_artist(Rectangle((0.045, 0.055), 0.91, 0.0025, transform=fig.transFigure, color="#E3E7EC"))
    fig.text(0.045, 0.03, "CorrFilter — consensus is not reliability under dependence", color=GRAY, fontsize=9)
    if page:
        fig.text(0.955, 0.03, f"{page}/{total}", color=GRAY, fontsize=9, ha="right")


def _bullets(fig, items, x=0.055, y0=0.76, dy=0.093, fs=15.5, wrap=92):
    y = y0
    for it in items:
        lead, body = (it if isinstance(it, tuple) else (None, it))
        fig.add_artist(Rectangle((x, y - 0.006), 0.013, 0.026, transform=fig.transFigure, color=RED))
        tx = x + 0.028
        if lead:
            fig.text(tx, y + 0.006, lead, fontsize=fs, fontweight="bold", color=INK, va="center")
            if body:
                bl = textwrap.wrap(body, wrap)
                fig.text(tx, y - 0.028, "\n".join(bl), fontsize=fs - 2.5, color="#3C4653", va="top")
                y -= dy + 0.028 * len(bl)
            else:
                y -= dy
        else:
            lines = textwrap.wrap(body, wrap)
            fig.text(tx, y + 0.004, "\n".join(lines), fontsize=fs, color=INK, va="center" if len(lines) == 1 else "top")
            y -= dy + 0.028 * (len(lines) - 1)
    return y


def _image(fig, path, box):
    ax = fig.add_axes(box); ax.axis("off")
    ax.imshow(mpimg.imread(str(ROOT / path)))


def _stat(fig, x, y, w, big, small, color=RED):
    fig.add_artist(FancyBboxPatch((x, y), w, 0.135, transform=fig.transFigure,
                   boxstyle="round,pad=0.006,rounding_size=0.012", fc="#F6F7F9", ec="#E3E7EC", lw=1))
    fig.text(x + w / 2, y + 0.088, big, ha="center", fontsize=24, fontweight="bold", color=color)
    fig.text(x + w / 2, y + 0.032, small, ha="center", fontsize=10.5, color="#3C4653")


def _table(fig, x, y, cols, headers, rows, fs=13.5, hl=None, dy=0.05):
    for cx, h in zip(cols, headers):
        fig.text(x + cx, y, h, fontsize=fs, fontweight="bold", color=NAVY)
    fig.add_artist(Rectangle((x, y - 0.018), cols[-1] + 0.22, 0.002, transform=fig.transFigure, color="#C9D2DC"))
    for j, r in enumerate(rows):
        yy = y - 0.052 - j * dy
        col = RED if hl == j else INK
        for cx, v in zip(cols, r):
            fig.text(x + cx, yy, v, fontsize=fs, color=col, fontweight="bold" if (col == RED or cx == cols[0]) else "normal")


def _paperref(fig, text):
    fig.text(0.955, 0.082, "Paper: " + text, ha="right", fontsize=9.5, color=BLUE, style="italic")


def new_slide():
    return plt.figure(figsize=(13.33, 7.5), dpi=200)


def divider(pdf, kicker, title):
    fig = new_slide()
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=NAVY))
    fig.add_artist(Rectangle((0.08, 0.44), 0.10, 0.006, transform=fig.transFigure, color=RED))
    fig.text(0.08, 0.56, kicker.upper(), color=RED, fontsize=15, fontweight="bold")
    fig.text(0.08, 0.50, title, color="white", fontsize=30, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)


def main():
    out = ROOT / "outputs/slides/corrfilter_update.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    T = 12
    with PdfPages(out) as pdf:
        # 1. TITLE
        fig = new_slide()
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=NAVY))
        fig.add_artist(Rectangle((0, 0.30), 1, 0.006, transform=fig.transFigure, color=RED))
        fig.text(0.5, 0.62, "Consensus is not reliability\nunder dependence", ha="center", color="white",
                 fontsize=34, fontweight="bold", linespacing=1.2)
        fig.text(0.5, 0.42, "Measuring and mitigating structured co-failure in LLM judge banks",
                 ha="center", color="#C9D2DC", fontsize=15)
        fig.text(0.5, 0.20, "Latest results & recent experiments  •  prepared for Dr. Lim",
                 ha="center", color="#8A97A6", fontsize=12.5)
        pdf.savefig(fig); plt.close(fig)

        # 2. CONTEXT — the core finding
        fig = new_slide()
        _chrome(fig, "Recap — LLM judge banks are strongly, structurally dependent", "Context", 2, T)
        for i, (b, s, c) in enumerate([("0.21", "mean error corr.  $\\bar\\rho$", RED),
                                        ("3.5", "effective judges (of 10)", BLUE),
                                        ("27%", "items majority jointly wrong", RED),
                                        ("4.9", "hard ceiling  $1/\\bar\\rho$", GRAY)]):
            _stat(fig, 0.055 + i * 0.232, 0.63, 0.205, b, s, c)
        _bullets(fig, [
            "Ten nominal judges carry only ~3.5 independent judges' worth of evidence.",
            "Diversity does not fix it: prompt-style gives zero decorrelation; model-family is inconsistent and never restores independence (H1 rejected).",
            "This deck covers the recent experiments that stress-test and extend that finding.",
        ], y0=0.50, dy=0.115, fs=15.5)
        pdf.savefig(fig); plt.close(fig)

        # 3. CROSS-DATASET incl PKU
        fig = new_slide()
        _chrome(fig, "Replicates across three datasets — worst on safety (PKU-SafeRLHF)", "Empirical scope", 3, T)
        _paperref(fig, "Measuring Dependence — \"A third dataset: PKU-SafeRLHF\"")
        _image(fig, "outputs/pku_saferlhf/fig_dependence_summary.png", [0.06, 0.30, 0.88, 0.50])
        fig.text(0.5, 0.205, r"PKU-SafeRLHF (human safety labels) is the most dependent:   "
                 r"$\bar\rho=0.27$,   $n_{\mathrm{eff}}=2.9$ of 10,   co-failure $=0.53$",
                 ha="center", fontsize=13, color=NAVY, fontweight="bold")
        fig.text(0.5, 0.12, "Same 10-judge bank, same estimators — dependence is not an artifact of one benchmark.",
                 ha="center", fontsize=12.5, color="#3C4653")
        pdf.savefig(fig); plt.close(fig)

        # 4. GRPO judges (own slide) + Table 2
        fig = new_slide()
        _chrome(fig, "GRPO-trained judges are MORE dependent, not less", "Preference optimization", 4, T)
        _paperref(fig, "§ GRPO Judges Are More Dependent (Table 2, Fig.)")
        _image(fig, "results/grpo_judges/figures/base_vs_grpo_dependence.png", [0.04, 0.30, 0.52, 0.52])
        fig.text(0.60, 0.79, "Table 2 — GRPO vs matched base", fontsize=13, fontweight="bold", color=GRAY)
        _table(fig, 0.60, 0.73, [0.0, 0.19, 0.30], ["quantity", "base", "GRPO"], [
            (r"$\bar\rho$", "0.176", "0.235"),
            (r"$n_{\mathrm{eff}}$ (of 6)", "3.17", "2.76"),
            ("eigen-rank", "4.90", "4.56"),
            ("majority co-fail", "—", "0.22"),
            ("supermaj. FRR", "—", "0.20"),
        ], fs=13)
        fig.text(0.60, 0.30, "CorrFilter still cuts false retention:", fontsize=12.5, color=INK)
        fig.text(0.60, 0.255, "4.2 pts  [2.8, 5.6]  @ 60% retention", fontsize=13.5, fontweight="bold", color=RED)
        fig.text(0.055, 0.115, "Optimizing judges for accuracy pulls different models toward the same decision boundary.", fontsize=13, color=NAVY, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        # 5. DPO judges + Table 3
        fig = new_slide()
        _chrome(fig, "Not GRPO-specific — DPO amplifies dependence even more", "Preference optimization", 5, T)
        _paperref(fig, "§ GRPO section, \"Not specific to GRPO: DPO\" (Table 3, Fig.)")
        _image(fig, "outputs/figures/summary_dependence_training_dpo.png", [0.05, 0.44, 0.90, 0.40])
        fig.text(0.10, 0.35, "Table 3 — base vs GRPO vs DPO (matched 6-judge bank)", fontsize=12.5, fontweight="bold", color=GRAY)
        _table(fig, 0.10, 0.305, [0.0, 0.28, 0.42, 0.58], ["training", r"$\bar\rho$", r"$n_{\mathrm{eff}}$", "CorrFilter FRR reduction @0.60"], [
            ("Base (untrained)", "0.178", "3.17", "—"),
            ("GRPO-trained", "0.235", "2.76", "4.2 pts  [2.8, 5.6]"),
            ("DPO-trained", "0.326", "2.28", "6.0 pts  [4.4, 7.7]"),
        ], fs=13, hl=2)
        fig.text(0.055, 0.10, r"Both methods raise $\bar\rho$ (DPO more); CorrFilter still cuts false retention "
                 r"on both trained banks (CIs exclude 0).", fontsize=12.5, color=NAVY, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        # 6. SCALING
        fig = new_slide()
        _chrome(fig, "Adding judges does not recover independence", "Bank scaling", 6, T)
        _paperref(fig, "Measuring Dependence, \"Scaling the bank\" (Fig.)")
        _image(fig, "outputs/extended_bank/fig_neff_saturation.png", [0.045, 0.16, 0.47, 0.62])
        _bullets(fig, [
            ("Grow the bank: 4 -> 16 judges.", "Five families, 1.5B-14B, two prompt styles."),
            ("n_eff grows only sublinearly.", "4.7 effective judges at n=16 -- about 29% of nominal, down from 71% at n=4."),
            ("The ratio keeps shrinking.", "Heterogeneous scaling improves coverage, not independence."),
        ], x=0.55, y0=0.71, dy=0.075, fs=14.5, wrap=36)
        pdf.savefig(fig); plt.close(fig)

        # 7. CORRUPTION TEST — gold-label noise (Figure 12)
        fig = new_slide()
        _chrome(fig, "Corruption test — dependence is stable under gold-label noise", "Robustness", 7, T)
        _paperref(fig, "Appendix — Gold-Label Sensitivity (Fig. 12)")
        _image(fig, "outputs/gold_sensitivity/fig_label_noise.png", [0.045, 0.15, 0.47, 0.64])
        _bullets(fig, [
            ("Flip 1/3/5/10% of gold labels (40 seeds).", ""),
            ("rho_bar barely moves: 0.21 -> 0.23.", "n_eff 3.5 -> 3.3 (of 10) -- never restores independence."),
            ("Pseudo-clean subset is MORE dependent.", "High-consensus items: rho_bar 0.34 -- not a label-noise artifact."),
        ], x=0.55, y0=0.70, dy=0.078, fs=14.5, wrap=37)
        fig.text(0.55, 0.135, "(this is Figure 12 in the paper)", fontsize=10.5, color=GRAY, style="italic")
        pdf.savefig(fig); plt.close(fig)

        # 8. NATURAL SUBGROUPS
        fig = new_slide()
        _chrome(fig, "Vulnerable subgroups occur naturally, not only under synthetic poisoning", "Regimes", 8, T)
        _paperref(fig, "§ Position-Aligned Poisoning, \"Natural subgroup signatures\"")
        for i, (b, s, c) in enumerate([("150 / 319", "consensus-wrong items with a\nnatural subgroup signature", RED),
                                        ("47%", "of wrong items: leave-cluster-out\nflips majority to correct", BLUE),
                                        ("36", "cases CorrFilter fails but the\nbias-cluster filter fixes", GREEN)]):
            fig.add_artist(FancyBboxPatch((0.06 + i * 0.31, 0.55), 0.28, 0.20, transform=fig.transFigure,
                           boxstyle="round,pad=0.008,rounding_size=0.014", fc="#F6F7F9", ec="#E3E7EC", lw=1))
            fig.text(0.20 + i * 0.31, 0.685, b, ha="center", fontsize=23, fontweight="bold", color=c)
            fig.text(0.20 + i * 0.31, 0.60, s, ha="center", fontsize=10.5, color="#3C4653")
        _bullets(fig, [
            "Cluster the base judges by error-correlation; on wrong-majority items, test whether removing one correlated cluster flips the label.",
            "An incorrect consensus supported primarily by a correlated subset is exactly the vulnerable-subgroup regime — found in held-out data with no synthetic poison.",
            "Reported as existence cases; prevalence depends on clustering granularity (not claimed as a population rate).",
        ], y0=0.45, dy=0.10, fs=14.5, wrap=104)
        pdf.savefig(fig); plt.close(fig)

        # 9. DEPLOYMENT — Table 16 low-rank R
        fig = new_slide()
        _chrome(fig, "Cheap to deploy — a low-rank R recovers full CorrFilter", "Deployment", 9, T)
        _paperref(fig, "Appendix — Deployment & Low-Rank Approximation (Table 16)")
        fig.text(0.10, 0.79, "Table 16 — low-rank approximation of R (matched bank, FRR reduction @ 0.60)",
                 fontsize=13, fontweight="bold", color=GRAY)
        _table(fig, 0.12, 0.72, [0.0, 0.30], [r"$R$ used in $\alpha_{\mathrm{subset}}$", "FRR reduction @ 0.60 (pts, 95% CI)"], [
            ("full R (rank 6)", "20.3   [17.9, 22.8]"),
            ("rank-3", "23.8   [20.9, 26.5]"),
            ("rank-2", "26.0   [23.0, 29.1]"),
            ("rank-1", "9.5   [6.6, 12.5]"),
            ("identity (no correction)", "11.4   [9.5, 13.3]"),
        ], fs=13.5, hl=1, dy=0.058)
        _bullets(fig, [
            ("R is estimated once per bank; scoring is O(|S|^2).", "No per-item training, no matrix inversion."),
            ("Rank-2/3 recovers (even denoises) the full-R benefit.", "Large banks need only a block/factor approximation of R."),
            ("Cluster C is learnable from ~100 labels, or label-free from position sensitivity.", ""),
        ], x=0.055, y0=0.36, dy=0.075, fs=13.5, wrap=100)
        pdf.savefig(fig); plt.close(fig)

        # 10. HONESTY — router + downstream null
        fig = new_slide()
        _chrome(fig, "Reported honestly — where mitigations do NOT (yet) win", "Rigor", 10, T)
        _paperref(fig, "§ Regime Routing; Appendix — Downstream DPO Validation")
        _bullets(fig, [
            ("Router: no clear improvement.", "A supervised dev-set router matches ~78% regime accuracy but does not beat the best single filter (gain -0.6 pts, 95% CI [-1.5, +0.1] includes 0). Fixed unsupervised router is diagnostic, not a universal policy."),
            ("Downstream DPO: a null result.", "CorrFilter cut training contamination 24.9% -> 20.5%, but a 1.5B DPO policy's reward accuracy was unchanged within uncertainty. We do NOT claim a downstream win; contamination-scaling is future work."),
            ("No universal filter.", "Different regimes need different filters; regime identification is the prerequisite (Theorem: no fixed filter dominates)."),
        ], y0=0.74, dy=0.085, fs=14.5, wrap=100)
        pdf.savefig(fig); plt.close(fig)

        # 11. SUMMARY of recent experiments
        fig = new_slide()
        _chrome(fig, "Summary of recent experiments", "Overview", 11, T)
        _table(fig, 0.055, 0.75, [0.0, 0.42, 0.66], ["experiment", "result", "verdict"], [
            ("3rd dataset: PKU-SafeRLHF", "rho_bar 0.27, n_eff 2.9/10", "most dependent"),
            ("GRPO-trained judges", "rho_bar 0.18->0.24, n_eff 3.2->2.8", "dependence up"),
            ("DPO-trained judges", "rho_bar 0.18->0.33, n_eff 3.2->2.3", "up, more than GRPO"),
            ("Extended bank 4->16", "n_eff 4.7 of 16 (~29% of nominal)", "no independence"),
            ("Gold-label corruption", "rho_bar 0.21->0.23 at 10% flips", "stable"),
            ("Natural subgroups", "150/319 wrong items", "regime is real"),
            ("Low-rank R (deploy)", "rank-2/3 recovers full CorrFilter", "cheap"),
            ("CorrFilter @ matched ret.", "cuts FRR 4-6 pts (CI excludes 0)", "effective"),
            ("Dev-set router", "gain -0.6 pts, CI includes 0", "no clear win"),
            ("Downstream 1.5B DPO", "accuracy unchanged", "null (honest)"),
        ], fs=12.5, dy=0.062)
        pdf.savefig(fig); plt.close(fig)

        # 12. BOTTOM LINE
        fig = new_slide()
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=NAVY))
        fig.add_artist(Rectangle((0, 0.74), 1, 0.006, transform=fig.transFigure, color=RED))
        fig.text(0.5, 0.85, "Bottom line", ha="center", color="white", fontsize=26, fontweight="bold")
        for i, t in enumerate([
            "Judge banks are strongly dependent (10 act like ~3.5); replicates on RewardBench, UltraFeedback, and PKU-SafeRLHF — worst on safety.",
            "Preference optimization (GRPO and DPO) amplifies dependence; scaling the bank does not recover independence; the finding is robust to gold-label noise.",
            "High agreement is not a safety certificate. CorrFilter (cheap, low-rank) is the mitigation; router and downstream nulls reported honestly.",
        ]):
            fig.add_artist(Rectangle((0.08, 0.605 - i * 0.155), 0.014, 0.03, transform=fig.transFigure, color=RED))
            fig.text(0.11, 0.62 - i * 0.155, "\n".join(textwrap.wrap(t, 80)), color="white", fontsize=15,
                     va="top", linespacing=1.35)
        fig.text(0.5, 0.10, "Next: contamination-scaling downstream • more expressive router • frontier/API judges",
                 ha="center", color="#8A97A6", fontsize=12)
        pdf.savefig(fig); plt.close(fig)

    print(f"[make_slides] wrote {T}-slide deck -> {out}  ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()

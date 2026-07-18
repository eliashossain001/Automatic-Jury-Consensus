"""Assemble paper-ready sub-tables + a dependence figure from dependence.csv (Bucket 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dependence", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.dependence).set_index("label")

    keep = ["rho_bar", "n_eff", "eigen_rank", "conditional_cofailure", "majority_cofailure",
            "prompt_contrast_delta", "family_contrast_delta", "abstention_rate"]

    # Cross-dataset replication (base 10-judge)
    if {"rewardbench_base_10", "ultrafeedback_base_10"} <= set(d.index):
        rep = d.loc[["rewardbench_base_10", "ultrafeedback_base_10"], keep].T
        rep.columns = ["RewardBench_v2", "UltraFeedback"]
        rep.to_csv(out / "replication_rb_vs_uf.csv")
        print("wrote", out / "replication_rb_vs_uf.csv")

    # GRPO base-vs-trained (matched 6-judge)
    if {"rewardbench_base_matched6", "grpo_trained_6"} <= set(d.index):
        g = d.loc[["rewardbench_base_matched6", "grpo_trained_6"], keep].T
        g.columns = ["Base_untrained_6", "GRPO_trained_6"]
        g.to_csv(out / "grpo_base_vs_trained.csv")
        print("wrote", out / "grpo_base_vs_trained.csv")

    # Contrasts table
    if "prompt_contrast_delta" in d.columns:
        d[["n_judges", "prompt_contrast_delta", "family_contrast_delta"]].to_csv(out / "contrasts.csv")
        print("wrote", out / "contrasts.csv")

    # n_eff vs nominal-n / ceiling figure
    if args.figure:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        sub = d.reset_index()
        ax.bar(range(len(sub)), sub["n_eff"], color="#A41E34", width=0.55, label=r"$n_{\mathrm{eff}}$")
        ax.plot(range(len(sub)), sub["n_judges"], "o--", color="#4C78A8", label="nominal $n$")
        ax.plot(range(len(sub)), sub["neff_ceiling_1_over_rho"], "s:", color="#666",
                label=r"ceiling $1/\bar\rho$")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("judges")
        ax.set_title(r"Effective ensemble size stays far below nominal $n$", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, bbox_inches="tight")
        fig.savefig(str(args.figure).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print("wrote", args.figure)


if __name__ == "__main__":
    main()

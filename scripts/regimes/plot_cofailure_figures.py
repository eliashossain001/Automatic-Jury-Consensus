"""Render the four CFI figures from the analysis CSVs.

Inputs:
    experiments/h2_cfi/results/{cfi_summary_by_bank.csv, consensus_accuracy_curves.csv,
                                false_retention_curves.csv, cfi_corrfilter_comparison.csv,
                                conditional_error_heatmap.npz}

Outputs (also mirrored into results/cfi/):
    experiments/h2_cfi/figures/consensus_vs_accuracy.png
    experiments/h2_cfi/figures/false_retention_vs_biased_ratio.png
    experiments/h2_cfi/figures/conditional_error_heatmap.png
    experiments/h2_cfi/figures/naive_vs_corrfilter_bars.png

Usage:
    python scripts/regimes/plot_cofailure_figures.py [--config configs/cfi.yaml]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from corrfilter.cfi.plots import (
    plot_conditional_error_heatmap,
    plot_consensus_vs_accuracy,
    plot_false_retention_vs_biased_ratio,
    plot_naive_vs_corrfilter_bars,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.regimes.plot_cofailure_figures")


def _mirror(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    dst.write_bytes(src.read_bytes())
    return dst


def _consensus_curves_per_variant(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    series: dict[str, dict[str, np.ndarray]] = {}
    for variant, sub in df.groupby("variant"):
        sub_sorted = sub.sort_values("bin_lo")
        series[str(variant)] = {
            "bin_lo": sub_sorted["bin_lo"].to_numpy(),
            "bin_hi": sub_sorted["bin_hi"].to_numpy(),
            "accuracy": sub_sorted["accuracy"].to_numpy(),
            "count": sub_sorted["count"].to_numpy(),
        }
    return series


def _select_focus_curves(curves: dict, mechanism: str | None = None) -> dict:
    """Subset the curves to a single mechanism for the headline figure."""
    if mechanism is None:
        return curves
    return {k: v for k, v in curves.items() if k.startswith(f"{mechanism}_pct")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cfi.yaml")
    parser.add_argument("--focus-mechanism", default=None, help="Optional single mechanism to focus the consensus-vs-accuracy figure on.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results_dir = Path(cfg["output"]["results_csv_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    headline_dir = Path(cfg["output"]["results_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    consensus_df = pd.read_csv(results_dir / "consensus_accuracy_curves.csv")
    fr_df = pd.read_csv(results_dir / "false_retention_curves.csv")
    cmp_df = pd.read_csv(results_dir / "cfi_corrfilter_comparison.csv")
    bank_df = pd.read_csv(results_dir / "cfi_summary_by_bank.csv")

    # ---- Figure 1: consensus vs accuracy ----
    curves = _consensus_curves_per_variant(consensus_df)
    if args.focus_mechanism:
        curves = _select_focus_curves(curves, args.focus_mechanism)
    fig1 = plot_consensus_vs_accuracy(
        curves,
        figures_dir / "consensus_vs_accuracy.png",
        title=f"Accuracy vs consensus level (focus: {args.focus_mechanism})" if args.focus_mechanism else "Accuracy vs consensus level (all variants)",
    )

    # ---- Figure 2: false retention vs biased ratio (per method) ----
    # One series per (mechanism, method); for the headline figure we focus on
    # the 'majority' and 'corrfilter_h1' methods aggregated by mechanism.
    series: dict[str, tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]] = {}
    for (mech, method), sub in fr_df.groupby(["mechanism", "method"]):
        if method not in {"majority", "corrfilter_h1", "supermajority"}:
            continue
        sub_sorted = sub.sort_values("biased_ratio")
        series[f"{mech} :: {method}"] = (
            sub_sorted["biased_ratio"].to_numpy(),
            sub_sorted["false_retention_rate"].to_numpy(),
            None,
        )
    fig2 = plot_false_retention_vs_biased_ratio(
        series,
        figures_dir / "false_retention_vs_biased_ratio.png",
        title="False retention vs biased-judge ratio",
    )

    # ---- Figure 3: per-judge × mechanism conditional error heatmap ----
    cond = np.load(results_dir / "conditional_error_heatmap.npz", allow_pickle=True)
    fig3 = plot_conditional_error_heatmap(
        cond["matrix"],
        list(cond["rows"]),
        list(cond["cols"]),
        figures_dir / "conditional_error_heatmap.png",
        title="Per-judge error rate on triggered items (100%-biased banks)",
    )

    # ---- Figure 4: naive vs CorrFilter at matched retention ----
    cmp_rows = cmp_df.to_dict(orient="records")
    fig4 = plot_naive_vs_corrfilter_bars(
        cmp_rows,
        figures_dir / "naive_vs_corrfilter_bars.png",
        title="False retention at matched retention: naive vs CorrFilter (H1 R)",
    )

    for fig in (fig1, fig2, fig3, fig4):
        mirrored = _mirror(fig, headline_dir)
        logger.info("figure → %s (mirror: %s)", fig, mirrored)

    print(f"figures → {figures_dir}")


if __name__ == "__main__":
    _ = bank_df_is_used = True  # bank_df is loaded for downstream notebooks
    main()

"""CFI plots.

Four figures cover the headline questions of Experiment 2:

1. ``plot_consensus_vs_accuracy`` — accuracy at consensus level, faceted by
   biased-ratio. The clean curve should be monotone; the all-biased curve
   should bend downward in the high-consensus region.
2. ``plot_false_retention_vs_biased_ratio`` — false retention rate as a
   function of biased-judge ratio, per method.
3. ``plot_conditional_error_heatmap`` — per-judge error rate conditional on
   each bias mechanism, as a (judge × mechanism) heatmap.
4. ``plot_naive_vs_corrfilter_bars`` — false retention rate at matched
   retention, naive majority vs CorrFilter, per (mechanism × ratio).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def plot_consensus_vs_accuracy(
    curves: dict[str, dict[str, np.ndarray]],
    out_path: str | Path,
    title: str = "Accuracy vs consensus level",
) -> Path:
    """One line per series; each ``curves[name] = {bin_lo, bin_hi, accuracy, count}``."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = plt.get_cmap("viridis")
    n = max(len(curves), 1)
    for i, (name, c) in enumerate(curves.items()):
        x = 0.5 * (c["bin_lo"] + c["bin_hi"])
        y = c["accuracy"]
        count = c.get("count")
        keep = count is None or (count > 0)
        if isinstance(keep, np.ndarray):
            x = x[keep]
            y = y[keep]
        ax.plot(x, y, marker="o", linewidth=1.6, color=palette(i / n), label=name)
    ax.set_xlabel("consensus level")
    ax.set_ylabel("accuracy when kept")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_false_retention_vs_biased_ratio(
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    out_path: str | Path,
    title: str = "False retention vs biased-judge ratio",
) -> Path:
    """``series[name] = (ratios, frrs, (lo, hi) | None)``."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = plt.get_cmap("tab10")
    for i, (name, payload) in enumerate(series.items()):
        ratios = payload[0]
        frrs = payload[1]
        ci = payload[2] if len(payload) > 2 else None
        color = palette(i % 10)
        ax.plot(ratios, frrs, marker="o", linewidth=1.6, color=color, label=name)
        if ci is not None:
            lo, hi = ci
            ax.fill_between(ratios, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("biased-judge ratio")
    ax.set_ylabel("false retention rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_conditional_error_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    out_path: str | Path,
    title: str = "Per-judge error rate conditional on bias mechanism",
) -> Path:
    """Heatmap of per-judge error rates conditional on each bias mechanism."""
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(col_labels) + 4), max(4, 0.5 * len(row_labels) + 2)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=7, color="black",
            )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_naive_vs_corrfilter_bars(
    rows: list[dict],
    out_path: str | Path,
    title: str = "Naive consensus vs CorrFilter at matched retention",
) -> Path:
    """Grouped bar chart: naive_frr vs corrfilter_frr per (mechanism, ratio)."""
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path

    labels = [f"{r['mechanism']}@{int(round(r['biased_ratio']*100)):03d}%" for r in rows]
    naive = np.array([r["false_retention_majority"] for r in rows], dtype=np.float64)
    cf = np.array([r["false_retention_corrfilter"] for r in rows], dtype=np.float64)
    x = np.arange(len(rows))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(rows)), 4.5))
    ax.bar(x - width / 2, naive, width=width, color="#c0392b", label="naive majority")
    ax.bar(x + width / 2, cf, width=width, color="#2c7fb8", label="CorrFilter")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("false retention rate (matched retention)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, max(1.02, float(max(naive.max(), cf.max())) * 1.1))
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


__all__ = [
    "plot_consensus_vs_accuracy",
    "plot_false_retention_vs_biased_ratio",
    "plot_conditional_error_heatmap",
    "plot_naive_vs_corrfilter_bars",
]

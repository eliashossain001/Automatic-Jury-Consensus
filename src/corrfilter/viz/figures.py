"""Diagnostic figures for the H1 measurement (proposal §3.4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

matplotlib.use("Agg")


def plot_correlation_heatmap(
    R: np.ndarray,
    labels: list[str],
    out_path: str | Path,
    title: str = "Judge error-correlation matrix R",
    vmin: float = -0.2,
    vmax: float = 1.0,
) -> Path:
    """Heatmap of R with judge labels on both axes."""
    n = R.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.55), max(5, n * 0.55)))
    im = ax.imshow(R, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_dendrogram(Z: np.ndarray, labels: list[str], out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.5), 4))
    dendrogram(Z, labels=labels, leaf_rotation=60, leaf_font_size=8, ax=ax)
    ax.set_title("Hierarchical clustering on 1 - R")
    ax.set_ylabel("distance (1 - R)")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_eigenspectrum(eigvals: np.ndarray, n_eff_eig: float, out_path: str | Path) -> Path:
    """Bar chart of the eigenvalues of R with n_eff^eig annotated."""
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, len(eigvals) + 1)
    ax.bar(xs, eigvals, color="#3060b0")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="independence baseline λ = 1")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue of R")
    ax.set_title(f"Eigenspectrum of R   (n_eff^eig = {n_eff_eig:.2f})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_neff_collapse(rho_bar: float, n_obs: int, out_path: str | Path, max_n: int = 32) -> Path:
    """n_eff(n; ρ̄) curve plotted against the independence baseline."""
    from corrfilter.correlation.effective_size import neff_curve

    ks = np.arange(1, max_n + 1)
    curve = neff_curve(rho_bar, max_n=max_n)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ks, linestyle="--", color="grey", label="ρ̄ = 0 (independence)")
    ax.plot(ks, curve, color="#a02040", linewidth=2.0, label=f"ρ̄ = {rho_bar:.3f} (observed)")
    if rho_bar > 0:
        cap = 1.0 / rho_bar
        ax.axhline(cap, color="#a02040", linestyle=":", linewidth=0.9, label=f"asymptote 1/ρ̄ = {cap:.2f}")
    ax.scatter([n_obs], [curve[n_obs - 1]], color="#a02040", zorder=5, s=40)
    ax.set_xlabel("bank size n")
    ax.set_ylabel("effective ensemble size n_eff")
    ax.set_title("Effective-ensemble-size collapse under measured ρ̄")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path

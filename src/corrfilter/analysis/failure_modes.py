"""Failure-mode decomposition from the eigenstructure of R.

Each leading eigenvector of R indexes a direction in judge-error space
along which a substantial fraction of the bank co-varies. Items whose
error pattern loads heavily on a leading eigenvector are candidates for
the latent failure mechanism that direction represents (proposal §3.5).

This module returns the top-k eigenvectors plus per-item loadings, so a
qualitative review can attach interpretation labels (verbosity, sycophancy,
refusal alignment, ...) to specific directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenDirection:
    """One eigenvector of R with its item-side loadings."""

    rank: int               # 1-indexed eigenvalue rank (1 = largest)
    eigenvalue: float
    variance_explained: float
    judge_loading: np.ndarray   # (n_judges,) eigenvector entries
    item_loading: np.ndarray    # (n_items,) projected scores E @ v


def top_eigen_directions(R: np.ndarray, E: np.ndarray, k: int = 3) -> list[EigenDirection]:
    """Return the top-``k`` failure-mode directions sorted by eigenvalue."""
    R_sym = (R + R.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(R_sym)
    # eigh returns ascending; reverse to descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(np.clip(eigvals, a_min=0.0, a_max=None).sum())
    out: list[EigenDirection] = []
    for r in range(min(k, len(eigvals))):
        v = eigvecs[:, r]
        item_loading = E @ v
        out.append(
            EigenDirection(
                rank=r + 1,
                eigenvalue=float(eigvals[r]),
                variance_explained=float(eigvals[r] / total) if total > 0 else 0.0,
                judge_loading=v,
                item_loading=item_loading,
            )
        )
    return out

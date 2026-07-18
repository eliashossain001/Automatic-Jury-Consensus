"""Hierarchical clustering on ``1 - R`` for dendrogram + reordering.

The proposal commits to a dendrogram view of the bank (§3.4) that should
expose family / prompt-style clusters as natural cohorts. The same linkage
matrix supplies a leaf order that puts correlated judges adjacent in the
heatmap, which makes block structure visually legible.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _to_distance(R: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to non-negative distance ``1 - R``.

    Clipped to ``[0, 2]`` to satisfy ``squareform``; off-diagonal NaNs are
    treated as fully dissimilar.
    """
    D = 1.0 - R
    D = np.nan_to_num(D, nan=2.0, posinf=2.0, neginf=0.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, a_min=0.0, a_max=2.0)
    return D


def hierarchical_order(R: np.ndarray, method: str = "average") -> tuple[np.ndarray, np.ndarray]:
    """Linkage matrix + leaf order that groups correlated judges adjacently."""
    D = _to_distance(R)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)
    return Z, order

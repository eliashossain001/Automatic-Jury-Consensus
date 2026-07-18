"""Consensus computation: majority vote, supermajority, and per-item consensus level.

Definitions match the proposal:

* ``consensus_level(item)`` is ``max(p1, 1 - p1)`` where ``p1`` is the
  fraction of non-abstaining judges that voted 1 on the item. It lives in
  ``[0.5, 1.0]`` and is the natural x-axis for the "consensus vs accuracy"
  curve.
* ``majority_consensus`` returns the per-item argmax of ``(p0, p1)`` with a
  deterministic tie-break (defaults to 1 because the gold label is 1 in our
  manifest, which keeps tie items in the "predicted positive" pool the same
  way as a deterministic tie-break would).
* ``supermajority_consensus`` only predicts when ``consensus_level >=
  threshold``; otherwise it abstains (encoded as ``-1``).
"""

from __future__ import annotations

import numpy as np

ABSTAIN = -1


def _votes_with_mask(V: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if V.shape != M.shape:
        raise ValueError(f"V {V.shape} must match M {M.shape}")
    if V.ndim != 2:
        raise ValueError(f"V must be 2D; got shape {V.shape}")
    return V.astype(np.int8), M.astype(bool)


def consensus_level(V: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per-item consensus level in ``[0.5, 1.0]`` (items with no voters → 0.5)."""
    V_, M_ = _votes_with_mask(V, M)
    counts = M_.sum(axis=1)
    ones = (V_ * M_).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = np.where(counts > 0, ones / np.maximum(counts, 1), 0.5)
    return np.maximum(p1, 1.0 - p1)


def vote_fraction(V: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per-item fraction of non-abstainers voting 1."""
    V_, M_ = _votes_with_mask(V, M)
    counts = M_.sum(axis=1)
    ones = (V_ * M_).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, ones / np.maximum(counts, 1), 0.5)


def majority_consensus(
    V: np.ndarray, M: np.ndarray, tie_break: int = 1
) -> np.ndarray:
    """Per-item majority prediction, defaulting to ``tie_break`` on ties."""
    if tie_break not in (0, 1):
        raise ValueError(f"tie_break must be 0 or 1; got {tie_break}")
    p1 = vote_fraction(V, M)
    pred = np.where(p1 > 0.5, 1, np.where(p1 < 0.5, 0, tie_break)).astype(np.int8)
    counts = M.astype(bool).sum(axis=1)
    pred = np.where(counts > 0, pred, ABSTAIN).astype(np.int8)
    return pred


def supermajority_consensus(
    V: np.ndarray, M: np.ndarray, threshold: float = 2 / 3
) -> np.ndarray:
    """Predict only when consensus_level ≥ threshold; otherwise return ABSTAIN."""
    if not 0.5 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0.5, 1.0]; got {threshold}")
    p1 = vote_fraction(V, M)
    level = np.maximum(p1, 1.0 - p1)
    pred = np.where(p1 >= 0.5, 1, 0).astype(np.int8)
    keep = level >= threshold
    return np.where(keep, pred, ABSTAIN).astype(np.int8)


def weighted_consensus(
    V: np.ndarray, M: np.ndarray, weights: np.ndarray, tie_break: int = 1
) -> np.ndarray:
    """Weighted majority vote with per-judge ``weights``.

    Used by the "independent weighted vote" baseline when judge-quality
    weights are available (e.g., inverse error rate from H1).
    """
    V_, M_ = _votes_with_mask(V, M)
    if weights.shape != (V_.shape[1],):
        raise ValueError(
            f"weights shape {weights.shape} != ({V_.shape[1]},) judges"
        )
    w = weights[None, :] * M_
    ones = (V_ * w).sum(axis=1)
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = np.where(total > 0, ones / np.maximum(total, 1e-12), 0.5)
    pred = np.where(p1 > 0.5, 1, np.where(p1 < 0.5, 0, tie_break)).astype(np.int8)
    counts = M_.sum(axis=1)
    return np.where(counts > 0, pred, ABSTAIN).astype(np.int8)


def accuracy_by_consensus_level(
    pred: np.ndarray, gold: np.ndarray, level: np.ndarray, bins: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin items by consensus level and compute accuracy per bin.

    Returns ``(bin_edges, accuracy_per_bin, count_per_bin)``. Items with
    ``pred == ABSTAIN`` are excluded.
    """
    if bins is None:
        bins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0001], dtype=np.float64)
    keep = pred != ABSTAIN
    pred_k = pred[keep]
    gold_k = gold[keep]
    level_k = level[keep]
    idx = np.digitize(level_k, bins[1:-1])  # right-open bins, last bin closed
    n_bins = len(bins) - 1
    acc = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = idx == b
        cnt[b] = int(sel.sum())
        if cnt[b] > 0:
            acc[b] = float((pred_k[sel] == gold_k[sel]).mean())
    return bins, acc, cnt


__all__ = [
    "ABSTAIN",
    "consensus_level",
    "vote_fraction",
    "majority_consensus",
    "supermajority_consensus",
    "weighted_consensus",
    "accuracy_by_consensus_level",
]

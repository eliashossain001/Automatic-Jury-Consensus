"""Retention and false-retention metrics for the CFI experiment.

Definitions:

* ``retention_rate`` = fraction of items kept by a method.
* ``false_retention_rate`` = among kept items, fraction whose retained label
  disagrees with the gold label.

Both are computed conditional on a per-item ``keep`` mask, so the same
machinery handles naive majority (keep ⇔ not-abstain), supermajority
(keep ⇔ consensus_level ≥ τ), and CorrFilter (keep ⇔ top-k by α_subset).
"""

from __future__ import annotations

import numpy as np

from corrfilter.cfi.consensus import ABSTAIN


def retention_rate(keep: np.ndarray) -> float:
    """Fraction of items in ``keep`` (boolean)."""
    if keep.size == 0:
        return 0.0
    return float(np.mean(keep.astype(bool)))


def false_retention_rate(
    keep: np.ndarray, retained_label: np.ndarray, gold: np.ndarray
) -> float:
    """Among kept items, fraction whose retained label disagrees with gold.

    Items where the retained label is ``ABSTAIN`` are treated as not kept.
    """
    keep = keep.astype(bool) & (retained_label != ABSTAIN)
    if not keep.any():
        return 0.0
    wrong = retained_label[keep] != gold[keep]
    return float(np.mean(wrong))


def accuracy_when_kept(
    keep: np.ndarray, retained_label: np.ndarray, gold: np.ndarray
) -> float:
    """Among kept items, fraction whose retained label matches gold (= 1 - FRR)."""
    keep = keep.astype(bool) & (retained_label != ABSTAIN)
    if not keep.any():
        return 0.0
    return float(np.mean(retained_label[keep] == gold[keep]))


def accuracy_by_consensus_curve(
    pred: np.ndarray,
    gold: np.ndarray,
    level: np.ndarray,
    bins: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute accuracy and item count per consensus-level bin.

    Returns a dict with keys ``bin_lo``, ``bin_hi``, ``accuracy``, ``count``.
    """
    if bins is None:
        bins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0001], dtype=np.float64)
    keep = pred != ABSTAIN
    p = pred[keep]
    g = gold[keep]
    lv = level[keep]
    idx = np.digitize(lv, bins[1:-1])
    n_bins = len(bins) - 1
    acc = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = idx == b
        cnt[b] = int(sel.sum())
        if cnt[b] > 0:
            acc[b] = float(np.mean(p[sel] == g[sel]))
    return {
        "bin_lo": bins[:-1],
        "bin_hi": bins[1:],
        "accuracy": acc,
        "count": cnt,
    }


def false_retention_by_ratio(
    variant_rows: list[dict],
) -> dict[str, np.ndarray]:
    """Aggregate per-variant rows into ``(ratio → false_retention_rate)`` curves.

    Each row should expose at least ``biased_ratio`` and one or more
    ``false_retention_<method>`` columns. Returns ``{method: (ratios, frrs)}``.
    """
    if not variant_rows:
        return {}
    keys = sorted({k for row in variant_rows for k in row if k.startswith("false_retention_")})
    out: dict[str, np.ndarray] = {}
    ratios = np.array([row["biased_ratio"] for row in variant_rows], dtype=np.float64)
    order = np.argsort(ratios)
    out["biased_ratio"] = ratios[order]
    for k in keys:
        vals = np.array([row.get(k, np.nan) for row in variant_rows], dtype=np.float64)
        out[k] = vals[order]
    return out


def conditional_error_rate(
    V: np.ndarray, M: np.ndarray, gold: np.ndarray, condition_mask: np.ndarray
) -> np.ndarray:
    """Per-judge error rate restricted to items where ``condition_mask`` is True.

    Used for the (judge × bias mechanism) heatmap: condition each cell on items
    where the bias mechanism is triggered and average per-judge errors.
    """
    if condition_mask.dtype != bool:
        condition_mask = condition_mask.astype(bool)
    if not condition_mask.any():
        return np.zeros(V.shape[1], dtype=np.float64)
    V_, M_ = V[condition_mask], M[condition_mask]
    gold_ = gold[condition_mask]
    err = (V_ != gold_[:, None]).astype(np.float64) * M_
    denom = M_.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(denom > 0, err.sum(axis=0) / np.maximum(denom, 1), 0.0)
    return rate


__all__ = [
    "retention_rate",
    "false_retention_rate",
    "accuracy_when_kept",
    "accuracy_by_consensus_curve",
    "false_retention_by_ratio",
    "conditional_error_rate",
]

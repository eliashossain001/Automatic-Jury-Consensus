"""CorrFilter v1 — filtering methods and evaluation (Bucket 3, Part C).

This module is a thin, well-typed dispatch layer over the primitives that
already live in the package:

* consensus rules — :mod:`corrfilter.cfi.consensus`
  (majority / supermajority / accuracy-weighted majority);
* the subset-aware decorrelated score α_subset and its thresholding —
  :mod:`corrfilter.cfi.corrfilter_score`.

It exposes five comparable filtering methods over a shared interface so the
CorrFilter run script can score them on the same vote set / R / gold:

1. ``naive_majority``                — keep every non-abstaining item.
2. ``naive_supermajority_75``        — keep items with consensus ≥ 0.75.
3. ``independent_accuracy_weighted`` — weighted majority by inverse-error rate.
4. ``corrfilter_subset``             — keep the top-retention items by α_subset.
5. ``corrfilter_subset_risk_tau``    — keep items with α_subset ≥ a risk floor.

A filter is treated as a binary classifier deciding which items to *keep*; the
"positive" class is an item whose retained (preferred) label matches gold.
Precision is therefore the accuracy among kept items (1 − false-retention) and
recall is the fraction of all correctly-labellable items that survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corrfilter.cfi.consensus import (
    ABSTAIN,
    consensus_level,
    majority_consensus,
    supermajority_consensus,
    vote_fraction,
    weighted_consensus,
)
from corrfilter.cfi.corrfilter_score import (
    corrfilter_score,
    retention_match_threshold,
)


@dataclass(frozen=True)
class FilterResult:
    """Outcome of one filtering method on a vote set."""

    method: str
    keep: np.ndarray            # (N,) bool — items retained
    retained_label: np.ndarray  # (N,) preferred label per item (ABSTAIN if none)
    score: np.ndarray | None = None  # (N,) ranking score where the method has one
    meta: dict = field(default_factory=dict)


def inverse_error_weights(
    V: np.ndarray, M: np.ndarray, gold: np.ndarray, floor: float = 1e-3
) -> np.ndarray:
    """Per-judge weight ∝ 1 / error_rate, estimated on the supplied votes.

    Used by the independent-accuracy-weighted baseline. When fed the clean H1
    votes this is the realistic "trust the historically-accurate judges more"
    weighting; the weights are clipped so a perfect judge does not dominate.
    """
    M_b = M.astype(bool)
    err = (V != gold[:, None]).astype(np.float64) * M_b
    denom = M_b.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(denom > 0, err.sum(axis=0) / np.maximum(denom, 1), 0.5)
    rate = np.clip(rate, floor, 1.0)
    w = 1.0 / rate
    return w / w.mean()


def filter_naive_majority(V, M, gold) -> FilterResult:
    label = majority_consensus(V, M)
    keep = (M.astype(bool).any(axis=1)) & (label != ABSTAIN)
    margin = np.abs(vote_fraction(V, M) - 0.5)  # confidence proxy for ranking
    return FilterResult("naive_majority", keep, label, score=margin)


def filter_supermajority(V, M, gold, threshold: float = 0.75) -> FilterResult:
    label = supermajority_consensus(V, M, threshold=threshold)
    keep = label != ABSTAIN
    return FilterResult(
        f"naive_supermajority_{int(round(threshold * 100))}",
        keep, label, score=consensus_level(V, M),
        meta={"threshold": threshold},
    )


def filter_independent_weighted(
    V, M, gold, weights, target_retention: float | None = None
) -> FilterResult:
    label = weighted_consensus(V, M, weights)
    valid = label != ABSTAIN
    # Weighted confidence margin around 0.5 as the ranking score.
    M_b = M.astype(bool)
    w = weights[None, :] * M_b
    ones = (V * w).sum(axis=1)
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = np.where(total > 0, ones / np.maximum(total, 1e-12), 0.5)
    margin = np.abs(p1 - 0.5)
    if target_retention is None:
        keep = valid
    else:
        n_keep = int(round(target_retention * V.shape[0]))
        score = np.where(valid, margin, -np.inf)
        keep, _ = retention_match_threshold(score, n_keep)
    return FilterResult("independent_accuracy_weighted", keep, label, score=margin)


def filter_corrfilter_subset(
    V, M, R, gold, target_retention: float
) -> FilterResult:
    label = majority_consensus(V, M)
    scores = corrfilter_score(V, M, R, label)
    n_keep = int(round(target_retention * V.shape[0]))
    keep, threshold = retention_match_threshold(scores.score, n_keep)
    return FilterResult(
        "corrfilter_subset", keep, label, score=scores.score,
        meta={"threshold": float(threshold), "target_retention": target_retention,
              "set_size": scores.set_size, "quad_form": scores.quad_form},
    )


def filter_corrfilter_risk_tau(V, M, R, gold, alpha_floor: float) -> FilterResult:
    label = majority_consensus(V, M)
    scores = corrfilter_score(V, M, R, label)
    keep = np.nan_to_num(scores.score, nan=-np.inf) >= alpha_floor
    return FilterResult(
        "corrfilter_subset_risk_tau", keep, label, score=scores.score,
        meta={"alpha_floor": alpha_floor, "set_size": scores.set_size},
    )


def evaluate_filter(result: FilterResult, gold: np.ndarray) -> dict:
    """Retention / false-retention / precision / recall / F1 for a filter.

    The positive class is "item whose retained label equals gold". Precision is
    accuracy among kept items; recall is the fraction of all such items kept.
    """
    keep = result.keep.astype(bool)
    label = result.retained_label
    n = label.shape[0]
    labellable = label != ABSTAIN
    correct = labellable & (label == gold)            # items we *should* keep
    kept = keep & labellable

    n_kept = int(kept.sum())
    retention = n_kept / n if n else 0.0
    kept_correct = int((kept & correct).sum())
    precision = kept_correct / n_kept if n_kept else 0.0
    false_retention = 1.0 - precision if n_kept else 0.0
    total_correct = int(correct.sum())
    recall = kept_correct / total_correct if total_correct else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    out = {
        "method": result.method,
        "n_items": n,
        "n_kept": n_kept,
        "retention_rate": round(retention, 4),
        "false_retention_rate": round(false_retention, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    if result.score is not None:
        kept_scores = result.score[kept]
        kept_scores = kept_scores[~np.isnan(kept_scores)]
        out["avg_score_kept"] = round(float(kept_scores.mean()), 4) if kept_scores.size else float("nan")
    return out


__all__ = [
    "FilterResult",
    "inverse_error_weights",
    "filter_naive_majority",
    "filter_supermajority",
    "filter_independent_weighted",
    "filter_corrfilter_subset",
    "filter_corrfilter_risk_tau",
    "evaluate_filter",
]

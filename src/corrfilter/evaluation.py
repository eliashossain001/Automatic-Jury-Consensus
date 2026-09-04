"""Corrected retention-matched evaluation primitives (P0-0).

Single source of truth for turning (V, M, gold) into an evaluable pool, a correctness
mask, and a retention-matched keep set. Introduced by the P0-0 remediation to replace
three near-duplicate implementations that each carried the same two defects
(``scripts/analysis/bootstrap_ci.py``, ``scripts/analysis/run_filters.py``,
``scripts/29_grpo_corrfilter_eval.py``).

The two defects, documented in ``outputs/dpo_judges/D5_RESOLUTION.md``:

1. **Tie-break aligned with constant gold.** ``majority_consensus(..., tie_break=1)``
   scored against ``gold == 1`` everywhere makes split-vote items correct by
   construction, so a filter is rewarded for retaining the items on which the bank is
   maximally uncertain. :func:`evaluable_and_correct` defaults to ``tie_policy="abstain"``
   and raises if the caller asks for the unsafe combination.

2. **Retention denominator.** ``k = round(retention * n_items)`` counts items on which the
   bank never produced a verdict. With abstention rates differing by bank (2% vs 43% for
   the base and trained judge banks), a nominally matched retention is not matched.
   :func:`matched_k` defaults to the evaluable pool.

Both defaults are the corrected behaviour. The legacy behaviour is reachable only by
passing it explicitly, which exists so the archived numbers can be reproduced.
"""

from __future__ import annotations

import numpy as np

from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus

TIE_POLICIES = ("abstain", "fixed")
RETENTION_MODES = ("evaluable", "all_items")


def evaluable_and_correct(V, M, gold, *, tie_policy: str = "abstain", tie_break: int = 1):
    """Return ``(label, evaluable, correct)`` for a bank against ``gold``.

    ``evaluable`` is the set of items carrying a majority verdict. ``correct`` is the
    subset of those whose verdict matches gold. False-retention rates and precisions are
    always computed within ``evaluable``, never over all items.

    Raises if ``tie_policy="fixed"`` is combined with a constant ``gold`` vector, the
    exact circularity that inflated the base-bank CorrFilter gain by 2.7x.
    """
    if tie_policy not in TIE_POLICIES:
        raise ValueError(f"tie_policy must be one of {TIE_POLICIES}; got {tie_policy!r}")
    gold = np.asarray(gold)
    if tie_policy == "fixed" and np.unique(gold).size <= 1:
        raise ValueError(
            "tie_policy='fixed' with a constant gold vector makes every tied item correct "
            "by construction. Use tie_policy='abstain'. See outputs/dpo_judges/"
            "D5_RESOLUTION.md."
        )
    label = majority_consensus(V, M, tie_break=tie_break, tie_policy=tie_policy)
    evaluable = label != ABSTAIN
    correct = evaluable & (label == gold)
    return label, evaluable, correct


def matched_k(retention: float, evaluable, *, retention_mode: str = "evaluable") -> int:
    """Number of items to keep at ``retention``, denominated on the evaluable pool.

    ``retention_mode="all_items"`` reproduces the legacy denominator and exists only for
    archive reproduction.
    """
    if retention_mode not in RETENTION_MODES:
        raise ValueError(f"retention_mode must be one of {RETENTION_MODES}; got {retention_mode!r}")
    evaluable = np.asarray(evaluable, dtype=bool)
    base = int(evaluable.sum()) if retention_mode == "evaluable" else int(evaluable.size)
    return int(round(retention * base))


def top_k_keep(score, k: int, evaluable=None) -> np.ndarray:
    """Keep the ``k`` highest-scoring items, restricted to ``evaluable`` if given.

    Non-evaluable items are pushed below every evaluable one, so they can never consume a
    retention slot. Ties in ``score`` are broken by item order via a stable sort; callers
    comparing a coarse score (consensus level, which takes at most n+1 values) against a
    continuous one should be aware that the coarse score's within-block order is arbitrary.
    """
    score = np.asarray(score, dtype=float)
    if evaluable is not None:
        score = np.where(np.asarray(evaluable, dtype=bool), score, -np.inf)
    keep = np.zeros(score.shape[0], dtype=bool)
    if k > 0:
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
        keep[order[:min(k, score.shape[0])]] = True
    return keep


def frr(keep, correct, evaluable, idx=None) -> float:
    """False-retention rate among kept, evaluable items (optionally on bootstrap ``idx``)."""
    if idx is None:
        idx = np.arange(np.asarray(keep).shape[0])
    kept = np.asarray(keep)[idx] & np.asarray(evaluable)[idx]
    nk = int(kept.sum())
    return float("nan") if nk == 0 else 1.0 - float((kept & np.asarray(correct)[idx]).sum()) / nk


def consensus_score(V, M, evaluable) -> np.ndarray:
    """Naive-consensus ranking score: consensus level, minus-infinity off the pool."""
    return np.where(np.asarray(evaluable, dtype=bool), consensus_level(V, M), -np.inf)

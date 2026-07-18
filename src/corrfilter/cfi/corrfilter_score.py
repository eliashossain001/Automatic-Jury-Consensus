"""CorrFilter scoring: subset-aware decorrelated agreement.

For each item, given the per-item majority-retained label and the set ``S`` of
judges that voted with the retained label, the CorrFilter score is

    α_subset(v) = |S| / √(1ᵀ_S R_S 1_S)

where ``R_S`` is the submatrix of the error-correlation matrix ``R`` indexed
by ``S``. This is the proposal Eq. 5 operationalisation: high agreement among
nearly independent judges produces a high α, while high agreement among
strongly correlated judges produces a low α even with the same vote count.

Three modes for the correlation matrix used in scoring are supported, matching
the design discussion:

* ``h1_clean_R`` (default for headline results): the R measured on the clean
  H1 calibration set. This is the realistic deployment setting where R is
  estimated once during calibration and reused on production items.
* ``cfi_variant_R`` (oracle / ablation): re-estimate R from the CFI variant's
  own votes. Tells us the upper bound on CorrFilter's performance when it
  can adapt to the biased bank.
* ``identity_R`` (ablation): treats judges as independent. Strips the
  correlation correction, recovering ``α_subset = √|S|``. The contrast
  against ``h1_clean_R`` quantifies how much the H1-measured correlation
  structure contributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RMode(str, Enum):
    """How CorrFilter sources its correlation matrix for scoring."""

    H1_CLEAN = "h1_clean_R"
    CFI_VARIANT = "cfi_variant_R"
    IDENTITY = "identity_R"


@dataclass(frozen=True)
class CorrFilterScores:
    """Per-item CorrFilter scores plus the retained label they correspond to."""

    score: np.ndarray  # (N,) α_subset value, NaN where the item has no voters
    retained_label: np.ndarray  # (N,) majority label used to define S
    set_size: np.ndarray  # (N,) |S|
    quad_form: np.ndarray  # (N,) 1ᵀ R_S 1


def _subset_quad_form(R: np.ndarray, mask: np.ndarray) -> float:
    """Compute ``1ᵀ_S R_S 1_S`` for the row-vector indicator ``mask`` over judges."""
    if not mask.any():
        return 0.0
    sub = R[np.ix_(mask, mask)]
    return float(sub.sum())


def corrfilter_score(
    V: np.ndarray,
    M: np.ndarray,
    R: np.ndarray,
    retained_label: np.ndarray,
) -> CorrFilterScores:
    """Compute per-item ``α_subset`` against the supplied correlation matrix.

    Parameters
    ----------
    V, M
        ``(N × n)`` vote matrix and availability mask.
    R
        ``(n × n)`` correlation matrix aligned to the columns of ``V``.
    retained_label
        ``(N,)`` per-item retained label (e.g., the majority prediction). The
        agreeing set ``S`` for each item is ``{j : V[i, j] == retained_label[i]
        and M[i, j]}``.

    Returns
    -------
    CorrFilterScores
        Score plus bookkeeping. Items with empty ``S`` get ``score = NaN``.
    """
    if V.shape != M.shape:
        raise ValueError(f"V {V.shape} must match M {M.shape}")
    n_items, n_judges = V.shape
    if R.shape != (n_judges, n_judges):
        raise ValueError(f"R {R.shape} != ({n_judges}, {n_judges})")
    if retained_label.shape != (n_items,):
        raise ValueError(
            f"retained_label {retained_label.shape} != ({n_items},)"
        )

    M_bool = M.astype(bool)
    scores = np.full(n_items, np.nan, dtype=np.float64)
    set_sizes = np.zeros(n_items, dtype=np.int64)
    quad = np.zeros(n_items, dtype=np.float64)

    R_sym = (R + R.T) / 2.0

    for i in range(n_items):
        label = int(retained_label[i])
        agree = (V[i] == label) & M_bool[i]
        s = int(agree.sum())
        set_sizes[i] = s
        if s == 0:
            continue
        q = _subset_quad_form(R_sym, agree)
        quad[i] = q
        if q <= 0:
            scores[i] = float(s)
        else:
            scores[i] = float(s) / float(np.sqrt(q))

    return CorrFilterScores(
        score=scores,
        retained_label=retained_label.astype(np.int64),
        set_size=set_sizes,
        quad_form=quad,
    )


def resolve_R(
    mode: RMode | str,
    R_h1_clean: np.ndarray | None,
    R_cfi_variant: np.ndarray | None,
    n_judges: int,
) -> np.ndarray:
    """Return the correlation matrix to feed into ``corrfilter_score``."""
    if isinstance(mode, str):
        mode = RMode(mode)
    if mode is RMode.IDENTITY:
        return np.eye(n_judges, dtype=np.float64)
    if mode is RMode.H1_CLEAN:
        if R_h1_clean is None:
            raise ValueError(
                "h1_clean_R mode requires the H1 clean correlation matrix"
            )
        if R_h1_clean.shape != (n_judges, n_judges):
            raise ValueError(
                f"R_h1_clean {R_h1_clean.shape} != ({n_judges}, {n_judges})"
            )
        return R_h1_clean
    if mode is RMode.CFI_VARIANT:
        if R_cfi_variant is None:
            raise ValueError(
                "cfi_variant_R mode requires the per-variant correlation matrix"
            )
        if R_cfi_variant.shape != (n_judges, n_judges):
            raise ValueError(
                f"R_cfi_variant {R_cfi_variant.shape} != ({n_judges}, {n_judges})"
            )
        return R_cfi_variant
    raise ValueError(f"unknown R mode {mode!r}")


def corrfilter_filter(
    scores: np.ndarray, target_retention: float
) -> tuple[np.ndarray, float]:
    """Keep the top ``target_retention`` fraction of items by ``α_subset``.

    Returns ``(keep_mask, threshold)``. Items with ``NaN`` score are dropped
    (counted as discarded). The threshold is the minimum score among kept
    items; items whose score equals the threshold are all kept to keep the
    retention rate ≥ ``target_retention`` exactly when ties straddle the cut.
    """
    if not 0.0 <= target_retention <= 1.0:
        raise ValueError(f"target_retention must be in [0, 1]; got {target_retention}")
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool), float("nan")
    n_keep = int(round(target_retention * n))
    if n_keep == 0:
        return np.zeros(n, dtype=bool), float("inf")
    valid = ~np.isnan(scores)
    if not valid.any():
        return np.zeros(n, dtype=bool), float("nan")
    # Sort descending; NaN sorts to end naturally with this rule.
    safe_scores = np.where(valid, scores, -np.inf)
    order = np.argsort(-safe_scores, kind="stable")
    cutoff_idx = order[min(n_keep, n) - 1]
    threshold = float(safe_scores[cutoff_idx])
    keep = safe_scores >= threshold
    # Drop items whose score is NaN explicitly even if -inf would have masked them.
    keep = keep & valid
    return keep, threshold


def retention_match_threshold(
    scores: np.ndarray, n_target_keep: int
) -> tuple[np.ndarray, float]:
    """Pick a score threshold that retains exactly ``n_target_keep`` items.

    Use when comparing CorrFilter against an alternative method whose retention
    count is fixed (e.g., naive majority retains all non-abstaining items): we
    keep the top-``n_target_keep`` items by score so the two methods can be
    compared on false-retention rate at matched retention.
    """
    n = scores.shape[0]
    valid = ~np.isnan(scores)
    if n_target_keep <= 0:
        return np.zeros(n, dtype=bool), float("inf")
    if n_target_keep >= n:
        # Keep everything we *can* — NaN-scored items are never retainable.
        return valid, float("-inf")
    safe_scores = np.where(valid, scores, -np.inf)
    order = np.argsort(-safe_scores, kind="stable")
    cutoff_idx = order[n_target_keep - 1]
    threshold = float(safe_scores[cutoff_idx])
    keep = safe_scores >= threshold
    # Tie-breaking: if multiple items share the cutoff score, keep exactly
    # n_target_keep by sorting order priority.
    if int(keep.sum()) > n_target_keep:
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[order[:n_target_keep]] = True
        keep = keep_mask
    keep = keep & valid
    return keep, threshold


__all__ = [
    "RMode",
    "CorrFilterScores",
    "corrfilter_score",
    "resolve_R",
    "corrfilter_filter",
    "retention_match_threshold",
]

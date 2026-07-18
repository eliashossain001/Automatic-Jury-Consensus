"""Inter-rater-agreement statistics for the judge bank (proposal §3.4).

These statistics sit alongside the error-correlation matrix R as
supplementary tables in the H1 report:

* Cohen's κ on every judge pair, computed on the items where both judges
  voted (pairwise complete). κ corrects raw agreement for chance agreement
  given each judge's marginal vote distribution.
* Krippendorff's α over the full bank on a nominal scale, which is the
  standard generalisation of κ to >2 raters and tolerates missing entries
  by design (it weights observed pairs by the inverse of items × raters).

The agreement layer reads the same parsed-vote inputs as the correlation
layer (V, M) and is independent of gold labels: it measures inter-judge
consistency, which is conceptually distinct from the error-rate correlation
that drives H1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgreementResult:
    """Pairwise κ matrix + bank-wide Krippendorff α + bookkeeping."""

    kappa: np.ndarray              # (n × n) pairwise Cohen's κ
    n_pairs: np.ndarray            # (n × n) items used per pair
    krippendorff_alpha: float      # nominal-scale α over the full bank
    n_judges: int
    n_items: int


def cohen_kappa(votes_i: np.ndarray, votes_k: np.ndarray) -> float:
    """Cohen's κ on two binary vote vectors of equal length.

    Returns 0 when either rater is constant (no chance-corrected signal).
    """
    if votes_i.shape != votes_k.shape:
        raise ValueError(f"shape mismatch {votes_i.shape} vs {votes_k.shape}")
    n = votes_i.size
    if n == 0:
        return 0.0
    p_o = float((votes_i == votes_k).mean())
    p1_i = float(votes_i.mean())
    p1_k = float(votes_k.mean())
    p_e = p1_i * p1_k + (1.0 - p1_i) * (1.0 - p1_k)
    if p_e >= 1.0:
        return 0.0
    return float((p_o - p_e) / (1.0 - p_e))


def pairwise_cohen_kappa(V: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Cohen's κ on every (i, k) using items where both judges voted.

    Returns ``(K, n_pairs)`` where ``K[i, k]`` is Cohen's κ and
    ``n_pairs[i, k]`` is the number of items used.
    """
    if V.shape != M.shape:
        raise ValueError(f"V {V.shape} != M {M.shape}")
    n_items, n_judges = V.shape
    K = np.eye(n_judges, dtype=np.float64)
    n_pairs = np.zeros((n_judges, n_judges), dtype=np.int64)
    M_bool = M.astype(bool)
    for i in range(n_judges):
        n_pairs[i, i] = int(M_bool[:, i].sum())
        for k in range(i + 1, n_judges):
            both = M_bool[:, i] & M_bool[:, k]
            m = int(both.sum())
            n_pairs[i, k] = m
            n_pairs[k, i] = m
            if m == 0:
                continue
            kappa = cohen_kappa(V[both, i], V[both, k])
            K[i, k] = kappa
            K[k, i] = kappa
    return K, n_pairs


def krippendorff_alpha_nominal(V: np.ndarray, M: np.ndarray) -> float:
    """Krippendorff's α on a nominal scale, tolerating missing entries.

    Implementation follows the standard "coincidence matrix" formulation
    (Krippendorff 2004, ch. 11). For each item ``j``, every ordered pair of
    distinct raters that both voted contributes a coincidence count, weighted
    by ``1 / (n_voters_j - 1)``. The nominal disagreement metric is the
    indicator ``1[c != c']``.

    Returns α ∈ (-∞, 1]. α = 1 ⇒ perfect agreement; α = 0 ⇒ chance; α < 0 ⇒
    systematic disagreement.
    """
    if V.shape != M.shape:
        raise ValueError(f"V {V.shape} != M {M.shape}")
    n_items, n_judges = V.shape
    M_bool = M.astype(bool)

    # Coincidence matrix over the value space {0, 1}.
    o = np.zeros((2, 2), dtype=np.float64)
    for j in range(n_items):
        voters = np.where(M_bool[j])[0]
        m_j = voters.size
        if m_j < 2:
            continue
        denom = m_j - 1
        # For each ordered pair of distinct voters: contribute 1 / denom.
        # Vectorise via outer count of each value in this item's voters.
        vals = V[j, voters].astype(np.int64)
        n0 = int((vals == 0).sum())
        n1 = int((vals == 1).sum())
        # Same-value ordered pairs: n0*(n0-1) zeros-with-zeros etc.
        o[0, 0] += n0 * (n0 - 1) / denom
        o[1, 1] += n1 * (n1 - 1) / denom
        o[0, 1] += n0 * n1 / denom
        o[1, 0] += n1 * n0 / denom

    n_c = o.sum(axis=0)  # marginal counts per value
    n_total = n_c.sum()
    if n_total == 0:
        return 0.0
    # Observed disagreement: weighted sum of off-diagonal coincidences.
    D_o = (o[0, 1] + o[1, 0]) / n_total
    # Expected disagreement under independence of marginals.
    if n_total <= 1:
        return 0.0
    D_e = 2.0 * n_c[0] * n_c[1] / (n_total * (n_total - 1))
    if D_e <= 0:
        return 1.0 if D_o == 0 else 0.0
    return float(1.0 - D_o / D_e)


def compute_agreement(V: np.ndarray, M: np.ndarray) -> AgreementResult:
    """Compute Cohen's κ per judge pair + Krippendorff's α over the bank."""
    K, n_pairs = pairwise_cohen_kappa(V, M)
    alpha = krippendorff_alpha_nominal(V, M)
    return AgreementResult(
        kappa=K,
        n_pairs=n_pairs,
        krippendorff_alpha=alpha,
        n_judges=V.shape[1],
        n_items=V.shape[0],
    )

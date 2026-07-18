"""Adaptive R-estimation modes for CorrFilter (non-oracle correlation recovery).

The first CFI run showed CorrFilter with the *clean* H1 R ties naive
supermajority, while CorrFilter with an *oracle* R re-estimated from the biased
bank's gold-derived errors wins decisively. The clean-R shortfall is
**correlation drift**: the deployment bank's error-correlation differs from the
calibration bank's, and clean R cannot see it.

This module implements R estimators that try to recover the oracle gain
*without* full gold-derived biased errors:

* ``oracle_variant`` — error-correlation of the biased bank using gold on every
  item. Upper bound only; NOT deployable.
* ``disagreement`` — correlation of the raw deployment *votes*, no gold at all
  (pairwise-complete + shrinkage). Under this benchmark's fixed-direction gold
  (chosen ≻ rejected ⇒ gold ≡ 1), the per-item error is ``1 − vote``, so the
  vote-correlation equals the error-correlation exactly: disagreement R is the
  gold-free twin of oracle R, differing only by item-set / estimator choices.
* ``small_gold`` — error-correlation from a small labeled subset of CFI items.
  With fixed-direction gold this is a pure sample-size ablation (labels add only
  the preference direction, which is already known).
* ``hybrid`` — ``λ·R_clean + (1−λ)·R_adaptive``, renormalised to unit diagonal.

Plus R-comparison diagnostics: Frobenius distance and top-k eigenvector subspace
overlap against the oracle R.
"""

from __future__ import annotations

import numpy as np

from corrfilter.correlation import (
    compute_error_matrix,
    correlation_shrunk,
    correlation_shrunk_pairwise,
)


def _renormalize_diag(R: np.ndarray) -> np.ndarray:
    """Force unit diagonal so the result is a valid correlation matrix."""
    d = np.sqrt(np.clip(np.diag(R), a_min=1e-12, a_max=None))
    R = R / (d[:, None] * d[None, :])
    np.fill_diagonal(R, 1.0)
    return R


def oracle_variant_R(V: np.ndarray, M: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Gold-derived biased-bank error-correlation (listwise + shrinkage). UPPER BOUND."""
    E, _ = compute_error_matrix(V, M, gold)
    if E.shape[0] < 2:
        return np.eye(V.shape[1])
    R, _ = correlation_shrunk(E)
    return R


def disagreement_R(V: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Gold-free correlation of deployment votes (pairwise-complete + shrinkage).

    Uses the raw 0/1 votes as the signal and the availability mask to restrict
    each pairwise correlation to items both judges voted on. No gold labels.
    """
    M_bool = M.astype(bool)
    R, _, _ = correlation_shrunk_pairwise(V.astype(np.float64), M_bool)
    return _renormalize_diag(R)


def small_gold_R(
    V: np.ndarray, M: np.ndarray, gold: np.ndarray, subset_idx: np.ndarray
) -> np.ndarray:
    """Error-correlation estimated from a small labeled subset of items."""
    Vs, Ms, gs = V[subset_idx], M[subset_idx], gold[subset_idx]
    E, _ = compute_error_matrix(Vs, Ms, gs)
    if E.shape[0] < 2:
        return np.eye(V.shape[1])
    R, _ = correlation_shrunk(E)
    return R


def hybrid_R(R_clean: np.ndarray, R_adaptive: np.ndarray, lam: float) -> np.ndarray:
    """``λ·R_clean + (1−λ)·R_adaptive`` renormalised to unit diagonal."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [0, 1]; got {lam}")
    R = lam * R_clean + (1.0 - lam) * R_adaptive
    return _renormalize_diag(0.5 * (R + R.T))


def frobenius_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Frobenius norm of the difference of two correlation matrices."""
    return float(np.linalg.norm(R1 - R2))


def eigenvector_overlap(R1: np.ndarray, R2: np.ndarray, k: int = 3) -> float:
    """Subspace overlap of the top-``k`` eigenvectors, in [0, 1] (1 = identical).

    Equals ``(1/k)·‖U1ᵀ U2‖_F²`` = the mean squared cosine of the principal
    angles between the two leading-``k`` eigenspaces.
    """
    def _topk(R):
        w, v = np.linalg.eigh((R + R.T) / 2.0)
        order = np.argsort(w)[::-1][:k]
        return v[:, order]

    U1, U2 = _topk(R1), _topk(R2)
    return float(np.linalg.norm(U1.T @ U2) ** 2 / k)


__all__ = [
    "oracle_variant_R",
    "disagreement_R",
    "small_gold_R",
    "hybrid_R",
    "frobenius_distance",
    "eigenvector_overlap",
]

"""Effective ensemble size diagnostics for a judge bank.

Implements the two summaries the proposal commits to (§3.2):

* ``effective_size``: the random-effects ``n / (1 + (n-1) ρ̄)`` formula
  using the mean off-diagonal of R as ρ̄. Equivalent to Lemma 1 /
  Proposition 1 of Appendix A.

* ``effective_eig_rank``: the participation-ratio-style
  ``(Σ λ)² / Σ λ²`` of the eigenspectrum, which captures structured
  correlation (a tight cluster inside an otherwise diverse bank).
"""

from __future__ import annotations

import numpy as np


def mean_off_diagonal(R: np.ndarray) -> float:
    """Mean of off-diagonal entries of a square matrix."""
    n = R.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(R[mask].mean())


def effective_size(R: np.ndarray) -> float:
    """``n_eff = n / (1 + (n-1) ρ̄)`` (proposal Eq. 2)."""
    n = R.shape[0]
    if n <= 1:
        return float(n)
    rho_bar = mean_off_diagonal(R)
    denom = 1.0 + (n - 1) * rho_bar
    if denom <= 0:
        return float(n)
    return float(n / denom)


def effective_eig_rank(R: np.ndarray) -> float:
    """``n_eff^eig = (Σ λ)² / Σ λ²`` (proposal Eq. 3)."""
    eigvals = np.linalg.eigvalsh((R + R.T) / 2.0)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 <= 0:
        return float(R.shape[0])
    return float(s1 * s1 / s2)


def neff_curve(rho_bar: float, max_n: int = 32) -> np.ndarray:
    """``n_eff`` as a function of bank size, holding ρ̄ fixed.

    Returns an array of length ``max_n`` with element ``k`` equal to
    ``(k+1) / (1 + k * ρ̄)`` so that index 0 is a one-judge bank.
    """
    if rho_bar <= 0:
        return np.arange(1, max_n + 1, dtype=np.float64)
    ks = np.arange(1, max_n + 1)
    return ks / (1.0 + (ks - 1) * rho_bar)

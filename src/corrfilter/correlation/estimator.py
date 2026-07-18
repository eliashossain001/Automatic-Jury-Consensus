"""Error-correlation matrix R: construction, shrinkage, bootstrap.

Given the vote matrix V (N × n) and gold labels y* (N,), the per-item
error indicators are E[j, i] = 1[V[j, i] != y*[j]]. R is the inter-judge
Pearson correlation of E across calibration items (proposal Eq. 1).

Four estimators are provided:

* ``correlation_pearson`` — plain Pearson on complete data, sanity check.
* ``correlation_shrunk`` — Ledoit-Wolf shrinkage toward the identity, the
  proposal's default for small calibration sets (§3.1).
* ``correlation_pairwise_complete`` — per-pair Pearson computed on items
  where both judges voted, then projected to the nearest PSD matrix. Used
  when listwise deletion would discard too many calibration items (e.g., a
  high-abstention judge in the bank).
* ``correlation_shrunk_pairwise`` — pairwise-complete R with Ledoit-Wolf-style
  shrinkage applied analytically on the eigenspectrum.

Bootstrap CIs resample calibration items with replacement, recomputing R
each draw under the chosen estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


def compute_error_matrix(
    V: np.ndarray, M: np.ndarray, gold: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (N × n) error matrix E and item-keep mask (listwise complete).

    An item is kept only if every judge produced a parseable vote (listwise
    deletion). This matches the proposal's exchangeable-judge formalism and is
    the cleanest input to Ledoit-Wolf, which requires complete data.
    """
    if V.shape != M.shape:
        raise ValueError(f"V {V.shape} and M {M.shape} must match")
    if gold.shape[0] != V.shape[0]:
        raise ValueError(f"gold {gold.shape} incompatible with V {V.shape}")

    complete = M.all(axis=1)  # (N,) True where every judge voted
    V_kept = V[complete]
    gold_kept = gold[complete]
    E = (V_kept != gold_kept[:, None]).astype(np.float64)
    return E, complete


def compute_error_matrix_pairwise(
    V: np.ndarray, M: np.ndarray, gold: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N × n) error matrix E and availability mask without dropping items.

    Returns ``(E, M_bool)`` where ``E[j, i] = 1[V[j, i] != gold[j]]`` for items
    where judge i voted, and 0 otherwise. ``M_bool`` is the availability mask
    so per-pair correlations can be computed on the items where both judges
    are available.
    """
    if V.shape != M.shape:
        raise ValueError(f"V {V.shape} and M {M.shape} must match")
    if gold.shape[0] != V.shape[0]:
        raise ValueError(f"gold {gold.shape} incompatible with V {V.shape}")

    E = (V != gold[:, None]).astype(np.float64)
    M_bool = M.astype(bool)
    # Zero out positions where the judge did not vote so they don't contaminate
    # column means; the availability mask is the source of truth for which
    # entries are real.
    E = E * M_bool
    return E, M_bool


def nearest_psd(M: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the nearest PSD matrix in Frobenius norm.

    Used after pairwise-complete correlation estimation, which produces a
    valid symmetric matrix but not always a PSD one. Clipping the negative
    eigenvalues to zero is the Higham (1988) projection onto the PSD cone.
    """
    sym = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    psd = (eigvecs * eigvals_clipped) @ eigvecs.T
    # Re-symmetrize to clean up float noise from the reconstruction.
    return (psd + psd.T) / 2.0


def _renormalize_diag(R: np.ndarray) -> np.ndarray:
    """Force unit diagonal so the result is a correlation matrix."""
    d = np.sqrt(np.clip(np.diag(R), a_min=1e-12, a_max=None))
    R = R / (d[:, None] * d[None, :])
    np.fill_diagonal(R, 1.0)
    return R


def _normalise_covariance(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix robustly."""
    d = np.sqrt(np.clip(np.diag(cov), a_min=1e-12, a_max=None))
    R = cov / (d[:, None] * d[None, :])
    np.fill_diagonal(R, 1.0)
    return R


def correlation_pearson(E: np.ndarray) -> np.ndarray:
    """Plain Pearson correlation of columns of E."""
    n = E.shape[1]
    if E.shape[0] < 2:
        return np.eye(n)
    R = np.corrcoef(E, rowvar=False)
    np.fill_diagonal(R, 1.0)
    return np.nan_to_num(R, nan=0.0)


def correlation_shrunk(E: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk correlation; returns (R, shrinkage)."""
    if E.shape[0] < 2:
        return np.eye(E.shape[1]), 1.0
    lw = LedoitWolf().fit(E)
    R = _normalise_covariance(lw.covariance_)
    return R, float(lw.shrinkage_)


def correlation_pairwise_complete(
    E: np.ndarray, available: np.ndarray, min_pairs: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson correlation, projected to nearest PSD.

    Each entry R[i, k] uses only items where both judges i and k voted. Pairs
    with fewer than ``min_pairs`` joint observations are treated as having
    zero correlation (degraded to independence) rather than fabricating a
    high-variance estimate.

    Returns ``(R_psd, n_pairs)`` where ``n_pairs[i, k]`` is the number of items
    used for that entry. The diagonal of R is 1 by construction; off-diagonals
    pass through the nearest-PSD projection so the result is a valid
    correlation matrix for downstream eigen analysis.
    """
    n_items, n_judges = E.shape
    if available.shape != E.shape:
        raise ValueError(f"availability mask {available.shape} != E {E.shape}")

    R = np.eye(n_judges, dtype=np.float64)
    n_pairs = np.zeros((n_judges, n_judges), dtype=np.int64)

    # Per-judge mean error restricted to its available items, for centering.
    sums = (E * available).sum(axis=0)
    counts = available.sum(axis=0)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    for i in range(n_judges):
        n_pairs[i, i] = int(counts[i])
        for k in range(i + 1, n_judges):
            both = available[:, i] & available[:, k]
            m = int(both.sum())
            n_pairs[i, k] = m
            n_pairs[k, i] = m
            if m < min_pairs:
                # Not enough joint obs; default to zero (independence).
                continue
            ei = E[both, i] - means[i]
            ek = E[both, k] - means[k]
            num = float((ei * ek).sum())
            den = float(np.sqrt((ei * ei).sum() * (ek * ek).sum()))
            r_ik = num / den if den > 1e-12 else 0.0
            r_ik = max(-1.0, min(1.0, r_ik))
            R[i, k] = r_ik
            R[k, i] = r_ik

    R_psd = nearest_psd(R)
    R_psd = _renormalize_diag(R_psd)
    return R_psd, n_pairs


def correlation_shrunk_pairwise(
    E: np.ndarray, available: np.ndarray, min_pairs: int = 30
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pairwise-complete R with eigenvalue shrinkage toward the identity.

    A pragmatic replacement for Ledoit-Wolf when listwise deletion is
    infeasible: estimate R via pairwise-complete Pearson, then shrink the
    eigenspectrum toward 1 with weight
    ``shrinkage = n_judges / (n_judges + harmonic_mean(n_pairs_off_diag))``.
    The shrinkage weight grows when pairwise observations are scarce, which
    matches the Ledoit-Wolf intuition without requiring complete data.

    Returns ``(R_shrunk, n_pairs, shrinkage)``.
    """
    R_raw, n_pairs = correlation_pairwise_complete(E, available, min_pairs=min_pairs)
    n = R_raw.shape[0]
    off_pairs = n_pairs[~np.eye(n, dtype=bool)]
    if off_pairs.size == 0 or off_pairs.max() == 0:
        return R_raw, n_pairs, 1.0
    valid = off_pairs[off_pairs > 0]
    if valid.size == 0:
        return R_raw, n_pairs, 1.0
    harmonic = float(valid.size / (1.0 / valid).sum())
    shrinkage = float(n / (n + harmonic))
    R_shrunk = (1.0 - shrinkage) * R_raw + shrinkage * np.eye(n)
    return R_shrunk, n_pairs, shrinkage


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution over the error-correlation matrix."""

    point: np.ndarray             # (n × n) point estimate on the full sample
    samples: np.ndarray           # (B × n × n) bootstrap draws
    lower: np.ndarray             # (n × n) lower bootstrap quantile
    upper: np.ndarray             # (n × n) upper bootstrap quantile

    @property
    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        return self.samples.std(axis=0)


def bootstrap_correlation(
    E: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20260601,
    ci: float = 0.95,
    shrunk: bool = True,
) -> BootstrapResult:
    """Bootstrap resampling of calibration items to estimate CI on R.

    With ``shrunk=True`` (default) each draw uses Ledoit-Wolf shrinkage,
    matching the proposal's headline R estimator.
    """
    n_items, n_judges = E.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, n_judges, n_judges), dtype=np.float64)
    point = correlation_shrunk(E)[0] if shrunk else correlation_pearson(E)

    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        E_b = E[idx]
        R_b = correlation_shrunk(E_b)[0] if shrunk else correlation_pearson(E_b)
        draws[b] = R_b

    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(draws, alpha, axis=0)
    upper = np.quantile(draws, 1.0 - alpha, axis=0)
    return BootstrapResult(point=point, samples=draws, lower=lower, upper=upper)

"""H1 hypothesis test: intra-group vs cross-group judge error correlation.

The proposal commits to (§3.7):

    H1 (operationalises C1). Intra-family or intra-prompt-style judge
    correlation exceeds inter-family correlation by at least 0.10 in Pearson
    terms on the error indicators e_{ij}, with bootstrap p < 0.01.

Implementation:

1. Partition off-diagonal entries of R into "intra-group" and "cross-group"
   sets according to the grouping variable (family or prompt_style).
2. Compute the contrast ``Δ = ρ̄_intra − ρ̄_cross`` on the point estimate.
3. Bootstrap calibration items B times, recomputing Δ each draw.
4. Report observed Δ, bootstrap CI on Δ, bootstrap p-value
   ``Pr[Δ_b ≤ 0]`` as a one-sided test against the H1 direction.

The function returns both group-specific tests; H1 is accepted if either
clears the (≥ 0.10, p < 0.01) bar (proposal language: "intra-family or
intra-prompt-style").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corrfilter.correlation.estimator import correlation_shrunk, correlation_shrunk_pairwise
from corrfilter.judges.base import JudgeSpec


@dataclass(frozen=True)
class H1Result:
    """Single intra-vs-cross contrast result."""

    grouping: str               # "family" or "prompt_style"
    rho_intra: float            # mean off-diagonal correlation within groups
    rho_cross: float            # mean off-diagonal correlation across groups
    delta: float                # rho_intra - rho_cross
    delta_lower: float
    delta_upper: float
    p_value: float              # bootstrap one-sided: Pr[Δ_b ≤ 0]
    n_pairs_intra: int
    n_pairs_cross: int
    threshold: float            # H1 acceptance threshold (default 0.10)
    accepts_h1: bool


def _group_masks(specs: list[JudgeSpec], grouping: str) -> tuple[np.ndarray, np.ndarray]:
    """Return intra-group and cross-group boolean masks over off-diagonal pairs."""
    n = len(specs)
    if grouping == "family":
        labels = np.array([s.family for s in specs])
    elif grouping == "prompt_style":
        labels = np.array([s.prompt_style for s in specs])
    else:
        raise ValueError(f"unknown grouping {grouping!r}")
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(n, dtype=bool)
    intra = same & off_diag
    cross = (~same) & off_diag
    return intra, cross


def _contrast(R: np.ndarray, intra: np.ndarray, cross: np.ndarray) -> tuple[float, float, float]:
    rho_intra = float(R[intra].mean()) if intra.any() else 0.0
    rho_cross = float(R[cross].mean()) if cross.any() else 0.0
    return rho_intra, rho_cross, rho_intra - rho_cross


def pairwise_contrast_bootstrap(
    E: np.ndarray,
    specs: list[JudgeSpec],
    grouping: str,
    availability: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 20260601,
    ci: float = 0.95,
    threshold: float = 0.10,
) -> H1Result:
    """Bootstrap the intra-vs-cross-group contrast on R.

    Each bootstrap draw resamples calibration items with replacement and
    recomputes shrunk R, then forms the partition-specific mean off-diagonals.

    If ``availability`` is None, ``E`` is treated as listwise-complete and
    Ledoit-Wolf is used (proposal §3.1 default). Otherwise ``E`` is treated as
    a full vote-aligned error matrix (zeros wherever the judge did not vote)
    and the pairwise-complete + shrinkage estimator is used (safety net for
    high-abstention banks).
    """
    if E.shape[1] != len(specs):
        raise ValueError(f"E has {E.shape[1]} judges; specs has {len(specs)}")
    if availability is not None and availability.shape != E.shape:
        raise ValueError(
            f"availability {availability.shape} must match E {E.shape}"
        )

    intra, cross = _group_masks(specs, grouping)

    def estimate_R(E_sample: np.ndarray, avail_sample: np.ndarray | None) -> np.ndarray:
        if avail_sample is None:
            R_est, _ = correlation_shrunk(E_sample)
        else:
            R_est, _, _ = correlation_shrunk_pairwise(E_sample, avail_sample)
        return R_est

    R_point = estimate_R(E, availability)
    rho_intra, rho_cross, delta_point = _contrast(R_point, intra, cross)

    rng = np.random.default_rng(seed)
    n_items = E.shape[0]
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        E_b = E[idx]
        avail_b = availability[idx] if availability is not None else None
        R_b = estimate_R(E_b, avail_b)
        _, _, delta_b = _contrast(R_b, intra, cross)
        deltas[b] = delta_b

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(deltas, alpha))
    upper = float(np.quantile(deltas, 1.0 - alpha))
    # One-sided bootstrap p-value: Pr[Δ_b ≤ 0] under the resampled distribution.
    p_value = float((deltas <= 0).mean())

    accepts = (delta_point >= threshold) and (p_value < 0.01)

    return H1Result(
        grouping=grouping,
        rho_intra=rho_intra,
        rho_cross=rho_cross,
        delta=delta_point,
        delta_lower=lower,
        delta_upper=upper,
        p_value=p_value,
        n_pairs_intra=int(intra.sum()),
        n_pairs_cross=int(cross.sum()),
        threshold=threshold,
        accepts_h1=accepts,
    )


def family_contrast(E, specs, **kwargs):
    return pairwise_contrast_bootstrap(E, specs, grouping="family", **kwargs)


def prompt_contrast(E, specs, **kwargs):
    return pairwise_contrast_bootstrap(E, specs, grouping="prompt_style", **kwargs)

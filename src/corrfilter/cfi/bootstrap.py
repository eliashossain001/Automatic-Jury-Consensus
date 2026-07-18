"""Item-level bootstrap confidence intervals for CFI metrics.

Each metric is recomputed on ``n_boot`` resamples of the calibration items
with replacement; the central CI is read off the bootstrap distribution
quantiles. Defaults to 1000 resamples as specified in the experiment plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    """Bootstrap CI summary for a scalar statistic."""

    point: float
    lower: float
    upper: float
    mean: float
    std: float
    n_boot: int


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n_items: int,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 20260601,
) -> BootstrapInterval:
    """Resample item indices with replacement, recompute statistic each draw.

    ``statistic`` is called with a 1D ``ndarray`` of resampled item indices and
    must return a float.
    """
    if n_items <= 0:
        return BootstrapInterval(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_items)
    point = float(statistic(all_indices))
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        draws[b] = float(statistic(idx))
    alpha = (1.0 - ci) / 2.0
    return BootstrapInterval(
        point=point,
        lower=float(np.quantile(draws, alpha)),
        upper=float(np.quantile(draws, 1.0 - alpha)),
        mean=float(draws.mean()),
        std=float(draws.std()),
        n_boot=n_boot,
    )


__all__ = ["BootstrapInterval", "bootstrap_ci"]

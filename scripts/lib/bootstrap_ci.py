"""Paired / marginal bootstrap confidence intervals — shared helper (Bucket 1).

The paired FRR-gain routine matches the methodology used for the GRPO results
(scripts/robustness/analyze_abstention_robustness.py): rank items by consensus level vs by the decorrelated alpha_subset
score, keep the top-`retention` fraction under each, and bootstrap the per-item
false-retention DIFFERENCE (consensus - corrfilter). A positive gain with a CI
excluding zero means CorrFilter reduces false retention at matched retention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402
from corrfilter.evaluation import (consensus_score, evaluable_and_correct,  # noqa: E402
                                   frr, matched_k, top_k_keep)

DEFAULT_SEED = 20260706


def paired_frr_gain(V, M, R, gold, retention=0.60, B=2000, seed=DEFAULT_SEED,
                    tie_policy="abstain", retention_mode="evaluable"):
    """Return dict: consensus_FRR - corrfilter_FRR at matched retention, with 95% CI.

    Corrected by P0-0. Ties abstain rather than resolving to a fixed value, and retention
    is denominated on the evaluable pool. Pass ``tie_policy="fixed"`` and
    ``retention_mode="all_items"`` only to reproduce the archived pre-P0-0 numbers
    (``outputs/_archive/pre_P0-0_2026-08-11/``); with a constant gold vector the former
    now raises. See ``outputs/dpo_judges/D5_RESOLUTION.md``.
    """
    n = V.shape[0]
    label, evaluable, correct = evaluable_and_correct(V, M, gold, tie_policy=tie_policy)
    sc_cons = consensus_score(V, M, evaluable)
    sc_cf = np.where(evaluable, corrfilter_score(V, M, R, label).score, -np.inf)
    k = matched_k(retention, evaluable, retention_mode=retention_mode)
    kc = top_k_keep(sc_cons, k, evaluable)
    kf = top_k_keep(sc_cf, k, evaluable)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = frr(kc, correct, evaluable, idx) - frr(kf, correct, evaluable, idx)
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "retention": retention,
        "n_evaluable": int(evaluable.sum()),
        "n_keep": int(k),
        "consensus_frr": round(float(frr(kc, correct, evaluable)), 4),
        "corrfilter_frr": round(float(frr(kf, correct, evaluable)), 4),
        "gain_pts": round(float(diffs.mean()) * 100, 2),
        "ci_low_pts": round(float(lo) * 100, 2),
        "ci_high_pts": round(float(hi) * 100, 2),
        "p_corrfilter_better": round(float((diffs > 0).mean()), 3),
        "tie_policy": tie_policy,
        "retention_mode": retention_mode,
        "seed": seed,
    }


def bootstrap_metric(values, B=2000, seed=DEFAULT_SEED):
    """Mean + 95% CI of a 1-D array via item bootstrap (for accuracy/win-rate etc.)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, values.size, values.size)].mean() for _ in range(B)])
    return {
        "mean": round(float(values.mean()), 4),
        "ci_low": round(float(np.quantile(means, 0.025)), 4),
        "ci_high": round(float(np.quantile(means, 0.975)), 4),
    }


def paired_diff_ci(a, b, B=2000, seed=DEFAULT_SEED):
    """Paired bootstrap of mean(a-b) over aligned per-item arrays (a=method1, b=method2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    rng = np.random.default_rng(seed)
    d = a - b
    if d.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_pos": np.nan}
    boots = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(B)])
    return {
        "mean": round(float(d.mean()), 4),
        "ci_low": round(float(np.quantile(boots, 0.025)), 4),
        "ci_high": round(float(np.quantile(boots, 0.975)), 4),
        "p_pos": round(float((boots > 0).mean()), 3),
    }

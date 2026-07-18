"""Paired / marginal bootstrap confidence intervals — shared helper (Bucket 1).

The paired FRR-gain routine matches the methodology used for the GRPO results
(scripts/31): rank items by consensus level vs by the decorrelated alpha_subset
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

from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402

DEFAULT_SEED = 20260706


def _topk(score, k):
    n = score.shape[0]
    keep = np.zeros(n, dtype=bool)
    if k > 0:
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
        keep[order[:min(k, n)]] = True
    return keep


def _frr(keep, correct, labellable, idx):
    kk = keep[idx] & labellable[idx]
    nk = int(kk.sum())
    return np.nan if nk == 0 else 1.0 - float((kk & correct[idx]).sum()) / nk


def paired_frr_gain(V, M, R, gold, retention=0.60, B=2000, seed=DEFAULT_SEED):
    """Return dict: consensus_FRR - corrfilter_FRR at matched retention, with 95% CI."""
    n = V.shape[0]
    maj = majority_consensus(V, M)
    level = consensus_level(V, M)
    labellable = maj != ABSTAIN
    correct = labellable & (maj == gold)
    sc_cons = np.where(labellable, level, -np.inf)
    sc_cf = corrfilter_score(V, M, R, maj).score
    k = int(round(retention * n))
    kc, kf = _topk(sc_cons, k), _topk(sc_cf, k)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = _frr(kc, correct, labellable, idx) - _frr(kf, correct, labellable, idx)
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    point_cons = _frr(kc, correct, labellable, np.arange(n))
    point_cf = _frr(kf, correct, labellable, np.arange(n))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "retention": retention,
        "consensus_frr": round(float(point_cons), 4),
        "corrfilter_frr": round(float(point_cf), 4),
        "gain_pts": round(float(diffs.mean()) * 100, 2),
        "ci_low_pts": round(float(lo) * 100, 2),
        "ci_high_pts": round(float(hi) * 100, 2),
        "p_corrfilter_better": round(float((diffs > 0).mean()), 3),
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

"""Tests for Cohen's κ and Krippendorff's α (proposal §3.4 supplementary stats)."""

from __future__ import annotations

import numpy as np

from corrfilter.analysis import (
    cohen_kappa,
    compute_agreement,
    krippendorff_alpha_nominal,
    pairwise_cohen_kappa,
)


def test_kappa_perfect_agreement_is_one():
    v = np.array([1, 0, 1, 0, 1, 1, 0])
    assert cohen_kappa(v, v.copy()) == 1.0


def test_kappa_random_agreement_near_zero():
    rng = np.random.default_rng(0)
    v1 = rng.integers(0, 2, size=2000)
    v2 = rng.integers(0, 2, size=2000)
    assert abs(cohen_kappa(v1, v2)) < 0.05


def test_kappa_perfect_disagreement_is_minus_one():
    v = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    assert cohen_kappa(v, 1 - v) == -1.0


def test_kappa_handles_constant_rater():
    """If one rater is constant, κ is undefined; we return 0 rather than NaN."""
    v_const = np.zeros(10, dtype=int)
    v_other = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    assert cohen_kappa(v_const, v_other) == 0.0


def test_pairwise_kappa_diagonal_is_one_and_symmetric():
    rng = np.random.default_rng(0)
    V = rng.integers(0, 2, size=(100, 4))
    M = np.ones_like(V)
    K, n_pairs = pairwise_cohen_kappa(V, M)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert (n_pairs[~np.eye(4, dtype=bool)] == 100).all()


def test_pairwise_kappa_respects_availability_mask():
    """If judge 1 abstains on half the items, n_pairs for (0,1) should be ~half."""
    V = np.zeros((100, 2), dtype=int)
    M = np.ones((100, 2), dtype=int)
    M[:50, 1] = 0  # judge 1 abstains on first 50 items
    K, n_pairs = pairwise_cohen_kappa(V, M)
    assert n_pairs[0, 1] == 50


def test_krippendorff_alpha_perfect_agreement_is_one():
    V = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
    M = np.ones_like(V)
    alpha = krippendorff_alpha_nominal(V, M)
    assert alpha == 1.0


def test_krippendorff_alpha_random_near_zero():
    rng = np.random.default_rng(0)
    V = rng.integers(0, 2, size=(2000, 5))
    M = np.ones_like(V)
    alpha = krippendorff_alpha_nominal(V, M)
    assert abs(alpha) < 0.05


def test_krippendorff_alpha_handles_missing_entries():
    """α should still produce a finite value when judges abstain on subsets."""
    rng = np.random.default_rng(0)
    V = rng.integers(0, 2, size=(200, 4))
    M = (rng.uniform(size=V.shape) > 0.3).astype(int)
    alpha = krippendorff_alpha_nominal(V, M)
    assert np.isfinite(alpha)


def test_compute_agreement_returns_consistent_object():
    rng = np.random.default_rng(0)
    V = rng.integers(0, 2, size=(100, 3))
    M = np.ones_like(V)
    res = compute_agreement(V, M)
    assert res.kappa.shape == (3, 3)
    assert res.n_pairs.shape == (3, 3)
    assert res.n_judges == 3
    assert res.n_items == 100
    assert np.isfinite(res.krippendorff_alpha)

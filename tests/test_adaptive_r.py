"""Tests for adaptive R-estimation modes.

Covers the structural identity that drives the experiment (under fixed-direction
gold, gold-free vote correlation == gold-derived error correlation), the hybrid
interpolation endpoints, and the R-comparison diagnostics.
"""

from __future__ import annotations

import numpy as np

from corrfilter.cfi.adaptive_r import (
    disagreement_R,
    eigenvector_overlap,
    frobenius_distance,
    hybrid_R,
    oracle_variant_R,
    small_gold_R,
)


def _random_votes(n_items=300, n_judges=6, seed=0):
    rng = np.random.default_rng(seed)
    # Correlated latent difficulty so the matrix is non-trivial.
    latent = rng.normal(size=(n_items, 1))
    noise = rng.normal(size=(n_items, n_judges))
    score = 0.7 * latent + 0.3 * noise
    V = (score < 0).astype(np.int8)  # 1 = "agree with chosen" on ~half
    M = np.ones((n_items, n_judges), dtype=np.int8)
    return V, M


def test_disagreement_equals_oracle_under_constant_gold():
    # Fixed-direction gold ≡ 1 ⇒ error = 1 − vote ⇒ vote-corr == error-corr.
    V, M = _random_votes()
    gold = np.ones(V.shape[0], dtype=np.int8)
    R_oracle = oracle_variant_R(V, M, gold)
    R_dis = disagreement_R(V, M)
    # Different shrinkage estimators, but the same underlying correlation:
    # top-eigenvector subspaces should be near-identical and Frobenius small.
    assert eigenvector_overlap(R_oracle, R_dis, k=3) > 0.95
    assert frobenius_distance(R_oracle, R_dis) < 0.6


def test_valid_correlation_matrices():
    V, M = _random_votes()
    gold = np.ones(V.shape[0], dtype=np.int8)
    for R in (oracle_variant_R(V, M, gold), disagreement_R(V, M),
              small_gold_R(V, M, gold, np.arange(50))):
        assert R.shape == (V.shape[1], V.shape[1])
        assert np.allclose(np.diag(R), 1.0)
        assert np.allclose(R, R.T, atol=1e-8)
        assert np.all(np.abs(R) <= 1.0 + 1e-8)


def test_hybrid_endpoints():
    rng = np.random.default_rng(1)
    A = np.eye(5)
    B = 0.3 * np.eye(5) + 0.7 * np.ones((5, 5))
    h1 = hybrid_R(A, B, lam=1.0)
    h0 = hybrid_R(A, B, lam=0.0)
    assert np.allclose(h1, A, atol=1e-8)       # all clean
    assert np.allclose(h0, B, atol=1e-8)       # all adaptive
    hmid = hybrid_R(A, B, lam=0.5)
    assert np.allclose(np.diag(hmid), 1.0)     # still a correlation matrix
    _ = rng


def test_frobenius_and_overlap_self():
    R = 0.5 * np.eye(4) + 0.5 * np.ones((4, 4))
    assert frobenius_distance(R, R) == 0.0
    assert abs(eigenvector_overlap(R, R, k=2) - 1.0) < 1e-9


def test_eigenvector_overlap_bounds():
    rng = np.random.default_rng(2)
    A = np.corrcoef(rng.normal(size=(200, 5)), rowvar=False)
    B = np.corrcoef(rng.normal(size=(200, 5)), rowvar=False)
    ov = eigenvector_overlap(A, B, k=2)
    assert 0.0 <= ov <= 1.0


def test_frame_flip_preserves_error_and_oracle_R():
    # The direction-randomized construct (script 11) flips vote AND gold on ~50%
    # of items. The per-item error 1[O != gold] must stay = 1 - V (invariant),
    # so the oracle error-correlation is unchanged by the flip.
    V, M = _random_votes(n_items=400, n_judges=6, seed=11)
    rng = np.random.default_rng(4)
    flip = rng.random(V.shape[0]) >= 0.5
    O = V.copy()
    O[flip] = 1 - V[flip]
    gold = np.where(flip, 0, 1).astype(np.int8)
    err_flipped = (O != gold[:, None]).astype(np.int8)
    err_fixed = (1 - V).astype(np.int8)
    assert np.array_equal(err_flipped, err_fixed)             # error invariant
    R_fixed = oracle_variant_R(V, M, np.ones(V.shape[0], dtype=np.int8))
    R_flip = oracle_variant_R(O, M, gold)
    assert frobenius_distance(R_fixed, R_flip) < 1e-9          # oracle R invariant


def test_small_gold_more_items_closer_to_oracle():
    # Larger labeled subsets give R closer to the full oracle R (less noise).
    V, M = _random_votes(n_items=400, n_judges=6, seed=3)
    gold = np.ones(V.shape[0], dtype=np.int8)
    R_oracle = oracle_variant_R(V, M, gold)
    rng = np.random.default_rng(7)
    d_small = np.mean([frobenius_distance(small_gold_R(V, M, gold, rng.choice(400, 25, replace=False)), R_oracle)
                       for _ in range(5)])
    d_large = np.mean([frobenius_distance(small_gold_R(V, M, gold, rng.choice(400, 200, replace=False)), R_oracle)
                       for _ in range(5)])
    assert d_large < d_small

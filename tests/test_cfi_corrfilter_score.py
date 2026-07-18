"""Tests for CorrFilter α_subset scoring and R-mode resolution."""

from __future__ import annotations

import numpy as np
import pytest

from corrfilter.cfi.corrfilter_score import RMode, corrfilter_score, resolve_R


def test_identity_R_gives_sqrt_set_size():
    """With R = I, α_subset(v) = |S| / sqrt(|S|) = sqrt(|S|)."""
    V = np.array([[1, 1, 1, 0]], dtype=np.int8)
    M = np.ones_like(V)
    retained = np.array([1], dtype=np.int8)
    R = np.eye(4)
    sc = corrfilter_score(V, M, R, retained)
    # S = {0, 1, 2} -> |S| = 3 -> sqrt(3)
    assert float(sc.score[0]) == pytest.approx(np.sqrt(3.0))
    assert sc.set_size[0] == 3
    assert sc.quad_form[0] == pytest.approx(3.0)


def test_perfectly_correlated_judges_collapse_score():
    """When all judges in S are perfectly correlated (R_S = J), α_subset = sqrt(1)."""
    V = np.ones((1, 4), dtype=np.int8)
    M = np.ones_like(V)
    retained = np.array([1], dtype=np.int8)
    R = np.ones((4, 4))
    sc = corrfilter_score(V, M, R, retained)
    # |S| = 4, 1^T R_S 1 = 16, alpha = 4 / sqrt(16) = 1
    assert float(sc.score[0]) == pytest.approx(1.0)


def test_partially_correlated_decreases_score():
    n = 4
    V = np.ones((1, n), dtype=np.int8)
    M = np.ones_like(V)
    retained = np.array([1], dtype=np.int8)
    R_indep = np.eye(n)
    R_corr = 0.5 * np.eye(n) + 0.5 * np.ones((n, n))
    sc_indep = corrfilter_score(V, M, R_indep, retained)
    sc_corr = corrfilter_score(V, M, R_corr, retained)
    assert float(sc_corr.score[0]) < float(sc_indep.score[0])


def test_empty_agreeing_set_returns_nan():
    V = np.zeros((1, 3), dtype=np.int8)
    M = np.zeros_like(V)
    retained = np.array([1], dtype=np.int8)
    R = np.eye(3)
    sc = corrfilter_score(V, M, R, retained)
    assert np.isnan(sc.score[0])
    assert sc.set_size[0] == 0


def test_resolve_R_identity_mode_returns_eye():
    R = resolve_R(RMode.IDENTITY, R_h1_clean=None, R_cfi_variant=None, n_judges=5)
    assert R.shape == (5, 5)
    assert np.allclose(R, np.eye(5))


def test_resolve_R_h1_clean_requires_R():
    with pytest.raises(ValueError):
        resolve_R(RMode.H1_CLEAN, R_h1_clean=None, R_cfi_variant=None, n_judges=3)


def test_resolve_R_returns_each_mode_correctly():
    R_h1 = np.eye(3) * 0.5
    R_v = np.eye(3) * 0.25
    assert np.allclose(resolve_R(RMode.H1_CLEAN, R_h1, R_v, 3), R_h1)
    assert np.allclose(resolve_R(RMode.CFI_VARIANT, R_h1, R_v, 3), R_v)
    assert np.allclose(resolve_R(RMode.IDENTITY, R_h1, R_v, 3), np.eye(3))


def test_resolve_R_accepts_string_mode():
    R = resolve_R("identity_R", R_h1_clean=None, R_cfi_variant=None, n_judges=2)
    assert np.allclose(R, np.eye(2))


def test_score_only_counts_agreeing_judges():
    """If S = judges agreeing with retained, judges that disagree should not enter R_S."""
    V = np.array([[1, 1, 0, 0]], dtype=np.int8)
    M = np.ones_like(V)
    retained = np.array([1], dtype=np.int8)
    R = np.eye(4)
    sc = corrfilter_score(V, M, R, retained)
    assert sc.set_size[0] == 2
    assert float(sc.score[0]) == pytest.approx(np.sqrt(2.0))

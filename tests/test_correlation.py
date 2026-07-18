"""Correlation, effective-size, and H1-contrast sanity tests on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from corrfilter.analysis import pairwise_contrast_bootstrap
from corrfilter.correlation import (
    bootstrap_correlation,
    compute_error_matrix_pairwise,
    correlation_pairwise_complete,
    correlation_pearson,
    correlation_shrunk,
    correlation_shrunk_pairwise,
    effective_eig_rank,
    effective_size,
    nearest_psd,
)
from corrfilter.correlation.effective_size import mean_off_diagonal, neff_curve
from corrfilter.judges.base import JudgeSpec


def _make_spec(base_id: str, family: str, prompt_style: str) -> JudgeSpec:
    return JudgeSpec(
        base_id=base_id,
        family=family,
        scale="7B",
        prompt_style=prompt_style,
        hf_model=f"org/{base_id}",
        quantization="none",
    )


def _sample_correlated_binary(
    n_items: int, n_judges: int, rho: float, p_err: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a binary error matrix with given mean error rate and pairwise correlation.

    Uses a one-factor model: each row j has a shared latent z_j ~ U(0, 1).
    Each judge i emits e_{ij} = 1[u_{ij} < p_err] where u_{ij} = sqrt(rho) z_j +
    sqrt(1 - rho) eps_{ij} and eps_{ij} ~ U(0, 1) iid. This produces non-negative
    pairwise correlation that grows with rho.
    """
    z = rng.uniform(size=n_items)[:, None]
    eps = rng.uniform(size=(n_items, n_judges))
    u = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps
    return (u < p_err).astype(np.float64)


def test_pearson_diagonal_is_one():
    rng = np.random.default_rng(0)
    E = (rng.uniform(size=(300, 6)) < 0.3).astype(np.float64)
    R = correlation_pearson(E)
    assert np.allclose(np.diag(R), 1.0)


def test_shrunk_diagonal_is_one_and_shrinkage_in_unit_interval():
    rng = np.random.default_rng(0)
    E = (rng.uniform(size=(300, 6)) < 0.3).astype(np.float64)
    R, s = correlation_shrunk(E)
    assert np.allclose(np.diag(R), 1.0)
    assert 0.0 <= s <= 1.0


def test_independent_judges_have_near_zero_correlation():
    rng = np.random.default_rng(7)
    E = (rng.uniform(size=(4000, 5)) < 0.4).astype(np.float64)
    R = correlation_pearson(E)
    off = R[~np.eye(5, dtype=bool)]
    assert np.abs(off.mean()) < 0.05


def test_planted_correlation_is_recovered():
    rng = np.random.default_rng(42)
    rho_truth = 0.5
    E = _sample_correlated_binary(n_items=3000, n_judges=5, rho=rho_truth, p_err=0.3, rng=rng)
    R = correlation_pearson(E)
    rho_hat = mean_off_diagonal(R)
    # Coupling between the latent-z model and Pearson correlation is monotone but
    # not identity; just check that we move clearly into a positive regime.
    assert rho_hat > 0.2, f"expected ρ̂ > 0.2 under planted rho={rho_truth}, got {rho_hat}"


def test_effective_size_collapses_under_high_correlation():
    n = 8
    R_indep = np.eye(n)
    R_high = 0.05 * np.eye(n) + 0.95 * np.ones((n, n))
    assert effective_size(R_indep) == pytest.approx(n)
    assert effective_size(R_high) < 1.5


def test_effective_eig_rank_matches_for_identity():
    n = 6
    assert effective_eig_rank(np.eye(n)) == pytest.approx(n)


def test_effective_size_bounded_by_one_over_rho():
    """Proposition 1: n_eff ≤ 1 / ρ̄ regardless of bank size."""
    for rho in (0.05, 0.1, 0.3, 0.7):
        curve = neff_curve(rho, max_n=200)
        assert curve.max() <= 1.0 / rho + 1e-6


def test_bootstrap_ci_envelopes_point_estimate():
    rng = np.random.default_rng(0)
    E = _sample_correlated_binary(n_items=600, n_judges=4, rho=0.4, p_err=0.3, rng=rng)
    boot = bootstrap_correlation(E, n_boot=200, seed=0, shrunk=False)
    in_band = (boot.lower <= boot.point) & (boot.point <= boot.upper)
    # Allow a few entries to fall outside (sampling variability); require most to be inside.
    frac_in_band = float(in_band.mean())
    assert frac_in_band > 0.8


def test_nearest_psd_clips_negative_eigenvalues():
    """Nearest-PSD projection should zero out negative eigenvalues."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    sym = (A + A.T) / 2.0
    psd = nearest_psd(sym)
    eigvals = np.linalg.eigvalsh(psd)
    assert (eigvals >= -1e-10).all()


def test_pairwise_complete_matches_listwise_when_no_abstentions():
    """With no missing entries, pairwise-complete and listwise R should agree."""
    rng = np.random.default_rng(0)
    E = (rng.uniform(size=(500, 4)) < 0.3).astype(np.float64)
    avail = np.ones_like(E, dtype=bool)
    R_pairwise, n_pairs = correlation_pairwise_complete(E, avail)
    R_listwise = correlation_pearson(E)
    # Off-diagonal entries should be close (pairwise has PSD projection that
    # can move things slightly; allow 0.03 tolerance).
    off = ~np.eye(4, dtype=bool)
    assert np.abs(R_pairwise[off] - R_listwise[off]).max() < 0.03
    # n_pairs should equal n_items for every entry when nothing is missing.
    assert (n_pairs == 500).all()


def test_pairwise_complete_recovers_planted_correlation_under_abstentions():
    """Inject 40% MCAR abstentions; pairwise-complete should still recover ρ̄."""
    rng = np.random.default_rng(123)
    n_items, n_judges, rho = 2000, 5, 0.4
    z = rng.uniform(size=(n_items, 1))
    eps = rng.uniform(size=(n_items, n_judges))
    u = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps
    E = (u < 0.3).astype(np.float64)
    avail_full = np.ones_like(E, dtype=bool)
    R_clean, _ = correlation_pairwise_complete(E, avail_full)

    # Now drop 40% of entries MCAR.
    drop = rng.uniform(size=E.shape) < 0.4
    avail = ~drop
    R_missing, n_pairs = correlation_pairwise_complete(E, avail)
    # Mean off-diagonal should track the clean estimate within sampling noise.
    off = ~np.eye(n_judges, dtype=bool)
    delta = float(np.abs(R_missing[off].mean() - R_clean[off].mean()))
    assert delta < 0.05, f"mean off-diag drifted by {delta:.3f} under 40% MCAR"


def test_shrunk_pairwise_returns_valid_correlation_matrix():
    rng = np.random.default_rng(0)
    E = (rng.uniform(size=(400, 6)) < 0.3).astype(np.float64)
    # Make judge 3 abstain on 50% of items.
    avail = np.ones_like(E, dtype=bool)
    avail[: E.shape[0] // 2, 3] = False
    R, n_pairs, shrinkage = correlation_shrunk_pairwise(E, avail)
    assert np.allclose(np.diag(R), 1.0)
    eigvals = np.linalg.eigvalsh((R + R.T) / 2.0)
    assert (eigvals >= -1e-8).all()
    assert 0.0 <= shrinkage <= 1.0


def test_compute_error_matrix_pairwise_preserves_item_count():
    V = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int8)
    M = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
    gold = np.array([1, 1, 1], dtype=np.int8)
    E, avail = compute_error_matrix_pairwise(V, M, gold)
    assert E.shape == (3, 3)
    assert avail.shape == (3, 3)
    # Where judge voted: error = (vote != gold).
    assert E[0, 0] == 0  # vote 1, gold 1 → no error
    assert E[0, 2] == 1  # vote 0, gold 1 → error
    # Where judge abstained: E is zero AND mask is False.
    assert E[1, 1] == 0 and avail[1, 1] is np.False_ or not avail[1, 1]
    assert E[2, 2] == 0 and not avail[2, 2]


def test_h1_contrast_detects_planted_intra_group_correlation():
    """Planted: 4 judges in family A correlated, 4 in family B independent; intra-family ρ̄ should beat cross."""
    rng = np.random.default_rng(123)
    # Family A: 4 judges share latent, family B: independent noise.
    n_items = 1500
    A = _sample_correlated_binary(n_items, 4, rho=0.6, p_err=0.3, rng=rng)
    B = (rng.uniform(size=(n_items, 4)) < 0.3).astype(np.float64)
    E = np.concatenate([A, B], axis=1)
    specs = [
        _make_spec("a0", "A", "pairwise"),
        _make_spec("a1", "A", "pairwise"),
        _make_spec("a2", "A", "likert"),
        _make_spec("a3", "A", "likert"),
        _make_spec("b0", "B", "pairwise"),
        _make_spec("b1", "B", "pairwise"),
        _make_spec("b2", "B", "likert"),
        _make_spec("b3", "B", "likert"),
    ]
    result = pairwise_contrast_bootstrap(E, specs, grouping="family", n_boot=200, seed=0)
    assert result.delta > 0.10, f"expected planted delta > 0.10, got {result.delta:.3f}"
    assert result.p_value < 0.05

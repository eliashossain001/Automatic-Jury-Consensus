"""Tests for CorrFilter v1 α_subset properties and deterministic thresholding.

Covers the proposal's defining identities for α_subset(v) = |S| / √(1ᵀ R_S 1):

* R = I  ⇒ α_subset = √|S|         (independent judges; no discount)
* R = 11ᵀ ⇒ α_subset = 1           (perfectly correlated; one effective judge)
* positive correlation reduces α relative to the independent case
* empty agreeing subset is handled safely (NaN score, no crash)
* retention-matching keeps exactly the requested count, deterministically
"""

from __future__ import annotations

import numpy as np

from corrfilter.cfi.corrfilter_score import (
    corrfilter_score,
    retention_match_threshold,
)
from corrfilter.filtering import (
    evaluate_filter,
    filter_corrfilter_subset,
    filter_naive_majority,
)


def _all_agree(n_items, n_judges, label=1):
    """Vote matrix where every judge votes ``label`` on every item."""
    V = np.full((n_items, n_judges), label, dtype=np.int8)
    M = np.ones((n_items, n_judges), dtype=np.int8)
    return V, M


def test_alpha_identity_gives_sqrt_set_size():
    n = 5
    V, M = _all_agree(3, n)
    R = np.eye(n)
    label = np.ones(3, dtype=np.int8)
    out = corrfilter_score(V, M, R, label)
    assert np.allclose(out.set_size, n)
    assert np.allclose(out.score, np.sqrt(n)), out.score


def test_alpha_all_ones_gives_one():
    n = 6
    V, M = _all_agree(4, n)
    R = np.ones((n, n))
    label = np.ones(4, dtype=np.int8)
    out = corrfilter_score(V, M, R, label)
    # 1ᵀ R_S 1 = |S|² ⇒ α = |S| / √(|S|²) = 1.
    assert np.allclose(out.score, 1.0), out.score


def test_positive_correlation_reduces_alpha():
    n = 4
    V, M = _all_agree(2, n)
    label = np.ones(2, dtype=np.int8)
    indep = corrfilter_score(V, M, np.eye(n), label).score[0]
    rho = 0.5
    R_corr = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
    corr = corrfilter_score(V, M, R_corr, label).score[0]
    assert corr < indep, (corr, indep)
    # And stronger correlation reduces it further.
    R_strong = 0.1 * np.eye(n) + 0.9 * np.ones((n, n))
    strong = corrfilter_score(V, M, R_strong, label).score[0]
    assert strong < corr < indep


def test_empty_agreeing_subset_is_safe():
    n = 3
    # No judge voted (all abstain via M=0): agreeing set is empty.
    V = np.zeros((2, n), dtype=np.int8)
    M = np.zeros((2, n), dtype=np.int8)
    label = np.ones(2, dtype=np.int8)
    out = corrfilter_score(V, M, R=np.eye(n), retained_label=label)
    assert np.all(out.set_size == 0)
    assert np.all(np.isnan(out.score))


def test_retention_match_is_deterministic_and_exact():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=50)
    for n_keep in (0, 1, 10, 25, 50):
        keep1, t1 = retention_match_threshold(scores, n_keep)
        keep2, t2 = retention_match_threshold(scores, n_keep)
        assert np.array_equal(keep1, keep2)  # deterministic
        assert t1 == t2
        assert int(keep1.sum()) == n_keep, (n_keep, int(keep1.sum()))


def test_retention_match_handles_ties():
    # All identical scores: must still keep exactly n_keep, not all-or-nothing.
    scores = np.ones(10)
    keep, _ = retention_match_threshold(scores, 4)
    assert int(keep.sum()) == 4


def test_corrfilter_keeps_fewer_correlated_agreements():
    # Two items, both unanimous. With strong correlation α is low for both, but
    # retention-matching still selects the requested fraction without error.
    n = 5
    V, M = _all_agree(10, n)
    gold = np.ones(10, dtype=np.int8)
    R = 0.2 * np.eye(n) + 0.8 * np.ones((n, n))
    res = filter_corrfilter_subset(V, M, R, gold, target_retention=0.5)
    assert int(res.keep.sum()) == 5
    m = evaluate_filter(res, gold)
    assert m["retention_rate"] == 0.5
    assert 0.0 <= m["false_retention_rate"] <= 1.0


def test_naive_majority_evaluation_fields():
    n = 5
    V, M = _all_agree(8, n)
    gold = np.ones(8, dtype=np.int8)
    res = filter_naive_majority(V, M, gold)
    m = evaluate_filter(res, gold)
    # Everyone agrees with gold ⇒ perfect precision, full retention.
    assert m["retention_rate"] == 1.0
    assert m["precision"] == 1.0
    assert m["false_retention_rate"] == 0.0
    assert m["recall"] == 1.0

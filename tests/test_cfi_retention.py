"""Tests for retention rate, false retention rate, and CorrFilter retention matching."""

from __future__ import annotations

import numpy as np
import pytest

from corrfilter.cfi.consensus import ABSTAIN
from corrfilter.cfi.corrfilter_score import corrfilter_filter, retention_match_threshold
from corrfilter.cfi.metrics import (
    accuracy_when_kept,
    conditional_error_rate,
    false_retention_rate,
    retention_rate,
)


def test_retention_rate_is_fraction_of_true():
    keep = np.array([True, False, True, True, False])
    assert retention_rate(keep) == pytest.approx(0.6)


def test_false_retention_rate_counts_wrong_among_kept():
    keep = np.array([True, True, True, False])
    retained = np.array([1, 1, 0, 1], dtype=np.int8)
    gold = np.array([1, 0, 1, 1], dtype=np.int8)
    # Among kept (idx 0,1,2): items 1 and 2 disagree → frr = 2/3
    assert false_retention_rate(keep, retained, gold) == pytest.approx(2.0 / 3.0)


def test_false_retention_ignores_abstain():
    keep = np.array([True, True, True])
    retained = np.array([1, ABSTAIN, 0], dtype=np.int8)
    gold = np.array([1, 1, 1], dtype=np.int8)
    # Abstain treated as not-kept, so among the 2 surviving kept items: 1 wrong → 0.5
    assert false_retention_rate(keep, retained, gold) == pytest.approx(0.5)


def test_accuracy_when_kept_complements_frr():
    keep = np.array([True, True, True])
    retained = np.array([1, 0, 1], dtype=np.int8)
    gold = np.array([1, 1, 1], dtype=np.int8)
    assert accuracy_when_kept(keep, retained, gold) == pytest.approx(2.0 / 3.0)


def test_retention_match_keeps_exact_count():
    scores = np.array([0.5, 1.2, 0.9, 0.1, 1.5])
    keep, threshold = retention_match_threshold(scores, n_target_keep=3)
    assert int(keep.sum()) == 3
    # The 3 highest scores are 1.5, 1.2, 0.9 → threshold = 0.9.
    assert threshold == pytest.approx(0.9)


def test_retention_match_breaks_ties_by_order():
    scores = np.array([1.0, 1.0, 1.0, 0.5])
    keep, _ = retention_match_threshold(scores, n_target_keep=2)
    assert int(keep.sum()) == 2


def test_retention_match_drops_nan_scores():
    scores = np.array([np.nan, 1.0, 0.5, np.nan])
    keep, _ = retention_match_threshold(scores, n_target_keep=4)
    # Only 2 valid scores → at most 2 kept even when target > available.
    assert int(keep.sum()) == 2


def test_corrfilter_filter_target_zero_keeps_none():
    scores = np.array([1.0, 2.0, 3.0])
    keep, _ = corrfilter_filter(scores, target_retention=0.0)
    assert int(keep.sum()) == 0


def test_corrfilter_filter_target_one_keeps_all_valid():
    scores = np.array([1.0, np.nan, 3.0])
    keep, _ = corrfilter_filter(scores, target_retention=1.0)
    assert int(keep.sum()) == 2  # NaN dropped


def test_conditional_error_rate_zero_when_no_triggers():
    V = np.array([[1, 1], [1, 1]], dtype=np.int8)
    M = np.ones_like(V)
    gold = np.array([1, 1], dtype=np.int8)
    mask = np.zeros(2, dtype=bool)
    rate = conditional_error_rate(V, M, gold, mask)
    assert rate.shape == (2,)
    assert (rate == 0).all()


def test_conditional_error_rate_restricts_to_mask():
    V = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    M = np.ones_like(V)
    gold = np.array([1, 1, 1], dtype=np.int8)
    mask = np.array([True, True, False])
    # Among items 0,1: judge 0 errs on item 1 (0.5), judge 1 errs on item 0 (0.5).
    rate = conditional_error_rate(V, M, gold, mask)
    assert rate.tolist() == [0.5, 0.5]

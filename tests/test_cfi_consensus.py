"""Tests for consensus / supermajority / weighted consensus."""

from __future__ import annotations

import numpy as np

from corrfilter.cfi.consensus import (
    ABSTAIN,
    accuracy_by_consensus_level,
    consensus_level,
    majority_consensus,
    supermajority_consensus,
    vote_fraction,
    weighted_consensus,
)


def test_consensus_level_is_bounded():
    V = np.array([[1, 0, 1, 0]], dtype=np.int8)
    M = np.ones_like(V)
    lvl = consensus_level(V, M)
    assert 0.5 <= float(lvl[0]) <= 1.0


def test_consensus_level_unanimous_is_one():
    V = np.ones((3, 5), dtype=np.int8)
    M = np.ones_like(V)
    assert np.allclose(consensus_level(V, M), 1.0)


def test_majority_consensus_picks_argmax():
    V = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
    M = np.ones_like(V)
    pred = majority_consensus(V, M)
    assert pred.tolist() == [1, 0, 0]


def test_majority_consensus_tie_break():
    V = np.array([[1, 1, 0, 0]], dtype=np.int8)
    M = np.ones_like(V)
    assert majority_consensus(V, M, tie_break=1)[0] == 1
    assert majority_consensus(V, M, tie_break=0)[0] == 0


def test_majority_abstains_when_no_voters():
    V = np.zeros((1, 3), dtype=np.int8)
    M = np.zeros_like(V)
    assert majority_consensus(V, M)[0] == ABSTAIN


def test_supermajority_abstains_below_threshold():
    V = np.array([[1, 1, 0, 0, 0]], dtype=np.int8)  # consensus_level = 0.6
    M = np.ones_like(V)
    assert supermajority_consensus(V, M, threshold=0.7)[0] == ABSTAIN
    assert supermajority_consensus(V, M, threshold=0.5)[0] in (0, 1)


def test_supermajority_keeps_when_above_threshold():
    V = np.array([[1, 1, 1, 1, 0]], dtype=np.int8)  # level 0.8
    M = np.ones_like(V)
    assert supermajority_consensus(V, M, threshold=0.75)[0] == 1


def test_weighted_consensus_respects_weights():
    V = np.array([[1, 0, 0]], dtype=np.int8)
    M = np.ones_like(V)
    # Equal weights -> 0 wins (2 of 3).
    assert weighted_consensus(V, M, np.array([1.0, 1.0, 1.0]))[0] == 0
    # Heavy weight on judge 0 (vote 1) -> 1 wins.
    assert weighted_consensus(V, M, np.array([10.0, 1.0, 1.0]))[0] == 1


def test_accuracy_by_consensus_level_buckets_correctly():
    rng = np.random.default_rng(0)
    n = 200
    pred = rng.integers(0, 2, size=n)
    gold = pred.copy()  # perfect agreement
    level = rng.uniform(0.5, 1.0, size=n)
    bins, acc, cnt = accuracy_by_consensus_level(pred, gold, level)
    assert acc.shape == (len(bins) - 1,)
    assert cnt.sum() == n
    assert np.allclose(acc[cnt > 0], 1.0)


def test_consensus_level_handles_partial_abstention():
    V = np.array([[1, 1, 0]], dtype=np.int8)
    M = np.array([[1, 1, 0]], dtype=np.int8)  # judge 2 abstains
    # Among voters, both voted 1 -> level 1.0.
    assert float(consensus_level(V, M)[0]) == 1.0
    # Vote fraction is 1.0 as well.
    assert float(vote_fraction(V, M)[0]) == 1.0

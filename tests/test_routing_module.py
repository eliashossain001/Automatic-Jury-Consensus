"""Tests for corrfilter.routing: parity with the regime/unified routers in scripts/routing_selector/ and generator properties."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrfilter.routing import (
    HARD_THR,
    REGIME_FILTER,
    agree_features,
    classify_regime_hard,
    filter_metrics,
    filter_scores,
    keep_at_matched_retention,
    pct_rank,
    position_poison_mask,
    position_sensitivity_cluster,
    regime_probs,
)


def _toy_votes(seed=0, n=60, m=6):
    rng = np.random.default_rng(seed)
    V = (rng.random((n, m)) < 0.7).astype(np.int8)
    M = np.ones((n, m), np.int8)
    return V, M


def test_filter_metrics_matches_definitions():
    gold = np.array([1, 1, 0, 0, 1])
    keep = np.array([True, True, True, False, False])
    m = filter_metrics(keep, gold)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["false_retention_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["n_kept"] == 3 and m["n_kept_clean"] == 2 and m["n_clean"] == 3


def test_classify_regime_hard_rules():
    ref = {"rho_bar": 0.2, "lco_flip": 0.1}
    assert classify_regime_hard({"rho_bar": 0.30, "lco_flip": 0.1, "eig_overlap": 0.95}, ref) == "global"
    assert classify_regime_hard({"rho_bar": 0.20, "lco_flip": 0.30, "eig_overlap": 0.95}, ref) == "subgroup"
    assert classify_regime_hard({"rho_bar": 0.20, "lco_flip": 0.10, "eig_overlap": 0.80}, ref) == "subgroup"
    assert classify_regime_hard({"rho_bar": 0.21, "lco_flip": 0.12, "eig_overlap": 0.95}, ref) == "weak"


def test_regime_probs_softmax_properties():
    lco = np.array([0.0, 1.0, 0.0])
    pw, pg, ps = regime_probs(0.5, 0.0, np.zeros(3), lco)
    assert np.allclose(pw + pg + ps, 1.0)
    assert ps[1] > ps[0]          # lco raises subgroup evidence
    pw2, pg2, _ = regime_probs(2.0, 0.0, np.zeros(3), lco)
    assert pg2[0] > pg[0]         # larger rho_bar deviation raises global prob


def test_pct_rank_handles_neg_inf():
    r = pct_rank(np.array([1.0, -np.inf, 3.0, 2.0]))
    assert r[1] == 0.0
    assert r[2] == 1.0
    assert r[0] < r[3] < r[2]


def test_keep_at_matched_retention_respects_pool_and_count():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    pool = np.array([True, True, False, True, True])
    keep = keep_at_matched_retention(score, pool, 2)
    assert keep.sum() == 2
    assert not keep[2]
    assert keep[0] and keep[1]


def test_position_sensitivity_cluster_finds_biased_judges():
    rng = np.random.default_rng(1)
    n, m = 400, 6
    V = (rng.random((n, m)) < 0.5).astype(np.int8)
    M = np.ones((n, m), np.int8)
    S = (rng.random((n, m)) < 0.5).astype(np.int8)
    # judges 0 and 1 always pick slot A: vote = A-pick decoded, so V = 1-S -> slotA pick = 1
    V[:, 0] = 1 - S[:, 0]
    V[:, 1] = 1 - S[:, 1]
    mask = position_sensitivity_cluster(V, M, S, k=2)
    assert mask[0] and mask[1] and mask.sum() == 2


def test_position_poison_mask_targets_weighted_wrong_votes():
    lids = ["mistral-7b-v0.3::pairwise", "phi-3.5-mini::likert"]   # weights 70.2, 11.9
    V = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], np.int8)
    M = np.ones_like(V)
    pm = position_poison_mask(V, M, lids, rate=0.25)
    # scores: item0 = .702, item1 = .119, item2 = 0, item3 = .821 -> item3 poisoned first
    assert pm[3] and pm.sum() == 1
    sub = np.array([0, 1, 2])
    pm2 = position_poison_mask(V, M, lids, rate=1 / 3, subset_idx=sub)
    assert pm2[0] and pm2.sum() == 1   # within subset, item0 has top score


def test_agree_features_and_scores_consistency():
    V, M = _toy_votes()
    label = np.ones(V.shape[0], np.int8)
    cmask = np.zeros(V.shape[1], bool)
    cmask[:2] = True
    R = np.eye(V.shape[1])
    feats = agree_features(V, M, label, cmask, R)
    scores = filter_scores(V, M, label, R, R, cmask, feats)
    assert set(scores) == {"naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"}
    # indep = agreeing judges outside the cluster
    i = 0
    agree = (V[i] == 1)
    assert feats["indep"][i] == agree.sum() - agree[:2].sum()
    for s in scores.values():
        assert s.shape == (V.shape[0],)


def test_parity_with_script18_agree_features():
    """The canonical module must reproduce scripts/routing_selector/run_regime_router.py's private implementation."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s18", Path(__file__).resolve().parents[1] / "scripts" / "routing_selector/run_regime_router.py")
    s18 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s18)

    V, M = _toy_votes(seed=3)
    rng = np.random.default_rng(4)
    label = (rng.random(V.shape[0]) < 0.8).astype(np.int8)
    cmask = np.zeros(V.shape[1], bool)
    cmask[[1, 4]] = True
    R = np.corrcoef(rng.random((V.shape[1], 50)))
    ours = agree_features(V, M, label, cmask, R)
    theirs = s18._agree_features(V, M, label, cmask, R)
    for k in ours:
        np.testing.assert_allclose(ours[k], theirs[k], err_msg=k)
    np.testing.assert_allclose(pct_rank(ours["indep"]), s18._pct(theirs["indep"]))
    assert classify_regime_hard(
        {"rho_bar": 0.3, "lco_flip": 0.0, "eig_overlap": 0.9}, {"rho_bar": 0.2, "lco_flip": 0.1}, HARD_THR
    ) == "global"


def test_regime_filter_mapping_unchanged():
    assert REGIME_FILTER == {"weak": "supermajority_75",
                             "global": "corrfilter_small_gold_R",
                             "subgroup": "bias_cluster"}

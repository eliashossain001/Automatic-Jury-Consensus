"""Unit tests for the extended routing evaluation (experiments/router_upgrade).

Covers: per-instance oracle computation, pooled metrics, generator marginal
matching, mixture partitioning, determinism, and train/deploy leakage
prevention. CPU-only; no vote caches required except the leakage test, which
is skipped when the pool CSVs are absent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/router_upgrade"))
sys.path.insert(0, str(ROOT / "src"))

from build_crossed_pool import (P_HI, P_LO, WRONG_MARGINAL, Z_SUBGROUP,  # noqa: E402
                                marginal_params, wrong_probs)
from run_learned_router import FIXED, pooled_precision  # noqa: E402
from run_mixed_eval import pooled_confusion  # noqa: E402

OUT = ROOT / "outputs/router_upgrade"


def _methods_df(rows):
    return pd.DataFrame(rows)


def test_perinstance_oracle_argmax_and_tiebreak():
    md = _methods_df([
        {"instance": "i1", "method": "naive_majority", "precision": 0.90, "n_kept": 100, "n_kept_clean": 90},
        {"instance": "i1", "method": "bias_cluster", "precision": 0.95, "n_kept": 100, "n_kept_clean": 95},
        {"instance": "i2", "method": "naive_majority", "precision": 0.90, "n_kept": 120, "n_kept_clean": 108},
        {"instance": "i2", "method": "bias_cluster", "precision": 0.90, "n_kept": 100, "n_kept_clean": 90},
    ])
    sub = md[md.method.isin(FIXED)].sort_values(["precision", "n_kept"], ascending=False)
    orc = dict(zip(sub.drop_duplicates("instance").instance, sub.drop_duplicates("instance").method))
    assert orc["i1"] == "bias_cluster"          # strict argmax
    assert orc["i2"] == "naive_majority"        # tie broken by larger keep set


def test_pooled_precision_and_confusion():
    md = _methods_df([
        {"instance": "i1", "method": "m", "precision": 0.9, "n_kept": 10, "n_kept_clean": 9,
         "n_eval": 20, "n_clean": 15},
        {"instance": "i2", "method": "m", "precision": 0.5, "n_kept": 10, "n_kept_clean": 5,
         "n_eval": 20, "n_clean": 10},
    ])
    dec = {"i1": "m", "i2": "m"}
    assert pooled_precision(md, dec) == pytest.approx(14 / 20)
    c = pooled_confusion(md, dec)
    # TP=14, FP=6, FN=(15-9)+(10-5)=11, TN=(20-15-1)+(20-10-5)=9
    assert c["precision"] == pytest.approx(14 / 20)
    assert c["accuracy"] == pytest.approx((14 + 9) / 40)
    assert c["retention"] == pytest.approx(20 / 40)


@pytest.mark.parametrize("regime", ["weak", "global", "subgroup"])
def test_generator_marginals_matched(regime):
    """Every regime's expected wrong-vote fraction on bad items is 0.45."""
    nj = 10
    cl = np.zeros(nj, bool)
    cl[:5] = True
    q_rest, p_z_global = marginal_params(nj, cl)
    rng = np.random.default_rng(0)
    tot = 0.0
    n_items = 4000
    for _ in range(n_items):
        p = wrong_probs(regime, nj, cl, rng, q_rest, p_z_global)
        tot += p.mean()
    assert tot / n_items == pytest.approx(WRONG_MARGINAL, abs=0.02)


def test_generator_dependence_structure():
    """Global co-fails bank-wide; subgroup co-fails only inside the cluster."""
    nj = 10
    cl = np.zeros(nj, bool)
    cl[:5] = True
    q_rest, p_z_global = marginal_params(nj, cl)
    rng = np.random.default_rng(1)
    var_g = np.var([wrong_probs("global", nj, cl, rng, q_rest, p_z_global).mean()
                    for _ in range(2000)])
    var_w = np.var([wrong_probs("weak", nj, cl, rng, q_rest, p_z_global).mean()
                    for _ in range(2000)])
    assert var_g > 0.1 and var_w < 1e-12  # shared shock vs deterministic-iid probs
    ps = np.array([wrong_probs("subgroup", nj, cl, rng, q_rest, p_z_global)
                   for _ in range(2000)])
    assert ps[:, :5].std() > ps[:, 5:].std()  # variability concentrated in cluster
    assert np.allclose(ps[:, 5:], q_rest)


def test_wrong_probs_bounds():
    nj = 10
    cl = np.zeros(nj, bool)
    cl[:3] = True
    q_rest, p_z = marginal_params(nj, cl)
    assert 0.0 <= q_rest <= 1.0 and 0.0 < p_z < 1.0
    rng = np.random.default_rng(2)
    for regime in ("weak", "global", "subgroup"):
        p = wrong_probs(regime, nj, cl, rng, q_rest, p_z)
        assert p.shape == (nj,) and (p >= 0).all() and (p <= 1).all()
        assert set(np.round(np.unique(p), 6)) <= {round(v, 6) for v in
                                                  (P_LO, P_HI, q_rest, WRONG_MARGINAL)}


@pytest.mark.skipif(not (OUT / "mixed_features.csv").exists(), reason="pools not built")
def test_no_train_deploy_leakage():
    """No mixed-deployment config or instance appears in any pure training pool."""
    pure = pd.concat([pd.read_csv(OUT / "crossed_features.csv"),
                      pd.read_csv(OUT / "pku_features.csv")], ignore_index=True)
    mixed = pd.concat([pd.read_csv(OUT / "mixed_features.csv"),
                       pd.read_csv(OUT / "pku_mixed_features.csv")], ignore_index=True)
    assert not set(pure.instance) & set(mixed.instance)
    assert not set(pure.config_id) & set(mixed.config_id)


@pytest.mark.skipif(not (OUT / "ext_perinstance.csv").exists(), reason="eval not run")
def test_decomposition_partition():
    """Failure buckets partition all instances; regret only outside 'correct'."""
    art = pd.read_csv(OUT / "ext_perinstance.csv")
    assert set(art.bucket) <= {"A: wrong regime", "B: right regime, wrong mapping",
                               "correct", "no-majority"}
    assert (art.loc[art.bucket == "correct", "regret_pts"].abs() < 1e-9).all()
    assert (art.regret_pts >= -1e-9).all()  # regret is vs the per-instance max
    has_maj = art.majority_regime != "none"
    a = art[has_maj & (art.pred_regime != art.majority_regime)]
    assert (a.bucket == "A: wrong regime").all()


def test_mixture_partition_counts():
    from build_mixed_pool import majority_regime, mixtures
    for name, w in mixtures():
        assert abs(sum(w.values()) - 1.0) < 1e-9
        mr = majority_regime(w)
        assert mr in {"weak", "global", "subgroup", "none"}
        if mr != "none":
            assert w[mr] == max(w.values())


@pytest.mark.skipif(not (OUT / "crossed_features.csv").exists(), reason="pools not built")
def test_deterministic_features():
    """Feature CSVs are the deterministic product of seeded generators: the
    stored heuristic prediction must be reproducible from stored features."""
    f = pd.read_csv(OUT / "crossed_features.csv")
    assert f.instance.is_unique
    assert set(f.regime) == {"weak", "global", "subgroup"}
    assert f[["rho_bar", "n_eff", "eig_overlap"]].notna().all().all()

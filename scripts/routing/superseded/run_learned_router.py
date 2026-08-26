#!/usr/bin/env python
"""Learned-router evaluation on the enlarged mixed-regime pool.

Experiments (research only; the paper is untouched):
  1  models: logistic regression, random forest, XGBoost, LightGBM, small MLP
  2  feature sets: BASE (the heuristic router's information), EXT (engineered),
     ALL (both)
  3  confidence-aware routing: below-threshold predictions fall back to the
     best fixed filter selected on the training folds
  5  oracle-gap accounting: heuristic vs learned vs oracle, routing accuracy
     and downstream pooled precision

Leakage control: instances are grouped by generator parameterisation
(`config_id`); Grouped 5-fold CV never places the same config in train and
test. Extrapolation stress tests hold out entire rates (weak+subgroup) or
mechanisms (global). Best-fixed baselines are always selected on training
folds only. Downstream metrics come from the cached per-instance filter
results (hard routing at instance level reproduces the chosen filter's keep
set exactly, per the scripts/routing/superseded/mixed_regime_benchmark.py identity).

Usage: python scripts/routing/superseded/run_learned_router.py
Outputs -> outputs/router_upgrade/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.routing import (
    BASE_FEATS,
    N_REPS,
    REGIME_FILTER,
    REGIMES,
    best_fixed_on,
    make_models,
    paired_boot_diff,
    pooled_precision,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/router_upgrade"


def grouped_cv(feats, methods_df, feat_cols, tag):
    """Grouped 5-fold CV x N_REPS: routing accuracy + out-of-fold decisions."""

    X = feats[feat_cols].to_numpy(float)
    y = feats["regime"].to_numpy()
    groups = feats["config_id"].to_numpy()
    insts = feats["instance"].to_numpy()

    acc_rows, oof_store = [], {}
    for rep in range(N_REPS):
        # shuffle group-to-fold assignment via GroupShuffleSplit-style permutation
        rng = np.random.default_rng(1000 + rep)
        uniq = rng.permutation(np.unique(groups))
        gmap = {g: k % 5 for k, g in enumerate(uniq)}
        fold = np.array([gmap[g] for g in groups])
        models = make_models(seed=rep)
        for mname, model in models.items():
            pred = np.empty(len(y), dtype=object)
            proba = np.zeros(len(y))
            for f in range(5):
                tr, te = fold != f, fold == f
                model.fit(X[tr], y[tr])
                pred[te] = model.predict(X[te])
                if hasattr(model, "predict_proba"):
                    proba[te] = model.predict_proba(X[te]).max(1)
            acc = float((pred == y).mean())
            bal = float(np.mean([(pred[y == r] == r).mean() for r in REGIMES]))
            acc_rows.append({"tag": tag, "rep": rep, "model": mname, "accuracy": acc, "balanced": bal})
            if rep == 0:
                oof_store[mname] = (dict(zip(insts, pred)), dict(zip(insts, proba)), fold)
    return pd.DataFrame(acc_rows), oof_store


def main() -> None:
    feats = pd.read_csv(OUT / "pool_features.csv")
    methods_df = pd.read_csv(OUT / "pool_methods.csv")
    ext_cols = [c for c in feats.columns if c not in BASE_FEATS +
                ["instance", "regime", "config_id", "seed_rep", "pred_hard", "pred_soft"]]
    sets = {"base": BASE_FEATS, "ext": ext_cols, "all": BASE_FEATS + ext_cols}
    print(f"pool: {len(feats)} instances | {len(BASE_FEATS)} base + {len(ext_cols)} extended features")

    # ---------- Exp 1+2: grouped CV over models x feature sets ----------
    all_acc, oof_all = [], {}
    for sname, cols in sets.items():
        acc_df, oof = grouped_cv(feats, methods_df, cols, sname)
        all_acc.append(acc_df)
        oof_all[sname] = oof
    acc = pd.concat(all_acc)
    summ = acc.groupby(["tag", "model"]).agg(acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                                             bal_mean=("balanced", "mean")).reset_index()
    # heuristic + soft baselines
    heur_acc = float((feats["pred_hard"] == feats["regime"]).mean())
    soft_acc = float((feats["pred_soft"] == feats["regime"]).mean())
    summ.to_csv(OUT / "cv_results.csv", index=False)
    print("\n== Routing accuracy (grouped 5-fold CV, mean±sd over 10 reps) ==")
    print(summ.sort_values("acc_mean", ascending=False).to_string(index=False))
    print(f"heuristic hard router: {heur_acc:.3f} | soft router: {soft_acc:.3f}")

    # ---------- Downstream (rep 0 out-of-fold decisions) ----------
    y_map = dict(zip(feats["instance"], feats["regime"]))
    heur_dec = {i: REGIME_FILTER[p] for i, p in zip(feats["instance"], feats["pred_hard"])}
    oracle_dec = {i: REGIME_FILTER[y_map[i]] for i in feats["instance"]}
    rows_ds = []
    best_model_key = None
    best_ds = -1
    for sname, oof in oof_all.items():
        for mname, (pred, proba, fold) in oof.items():
            dec = {i: REGIME_FILTER[p] for i, p in pred.items()}
            # train-fold best fixed per fold for fallback + baseline
            inst_fold = dict(zip(feats["instance"], fold))
            bf_by_fold = {}
            for f in range(5):
                tr_insts = [i for i in feats["instance"] if inst_fold[i] != f]
                bf_by_fold[f] = best_fixed_on(methods_df, tr_insts)
            bf_dec = {i: bf_by_fold[inst_fold[i]] for i in feats["instance"]}
            p_learn = pooled_precision(methods_df, dec)
            rows_ds.append({"feat_set": sname, "model": mname,
                            "learned": p_learn,
                            "heuristic": pooled_precision(methods_df, heur_dec),
                            "best_fixed_cv": pooled_precision(methods_df, bf_dec),
                            "oracle": pooled_precision(methods_df, oracle_dec)})
            if p_learn > best_ds:
                best_ds, best_model_key = p_learn, (sname, mname)
    ds = pd.DataFrame(rows_ds)
    ds.to_csv(OUT / "downstream_results.csv", index=False)
    print("\n== Downstream pooled precision (rep-0 out-of-fold) ==")
    print(ds.sort_values("learned", ascending=False).head(8).to_string(index=False))

    # significance for the best learned router
    sname, mname = best_model_key
    pred, proba, fold = oof_all[sname][mname]
    dec = {i: REGIME_FILTER[p] for i, p in pred.items()}
    inst_fold = dict(zip(feats["instance"], fold))
    bf_by_fold = {f: best_fixed_on(methods_df, [i for i in feats["instance"] if inst_fold[i] != f])
                  for f in range(5)}
    bf_dec = {i: bf_by_fold[inst_fold[i]] for i in feats["instance"]}
    sig = {}
    for label, other in (("learned_minus_heuristic", heur_dec), ("learned_minus_bestfixed", bf_dec),
                         ("oracle_minus_bestfixed", None)):
        a = oracle_dec if label.startswith("oracle") else dec
        b = bf_dec if other is None else other
        point, lo, hi = paired_boot_diff(methods_df, a, b)
        sig[label] = (round(100 * point, 2), round(100 * lo, 2), round(100 * hi, 2))
    pd.DataFrame([{"comparison": k, "gain_pts": v[0], "ci_lo": v[1], "ci_hi": v[2],
                   "significant": v[1] > 0 or v[2] < 0} for k, v in sig.items()]
                 ).to_csv(OUT / "significance.csv", index=False)
    print(f"\nbest learned router: {mname} on '{sname}' features")
    for k, v in sig.items():
        print(f"  {k}: {v[0]:+.2f} pts, 95% CI [{v[1]:+.2f}, {v[2]:+.2f}]")

    # ---------- Exp 3: confidence-aware fallback ----------
    conf_rows = []
    for tau in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        dec_t = {i: (REGIME_FILTER[pred[i]] if proba[i] >= tau else bf_dec[i]) for i in feats["instance"]}
        conf_rows.append({"tau": tau,
                          "routed_frac": float(np.mean([proba[i] >= tau for i in feats["instance"]])),
                          "pooled_precision": pooled_precision(methods_df, dec_t)})
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(OUT / "confidence_sweep.csv", index=False)
    print("\n== Confidence-aware fallback (best model) ==")
    print(conf.to_string(index=False))

    # ---------- Extrapolation stress tests ----------
    rows_ex = []
    models = make_models(seed=0)
    X_all = {s: feats[c].to_numpy(float) for s, c in sets.items()}
    y = feats["regime"].to_numpy()
    for held_rate in ["0.05", "0.075", "0.1", "0.15", "0.2"]:
        te = feats["config_id"].str.contains(f"_r{held_rate}$", regex=True).to_numpy()
        if te.sum() == 0:
            continue
        tr = ~te
        for sname in sets:
            m = make_models(seed=0)["lightgbm"]
            m.fit(X_all[sname][tr], y[tr])
            accte = float((m.predict(X_all[sname][te]) == y[te]).mean())
            rows_ex.append({"split": f"leave-rate-{held_rate}", "feat_set": sname,
                            "model": "lightgbm", "test_n": int(te.sum()), "accuracy": accte})
    for mech in ["position", "polite_h", "verbosit"]:
        te = feats["config_id"].str.contains(f"global_{mech}").to_numpy()
        tr = ~te
        for sname in sets:
            m = make_models(seed=0)["lightgbm"]
            m.fit(X_all[sname][tr], y[tr])
            accte = float((m.predict(X_all[sname][te]) == y[te]).mean())
            rows_ex.append({"split": f"leave-mech-{mech}", "feat_set": sname,
                            "model": "lightgbm", "test_n": int(te.sum()), "accuracy": accte})
    ex = pd.DataFrame(rows_ex)
    ex.to_csv(OUT / "extrapolation_results.csv", index=False)
    print("\n== Extrapolation stress tests (LightGBM) ==")
    print(ex.pivot_table(index="split", columns="feat_set", values="accuracy").round(3).to_string())

    # ---------- Exp 4 artifacts: confusion + importances ----------
    cm = pd.crosstab(pd.Series(y, name="true"),
                     pd.Series([pred[i] for i in feats["instance"]], name="pred"))
    cm.to_csv(OUT / f"confusion_{mname}_{sname}.csv")
    print(f"\n== Confusion (best model, out-of-fold) ==\n{cm.to_string()}")
    from sklearn.inspection import permutation_importance
    m = make_models(seed=0)["lightgbm"]
    cols = sets["all"]
    m.fit(feats[cols].to_numpy(float), y)
    imp = permutation_importance(m, feats[cols].to_numpy(float), y, n_repeats=10, random_state=0)
    imp_df = pd.DataFrame({"feature": cols, "importance": imp.importances_mean}).sort_values(
        "importance", ascending=False)
    imp_df.to_csv(OUT / "feature_importance.csv", index=False)
    print("\n== Top-10 permutation importances (LightGBM, all features) ==")
    print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

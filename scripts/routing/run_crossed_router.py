#!/usr/bin/env python
"""Routing evaluation on the crossed (regime x source) benchmark.

Protocols:
  A  grouped 5-fold CV by config (10 reps): routing accuracy + downstream
  B  leave-one-source-out: train on UF, test on RB, and vice versa
  C  leave-one-rate-out; leave-CFI-configs-out (flip-generators -> CFI transfer)
  D  source-shortcut ceiling: what regime accuracy would source identity alone
     give on this pool (the confound's residual value)
  E  alternative objective (Task 4): per-filter performance regression, select
     the filter with highest predicted precision (no regime label used)

Baselines: per-source heuristic router (computed at pool build), best fixed
filter selected on training folds, oracle regime router, oracle filter.
Primary pre-registered model: logistic regression (per verification report);
random forest / LightGBM reported as secondary.

Usage: python scripts/routing/run_crossed_router.py
Outputs -> outputs/router_upgrade/crossed_*.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.routing import (
    BASE_FEATS,
    FIXED,
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


def cv_folds(groups, rep):
    rng = np.random.default_rng(1000 + rep)
    uniq = rng.permutation(np.unique(groups))
    gmap = {g: k % 5 for k, g in enumerate(uniq)}
    return np.array([gmap[g] for g in groups])


def fit_predict(model_name, X, y, tr, te, rep=0):
    m = make_models(seed=rep)[model_name]
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def main() -> None:
    f = pd.read_csv(OUT / "crossed_features.csv")
    md = pd.read_csv(OUT / "crossed_methods.csv")
    ext = [c for c in f.columns if c not in BASE_FEATS +
           ["instance", "regime", "source", "config_id", "pred_hard"]]
    sets = {"base": BASE_FEATS, "all": BASE_FEATS + ext}
    y = f["regime"].to_numpy()
    groups = f["config_id"].to_numpy()
    insts = f["instance"].to_numpy()
    src = f["source"].to_numpy()
    print(f"crossed pool: {len(f)} instances, {f.config_id.nunique()} configs")
    print(f.groupby(['regime', 'source']).size().unstack(fill_value=0).to_string())

    # ---- D: source-shortcut ceiling ----
    ceiling = 0.0
    for s in np.unique(src):
        sub = y[src == s]
        ceiling += max(np.mean(sub == r) for r in REGIMES) * np.mean(src == s)
    heur_acc = float((f["pred_hard"] == y).mean())
    print(f"\nsource-shortcut ceiling (predict regime from source alone): {ceiling:.3f}")
    print(f"per-source heuristic router accuracy: {heur_acc:.3f}")

    # ---- A: grouped CV ----
    rows = []
    oof = {}
    for sname, cols in sets.items():
        X = f[cols].to_numpy(float)
        for mname in ["logreg", "rf", "lightgbm"]:
            accs, bals = [], []
            for rep in range(10):
                fold = cv_folds(groups, rep)
                pred = np.empty(len(y), dtype=object)
                for k in range(5):
                    pred[fold == k] = fit_predict(mname, X, y, fold != k, fold == k, rep)
                accs.append(float((pred == y).mean()))
                bals.append(float(np.mean([(pred[y == r] == r).mean() for r in REGIMES])))
                if rep == 0:
                    oof[(sname, mname)] = (dict(zip(insts, pred)), fold)
            rows.append({"protocol": "grouped_cv", "feat_set": sname, "model": mname,
                         "acc_mean": np.mean(accs), "acc_sd": np.std(accs),
                         "bal_mean": np.mean(bals)})
    acc_df = pd.DataFrame(rows)
    acc_df.to_csv(OUT / "crossed_cv_results.csv", index=False)
    print("\n== Grouped CV routing accuracy ==")
    print(acc_df.sort_values("acc_mean", ascending=False).to_string(index=False))

    # ---- downstream for pre-registered primary (logreg / base and all) ----
    y_map = dict(zip(insts, y))
    heur_dec = {i: REGIME_FILTER[p] for i, p in zip(insts, f["pred_hard"])}
    orc_dec = {i: REGIME_FILTER[y_map[i]] for i in insts}
    ds_rows, sig_rows = [], []
    for sname in sets:
        pred, fold = oof[(sname, "logreg")]
        dec = {i: REGIME_FILTER[p] for i, p in pred.items()}
        inst_fold = dict(zip(insts, fold))
        bf = {k: best_fixed_on(md, [i for i in insts if inst_fold[i] != k]) for k in range(5)}
        bf_dec = {i: bf[inst_fold[i]] for i in insts}
        ds_rows.append({"feat_set": sname,
                        "learned": pooled_precision(md, dec),
                        "heuristic": pooled_precision(md, heur_dec),
                        "best_fixed_cv": pooled_precision(md, bf_dec),
                        "oracle_router": pooled_precision(md, orc_dec)})
        for lbl, a, b in (("learned-heuristic", dec, heur_dec),
                          ("learned-bestfixed", dec, bf_dec),
                          ("oracle-bestfixed", orc_dec, bf_dec),
                          ("learned-oracle", dec, orc_dec)):
            pt, lo, hi = paired_boot_diff(md, a, b)
            sig_rows.append({"feat_set": sname, "comparison": lbl, "gain_pts": round(100 * pt, 2),
                             "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2),
                             "significant": lo > 0 or hi < 0})
    pd.DataFrame(ds_rows).to_csv(OUT / "crossed_downstream.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(OUT / "crossed_significance.csv", index=False)
    print("\n== Downstream pooled precision (logreg, rep-0 OOF) ==")
    print(pd.DataFrame(ds_rows).to_string(index=False))
    print(pd.DataFrame(sig_rows).to_string(index=False))

    # ---- B: leave-one-source-out ----
    print("\n== Leave-one-source-out (train one source, test the other) ==")
    loso_rows = []
    for test_src in ("uf", "rb"):
        te = src == test_src
        tr = ~te
        for sname, cols in sets.items():
            X = f[cols].to_numpy(float)
            for mname in ["logreg", "lightgbm"]:
                pred = fit_predict(mname, X, y, tr, te)
                accs = float((pred == y[te]).mean())
                bal = float(np.mean([(pred[y[te] == r] == r).mean() for r in REGIMES if (y[te] == r).any()]))
                dec = {i: REGIME_FILTER[p] for i, p in zip(insts[te], pred)}
                sub_h = {i: heur_dec[i] for i in insts[te]}
                sub_o = {i: orc_dec[i] for i in insts[te]}
                bf_tr = best_fixed_on(md, list(insts[tr]))
                sub_b = {i: bf_tr for i in insts[te]}
                loso_rows.append({"test_source": test_src, "feat_set": sname, "model": mname,
                                  "accuracy": accs, "balanced": bal,
                                  "ds_learned": pooled_precision(md, dec),
                                  "ds_heuristic": pooled_precision(md, sub_h),
                                  "ds_bestfixed_train": pooled_precision(md, sub_b),
                                  "ds_oracle": pooled_precision(md, sub_o)})
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(OUT / "crossed_loso.csv", index=False)
    print(loso.to_string(index=False))

    # ---- C: leave-one-rate-out and CFI transfer ----
    print("\n== Leave-one-rate-out / CFI transfer (logreg) ==")
    strs_rows = []
    for rate in ["0.05", "0.1", "0.15", "0.2"]:
        te = pd.Series(groups).str.endswith(f"_r{rate}").to_numpy()
        if te.sum() == 0:
            continue
        for sname, cols in sets.items():
            X = f[cols].to_numpy(float)
            pred = fit_predict("logreg", X, y, ~te, te)
            strs_rows.append({"split": f"leave-rate-{rate}", "feat_set": sname,
                              "n_test": int(te.sum()), "accuracy": float((pred == y[te]).mean())})
    te = pd.Series(groups).str.contains("globalCFI").to_numpy()
    for sname, cols in sets.items():
        X = f[cols].to_numpy(float)
        pred = fit_predict("logreg", X, y, ~te, te)
        strs_rows.append({"split": "flipgens->CFI", "feat_set": sname,
                          "n_test": int(te.sum()), "accuracy": float((pred == y[te]).mean())})
    strs = pd.DataFrame(strs_rows)
    strs.to_csv(OUT / "crossed_stress.csv", index=False)
    print(strs.to_string(index=False))

    # ---- E: Task 4 — per-filter performance regression ----
    print("\n== Alternative objective: predict per-filter precision, pick argmax ==")
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    prec = md.pivot_table(index="instance", columns="method", values="precision")
    alt_rows = []
    for sname, cols in sets.items():
        X = f[cols].to_numpy(float)
        for reg_name, mk in (("ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                             ("lgbm_reg", lambda: LGBMRegressor(n_estimators=300, max_depth=4,
                                                                learning_rate=0.1, verbose=-1))):
            fold = cv_folds(groups, 0)
            pred_filter = {}
            for k in range(5):
                tr, te = fold != k, fold == k
                preds = {}
                for filt in FIXED:
                    target = prec.loc[insts, filt].to_numpy()
                    r = mk()
                    r.fit(X[tr], target[tr])
                    preds[filt] = r.predict(X[te])
                choice = np.array(FIXED)[np.argmax(np.column_stack([preds[fl] for fl in FIXED]), axis=1)]
                for i, c in zip(insts[te], choice):
                    pred_filter[i] = c
            ds_alt = pooled_precision(md, pred_filter)
            agree_oracle = float(np.mean([pred_filter[i] == orc_dec[i] for i in insts]))
            alt_rows.append({"feat_set": sname, "regressor": reg_name,
                             "ds_direct_selection": ds_alt, "picks_oracle_filter": agree_oracle})
    alt = pd.DataFrame(alt_rows)
    alt.to_csv(OUT / "crossed_alt_objective.csv", index=False)
    print(alt.to_string(index=False))


if __name__ == "__main__":
    main()

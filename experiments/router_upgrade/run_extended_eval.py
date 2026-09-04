#!/usr/bin/env python
"""Extended routing evaluation across three judge-vote sources (uf, rb, pku).

Sections:
  A  regime identification on the 300-instance pure pool: grouped CV + LOSO
     over three sources
  B  deployment on 456 mixed instances with a full metric matrix; two training
     regimes: in-domain (train on all pure) and cross-dataset (train on uf+rb
     pure only, deploy on pku mixed, which no router has ever seen)
  C  failure decomposition on mixed deployments: (A) wrong majority-regime
     prediction vs (B) right regime, wrong regime-to-filter mapping
  D  router improvements: confidence fallback and direct-prediction margin
     fallback (thresholds selected by CV on the pure pool ONLY), probability
     calibration
  E  router training-size study (labelled pure instances: 25..300)

Terminology (fixed here and in the paper):
  best fixed filter      pooled-precision argmax over FIXED on the training pool
  regime-oracle router   filter of the true (majority) regime
  per-instance oracle    per-instance best FIXED filter by realized precision
  keep-set oracle        gold-based keep mask (upper bound 1.0; not a router)

Usage: python experiments/router_upgrade/run_extended_eval.py
Outputs -> outputs/router_upgrade/ext_*.csv (+ per-instance artifact with
predictions, probabilities, selected and oracle filters)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/router_upgrade"
sys.path.insert(0, str(ROOT / "experiments/router_upgrade"))

from run_learned_router import (  # noqa: E402
    BASE_FEATS, FIXED, REGIME_FILTER, REGIMES, best_fixed_on, make_models,
    paired_boot_diff, pooled_precision)
from run_mixed_eval import META, pooled_confusion  # noqa: E402

TAU_GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MARGIN_GRID = [0.0, 0.001, 0.002, 0.005, 0.01]
TRAIN_SIZES = [25, 50, 100, 200, 300]


def cv_folds(groups, rep):
    rng = np.random.default_rng(1000 + rep)
    uniq = rng.permutation(np.unique(groups))
    gmap = {g: k % 5 for k, g in enumerate(uniq)}
    return np.array([gmap[g] for g in groups])


def feature_cols(df):
    ext = [c for c in df.columns if c not in BASE_FEATS + META]
    return {"base": BASE_FEATS, "all": BASE_FEATS + ext}


def direct_pred(tr_f, tr_md, te_f, cols, kind="ridge"):
    """Per-filter precision regressors trained on tr, argmax choice on te."""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    prec = tr_md.pivot_table(index="instance", columns="method", values="precision")
    prec = prec.loc[tr_f.instance]
    preds = {}
    for filt in FIXED:
        r = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if kind == "ridge"
             else LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.1, verbose=-1))
        r.fit(tr_f[cols].to_numpy(float), prec[filt].to_numpy())
        preds[filt] = r.predict(te_f[cols].to_numpy(float))
    P = np.column_stack([preds[f] for f in FIXED])
    return np.array(FIXED)[np.argmax(P, axis=1)], P


def metric_row(md, dec, orc_dec, fixed_dec, naive_dec):
    c = pooled_confusion(md, dec)
    p_orc = pooled_precision(md, {i: orc_dec[i] for i in dec})
    p_fix = pooled_precision(md, {i: fixed_dec[i] for i in dec})
    p_nai = pooled_precision(md, {i: naive_dec[i] for i in dec})
    head = p_orc - p_fix
    return {**{k: round(v, 4) for k, v in c.items()},
            "false_retention": round(1 - c["precision"], 4),
            "vs_naive_pts": round(100 * (c["precision"] - p_nai), 2),
            "vs_fixed_pts": round(100 * (c["precision"] - p_fix), 2),
            "oracle_gap_pts": round(100 * (p_orc - c["precision"]), 2),
            "headroom_recovered": round((c["precision"] - p_fix) / head, 3) if head > 1e-9 else np.nan}


def main() -> None:
    pure_f = pd.concat([pd.read_csv(OUT / "crossed_features.csv"),
                        pd.read_csv(OUT / "pku_features.csv")], ignore_index=True)
    pure_md = pd.concat([pd.read_csv(OUT / "crossed_methods.csv"),
                         pd.read_csv(OUT / "pku_methods.csv")], ignore_index=True)
    mix_f = pd.concat([pd.read_csv(OUT / "mixed_features.csv"),
                       pd.read_csv(OUT / "pku_mixed_features.csv")], ignore_index=True)
    mix_md = pd.concat([pd.read_csv(OUT / "mixed_methods.csv"),
                        pd.read_csv(OUT / "pku_mixed_methods.csv")], ignore_index=True)
    sets = feature_cols(pure_f)
    y_p = pure_f.regime.to_numpy()
    g_p = pure_f.config_id.to_numpy()
    src_p = pure_f.source.to_numpy()
    print(f"pure pool: {len(pure_f)} instances ({pd.Series(src_p).value_counts().to_dict()}); "
          f"mixed pool: {len(mix_f)}")

    # ---- A: identification ----
    rows = []
    rows.append({"protocol": "per-source refs", "router": "heuristic", "feat": "-",
                 "acc": round(float((pure_f.pred_hard == y_p).mean()), 3)})
    for sname, cols in sets.items():
        X = pure_f[cols].to_numpy(float)
        for mname in ["logreg", "lightgbm"]:
            accs = []
            for rep in range(10):
                fold = cv_folds(g_p, rep)
                pred = np.empty(len(y_p), dtype=object)
                for k in range(5):
                    m = make_models(seed=rep)[mname]
                    m.fit(X[fold != k], y_p[fold != k])
                    pred[fold == k] = m.predict(X[fold == k])
                accs.append(float((pred == y_p).mean()))
            rows.append({"protocol": "grouped CV", "router": mname, "feat": sname,
                         "acc": round(np.mean(accs), 3), "sd": round(np.std(accs), 3)})
            for hold in ("uf", "rb", "pku"):
                te = src_p == hold
                m = make_models(seed=0)[mname]
                m.fit(X[~te], y_p[~te])
                rows.append({"protocol": f"LOSO->{hold}", "router": mname, "feat": sname,
                             "acc": round(float((m.predict(X[te]) == y_p[te]).mean()), 3)})
    ident = pd.DataFrame(rows)
    ident.to_csv(OUT / "ext_ident.csv", index=False)
    print("\n== A: regime identification (300 pure instances, 3 sources) ==")
    print(ident.to_string(index=False))

    # ---- deployment decision builders ----
    def build_decisions(tr_f, tr_md, te_f):
        insts = te_f.instance.to_numpy()
        dec, proba = {}, {}
        for sname, cols in sets.items():
            m = make_models(seed=0)["logreg"]
            m.fit(tr_f[cols].to_numpy(float), tr_f.regime.to_numpy())
            pred = m.predict(te_f[cols].to_numpy(float))
            dec[f"logreg_{sname}"] = {i: REGIME_FILTER[p] for i, p in zip(insts, pred)}
            proba[f"logreg_{sname}"] = (m.classes_, m.predict_proba(te_f[cols].to_numpy(float)), pred)
        for kind in ("ridge", "lgbm"):
            choice, P = direct_pred(tr_f, tr_md, te_f, sets["all"], kind)
            dec[f"direct_{kind}"] = dict(zip(insts, choice))
            proba[f"direct_{kind}"] = P
        bf = best_fixed_on(tr_md, list(tr_f.instance))
        dec["best_fixed"] = dict.fromkeys(insts, bf)
        for filt in FIXED:
            dec[f"fixed_{filt}"] = dict.fromkeys(insts, filt)
        dec["heuristic"] = {i: REGIME_FILTER[p] for i, p in zip(insts, te_f.pred_hard)}
        maj = dict(zip(insts, te_f.majority_regime))
        dec["regime_oracle"] = {i: REGIME_FILTER[maj[i]] for i in insts if maj[i] != "none"}
        sub = mix_md[mix_md.method.isin(FIXED) & mix_md.instance.isin(insts)]
        orc = sub.sort_values(["precision", "n_kept"], ascending=False).drop_duplicates("instance")
        dec["perinstance_oracle"] = dict(zip(orc.instance, orc.method))
        return insts, dec, proba, bf

    def deploy_report(tag, tr_f, tr_md, te_f):
        insts, dec, proba, bf = build_decisions(tr_f, tr_md, te_f)
        md = mix_md[mix_md.instance.isin(insts)]
        orc_dec, fix_dec = dec["perinstance_oracle"], dec["best_fixed"]
        naive_dec = dec["fixed_naive_majority"]
        rows, sig = [], []
        for name, d in dec.items():
            if name.startswith("fixed_") and name != f"fixed_{bf}":
                label = name
            rows.append({"deploy": tag, "method": name, "n_inst": len(d),
                         **metric_row(md, d, orc_dec, fix_dec, naive_dec)})
        for name in ("heuristic", "logreg_base", "logreg_all", "direct_ridge",
                     "direct_lgbm", "regime_oracle", "perinstance_oracle"):
            d = dec[name]
            sub_fix = {i: fix_dec[i] for i in d}
            pt, lo, hi = paired_boot_diff(md, d, sub_fix)
            sig.append({"deploy": tag, "comparison": f"{name} - best_fixed",
                        "gain_pts": round(100 * pt, 2), "ci_lo": round(100 * lo, 2),
                        "ci_hi": round(100 * hi, 2), "significant": lo > 0 or hi < 0})
        return pd.DataFrame(rows), pd.DataFrame(sig), dec, proba

    # ---- B: two training regimes ----
    uf_rb_f = pure_f[pure_f.source != "pku"]
    uf_rb_md = pure_md[pure_md.instance.isin(uf_rb_f.instance)]
    pku_mix_f = mix_f[mix_f.source == "pku"]
    r1, s1, dec_all, proba_all = deploy_report("in-domain (train all pure, deploy all mixed)",
                                               pure_f, pure_md, mix_f)
    r2, s2, dec_x, proba_x = deploy_report("cross-dataset (train uf+rb pure, deploy pku mixed)",
                                           uf_rb_f, uf_rb_md, pku_mix_f)
    dep = pd.concat([r1, r2], ignore_index=True)
    sig = pd.concat([s1, s2], ignore_index=True)
    dep.to_csv(OUT / "ext_deploy_metrics.csv", index=False)
    sig.to_csv(OUT / "ext_significance.csv", index=False)
    print("\n== B: deployment metric matrix ==")
    print(dep.to_string(index=False))
    print(sig.to_string(index=False))

    # ---- C: failure decomposition (in-domain deployment, logreg_all) ----
    insts = mix_f.instance.to_numpy()
    classes, P, pred = proba_all["logreg_all"]
    prec_i = mix_md.pivot_table(index="instance", columns="method", values="precision")
    orc_dec = dec_all["perinstance_oracle"]
    maj = dict(zip(insts, mix_f.majority_regime))
    art = pd.DataFrame({
        "instance": insts, "source": mix_f.source, "mix_name": mix_f.mix_name,
        "rate": mix_f.rate, "w_weak": mix_f.w_weak, "w_global": mix_f.w_global,
        "w_subgroup": mix_f.w_subgroup, "majority_regime": mix_f.majority_regime,
        "pred_regime": pred, "conf": P.max(1),
        **{f"p_{c}": P[:, k] for k, c in enumerate(classes)},
        "chosen_filter": [REGIME_FILTER[p] for p in pred],
        "heuristic_filter": [REGIME_FILTER[p] for p in mix_f.pred_hard],
        "oracle_filter": [orc_dec[i] for i in insts],
        "best_fixed": dec_all["best_fixed"][insts[0]],
    })
    art["regret_pts"] = [100 * (prec_i.loc[i, orc_dec[i]] - prec_i.loc[i, c])
                         for i, c in zip(insts, art.chosen_filter)]
    has_maj = art.majority_regime != "none"
    art["bucket"] = np.where(~has_maj, "no-majority",
                     np.where(art.pred_regime != art.majority_regime, "A: wrong regime",
                     np.where(art.chosen_filter != art.oracle_filter,
                              "B: right regime, wrong mapping", "correct")))
    art.to_csv(OUT / "ext_perinstance.csv", index=False)
    dcmp = (art.groupby("bucket").agg(n=("instance", "size"),
                                      mean_regret_pts=("regret_pts", "mean"),
                                      total_regret=("regret_pts", "sum")).round(3)
            .reset_index())
    dcmp["regret_share"] = (dcmp.total_regret / dcmp.total_regret.sum()).round(3)
    dcmp.to_csv(OUT / "ext_decomposition.csv", index=False)
    print("\n== C: failure decomposition (logreg_all, 456 mixed instances) ==")
    print(dcmp.to_string(index=False))

    # ---- D: improvements; thresholds selected on PURE pool CV only ----
    X_all = pure_f[sets["all"]].to_numpy(float)
    fold = cv_folds(g_p, 0)
    oof_pred = np.empty(len(y_p), dtype=object)
    oof_conf = np.zeros(len(y_p))
    for k in range(5):
        m = make_models(seed=0)["logreg"]
        m.fit(X_all[fold != k], y_p[fold != k])
        Pk = m.predict_proba(X_all[fold == k])
        oof_pred[fold == k] = m.classes_[Pk.argmax(1)]
        oof_conf[fold == k] = Pk.max(1)
    bf_pure = best_fixed_on(pure_md, list(pure_f.instance))
    imp_rows = []
    best_tau, best_prec = 0.0, -1
    for tau in TAU_GRID:
        d = {i: (REGIME_FILTER[p] if c >= tau else bf_pure)
             for i, p, c in zip(pure_f.instance, oof_pred, oof_conf)}
        p = pooled_precision(pure_md, d)
        imp_rows.append({"variant": "conf_fallback", "threshold": tau,
                         "pure_cv_precision": round(p, 4),
                         "fallback_rate": round(float(np.mean(oof_conf < tau)), 3)})
        if p > best_prec:
            best_tau, best_prec = tau, p
    # apply frozen tau* to mixed
    classes, P, pred = proba_all["logreg_all"]
    conf = P.max(1)
    bf_all = dec_all["best_fixed"][insts[0]]
    d_fb = {i: (REGIME_FILTER[p] if c >= best_tau else bf_all)
            for i, p, c in zip(insts, pred, conf)}
    pt, lo, hi = paired_boot_diff(mix_md, d_fb, dec_all["best_fixed"])
    imp_rows.append({"variant": f"conf_fallback DEPLOYED tau*={best_tau}",
                     "mixed_precision": round(pooled_precision(mix_md, d_fb), 4),
                     "gain_vs_fixed_pts": round(100 * pt, 2),
                     "ci": f"[{100*lo:+.2f},{100*hi:+.2f}]"})
    # direct-pred margin fallback (margin selected on pure CV)
    prec_pure = pure_md.pivot_table(index="instance", columns="method", values="precision")
    prec_pure = prec_pure.loc[pure_f.instance]
    oof_P = np.zeros((len(y_p), len(FIXED)))
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for k in range(5):
        for fi, filt in enumerate(FIXED):
            r = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            r.fit(X_all[fold != k], prec_pure[filt].to_numpy()[fold != k])
            oof_P[fold == k, fi] = r.predict(X_all[fold == k])
    fi_bf = FIXED.index(bf_pure)
    best_m, best_prec_m = 0.0, -1
    for marg in MARGIN_GRID:
        choice = np.where(oof_P.max(1) - oof_P[:, fi_bf] > marg,
                          np.array(FIXED)[oof_P.argmax(1)], bf_pure)
        p = pooled_precision(pure_md, dict(zip(pure_f.instance, choice)))
        imp_rows.append({"variant": "direct_margin", "threshold": marg,
                         "pure_cv_precision": round(p, 4)})
        if p > best_prec_m:
            best_m, best_prec_m = marg, p
    Pmix = proba_all["direct_ridge"]
    fi_bf_all = FIXED.index(bf_all)
    choice = np.where(Pmix.max(1) - Pmix[:, fi_bf_all] > best_m,
                      np.array(FIXED)[Pmix.argmax(1)], bf_all)
    d_dm = dict(zip(insts, choice))
    pt, lo, hi = paired_boot_diff(mix_md, d_dm, dec_all["best_fixed"])
    imp_rows.append({"variant": f"direct_margin DEPLOYED m*={best_m}",
                     "mixed_precision": round(pooled_precision(mix_md, d_dm), 4),
                     "gain_vs_fixed_pts": round(100 * pt, 2),
                     "ci": f"[{100*lo:+.2f},{100*hi:+.2f}]"})
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(OUT / "ext_improvements.csv", index=False)
    print("\n== D: improvements (thresholds selected on pure-pool CV only) ==")
    print(imp.to_string(index=False))

    # ---- E: router training-size study ----
    ts_rows = []
    rng = np.random.default_rng(20260801)
    cfgs = np.unique(g_p)
    cfg_lab = {c: y_p[g_p == c][0] for c in cfgs}
    for n_inst in TRAIN_SIZES:
        for draw in range(5):
            take_cfg = []
            per = {r: [c for c in cfgs if cfg_lab[c] == r] for r in REGIMES}
            n_cfg = max(3, int(round(n_inst / (len(pure_f) / len(cfgs)))))
            for r in REGIMES:
                k = max(1, n_cfg // 3)
                take_cfg += list(rng.choice(per[r], min(k, len(per[r])), replace=False))
            tr_mask = pd.Series(g_p).isin(take_cfg).to_numpy()
            sub_f = pure_f[tr_mask]
            m = make_models(seed=draw)["logreg"]
            m.fit(sub_f[sets["all"]].to_numpy(float), sub_f.regime.to_numpy())
            pred_m = m.predict(mix_f[sets["all"]].to_numpy(float))
            d = {i: REGIME_FILTER[p] for i, p in zip(insts, pred_m)}
            ok = mix_f.majority_regime != "none"
            acc = float((pred_m[ok.to_numpy()] == mix_f.majority_regime.to_numpy()[ok.to_numpy()]).mean())
            ts_rows.append({"n_train_inst": int(tr_mask.sum()), "draw": draw,
                            "majority_match": round(acc, 3),
                            "mixed_precision": round(pooled_precision(mix_md, d), 4)})
    ts = pd.DataFrame(ts_rows)
    agg = ts.groupby("n_train_inst").agg(maj=("majority_match", "mean"),
                                         maj_sd=("majority_match", "std"),
                                         prec=("mixed_precision", "mean"),
                                         prec_sd=("mixed_precision", "std")).round(4)
    ts.to_csv(OUT / "ext_trainsize.csv", index=False)
    print("\n== E: router training-size study (mixed deployment) ==")
    print(agg.to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate routing on mixed-regime deployments (paper untouched).

Routers are trained ONLY on the pure-regime crossed pool (204 instances) and
deployed frozen on the 304 mixed-regime instances. Primary metrics are
downstream: pooled precision / accuracy / F1, per-instance regret vs the
deployable oracle (per-instance best fixed filter), paired instance-bootstrap
CIs. Routing accuracy vs the majority mixture regime is secondary.

Methods:
  fixed_best     single best filter chosen on the TRAINING pool
  heuristic      per-source clean-referenced hard router (computed at build)
  logreg_base/all  pre-registered logistic-regression regime router
  direct_ridge/lgbm  per-filter precision regression + argmax (no regime label)
  majority_filter  filter of the true majority regime (regime-level reference)
  oracle_choice  per-instance best FIXED filter by realized precision (bound)

Usage: python experiments/router_upgrade/run_mixed_eval.py
Outputs -> outputs/router_upgrade/mixed_{downstream,significance,by_mixture,
calibration,failure,routeacc}.csv and mixed_plots/*.png
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
PLOTS = OUT / "mixed_plots"
sys.path.insert(0, str(ROOT / "experiments/router_upgrade"))

from run_learned_router import (  # noqa: E402
    BASE_FEATS, FIXED, REGIME_FILTER, best_fixed_on, make_models,
    paired_boot_diff, pooled_precision)

META = ["instance", "regime", "source", "config_id", "pred_hard", "mix_name",
        "rate", "majority_regime", "w_weak", "w_global", "w_subgroup",
        "n_bad_weak", "n_bad_global", "n_bad_subgroup"]


def pooled_confusion(md, decisions):
    """Pooled precision/accuracy/F1/retention over decided filters."""
    idx = md.set_index(["instance", "method"])
    tp = fp = fn = tn = 0
    for inst, m in decisions.items():
        r = idx.loc[(inst, m)]
        kc, nk, nc, n = int(r.n_kept_clean), int(r.n_kept), int(r.n_clean), int(r.n_eval)
        tp += kc
        fp += nk - kc
        fn += nc - kc
        tn += n - nc - (nk - kc)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "retention": (tp + fp) / max(tp + fp + fn + tn, 1)}


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    tr_f = pd.read_csv(OUT / "crossed_features.csv")
    tr_md = pd.read_csv(OUT / "crossed_methods.csv")
    mx_f = pd.read_csv(OUT / "mixed_features.csv")
    mx_md = pd.read_csv(OUT / "mixed_methods.csv")
    ext = [c for c in tr_f.columns if c not in BASE_FEATS + META]
    sets = {"base": BASE_FEATS, "all": BASE_FEATS + ext}
    insts = mx_f.instance.to_numpy()

    # ---- frozen training on the pure-regime pool ----
    y_tr = tr_f.regime.to_numpy()
    dec = {}
    proba = {}
    for sname, cols in sets.items():
        m = make_models(seed=0)["logreg"]
        m.fit(tr_f[cols].to_numpy(float), y_tr)
        pred = m.predict(mx_f[cols].to_numpy(float))
        dec[f"logreg_{sname}"] = {i: REGIME_FILTER[p] for i, p in zip(insts, pred)}
        proba[f"logreg_{sname}"] = (m.classes_, m.predict_proba(mx_f[cols].to_numpy(float)), pred)

    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    prec_tr = tr_md.pivot_table(index="instance", columns="method", values="precision")
    prec_tr = prec_tr.loc[tr_f.instance]
    for rname, mk in (("ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                      ("lgbm", lambda: LGBMRegressor(n_estimators=300, max_depth=4,
                                                     learning_rate=0.1, verbose=-1))):
        preds = {}
        for filt in FIXED:
            r = mk()
            r.fit(tr_f[sets["all"]].to_numpy(float), prec_tr[filt].to_numpy())
            preds[filt] = r.predict(mx_f[sets["all"]].to_numpy(float))
        choice = np.array(FIXED)[np.argmax(np.column_stack([preds[f] for f in FIXED]), axis=1)]
        dec[f"direct_{rname}"] = dict(zip(insts, choice))

    dec["fixed_best"] = dict.fromkeys(insts, best_fixed_on(tr_md, list(tr_f.instance)))
    dec["heuristic"] = {i: REGIME_FILTER[p] for i, p in zip(insts, mx_f.pred_hard)}
    maj = dict(zip(insts, mx_f.majority_regime))
    dec["majority_filter"] = {i: REGIME_FILTER[maj[i]] for i in insts if maj[i] != "none"}

    # deployable oracle: per-instance best FIXED filter by realized precision
    sub = mx_md[mx_md.method.isin(FIXED)].sort_values(["precision", "n_kept"], ascending=False)
    orc = sub.drop_duplicates("instance").set_index("instance")
    dec["oracle_choice"] = {i: orc.loc[i, "method"] for i in insts}

    # ---- downstream table ----
    prec_i = mx_md.pivot_table(index="instance", columns="method", values="precision")
    orc_prec = {i: prec_i.loc[i, dec["oracle_choice"][i]] for i in insts}
    rows = []
    for name, d in dec.items():
        c = pooled_confusion(mx_md, d)
        reg = np.array([orc_prec[i] - prec_i.loc[i, d[i]] for i in d])
        rows.append({"method": name, "n_inst": len(d), **{k: round(v, 4) for k, v in c.items()},
                     "mean_regret_pts": round(100 * reg.mean(), 3),
                     "p90_regret_pts": round(100 * np.quantile(reg, 0.9), 3)})
    ds = pd.DataFrame(rows).sort_values("precision", ascending=False)
    ds.to_csv(OUT / "mixed_downstream.csv", index=False)
    print("== Downstream on 304 mixed deployments ==")
    print(ds.to_string(index=False))

    # ---- significance (paired instance bootstrap) ----
    comps = [("logreg_all", "fixed_best"), ("logreg_all", "heuristic"),
             ("logreg_base", "fixed_best"), ("direct_lgbm", "fixed_best"),
             ("direct_ridge", "fixed_best"), ("heuristic", "fixed_best"),
             ("oracle_choice", "fixed_best"), ("logreg_all", "oracle_choice"),
             ("logreg_all", "direct_lgbm")]
    sig = []
    for a, b in comps:
        pt, lo, hi = paired_boot_diff(mx_md, dec[a], dec[b])
        sig.append({"comparison": f"{a} - {b}", "gain_pts": round(100 * pt, 2),
                    "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2),
                    "significant": lo > 0 or hi < 0})
    sig = pd.DataFrame(sig)
    sig.to_csv(OUT / "mixed_significance.csv", index=False)
    print("\n== Paired bootstrap (pooled precision, 95% CI) ==")
    print(sig.to_string(index=False))

    # ---- per-mixture breakdown + consistency ----
    fam = mx_f.mix_name.str.rsplit("_", n=1)
    mx_f["family"] = np.where(mx_f.mix_name.str.startswith("3way"), "3way", fam.str[0])
    mx_f["share"] = np.where(mx_f.family == "3way", mx_f.mix_name.str[5:],
                             fam.str[1])
    brows = []
    for (family, share), g in mx_f.groupby(["family", "share"]):
        ii = g.instance.tolist()
        row = {"family": family, "share": share, "n": len(ii)}
        for name in ("fixed_best", "heuristic", "logreg_all", "direct_lgbm", "oracle_choice"):
            row[name] = round(pooled_precision(mx_md, {i: dec[name][i] for i in ii}), 4)
        smd = mx_md[mx_md.instance.isin(ii)]
        pt, lo, hi = paired_boot_diff(smd, {i: dec["logreg_all"][i] for i in ii},
                                      {i: dec["fixed_best"][i] for i in ii})
        row.update({"lr_vs_fixed_pts": round(100 * pt, 2), "ci_lo": round(100 * lo, 2),
                    "ci_hi": round(100 * hi, 2)})
        brows.append(row)
    bym = pd.DataFrame(brows)
    bym.to_csv(OUT / "mixed_by_mixture.csv", index=False)
    print("\n== By mixture (pooled precision; logreg_all - fixed_best CI) ==")
    print(bym.to_string(index=False))

    # ---- secondary: routing accuracy vs majority regime ----
    ok = mx_f.majority_regime != "none"
    racc = []
    for name in ("heuristic", "logreg_base", "logreg_all"):
        if name == "heuristic":
            pred = mx_f.pred_hard.to_numpy()
        else:
            pred = proba[name][2] if name in proba else None
        agree = (pred[ok.to_numpy()] == mx_f.majority_regime.to_numpy()[ok.to_numpy()]).mean()
        racc.append({"router": name, "majority_match": round(float(agree), 4), "n": int(ok.sum())})
    racc = pd.DataFrame(racc)
    racc.to_csv(OUT / "mixed_routeacc.csv", index=False)
    print("\n== Secondary: agreement with majority regime (n=272) ==")
    print(racc.to_string(index=False))

    # ---- calibration of logreg_all confidence ----
    classes, P, pred = proba["logreg_all"]
    conf = P.max(1)
    picked = np.array([dec["logreg_all"][i] for i in insts])
    orc_f = np.array([dec["oracle_choice"][i] for i in insts])
    near_opt = np.array([prec_i.loc[i, p] >= orc_prec[i] - 1e-9
                         for i, p in zip(insts, picked)])
    bins = np.quantile(conf, np.linspace(0, 1, 6))
    bins[0], bins[-1] = 0, 1.0001
    crows = []
    for k in range(5):
        m = (conf >= bins[k]) & (conf < bins[k + 1])
        crows.append({"conf_bin": f"[{bins[k]:.2f},{bins[k+1]:.2f})", "n": int(m.sum()),
                      "mean_conf": round(float(conf[m].mean()), 3),
                      "picked_optimal_rate": round(float(near_opt[m].mean()), 3),
                      "majority_match_rate": round(float((pred[m & ok.to_numpy()] ==
                                                          mx_f.majority_regime.to_numpy()[m & ok.to_numpy()]).mean()), 3)})
    cal = pd.DataFrame(crows)
    cal.to_csv(OUT / "mixed_calibration.csv", index=False)
    print("\n== logreg_all confidence vs outcomes ==")
    print(cal.to_string(index=False))
    # ambiguity: mixture entropy vs correctness
    W = mx_f[["w_weak", "w_global", "w_subgroup"]].to_numpy()
    Went = -(W * np.log(np.maximum(W, 1e-12))).sum(1)
    Pent = -(P * np.log(np.maximum(P, 1e-12))).sum(1)
    from scipy.stats import pearsonr
    r1 = pearsonr(Went, Pent)
    r2 = pearsonr(Went[ok], (pred[ok.to_numpy()] == mx_f.majority_regime.to_numpy()[ok.to_numpy()]).astype(float))
    print(f"corr(mixture entropy, router entropy) = {r1.statistic:.3f} (p={r1.pvalue:.1e})")
    print(f"corr(mixture entropy, majority-match) = {r2.statistic:.3f} (p={r2.pvalue:.1e})")

    # ---- failure analysis: regret by config for learned router ----
    mx_f["regret_lr"] = [100 * (orc_prec[i] - prec_i.loc[i, dec["logreg_all"][i]]) for i in insts]
    mx_f["regret_fx"] = [100 * (orc_prec[i] - prec_i.loc[i, dec["fixed_best"][i]]) for i in insts]
    fail = (mx_f.groupby(["family", "share", "rate"])[["regret_lr", "regret_fx"]]
            .mean().round(3).reset_index().sort_values("regret_lr", ascending=False))
    fail.to_csv(OUT / "mixed_failure.csv", index=False)
    print("\n== Worst mean regret (pts) for logreg_all by config ==")
    print(fail.head(10).to_string(index=False))

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"fixed_best": "#2a78d6", "heuristic": "#eb6834", "logreg_all": "#1baf7a",
           "direct_lgbm": "#eda100", "oracle_choice": "#52514e"}
    LBL = {"fixed_best": "Best fixed filter", "heuristic": "Heuristic router",
           "logreg_all": "LogReg router (pre-reg.)", "direct_lgbm": "Direct perf. prediction",
           "oracle_choice": "Oracle (per-instance best)"}
    fams = [("wg", "weak + global", "share of global"),
            ("ws", "weak + subgroup", "share of subgroup"),
            ("gs", "global + subgroup", "share of subgroup")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for ax, (fam_k, title, xlab) in zip(axes, fams):
        sub = bym[bym.family == fam_k].copy()
        sub["x"] = sub.share.astype(int)
        sub = sub.sort_values("x")
        for name in COL:
            ls = "--" if name == "oracle_choice" else "-"
            ax.plot(sub.x, sub[name], ls, color=COL[name], lw=2,
                    marker="o", ms=4, label=LBL[name])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlab + " (%)", fontsize=9)
        ax.set_xticks([10, 25, 50, 75, 90])
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("pooled downstream precision", fontsize=9)
    axes[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "mixed_precision_by_share.png", dpi=200)

    fig2, ax = plt.subplots(figsize=(6.4, 3.4))
    piv = fail.groupby(["family", "share"])[["regret_lr", "regret_fx"]].mean().reset_index()
    xs = np.arange(len(piv))
    ax.bar(xs - 0.2, piv.regret_fx, 0.38, color="#2a78d6", label="Best fixed filter")
    ax.bar(xs + 0.2, piv.regret_lr, 0.38, color="#1baf7a", label="LogReg router")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f}-{s}" for f, s in zip(piv.family, piv.share)],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean regret vs oracle (pts)", fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    fig2.savefig(PLOTS / "mixed_regret_by_mixture.png", dpi=200)

    fig3, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(cal.mean_conf, cal.majority_match_rate, "-o", color="#2a78d6", lw=2, ms=5,
            label="matched majority regime")
    ax.plot(cal.mean_conf, cal.picked_optimal_rate, "-o", color="#eb6834", lw=2, ms=5,
            label="chose instance-optimal filter")
    ax.plot([0.55, 1.0], [0.55, 1.0], ":", color="#52514e", lw=1, label="perfect calibration")
    ax.set_xlabel("router confidence (bin mean)", fontsize=9)
    ax.set_ylabel("empirical rate", fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig3.tight_layout()
    fig3.savefig(PLOTS / "mixed_calibration.png", dpi=200)
    print(f"\nplots -> {PLOTS}")


if __name__ == "__main__":
    main()

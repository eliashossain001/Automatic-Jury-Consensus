"""Bucket E — dev-set supervised router, leave-one-deployment-out (honest, anti-overfit).

There are only 9 deployment instances (3 regimes x 3), so this is a small-sample check, not
a deployable model. We train a logistic-regression regime classifier on the unsupervised
diagnostics with leave-one-out CV, map the predicted regime to its matched filter, and
compare the resulting false-retention against (a) the best single filter (CorrFilter) and
(b) the oracle regime-matched selection. We report the gap honestly with the n=9 caveat.

Outputs outputs/router_dev/table_router_comparison.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "outputs/regime_router/regime_diagnostics.csv"
RES = ROOT / "outputs/regime_router/regime_router_results.csv"
REGIME_FILTER = {"weak": "supermajority_75", "global": "corrfilter_small_gold_R", "subgroup": "bias_cluster"}
BEST_SINGLE = "corrfilter_small_gold_R"
NAIVE = "naive_majority"
FEATS = ["rho_bar", "n_eff", "eig_overlap", "conc", "rho_S", "disagree", "lco_flip"]


def frr_for(res, dataset, method):
    r = res[(res.dataset == dataset) & (res.method == method)]
    return float(r["false_retention_rate"].iloc[0]) if len(r) else np.nan


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    diag = pd.read_csv(DIAG)
    res = pd.read_csv(RES)
    X = diag[FEATS].values
    y = diag["true_regime"].values
    n = len(diag)

    # leave-one-deployment-out regime classification
    preds = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        preds.append(clf.predict(sc.transform(X[i:i+1]))[0])
    diag["pred_regime"] = preds
    loo_acc = float((diag.pred_regime == diag.true_regime).mean())

    # filter selection FRR under each policy
    rows = []
    for _, r in diag.iterrows():
        ds = r["dataset"]
        rows.append({
            "dataset": ds, "true_regime": r["true_regime"], "pred_regime": r["pred_regime"],
            "frr_router_dev": frr_for(res, ds, REGIME_FILTER[r["pred_regime"]]),
            "frr_oracle_regime": frr_for(res, ds, REGIME_FILTER[r["true_regime"]]),
            "frr_best_single_corrfilter": frr_for(res, ds, BEST_SINGLE),
            "frr_naive_consensus": frr_for(res, ds, NAIVE),
        })
    comp = pd.DataFrame(rows)
    out = ROOT / "outputs/router_dev"; out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "table_router_comparison.csv", index=False)

    means = comp[["frr_router_dev", "frr_oracle_regime", "frr_best_single_corrfilter", "frr_naive_consensus"]].mean()
    # paired bootstrap: router_dev - best_single (negative = router better)
    d = (comp.frr_router_dev - comp.frr_best_single_corrfilter).values
    rng = np.random.default_rng(20260708)
    bo = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    print(comp.to_string(index=False))
    print(f"\nLeave-one-out regime accuracy: {loo_acc:.3f} (n={n})")
    print("mean FRR: router_dev=%.3f  oracle=%.3f  best_single(CorrFilter)=%.3f  naive=%.3f"
          % (means.frr_router_dev, means.frr_oracle_regime, means.frr_best_single_corrfilter, means.frr_naive_consensus))
    print("router_dev - best_single mean %.4f, 95%% CI [%.4f, %.4f], P(router better)=%.2f"
          % (d.mean(), np.quantile(bo, .025), np.quantile(bo, .975), (bo < 0).mean()))
    print(f"-> {out}/table_router_comparison.csv")


if __name__ == "__main__":
    main()

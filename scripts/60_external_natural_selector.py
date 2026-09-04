#!/usr/bin/env python
"""Frozen deployment selector on natural external AggreFact deployments.

Training uses only the 456 preference deployments produced by script 59.
Evaluation uses eight human-labelled AggreFact components with cached natural
votes (no error injection and no API/model inference). Each component is
re-evaluated under 20 deterministic 100-calibration/39--50-held-out splits.
Uncertainty is clustered by the eight dataset components, so repeated splits
are never treated as independent evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import datasets  # initialize optional torch detection before lightweight shim

if importlib.util.find_spec("torch") is None:
    t = types.ModuleType("torch")
    t.dtype = object
    t.bfloat16, t.float16, t.float32 = object(), object(), object()
    t.inference_mode = lambda: (lambda fn: fn)
    sys.modules["torch"] = t

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "router_upgrade"
OUT = ROOT / "outputs" / "deployment_selector"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP))

from build_crossed_pool import Source, evaluate  # noqa: E402
from corrfilter.cfi.adaptive_r import disagreement_R  # noqa: E402
from corrfilter.cfi.consensus import majority_consensus  # noqa: E402
from corrfilter.data.generic_task import batch_from_cache  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.routing import agree_features, correlation_cluster  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

FIXED = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
BASE = ["rho_bar", "n_eff", "eig_overlap", "lco_mean", "conc_mean", "rho_S_mean", "disagree_mean"]
GOLD = ["gold_rho_bar", "gold_n_eff", "gold_top1_share", "gold_top2_share",
        "gold_eig_rank_ratio", "gold_error_rate_mean", "gold_error_rate_std",
        "gold_error_rate_min", "gold_error_rate_max"]
CORE = BASE + GOLD
CAL_PERF = [f"cal_precision__{m}" for m in FIXED]
N_SPLITS = 20
N_BOOT = 20_000
SEED = 20260831


def load_source():
    root = ROOT / "outputs" / "screening_bank"
    man = pd.read_parquet(root / "manifest_aggrefact_screen.parquet")
    bank = load_bank_config(ROOT / "configs" / "judge_bank.yaml")
    models = sorted({s.base_id for s in bank.specs})
    lids = [f"{m}::{style}" for m in models for style in ("pw_direct", "pw_analysis")]
    batch = batch_from_cache(man, VoteCache(root / "votes" / "aggrefact"), lids)
    V, M, gold = batch.judge_votes, batch.availability, batch.gold_labels
    R = disagreement_R(V, M)
    src = Source("aggrefact", V, M, np.zeros_like(V), man.doc_chars.to_numpy(),
                 man.claim_chars.to_numpy(), lids, R)
    src.cpos = correlation_cluster(R, k=5)
    label = majority_consensus(V, M)
    af = agree_features(V, M, label, src.cpos, R)
    off = R[~np.eye(R.shape[0], dtype=bool)]
    src.ref = {"rho_bar": float(off.mean()), "rho_S": float(np.nanmean(af["rho"])),
               "lco_flip": float(np.nanmean(af["lco_flip"]))}
    return src, R, gold, man


def build_external():
    src, R, gold, man = load_source()
    frows, mrows = [], []
    components = sorted(man.source_dataset.unique())
    for ci, comp in enumerate(components):
        ids = np.where(man.source_dataset.to_numpy() == comp)[0]
        if len(ids) < 120:
            continue
        for split in range(N_SPLITS):
            counter = 400_000 + ci * 100 + split
            metrics, fvec, _, _ = evaluate(
                src, src.V[ids], src.M[ids], gold[ids], counter, R,
                retention_mode="affirmed", selector_features_from_calibration=True,
            )
            inst = f"{comp}_split{split:02d}"
            fvec.update({"instance": inst, "component": comp, "split": split,
                         "n_items": len(ids)})
            frows.append(fvec)
            for method, metric in metrics.items():
                mrows.append({"instance": inst, "component": comp,
                              "method": method, **metric})
    f = pd.DataFrame(frows)
    m = pd.DataFrame(mrows)
    f.to_csv(OUT / "external_natural_features.csv", index=False)
    m.to_csv(OUT / "external_natural_methods.csv", index=False)
    return f, m


def fit_selector(train_f, train_m, test_f):
    target = train_m[train_m.method.isin(FIXED)].pivot_table(
        index="instance", columns="method", values="precision").loc[train_f.instance, FIXED]
    pred = []
    for method in FIXED:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_f[CORE].to_numpy(float), target[method].to_numpy(float))
        pred.append(model.predict(test_f[CORE].to_numpy(float)))
    P = np.column_stack(pred)
    return np.asarray(FIXED)[P.argmax(1)], P


def best_fixed(train_m):
    sub = train_m[train_m.method.isin(FIXED)].groupby("method")[["n_kept", "n_kept_clean"]].sum()
    return max(FIXED, key=lambda m: (sub.loc[m, "n_kept_clean"] / max(sub.loc[m, "n_kept"], 1),
                                     -FIXED.index(m)))


def oracle_choices(methods):
    sub = methods[methods.method.isin(FIXED)].copy()
    order = {m: i for i, m in enumerate(FIXED)}
    sub["ord"] = sub.method.map(order)
    win = sub.sort_values(["instance", "precision", "n_kept", "ord"],
                          ascending=[True, False, False, True]).drop_duplicates("instance")
    return dict(zip(win.instance, win.method))


def component_boot(per_instance, a, b):
    d = per_instance[a] - per_instance[b]
    dc = d.groupby(per_instance.component).mean()
    rng = np.random.default_rng(SEED)
    vals = dc.to_numpy(float)
    boot = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(vals.mean()), float(lo), float(hi)


def evaluate_external(f, m):
    train_f = pd.read_csv(OUT / "strict_features.csv")
    train_m = pd.read_csv(OUT / "strict_methods.csv")
    choice, P = fit_selector(train_f, train_m, f)
    bf = best_fixed(train_m)
    oracle = oracle_choices(m)
    cal = np.asarray(FIXED)[f[CAL_PERF].to_numpy(float).argmax(1)]
    # Non-deployable distribution-level reference: the best single filter
    # selected with all external labels. It shows whether the frozen selector
    # recovered the correct action under aggregation-rule shift.
    ext = m[m.method.isin(FIXED)].groupby("method").precision.mean()
    external_best = max(FIXED, key=lambda method: (ext.loc[method], -FIXED.index(method)))
    decisions = {
        "frozen_selector": dict(zip(f.instance, choice)),
        "transferred_fixed": dict.fromkeys(f.instance, bf),
        "calibration_best": dict(zip(f.instance, cal)),
        "external_best_fixed_oracle": dict.fromkeys(f.instance, external_best),
        "perdeployment_oracle": oracle,
    }
    piv = m[m.method.isin(FIXED)].pivot_table(index="instance", columns="method", values="precision")
    rows = []
    for _, r in f.iterrows():
        row = {"instance": r.instance, "component": r.component, "split": r.split}
        for name, dec in decisions.items():
            row[name] = float(piv.loc[r.instance, dec[r.instance]])
            row[f"choice__{name}"] = dec[r.instance]
        rows.append(row)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "external_natural_per_split.csv", index=False)

    summary = []
    for name in decisions:
        component_means = per.groupby("component")[name].mean()
        summary.append({"method": name, "component_balanced_precision": component_means.mean(),
                        "component_sd": component_means.std(),
                        "min_component": component_means.min(),
                        "max_component": component_means.max()})
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "external_natural_summary.csv", index=False)

    sig = []
    for a, b in [("frozen_selector", "transferred_fixed"),
                 ("calibration_best", "transferred_fixed"),
                 ("external_best_fixed_oracle", "transferred_fixed"),
                 ("perdeployment_oracle", "transferred_fixed"),
                 ("frozen_selector", "external_best_fixed_oracle"),
                 ("frozen_selector", "perdeployment_oracle")]:
        pt, lo, hi = component_boot(per, a, b)
        sig.append({"comparison": f"{a} - {b}", "gain_pts": 100 * pt,
                    "ci95_low": 100 * lo, "ci95_high": 100 * hi,
                    "n_components": per.component.nunique(),
                    "significant": bool(lo > 0 or hi < 0)})
    sig = pd.DataFrame(sig)
    sig.to_csv(OUT / "external_natural_significance.csv", index=False)

    bycomp = per.groupby("component").agg(
        n_splits=("split", "size"), selector=("frozen_selector", "mean"),
        fixed=("transferred_fixed", "mean"), oracle=("perdeployment_oracle", "mean"),
    ).reset_index()
    bycomp["selector_gain_pts"] = 100 * (bycomp.selector - bycomp.fixed)
    bycomp["oracle_headroom_pts"] = 100 * (bycomp.oracle - bycomp.fixed)
    bycomp.to_csv(OUT / "external_natural_by_component.csv", index=False)

    choice_counts = {name: per[f"choice__{name}"].value_counts().to_dict() for name in decisions}
    (OUT / "external_natural_choices.json").write_text(json.dumps(choice_counts, indent=2))
    return summ, sig, bycomp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, m = build_external()
    assert f.component.nunique() == 8 and len(f) == 8 * N_SPLITS
    summ, sig, bycomp = evaluate_external(f, m)
    print("External natural AggreFact summary")
    print(summ.to_string(index=False))
    print("\nComponent-clustered comparisons")
    print(sig.to_string(index=False))
    print("\nBy component")
    print(bycomp.to_string(index=False))


if __name__ == "__main__":
    main()

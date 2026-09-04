#!/usr/bin/env python
"""Leakage-safe deployment-level filter selection.

This is the strict version of the experiment proposed in review:

* one deployment contains 300 items;
* 100 trusted calibration items produce every selector feature and CorrFilter R;
* the remaining 200 items are used only to measure filter outcomes;
* a selector predicts held-out precision for each candidate filter;
* grouped outer folds hold out an entire mixture/rate configuration across all
  three data sources, so related deployment replicates never cross a fold.

The per-deployment oracle is an upper bound over the same fixed filter menu and
cannot legitimately be beaten. The regime oracle is not an upper bound: it maps
true majority regime to a prescribed filter and can be beaten by direct outcome
prediction.

No model inference or API calls are made; the pool is reconstructed from cached
votes. Outputs are written to outputs/deployment_selector/.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import datasets  # initialize optional-dependency detection before the torch stub

# Cached-vote analysis does not use PyTorch. The package exposes HFJudge eagerly,
# so provide only the dtype names needed while importing the routing utilities in
# lightweight CPU environments where torch is intentionally absent.
if importlib.util.find_spec("torch") is None:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = object
    torch_stub.bfloat16 = object()
    torch_stub.float16 = object()
    torch_stub.float32 = object()
    torch_stub.inference_mode = lambda: (lambda fn: fn)
    sys.modules["torch"] = torch_stub

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "router_upgrade"
OUT = ROOT / "outputs" / "deployment_selector"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP))

from build_crossed_pool import evaluate, load_sources  # noqa: E402
from build_mixed_pool import make_mixed_instance, majority_regime, mixtures  # noqa: E402
from extend_pku import load_pku_source  # noqa: E402

FIXED = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
REGIME_FILTER = {
    "weak": "supermajority_75",
    "global": "corrfilter_small_gold_R",
    "subgroup": "bias_cluster",
}
BASE = ["rho_bar", "n_eff", "eig_overlap", "lco_mean", "conc_mean", "rho_S_mean", "disagree_mean"]
GOLD = [
    "gold_rho_bar", "gold_n_eff", "gold_top1_share", "gold_top2_share",
    "gold_eig_rank_ratio", "gold_error_rate_mean", "gold_error_rate_std",
    "gold_error_rate_min", "gold_error_rate_max",
]
CAL_PERF = [f"cal_precision__{method}" for method in FIXED]
META = {
    "instance", "config_id", "config_group", "source", "mix_name", "rate",
    "majority_regime", "pred_hard", "w_weak", "w_global", "w_subgroup",
    "n_bad_weak", "n_bad_global", "n_bad_subgroup",
}
BUILD_SEED_OFFSETS = {"uf": 0, "rb": None, "pku": 200_000}
N_BOOT = 10_000
BOOT_SEED = 20260831


def build_strict_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the 456 deployments with calibration-only selector features."""
    OUT.mkdir(parents=True, exist_ok=True)
    uf, rb, lids, R_h1 = load_sources()
    pku = load_pku_source(lids, R_h1)
    feat_rows: list[dict] = []
    method_rows: list[dict] = []

    # Match the original RNG streams exactly: UF then RB share a 0..303 counter;
    # PKU uses the pre-existing disjoint offset from extend_pku.py.
    counter = 0
    for src in (uf, rb):
        for mix_name, weights in mixtures():
            for rate in (0.10, 0.20):
                for seed_rep in range(4):
                    O, M, gold, counts = make_mixed_instance(src, weights, rate, counter)
                    name = f"{src.name}_{mix_name}_r{rate:g}_s{seed_rep}"
                    cfg = f"{src.name}_{mix_name}_r{rate:g}"
                    metrics, fvec, pred_hard, _ = evaluate(
                        src, O, M, gold, counter, R_h1,
                        selector_features_from_calibration=True,
                    )
                    counter += 1
                    for method, metric in metrics.items():
                        method_rows.append({"instance": name, "method": method, **metric})
                    fvec.update({
                        "instance": name, "config_id": cfg,
                        "config_group": f"{mix_name}_r{rate:g}",
                        "source": src.name, "mix_name": mix_name, "rate": rate,
                        "majority_regime": majority_regime(weights), "pred_hard": pred_hard,
                        **{f"w_{r}": round(weights.get(r, 0.0), 4)
                           for r in ("weak", "global", "subgroup")},
                        **{f"n_bad_{r}": counts.get(r, 0)
                           for r in ("weak", "global", "subgroup")},
                    })
                    feat_rows.append(fvec)

    counter = BUILD_SEED_OFFSETS["pku"]
    for mix_name, weights in mixtures():
        for rate in (0.10, 0.20):
            for seed_rep in range(4):
                O, M, gold, counts = make_mixed_instance(pku, weights, rate, counter)
                name = f"pku_{mix_name}_r{rate:g}_s{seed_rep}"
                cfg = f"pku_{mix_name}_r{rate:g}"
                metrics, fvec, pred_hard, _ = evaluate(
                    pku, O, M, gold, counter, R_h1,
                    selector_features_from_calibration=True,
                )
                counter += 1
                for method, metric in metrics.items():
                    method_rows.append({"instance": name, "method": method, **metric})
                fvec.update({
                    "instance": name, "config_id": cfg,
                    "config_group": f"{mix_name}_r{rate:g}",
                    "source": "pku", "mix_name": mix_name, "rate": rate,
                    "majority_regime": majority_regime(weights), "pred_hard": pred_hard,
                    **{f"w_{r}": round(weights.get(r, 0.0), 4)
                       for r in ("weak", "global", "subgroup")},
                    **{f"n_bad_{r}": counts.get(r, 0)
                       for r in ("weak", "global", "subgroup")},
                })
                feat_rows.append(fvec)

    feats = pd.DataFrame(feat_rows)
    methods = pd.DataFrame(method_rows)
    feats.to_csv(OUT / "strict_features.csv", index=False)
    methods.to_csv(OUT / "strict_methods.csv", index=False)
    return feats, methods


def fold_assignment(groups: pd.Series, rep: int) -> np.ndarray:
    """Five outer folds, grouped across source by mixture/rate configuration."""
    rng = np.random.default_rng(BOOT_SEED + rep)
    uniq = rng.permutation(groups.unique())
    mapping = {g: i % 5 for i, g in enumerate(uniq)}
    return groups.map(mapping).to_numpy(int)


def best_fixed(methods: pd.DataFrame, instances: list[str]) -> str:
    sub = methods[methods.instance.isin(instances) & methods.method.isin(FIXED)]
    agg = sub.groupby("method")[["n_kept", "n_kept_clean"]].sum()
    return max(FIXED, key=lambda m: (agg.loc[m, "n_kept_clean"] /
                                     max(agg.loc[m, "n_kept"], 1), -FIXED.index(m)))


def method_lookup(methods: pd.DataFrame) -> pd.DataFrame:
    return methods[methods.method.isin(FIXED)].set_index(["instance", "method"])


def pooled_precision(idx: pd.DataFrame, decisions: dict[str, str]) -> float:
    rows = [idx.loc[(inst, method)] for inst, method in decisions.items()]
    kept = sum(int(r.n_kept) for r in rows)
    clean = sum(int(r.n_kept_clean) for r in rows)
    return clean / max(kept, 1)


def oracle_decisions(methods: pd.DataFrame, instances: list[str]) -> dict[str, str]:
    sub = methods[methods.instance.isin(instances) & methods.method.isin(FIXED)].copy()
    sub["method_order"] = sub.method.map({m: i for i, m in enumerate(FIXED)})
    # Deterministic: precision, then kept count, then declared menu order.
    sub = sub.sort_values(
        ["instance", "precision", "n_kept", "method_order"],
        ascending=[True, False, False, True],
    )
    win = sub.drop_duplicates("instance")
    return dict(zip(win.instance, win.method))


def fit_precision_selector(
    train_f: pd.DataFrame,
    test_f: pd.DataFrame,
    methods: pd.DataFrame,
    cols: list[str],
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    targets = methods[methods.method.isin(FIXED)].pivot_table(
        index="instance", columns="method", values="precision"
    ).loc[train_f.instance, FIXED]
    predictions = []
    for method in FIXED:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_f[cols].to_numpy(float), targets[method].to_numpy(float))
        predictions.append(model.predict(test_f[cols].to_numpy(float)))
    pred = np.column_stack(predictions)
    return np.asarray(FIXED)[pred.argmax(axis=1)], pred


def fit_choice_classifier(
    train_f: pd.DataFrame,
    test_f: pd.DataFrame,
    methods: pd.DataFrame,
    cols: list[str],
) -> np.ndarray:
    oracle = oracle_decisions(methods, train_f.instance.tolist())
    y = np.array([oracle[i] for i in train_f.instance])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=0.25, class_weight="balanced", random_state=0),
    )
    model.fit(train_f[cols].to_numpy(float), y)
    return model.predict(test_f[cols].to_numpy(float))


def grouped_bootstrap(
    methods: pd.DataFrame,
    dec_a: dict[str, str],
    dec_b: dict[str, str],
    group_map: dict[str, str],
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Cluster bootstrap over mixture/rate configurations, not seed replicates."""
    idx = method_lookup(methods)
    groups = sorted({group_map[i] for i in dec_a})
    a = {}; b = {}
    for group in groups:
        insts = [i for i in dec_a if group_map[i] == group]
        ar = [idx.loc[(i, dec_a[i])] for i in insts]
        br = [idx.loc[(i, dec_b[i])] for i in insts]
        a[group] = (sum(int(r.n_kept) for r in ar), sum(int(r.n_kept_clean) for r in ar))
        b[group] = (sum(int(r.n_kept) for r in br), sum(int(r.n_kept_clean) for r in br))

    def score(table: dict, sampled: list[str]) -> float:
        kept = sum(table[g][0] for g in sampled)
        clean = sum(table[g][1] for g in sampled)
        return clean / max(kept, 1)

    point = score(a, groups) - score(b, groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sampled = list(rng.choice(groups, size=len(groups), replace=True))
        draws[k] = score(a, sampled) - score(b, sampled)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def evaluate_grouped(feats: pd.DataFrame, methods: pd.DataFrame, reps: int = 10):
    numeric = [c for c in feats.columns if c not in META and pd.api.types.is_numeric_dtype(feats[c])]
    feature_sets = {"core": BASE + GOLD, "core_perf": BASE + GOLD + CAL_PERF,
                    "all": numeric}
    idx = method_lookup(methods)
    all_instances = feats.instance.tolist()
    oracle = oracle_decisions(methods, all_instances)
    group_map = dict(zip(feats.instance, feats.config_group))
    rows: list[dict] = []
    prediction_rows: list[dict] = []
    primary_decisions = None

    for rep in range(reps):
        fold = fold_assignment(feats.config_group, rep)
        decisions = {"ridge_core": {}, "ridge_core_perf": {}, "ridge_all": {},
                     "choice_logreg": {}, "calibration_best": {}, "best_fixed": {},
                     "regime_policy": {}, "regime_oracle_defined": {},
                     "perdeployment_oracle": oracle}
        for k in range(5):
            train = feats[fold != k]
            test = feats[fold == k]
            assert set(train.config_group).isdisjoint(set(test.config_group))
            bf = best_fixed(methods, train.instance.tolist())
            for tag, cols in feature_sets.items():
                choice, _ = fit_precision_selector(train, test, methods, cols)
                decisions[f"ridge_{tag}"].update(dict(zip(test.instance, choice)))
            cls = fit_choice_classifier(train, test, methods, feature_sets["core"])
            decisions["choice_logreg"].update(dict(zip(test.instance, cls)))
            cal_choice = np.asarray(FIXED)[test[CAL_PERF].to_numpy(float).argmax(axis=1)]
            decisions["calibration_best"].update(dict(zip(test.instance, cal_choice)))
            decisions["best_fixed"].update(dict.fromkeys(test.instance, bf))
            for _, row in test.iterrows():
                r = row.majority_regime
                decisions["regime_policy"][row.instance] = REGIME_FILTER.get(r, bf)
                if r in REGIME_FILTER:
                    decisions["regime_oracle_defined"][row.instance] = REGIME_FILTER[r]

        for method, dec in decisions.items():
            p = pooled_precision(idx, dec)
            match = np.mean([dec[i] == oracle[i] for i in dec])
            rows.append({"rep": rep, "method": method, "precision": p,
                         "oracle_choice_match": match})
        if rep == 0:
            primary_decisions = decisions
            for i in all_instances:
                prediction_rows.append({
                    "instance": i, "config_group": group_map[i],
                    **{m: d.get(i, "") for m, d in decisions.items()},
                })

    result = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    assert primary_decisions is not None

    comps = [
        ("ridge_core", "best_fixed"),
        ("ridge_core_perf", "best_fixed"),
        ("ridge_all", "best_fixed"),
        ("calibration_best", "best_fixed"),
        ("choice_logreg", "best_fixed"),
        ("ridge_core", "regime_policy"),
        ("ridge_core", "regime_oracle_defined"),
        ("ridge_core_perf", "regime_oracle_defined"),
        ("ridge_all", "regime_oracle_defined"),
        ("regime_policy", "best_fixed"),
        ("regime_oracle_defined", "best_fixed"),
        ("perdeployment_oracle", "best_fixed"),
        ("ridge_core", "perdeployment_oracle"),
        ("ridge_core_perf", "perdeployment_oracle"),
    ]
    sig = []
    for a, b in comps:
        common = set(primary_decisions[a]) & set(primary_decisions[b])
        dec_a = {i: primary_decisions[a][i] for i in common}
        dec_b = {i: primary_decisions[b][i] for i in common}
        pt, lo, hi = grouped_bootstrap(methods, dec_a, dec_b, group_map)
        sig.append({"comparison": f"{a} - {b}", "gain_pts": 100 * pt,
                    "ci95_low": 100 * lo, "ci95_high": 100 * hi,
                    "significant": bool(lo > 0 or hi < 0), "n_deployments": len(common)})
    return result, predictions, pd.DataFrame(sig), feature_sets


def evaluate_loso(feats: pd.DataFrame, methods: pd.DataFrame, feature_sets: dict[str, list[str]]):
    idx = method_lookup(methods)
    oracle = oracle_decisions(methods, feats.instance.tolist())
    rows = []
    for hold in sorted(feats.source.unique()):
        train, test = feats[feats.source != hold], feats[feats.source == hold]
        bf = best_fixed(methods, train.instance.tolist())
        fixed = dict.fromkeys(test.instance, bf)
        for tag, cols in feature_sets.items():
            choice, _ = fit_precision_selector(train, test, methods, cols)
            dec = dict(zip(test.instance, choice))
            rows.append({"held_out_source": hold, "method": f"ridge_{tag}",
                         "precision": pooled_precision(idx, dec),
                         "best_fixed": pooled_precision(idx, fixed),
                         "oracle": pooled_precision(idx, {i: oracle[i] for i in test.instance}),
                         "gain_vs_fixed_pts": 100 * (pooled_precision(idx, dec) -
                                                      pooled_precision(idx, fixed))})
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame, sig: pd.DataFrame, loso: pd.DataFrame):
    primary = results[results.rep == 0].set_index("method")
    robust = results.groupby("method").precision.agg(["mean", "std", "min", "max"])
    s = sig.set_index("comparison")
    lines = [
        "# Deployment-Level Filter Selector (Strict Calibration Split)", "",
        "All selector features come from 100 trusted calibration items; all filter outcomes",
        "come from a disjoint 200-item held-out split. The primary outer split groups the same",
        "mixture/rate configuration across UF, RewardBench, and PKU-SafeRLHF.", "",
        "## Primary result (fixed ridge/core analysis, outer split seed 20260831)", "",
        f"- selector precision: {primary.loc['ridge_core', 'precision']:.4f}",
        f"- best-fixed precision: {primary.loc['best_fixed', 'precision']:.4f}",
        f"- regime-informed policy precision (fixed fallback when tied): {primary.loc['regime_policy', 'precision']:.4f}",
        f"- regime-oracle precision (360 deployments with a unique dominant regime): {primary.loc['regime_oracle_defined', 'precision']:.4f}",
        f"- per-deployment oracle precision: {primary.loc['perdeployment_oracle', 'precision']:.4f}",
        "", "## Exploratory calibration-performance augmentation", "",
        f"- ridge core+calibration-performance precision: {primary.loc['ridge_core_perf', 'precision']:.4f}",
        f"- direct choose-best-on-calibration precision: {primary.loc['calibration_best', 'precision']:.4f}",
    ]
    for comp in ["ridge_core - best_fixed", "ridge_core - regime_policy",
                 "ridge_core - regime_oracle_defined", "perdeployment_oracle - best_fixed",
                 "ridge_core - perdeployment_oracle"]:
        r = s.loc[comp]
        lines.append(f"- {comp}: {r.gain_pts:+.3f} pts "
                     f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]")
    lines += ["", "## Repeated grouped-split robustness", "", robust.to_markdown(),
              "", "## Leave-one-dataset-out transfer", "", loso.to_markdown(index=False), "",
              "The per-deployment oracle is the best realized filter on held-out labels and is",
              "therefore an upper bound over this filter menu. Claims of beating it would be invalid.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="reuse strict_features/methods if present")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    fpath, mpath = OUT / "strict_features.csv", OUT / "strict_methods.csv"
    if args.reuse and fpath.exists() and mpath.exists():
        feats, methods = pd.read_csv(fpath), pd.read_csv(mpath)
    else:
        feats, methods = build_strict_pool()

    assert len(feats) == 456, len(feats)
    assert feats.config_group.nunique() == 38
    assert set(FIXED).issubset(set(methods.method))
    results, predictions, sig, feature_sets = evaluate_grouped(feats, methods, args.reps)
    loso = evaluate_loso(feats, methods, feature_sets)
    results.to_csv(OUT / "repeated_grouped_results.csv", index=False)
    predictions.to_csv(OUT / "primary_oof_predictions.csv", index=False)
    sig.to_csv(OUT / "primary_significance.csv", index=False)
    loso.to_csv(OUT / "loso_results.csv", index=False)
    write_report(results, sig, loso)

    print("\nPrimary outer-fold results")
    print(results[results.rep == 0].sort_values("precision", ascending=False).to_string(index=False))
    print("\nCluster-bootstrap comparisons")
    print(sig.to_string(index=False))
    print("\nLOSO transfer")
    print(loso.to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()

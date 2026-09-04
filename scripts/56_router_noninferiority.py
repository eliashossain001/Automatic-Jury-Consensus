#!/usr/bin/env python
"""Step 1: pre-registered non-inferiority test, router vs regime-oracle.

CPU-only, no inference. Consumes cached per-instance precision artifacts only.
Pre-registration (margin fixed before results): experiments/router_ni/PREREG.md

Delta = precision(router) - precision(regime_oracle), in points, paired by instance.
Non-inferior iff the one-sided 95% lower bound on mean Delta exceeds -MARGIN.
Significantly worse iff the two-sided 95% upper bound is below 0.

Outputs -> outputs/router_ni/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/router_ni"
RU = ROOT / "outputs/router_upgrade"

MARGIN = 0.5          # points; pre-registered
B = 20000
SEED = 20260706
REGIME_FILTER = {"weak": "supermajority_75", "global": "corrfilter_small_gold_R",
                 "subgroup": "bias_cluster"}


def paired_ni(router: np.ndarray, oracle: np.ndarray, seed=SEED, B=B) -> dict:
    """Paired instance bootstrap on mean (router - oracle), in points."""
    d = (np.asarray(router, float) - np.asarray(oracle, float)) * 100.0
    d = d[~np.isnan(d)]
    n = d.size
    if n < 5:
        return {"n": n, "mean_delta": np.nan}
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo1 = float(np.quantile(boot, 0.05))                    # one-sided 95% lower bound
    lo2, hi2 = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    return {
        "n": int(n),
        "mean_delta": round(float(d.mean()), 4),
        "ni_lower_bound_1sided": round(lo1, 4),
        "non_inferior": bool(lo1 > -MARGIN),
        "ci95_low": round(lo2, 4), "ci95_high": round(hi2, 4),
        "significantly_worse": bool(hi2 < 0),
        "significantly_better": bool(lo2 > 0),
    }


def family_a() -> pd.DataFrame:
    """Pools that store router/oracle precision directly."""
    sources = {
        "A: pool_methods (mixed pools)": RU / "pool_methods.csv",
        "A: pool_methods_eqsize": RU / "pool_methods_eqsize.csv",
        "A: mixed_regime results_by_instance": ROOT / "outputs/mixed_regime/results_by_instance.csv",
    }
    rows = []
    for label, path in sources.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        wide = df.pivot_table(index="instance", columns="method", values="precision")
        regimes = df.drop_duplicates("instance").set_index("instance")["true_regime"]
        if "oracle_router" not in wide.columns:
            continue
        for router in ("router_hard", "router_soft"):
            if router not in wide.columns:
                continue
            sub = wide[[router, "oracle_router"]].dropna()
            rows.append({"family": label, "setting": "all instances", "router": router,
                         **paired_ni(sub[router].values, sub["oracle_router"].values)})
            for reg in sorted(regimes.dropna().unique()):
                idx = regimes[regimes == reg].index
                s2 = wide.loc[wide.index.intersection(idx), [router, "oracle_router"]].dropna()
                if len(s2) >= 5:
                    rows.append({"family": label, "setting": f"true_regime={reg}",
                                 "router": router,
                                 **paired_ni(s2[router].values, s2["oracle_router"].values)})
    return pd.DataFrame(rows)


def family_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extended pools: reconstruct router precision from its per-instance filter choice."""
    e = pd.read_csv(RU / "ext_perinstance.csv")
    m = pd.read_csv(RU / "mixed_methods.csv")
    prec = m.pivot_table(index="instance", columns="method", values="precision")

    def pick(inst, filt):
        if pd.isna(filt) or inst not in prec.index or filt not in prec.columns:
            return np.nan
        return prec.at[inst, filt]

    e = e[e.instance.isin(prec.index)].copy()
    e["p_logreg_all"] = [pick(i, f) for i, f in zip(e.instance, e.chosen_filter)]
    e["p_heuristic"] = [pick(i, f) for i, f in zip(e.instance, e.heuristic_filter)]
    e["p_regime_oracle"] = [pick(i, REGIME_FILTER.get(r))
                            for i, r in zip(e.instance, e.majority_regime)]
    e["p_perinstance_oracle"] = [pick(i, f) for i, f in zip(e.instance, e.oracle_filter)]

    settings = {"in-domain (all)": e}
    for src in sorted(e.source.dropna().unique()):
        settings[f"LODO deploy={src}"] = e[e.source == src]

    rows = []
    for label, sub in settings.items():
        for router, col in (("logreg_all", "p_logreg_all"), ("heuristic", "p_heuristic")):
            s = sub[[col, "p_regime_oracle"]].dropna()
            if len(s) >= 5:
                rows.append({"family": "B: extended/crossed", "setting": label,
                             "router": router,
                             **paired_ni(s[col].values, s["p_regime_oracle"].values)})
        s = sub[["p_regime_oracle", "p_perinstance_oracle"]].dropna()
        if len(s) >= 5:
            rows.append({"family": "B: extended/crossed", "setting": label,
                         "router": "regime_oracle (vs per-instance oracle)",
                         **paired_ni(s["p_regime_oracle"].values,
                                     s["p_perinstance_oracle"].values)})
    return pd.DataFrame(rows), e


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = family_a()
    b, e = family_b()
    res = pd.concat([a, b], ignore_index=True)
    res.to_csv(OUT / "noninferiority.csv", index=False)

    # pre-specified reconstruction validity check against the published aggregates
    pub = pd.read_csv(RU / "lim_router_vs_regime_oracle.csv")
    pub_map = {("in-domain (all)", "logreg_all"): 0.06, ("LODO deploy=pku", "logreg_all"): -0.07,
               ("LODO deploy=uf", "logreg_all"): 0.13, ("LODO deploy=rb", "logreg_all"): 0.12,
               ("in-domain (all)", "heuristic"): -0.13, ("LODO deploy=pku", "heuristic"): -0.09,
               ("LODO deploy=uf", "heuristic"): -0.11, ("LODO deploy=rb", "heuristic"): -0.22}
    checks = []
    for (setting, router), published in pub_map.items():
        got = b[(b.setting == setting) & (b.router == router)]
        if len(got):
            d = float(got.iloc[0].mean_delta)
            checks.append({"setting": setting, "router": router, "published": published,
                           "reconstructed": round(d, 4), "abs_diff": round(abs(d - published), 4),
                           "within_0.05": bool(abs(d - published) <= 0.05)})
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "reconstruction_check.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== reconstruction validity check (Family B) ===")
    print(chk.to_string(index=False) if len(chk) else "no overlap")
    print(f"\n=== non-inferiority, margin = -{MARGIN} pts, one-sided 95% lower bound ===")
    cols = ["family", "setting", "router", "n", "mean_delta", "ni_lower_bound_1sided",
            "non_inferior", "ci95_low", "ci95_high", "significantly_worse"]
    print(res[cols].to_string(index=False))

    tested = res[~res.router.str.startswith("regime_oracle")]
    print(f"\ncells tested: {len(tested)}   non-inferior: {int(tested.non_inferior.sum())}"
          f"   significantly worse: {int(tested.significantly_worse.sum())}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

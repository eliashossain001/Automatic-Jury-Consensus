#!/usr/bin/env python
"""Zero-shot cross-task routing: preference-trained routers on factuality.

BLOCKED unless outputs/router_upgrade/fact_gate_PASSED.txt exists (the
construct-validity gate of build_factuality_pools.py).

Leakage contract: routers, the transferred fixed filter, direct-prediction
regressors, and every threshold are fit on the THREE PREFERENCE pure pools
only (uf+rb+pku, 300 instances). Factuality gold labels are used only inside
the frozen evaluation protocol (per-instance small-gold split for CorrFilter,
metrics, bootstrap). Features: the 7 position-free pre-registered BASE
features (primary) and the full 35-feature set (secondary; its fields are all
swap-free, see report).

Outputs (Tables C, D, E + figures):
  fact_transfer_results.csv   method matrix on factuality deployments
  fact_rvsro.csv              router - regime-oracle (primary), incl.
                              practical-equivalence call at +-0.5 pts
  fact_decomposition.csv      regret decomposition (4 causes)
  mixed_plots/fact_*.png      figures

Usage: python scripts/crosstask/run_factuality_transfer.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from run_extended_eval import direct_pred, feature_cols

from corrfilter.routing import (
    FIXED,
    REGIME_FILTER,
    best_fixed_on,
    make_models,
    paired_boot_diff,
    pooled_precision,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/router_upgrade"
PLOTS = OUT / "mixed_plots"


EQUIV_MARGIN = 0.5  # precision points; pre-declared practical-equivalence band


def main() -> None:
    if not (OUT / "fact_gate_PASSED.txt").exists():
        raise SystemExit("construct-validity gate not passed; refusing to run "
                         "(see fact_gate.csv / fact_gate_FAILED.txt)")
    pure_f = pd.concat([pd.read_csv(OUT / "crossed_features.csv"),
                        pd.read_csv(OUT / "pku_features.csv")], ignore_index=True)
    pure_md = pd.concat([pd.read_csv(OUT / "crossed_methods.csv"),
                         pd.read_csv(OUT / "pku_methods.csv")], ignore_index=True)
    fp = pd.read_csv(OUT / "fact_features.csv")
    fp["variant"] = fp.get("variant", pd.Series(index=fp.index, dtype=object)).fillna("ctrl")
    fmd = pd.read_csv(OUT / "fact_methods.csv")
    fmx = pd.read_csv(OUT / "fact_mixed_features.csv")
    fmmd = pd.read_csv(OUT / "fact_mixed_methods.csv")
    sets = feature_cols(pure_f)

    def decisions(te_f, te_md, majority):
        insts = te_f.instance.to_numpy()
        dec = {}
        bf = best_fixed_on(pure_md, list(pure_f.instance))     # source-only
        dec["transferred_fixed"] = dict.fromkeys(insts, bf)
        for filt in FIXED:
            dec[f"fixed_{filt}"] = dict.fromkeys(insts, filt)
        dec["heuristic"] = {i: REGIME_FILTER[p] for i, p in zip(insts, te_f.pred_hard)}
        for sname in ("base", "all"):
            m = make_models(seed=0)["logreg"]
            m.fit(pure_f[sets[sname]].to_numpy(float), pure_f.regime.to_numpy())
            pred = m.predict(te_f[sets[sname]].to_numpy(float))
            dec[f"logreg_{sname}"] = {i: REGIME_FILTER[p] for i, p in zip(insts, pred)}
            if sname == "base":
                dec["_pred_base"] = dict(zip(insts, pred))
        ch, _ = direct_pred(pure_f, pure_md, te_f, sets["all"], "ridge")
        dec["direct_ridge"] = dict(zip(insts, ch))
        dec["regime_oracle"] = {i: REGIME_FILTER[majority[i]] for i in insts
                                if majority.get(i, "none") != "none"}
        sub = te_md[te_md.method.isin(FIXED)].sort_values(
            ["precision", "n_kept"], ascending=False).drop_duplicates("instance")
        dec["perinstance_oracle"] = dict(zip(sub.instance, sub.method))
        return insts, dec, bf

    # deployment sets: pure controlled, pure real, mixed, combined
    maj_pure = dict(zip(fp.instance, fp.regime))
    maj_mix = dict(zip(fmx.instance, fmx.majority_regime))
    fp_ctrl = fp[fp.variant == "ctrl"]
    fp_real = fp[fp.variant == "real"]
    sets_map = {
        "pure-controlled": (fp_ctrl, fmd[fmd.instance.isin(fp_ctrl.instance)], maj_pure),
        "pure-real": (fp_real, fmd[fmd.instance.isin(fp_real.instance)], maj_pure),
        "mixed": (fmx, fmmd, maj_mix),
    }
    comb_f = pd.concat([fp[["instance", "pred_hard"] + sets["all"]],
                        fmx[["instance", "pred_hard"] + sets["all"]]], ignore_index=True)
    comb_md = pd.concat([fmd, fmmd], ignore_index=True)
    sets_map["combined"] = (comb_f, comb_md, {**maj_pure, **maj_mix})

    rows, rvsro, dcmp_rows = [], [], []
    for tag, (te_f, te_md, majority) in sets_map.items():
        insts, dec, bf = decisions(te_f, te_md, majority)
        prec_i = te_md.pivot_table(index="instance", columns="method", values="precision")
        orc = dec["perinstance_oracle"]
        orc_p = {i: prec_i.loc[i, orc[i]] for i in insts}
        fix = dec["transferred_fixed"]
        p_fix = pooled_precision(te_md, fix)
        p_orc = pooled_precision(te_md, orc)
        for name in ("transferred_fixed", "fixed_naive_majority",
                     "fixed_corrfilter_small_gold_R", "fixed_bias_cluster",
                     "heuristic", "logreg_base", "logreg_all", "direct_ridge",
                     "regime_oracle", "perinstance_oracle"):
            d = dec[name]
            p = pooled_precision(te_md, d)
            reg = float(np.mean([orc_p[i] - prec_i.loc[i, d[i]] for i in d]))
            row = {"deploy": tag, "method": name, "n": len(d),
                   "precision": round(p, 4),
                   "mean_regret_pts": round(100 * reg, 3)}
            if name != "transferred_fixed":
                pt, lo, hi = paired_boot_diff(te_md, d, {i: fix[i] for i in d})
                row.update({"vs_fixed_pts": round(100 * pt, 2),
                            "fix_ci": f"[{100*lo:+.2f},{100*hi:+.2f}]"})
                head = p_orc - p_fix
                row["headroom_recovered"] = round((p - p_fix) / head, 3) if head > 1e-9 else np.nan
            rows.append(row)
        # primary: router - regime-oracle (majority-defined subset)
        ro = dec["regime_oracle"]
        for rname in ("logreg_base", "logreg_all", "heuristic", "direct_ridge"):
            d = {i: dec[rname][i] for i in ro}
            pt, lo, hi = paired_boot_diff(te_md, d, ro)
            equiv = (100 * lo > -EQUIV_MARGIN) and (100 * hi < EQUIV_MARGIN)
            rvsro.append({"deploy": tag, "router": rname, "n": len(ro),
                          "router_pts": round(100 * pooled_precision(te_md, d), 2),
                          "ro_pts": round(100 * pooled_precision(te_md, ro), 2),
                          "diff_pts": round(100 * pt, 2),
                          "ci": f"[{100*lo:+.2f},{100*hi:+.2f}]",
                          "ci_width": round(100 * (hi - lo), 2),
                          "significant": lo > 0 or hi < 0,
                          "practically_equivalent": bool(equiv)})
        pt, lo, hi = paired_boot_diff(te_md, ro, {i: orc[i] for i in ro})
        rvsro.append({"deploy": tag, "router": "regime_oracle_minus_perinstance",
                      "n": len(ro), "diff_pts": round(100 * pt, 2),
                      "ci": f"[{100*lo:+.2f},{100*hi:+.2f}]",
                      "significant": lo > 0 or hi < 0})
        # decomposition (logreg_base primary router) on this deployment set
        if tag == "combined":
            pred = {**{i: fp.regime[fp.instance == i].iloc[0] for i in []}}  # placeholder no-op
            predmap = dec["_pred_base"]
            for i in insts:
                m = majority.get(i, "none")
                chosen = dec["logreg_base"][i]
                r = 100 * (orc_p[i] - prec_i.loc[i, chosen])
                if m == "none":
                    b = "no-majority"
                elif predmap[i] != m:
                    b = "A: wrong regime"
                elif chosen != orc[i]:
                    b = "B: right regime, wrong mapping"
                else:
                    b = "correct"
                dcmp_rows.append({"instance": i, "bucket": b, "regret_pts": r})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fact_transfer_results.csv", index=False)
    rv = pd.DataFrame(rvsro)
    rv.to_csv(OUT / "fact_rvsro.csv", index=False)
    dc = pd.DataFrame(dcmp_rows)
    agg = dc.groupby("bucket").agg(n=("instance", "size"),
                                   total=("regret_pts", "sum")).reset_index()
    agg["share_instances"] = (agg.n / agg.n.sum()).round(3)
    agg["share_regret"] = (agg.total / max(agg.total.sum(), 1e-9)).round(3)
    agg.to_csv(OUT / "fact_decomposition.csv", index=False)

    print("== Table C: zero-shot cross-task routing ==")
    print(res.to_string(index=False))
    print("\n== Table D: router vs regime-oracle (primary; margin +-0.5) ==")
    print(rv.to_string(index=False))
    print("\n== Table E: regret decomposition (combined, logreg_base) ==")
    print(agg.to_string(index=False))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    C = {"transferred_fixed": "#2a78d6", "heuristic": "#eb6834",
         "logreg_base": "#1baf7a", "direct_ridge": "#eda100",
         "regime_oracle": "#52514e", "perinstance_oracle": "#0b0b0b"}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sub = res[res.deploy == "combined"].set_index("method")
    names = [n for n in C if n in sub.index]
    xs = np.arange(len(names))
    ax.bar(xs, [sub.loc[n, "precision"] for n in names],
           color=[C[n] for n in names], width=0.62)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7.5)
    ax.set_ylabel("pooled precision (factuality, combined)", fontsize=9)
    lo = min(sub.loc[n, "precision"] for n in names)
    ax.set_ylim(lo - 0.01, sub.precision.max() + 0.004)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS / "fact_router_comparison.png", dpi=200)

    fig2, ax = plt.subplots(figsize=(5.2, 3.0))
    a = agg.set_index("bucket")
    order = [b for b in ["A: wrong regime", "B: right regime, wrong mapping",
                         "no-majority", "correct"] if b in a.index]
    ax.barh(np.arange(len(order)), [a.loc[b, "share_regret"] for b in order],
            color="#2a78d6", height=0.55)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("share of total regret", fontsize=9)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(PLOTS / "fact_regret_decomposition.png", dpi=200)
    print(f"\nfigures -> {PLOTS}")


if __name__ == "__main__":
    main()

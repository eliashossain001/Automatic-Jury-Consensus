#!/usr/bin/env python
"""Zero-shot routing transfer to a genuinely different judge ensemble.

Fourth source `fb`: the six-judge FRONTIER bank (gemini-3.1-pro, 3.6-flash,
3.5-flash-lite, gpt-5.6-sol, claude-opus-5, grok-4.5) on the 400-item
RewardBench intersection of the cached Gemini and OpenRouter votes. This bank
differs from the ten-judge open bank in size (6 vs 10), capability tier, and
dependence level (frontier rho_bar ~0.56 vs 0.21), so it tests whether routing
transfers to a different LLM-judge aggregation task, not just a different
dataset. $0 spent: cached votes only.

Protocol: pure fb instances are built ONLY for the construct-validity
signature gate; routers are trained on the 300 open-bank pure instances
(uf+rb+pku) and deployed zero-shot on fb mixed deployments. The transferred
fixed filter is selected on the same training pools. The vulnerable cluster
uses k=3 (half the bank, matching the open bank's 5-of-10 convention). The
clean reference for fb is its own label-free disagreement R. Because the item
pool has only 400 items, 300-item instances overlap in items; the bootstrap
resamples instances and does not correct for this overlap (disclosed).

Usage: python experiments/router_upgrade/frontier_bank_transfer.py
Outputs -> outputs/router_upgrade/fb_{features,methods,mixed_features,
mixed_methods}.csv, fb_transfer_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP))

import build_crossed_pool as bcp  # noqa: E402
from build_crossed_pool import Source, evaluate, make_instance  # noqa: E402
from build_mixed_pool import make_mixed_instance, majority_regime, mixtures  # noqa: E402
from run_learned_router import (  # noqa: E402
    BASE_FEATS, FIXED, REGIME_FILTER, best_fixed_on, make_models,
    paired_boot_diff, pooled_precision)
from run_mixed_eval import META  # noqa: E402
from run_extended_eval import direct_pred, feature_cols  # noqa: E402

from corrfilter.cfi.adaptive_r import disagreement_R  # noqa: E402
from corrfilter.cfi.consensus import majority_consensus  # noqa: E402
from corrfilter.routing import (agree_features, load_votes_swaps,  # noqa: E402
                                position_sensitivity_cluster)
from corrfilter.voting import VoteCache  # noqa: E402

FB_LIDS = ["gemini-3.1-pro::pairwise", "gemini-3.6-flash::pairwise",
           "gemini-3.5-flash-lite::pairwise", "gpt-5.6-sol::pairwise",
           "claude-opus-5::pairwise", "grok-4.5::pairwise"]
FB_CLUSTER_K = 3
PURE_OFFSET = 400_000
MIX_OFFSET = 500_000
RATES = [0.10, 0.20]
N_SEEDS = 4


def load_fb_source():
    man = pd.read_parquet(ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet")
    man = man.drop_duplicates("item_id", keep="last")
    gem = VoteCache(ROOT / "outputs/gemini_bank/votes/rewardbench")
    orb = VoteCache(ROOT / "outputs/openrouter_bank/votes/frontier/rewardbench")
    or_ids = set(pd.read_parquet(next((ROOT / "outputs/openrouter_bank/votes/frontier/rewardbench").glob("*.parquet"))).item_id.astype(str))
    ids = [i for i in man.item_id.astype(str) if i in or_ids]
    man = man.set_index(man.item_id.astype(str)).loc[ids]

    def cache_load(lid):
        base = lid.split("::")[0]
        for c in (gem, orb):
            d = c.load(f"{base}::pairwise") if hasattr(c, "load") else None
            if d is not None:
                return d
        return None

    class _Multi:
        def load(self, lid):
            base = lid.split("::")[0]
            d = gem.load(lid)
            if d is None:
                d = orb.load(lid)
            return d
    V, M, S = load_votes_swaps(_Multi(), FB_LIDS, ids)
    R_fb = disagreement_R(V, M)
    src = Source("fb", V, M, S, man.chosen.str.len().to_numpy(),
                 man.rejected.str.len().to_numpy(), FB_LIDS, R_fb)
    # half-bank vulnerable cluster (open-bank convention is 5 of 10)
    src.cpos = position_sensitivity_cluster(V, M, S, k=FB_CLUSTER_K)
    lab = majority_consensus(V, M)
    fc = agree_features(V, M, lab, src.cpos, R_fb)
    nj = V.shape[1]
    src.ref = {"rho_bar": float(np.mean(R_fb[~np.eye(nj, dtype=bool)])),
               "rho_S": float(np.nanmean(fc["rho"])),
               "lco_flip": float(np.nanmean(fc["lco_flip"]))}
    return src, R_fb


def main() -> None:
    out = ROOT / "outputs/router_upgrade"
    fb, R_fb = load_fb_source()
    nj = fb.V.shape[1]
    print(f"fb source: {fb.n} items, {nj} judges, clean rho_bar {fb.ref['rho_bar']:.3f}, "
          f"cluster {int(fb.cpos.sum())}, coverage {fb.M.mean():.3f}")

    # ---- pure instances: signature gate only (never used for training) ----
    counter = PURE_OFFSET
    sig_rows, pure_feat, pure_md_rows = [], [], []
    for regime in ("weak", "subgroup", "global"):
        for rate in RATES:
            for s in range(N_SEEDS):
                O, Md, gold = make_instance(fb, regime, rate, counter)
                metrics, fvec, pred_hard, d = evaluate(fb, O, Md, gold, counter, R_fb)
                counter += 1
                name = f"fb_{regime}_r{rate:g}_s{s}"
                for f, m in metrics.items():
                    pure_md_rows.append({"instance": name, "method": f, **m})
                fvec.update({"instance": name, "regime": regime, "pred_hard": pred_hard})
                pure_feat.append(fvec)
                sig_rows.append({"config": f"fb_{regime}_r{rate:g}", "regime": regime,
                                 "rate": rate, **{k: round(v, 4) for k, v in d.items()}})
    pd.DataFrame(pure_feat).to_csv(out / "fb_features.csv", index=False)
    pd.DataFrame(pure_md_rows).to_csv(out / "fb_methods.csv", index=False)
    sig = pd.DataFrame(sig_rows).groupby(["config", "regime"]).mean().reset_index()
    print("\nsignature gate (fb):")
    print(sig.to_string(index=False))

    # ---- mixed deployments ----
    counter = MIX_OFFSET
    feat_rows, md_rows = [], []
    for mix_name, weights in mixtures():
        for rate in RATES:
            for s in range(N_SEEDS):
                O, Md, gold, counts = make_mixed_instance(fb, weights, rate, counter)
                metrics, fvec, pred_hard, d = evaluate(fb, O, Md, gold, counter, R_fb)
                counter += 1
                name = f"fb_{mix_name}_r{rate:g}_s{s}"
                for f, m in metrics.items():
                    md_rows.append({"instance": name, "method": f, **m})
                fvec.update({"instance": name, "config_id": f"fb_{mix_name}_r{rate:g}",
                             "source": "fb", "mix_name": mix_name, "rate": rate,
                             "majority_regime": majority_regime(weights),
                             "pred_hard": pred_hard,
                             **{f"w_{r}": round(weights.get(r, 0.0), 4)
                                for r in ("weak", "global", "subgroup")},
                             **{f"n_bad_{r}": counts.get(r, 0)
                                for r in ("weak", "global", "subgroup")}})
                feat_rows.append(fvec)
    fb_mix_f = pd.DataFrame(feat_rows)
    fb_mix_md = pd.DataFrame(md_rows)
    fb_mix_f.to_csv(out / "fb_mixed_features.csv", index=False)
    fb_mix_md.to_csv(out / "fb_mixed_methods.csv", index=False)
    print(f"\nfb mixed: {len(fb_mix_f)} instances")

    # ---- zero-shot deployment: routers trained on open-bank pure pools ----
    pure_f = pd.concat([pd.read_csv(out / "crossed_features.csv"),
                        pd.read_csv(out / "pku_features.csv")], ignore_index=True)
    pure_md = pd.concat([pd.read_csv(out / "crossed_methods.csv"),
                         pd.read_csv(out / "pku_methods.csv")], ignore_index=True)
    sets = feature_cols(pure_f)
    insts = fb_mix_f.instance.to_numpy()
    bf = best_fixed_on(pure_md, list(pure_f.instance))
    print(f"\ntransferred fixed (chosen on uf+rb+pku pure): {bf}; "
          f"best on fb would be: {best_fixed_on(fb_mix_md, list(insts))}")
    dec = {"transferred_fixed": dict.fromkeys(insts, bf),
           "heuristic": {i: REGIME_FILTER[p] for i, p in zip(insts, fb_mix_f.pred_hard)}}
    m = make_models(seed=0)["logreg"]
    m.fit(pure_f[sets["all"]].to_numpy(float), pure_f.regime.to_numpy())
    dec["logreg_all"] = {i: REGIME_FILTER[p]
                         for i, p in zip(insts, m.predict(fb_mix_f[sets["all"]].to_numpy(float)))}
    for kind in ("ridge", "lgbm"):
        ch, _ = direct_pred(pure_f, pure_md, fb_mix_f, sets["all"], kind)
        dec[f"direct_{kind}"] = dict(zip(insts, ch))
    maj = dict(zip(insts, fb_mix_f.majority_regime))
    dec["regime_oracle"] = {i: REGIME_FILTER[maj[i]] for i in insts if maj[i] != "none"}
    sub = fb_mix_md[fb_mix_md.method.isin(FIXED)].sort_values(
        ["precision", "n_kept"], ascending=False).drop_duplicates("instance")
    dec["perinstance_oracle"] = dict(zip(sub.instance, sub.method))

    rows = []
    d_fx = dec["transferred_fixed"]
    for name, d in dec.items():
        p = pooled_precision(fb_mix_md, d)
        row = {"method": name, "n": len(d), "precision": round(p, 4)}
        if name != "transferred_fixed":
            sub_fx = {i: d_fx[i] for i in d}
            pt, lo, hi = paired_boot_diff(fb_mix_md, d, sub_fx)
            row.update({"vs_transferred_pts": round(100 * pt, 2),
                        "ci": f"[{100*lo:+.2f},{100*hi:+.2f}]", "sig": lo > 0 or hi < 0})
        if name not in ("transferred_fixed", "regime_oracle", "perinstance_oracle"):
            ro = {i: dec["regime_oracle"][i] for i in d if i in dec["regime_oracle"]}
            pt, lo, hi = paired_boot_diff(fb_mix_md, {i: d[i] for i in ro}, ro)
            row.update({"vs_regime_oracle_pts": round(100 * pt, 2),
                        "ro_ci": f"[{100*lo:+.2f},{100*hi:+.2f}]"})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "fb_transfer_results.csv", index=False)
    print("\n== Zero-shot deployment on the frontier bank ==")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Mixed-regime end-to-end routing benchmark (analysis-only, replays cached votes).

Builds a pool of synthetic deployment instances from the three existing regime
generators (ground-truth regime = generator, by construction):

  weak      content-poisoned UltraFeedback (verbosity/length/polite/style flips),
            sampled at rates {5,10,20}% x 6 item-subsample seeds  (18 instances)
  subgroup  position-aligned poisoning (H1-weighted wrong-vote mass), same
            rates x 6 seeds                                        (18 instances)
  global    CFI prompt-injected biased banks, mechanisms x biased-judge ratio
            {1.0, 0.75} x 3 seeds                                  (18 instances)

Every instance is evaluated with the the regime/unified routers in scripts/routing_selector/ protocol: 100-item labeled
calibration split for small_gold R (held out from eval), matched retention at
the supermajority-0.75 operating point, identical metric definitions.

Methods: naive_majority, supermajority_75, corrfilter_small_gold_R (always-
CorrFilter), bias_cluster (always-bias-cluster), best fixed filter (selected
globally per mixture), router_hard (scripts/routing_selector/run_regime_router.py clean-referenced rule),
router_soft (scripts/routing_selector/run_unified_router.py softmax blend), oracle_router, oracle_filter.

Mixtures: balanced (12/12/12), weak-dominant (18/6/6), global-dominant
(6/18/6), subgroup-dominant (6/6/18); instance-stratified bootstrap CIs; the
router-oracle gap is decomposed into misroute rate x per-misroute cost, against
the oracle headroom over the best fixed filter.

Usage: python scripts/routing_selector/run_mixed_regime_benchmark.py --project-root .
Outputs -> outputs/mixed_regime/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.cfi.adaptive_r import disagreement_R, eigenvector_overlap, small_gold_R  # noqa: E402
from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import majority_consensus, supermajority_consensus  # noqa: E402
from corrfilter.correlation import effective_size  # noqa: E402
from corrfilter.correlation.effective_size import mean_off_diagonal  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.routing import (  # noqa: E402
    ABSTAIN, ALL_SIGNALS, REGIME_FILTER, REGIMES, SG_SIZE, TEMP,
    agree_features, classify_regime_hard, filter_metrics, filter_scores,
    keep_at_matched_retention, load_votes_swaps, pct_rank,
    position_poison_mask, position_sensitivity_cluster, regime_probs,
)
from corrfilter.voting import VoteCache  # noqa: E402

BASE_SEED = 20260601
RATES = [0.05, 0.10, 0.20]
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]
CFI_RATIOS = [1.0, 0.75]
N_SEEDS_UF = 6
N_SEEDS_CFI = 3
UF_INSTANCE_SIZE = 400
CFI_INSTANCE_SIZE = 300
FIXED_FILTERS = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
ROUTERS = ["router_hard", "router_soft", "oracle_router"]
MIXTURES = {
    "balanced": {"weak": 12, "global": 12, "subgroup": 12},
    "weak_dominant": {"weak": 18, "global": 6, "subgroup": 6},
    "global_dominant": {"weak": 6, "global": 18, "subgroup": 6},
    "subgroup_dominant": {"weak": 6, "global": 6, "subgroup": 18},
}
N_BOOT = 2000
BOOT_SEED = 20260706


def build_instances(root: Path):
    """Yield (instance_id, regime, O, M, gold, affirmed_only, rng) tuples."""
    bank_cfg = load_bank_config(root / "configs/judge_bank.yaml")
    lids = [s.logical_id for s in bank_cfg.specs]
    man = pd.read_csv(root / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    item_ids = man["item_id"].astype(str).tolist()
    V, M, Sw = load_votes_swaps(VoteCache(root / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"),
                                lids, item_ids)
    n_uf = len(item_ids)
    cpos_mask = position_sensitivity_cluster(V, M, Sw)
    R_h1 = np.load(root / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)["R"]
    R_clean_uf = disagreement_R(V, M)

    # clean-bank reference (scripts/routing_selector/run_regime_router.py)
    lab_clean = majority_consensus(V, M)
    feat_clean = agree_features(V, M, lab_clean, cpos_mask, R_h1)
    ref = {"rho_bar": mean_off_diagonal(R_clean_uf),
           "rho_S": float(np.nanmean(feat_clean["rho"])),
           "lco_flip": float(np.nanmean(feat_clean["lco_flip"]))}

    poison20 = man["poisoned_20"].to_numpy(bool)
    pool_poison = np.where(poison20)[0]
    pool_clean = np.where(~poison20)[0]

    instances = []
    counter = 0

    def rng_for(k):
        return np.random.default_rng(BASE_SEED + 7919 * k)

    # ---- weak: content poisoning, sampled composition ----
    for s in range(N_SEEDS_UF):
        for rate in RATES:
            rng = rng_for(counter)
            n_p = int(round(rate * UF_INSTANCE_SIZE))
            pi = rng.choice(pool_poison, size=n_p, replace=False)
            ci = rng.choice(pool_clean, size=UF_INSTANCE_SIZE - n_p, replace=False)
            idx = np.sort(np.concatenate([pi, ci]))
            pm = np.isin(idx, pi)
            O = V[idx].copy()
            O[pm] = 1 - O[pm]
            gold = np.where(pm, 0, 1).astype(np.int8)
            instances.append((f"weak_r{int(rate*100):02d}_s{s}", "weak", O, M[idx], gold, True, counter))
            counter += 1

    # ---- subgroup: position-aligned poisoning within a sampled subset ----
    for s in range(N_SEEDS_UF):
        for rate in RATES:
            rng = rng_for(counter)
            idx = np.sort(rng.choice(n_uf, size=UF_INSTANCE_SIZE, replace=False))
            pm_full = position_poison_mask(V, M, lids, rate, subset_idx=idx)
            pm = pm_full[idx]
            O = V[idx].copy()
            O[pm] = 1 - O[pm]
            gold = np.where(pm, 0, 1).astype(np.int8)
            instances.append((f"subgroup_r{int(rate*100):02d}_s{s}", "subgroup", O, M[idx], gold, True, counter))
            counter += 1

    # ---- global: CFI biased-bank variants ----
    cfg = yaml.safe_load((root / "configs/cfi_bias_prompts.yaml").read_text())
    bias_bank = load_bias_bank(str(root / "configs/cfi_bias_prompts.yaml"))
    cal = yaml.safe_load((root / cfg["sources"]["h1_calibration_config"]).read_text())
    keep_ids = {l.strip() for l in (root / "outputs/cfi/cfi_subset_items.txt").read_text().splitlines() if l.strip()}
    citems = [it for it in load_calibration_set(root / cal["output"]["manifest_path"]) if it.item_id in keep_ids]
    clean_views = load_vote_views(root / cfg["sources"]["h1_votes_dir"], lids)
    for mech in CFI_MECHS:
        bv = load_vote_views(root / "outputs/cfi/votes" / mech, lids)
        if sum(len(x) for x in bv.values()) == 0:
            print(f"[37] WARNING: no CFI votes for {mech}; skipping")
            continue
        for ratio in CFI_RATIOS:
            var = assemble_variant(citems, lids, bias_bank.by_name(mech), ratio, bv, clean_views, bias_bank.seed)
            gold_all = (majority_consensus(var.V, var.M) == 1).astype(np.int8)
            for s in range(N_SEEDS_CFI):
                rng = rng_for(counter)
                idx = np.sort(rng.choice(var.V.shape[0], size=min(CFI_INSTANCE_SIZE, var.V.shape[0]), replace=False))
                instances.append((f"global_{mech[:8]}_x{int(ratio*100)}_s{s}", "global",
                                  var.V[idx], var.M[idx], gold_all[idx], False, counter))
                counter += 1

    return instances, lids, cpos_mask, R_h1, R_clean_uf, ref


def evaluate_instance(inst, lids, cpos_mask, R_h1, R_clean_uf, ref):
    """Scripts/18-19 protocol on one instance; returns per-method metrics + preds."""
    name, regime, O, Md, gold, affirmed, counter = inst
    N = len(gold)
    rng = np.random.default_rng(BASE_SEED + 104729 * (counter + 1))
    cal_idx = rng.choice(N, size=min(SG_SIZE, N), replace=False)
    eval_mask = np.ones(N, bool)
    eval_mask[cal_idx] = False
    ei = np.where(eval_mask)[0]
    R_sg = small_gold_R(O, Md, gold, cal_idx)

    Oe, Me, ge = O[ei], Md[ei], gold[ei]
    label = majority_consensus(Oe, Me)
    n_match = max(int((supermajority_consensus(Oe, Me, 0.75) != ABSTAIN).sum()), 1)
    pool = (label == 1) if affirmed else np.ones(len(ge), bool)
    feats = agree_features(Oe, Me, label, cpos_mask, R_h1)
    scores = filter_scores(Oe, Me, label, R_h1, R_sg, cpos_mask, feats)

    R_O = disagreement_R(Oe, Me)
    d = {"rho_bar": mean_off_diagonal(R_O), "n_eff": effective_size(R_O),
         "eig_overlap": eigenvector_overlap(R_O, R_clean_uf, 3),
         "lco_flip": float(np.nanmean(feats["lco_flip"][pool]))}

    hard_pred = classify_regime_hard(d, ref)
    rho_bar_dev = (d["rho_bar"] - ref["rho_bar"]) / max(ref["rho_bar"], 1e-6)
    eig_drift = max(0.0, (0.90 - d["eig_overlap"]) / 0.90) * 1.5
    rho_S_dev = (feats["rho"] - ref["rho_S"]) / max(ref["rho_S"], 1e-6)
    pw, pg, ps = regime_probs(rho_bar_dev, eig_drift, rho_S_dev, feats["lco_flip"], TEMP, ALL_SIGNALS)
    pcts = {f: pct_rank(s) for f, s in scores.items()}
    soft = pw * pcts["supermajority_75"] + pg * pcts["corrfilter_small_gold_R"] + ps * pcts["bias_cluster"]
    soft_pred = REGIMES[int(np.argmax([pw.mean(), pg.mean(), ps.mean()]))]

    keeps = {f: keep_at_matched_retention(s, pool, n_match) for f, s in scores.items()}
    keeps["router_hard"] = keeps[REGIME_FILTER[hard_pred]]
    keeps["router_soft"] = keep_at_matched_retention(soft, pool, n_match)
    keeps["oracle_router"] = keeps[REGIME_FILTER[regime]]
    keeps["oracle_filter"] = keep_at_matched_retention(np.where(ge == 1, 1.0, -np.inf), pool, n_match)

    rows = []
    for method, keep in keeps.items():
        m = filter_metrics(keep, ge)
        m.update({"instance": name, "true_regime": regime, "method": method,
                  "pred_hard": hard_pred, "pred_soft": soft_pred})
        rows.append(m)
    diag = {"instance": name, "true_regime": regime, "pred_hard": hard_pred, "pred_soft": soft_pred,
            **{k: round(float(v), 4) for k, v in d.items()}}
    return rows, diag


def pooled(df_sel: pd.DataFrame) -> dict:
    nk = df_sel["n_kept"].sum()
    kc = df_sel["n_kept_clean"].sum()
    nc = df_sel["n_clean"].sum()
    ne = df_sel["n_eval"].sum()
    prec = kc / nk if nk else 0.0
    rec = kc / nc if nc else 0.0
    return {"precision": prec, "false_retention_rate": 1 - prec if nk else 0.0, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "retention_rate": nk / ne if ne else 0.0}


def select_mixture(inst_names_by_regime: dict[str, list[str]], mixture: dict[str, int]) -> list[str]:
    chosen = []
    for reg, k in mixture.items():
        chosen.extend(inst_names_by_regime[reg][:k])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(ROOT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    root = Path(args.project_root)
    out_dir = root / "outputs" / "mixed_regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    instances, lids, cpos_mask, R_h1, R_clean_uf, ref = build_instances(root)
    print(f"[37] built {len(instances)} instances "
          f"({pd.Series([i[1] for i in instances]).value_counts().to_dict()})")

    all_rows, diags = [], []
    for inst in instances:
        rows, diag = evaluate_instance(inst, lids, cpos_mask, R_h1, R_clean_uf, ref)
        all_rows.extend(rows)
        diags.append(diag)
        print(f"[37]   {diag['instance']}: true={diag['true_regime']} "
              f"hard={diag['pred_hard']} soft={diag['pred_soft']}")
    res = pd.DataFrame(all_rows)
    diag_df = pd.DataFrame(diags)
    res.to_csv(out_dir / "results_by_instance.csv", index=False)
    diag_df.to_csv(out_dir / "instance_diagnostics.csv", index=False)

    # ---- router accuracy on the full instance pool ----
    acc_rows = []
    for router in ["hard", "soft"]:
        y_true, y_pred = diag_df["true_regime"], diag_df[f"pred_{router}"]
        recalls = []
        for r in REGIMES:
            sel = y_true == r
            recalls.append(float((y_pred[sel] == r).mean()))
        conf = pd.crosstab(y_true, y_pred).reindex(index=REGIMES, columns=REGIMES, fill_value=0)
        conf.to_csv(out_dir / f"router_confusion_{router}.csv")
        acc_rows.append({"router": router, "n_instances": len(diag_df),
                         "accuracy": round(float((y_true == y_pred).mean()), 3),
                         "balanced_accuracy": round(float(np.mean(recalls)), 3),
                         **{f"recall_{r}": round(rc, 3) for r, rc in zip(REGIMES, recalls)}})
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(out_dir / "router_accuracy_mixed.csv", index=False)

    # ---- mixtures ----
    names_by_regime = {r: [i[0] for i in instances if i[1] == r] for r in REGIMES}
    rng = np.random.default_rng(BOOT_SEED)
    mix_rows, ci_rows, decomp_rows = [], [], []
    for mix_name, mix in MIXTURES.items():
        chosen = select_mixture(names_by_regime, mix)
        sel = res[res["instance"].isin(chosen)]

        pooled_by_method = {m: pooled(sel[sel["method"] == m]) for m in sel["method"].unique()}
        best_fixed = max(FIXED_FILTERS, key=lambda m: pooled_by_method[m]["precision"])
        for m, pm in pooled_by_method.items():
            mix_rows.append({"mixture": mix_name, "method": m, "n_instances": len(chosen),
                             "is_best_fixed": m == best_fixed,
                             **{k: round(v, 4) for k, v in pm.items()}})
        # per-regime pooled within mixture
        for reg in REGIMES:
            sub = sel[sel["true_regime"] == reg]
            for m in ("naive_majority", best_fixed, "router_hard", "router_soft", "oracle_router"):
                pm = pooled(sub[sub["method"] == m])
                mix_rows.append({"mixture": f"{mix_name}:{reg}", "method": m,
                                 "n_instances": len(sub["instance"].unique()),
                                 "is_best_fixed": m == best_fixed,
                                 **{k: round(v, 4) for k, v in pm.items()}})

        # ---- instance-stratified bootstrap over mixture gains ----
        per_inst = {m: sel[sel["method"] == m].set_index("instance")
                    for m in ["naive_majority", best_fixed, "router_hard", "router_soft", "oracle_router"]}
        chosen_by_reg = {r: [c for c in chosen if per_inst["naive_majority"].loc[c, "true_regime"] == r]
                         for r in REGIMES}
        gain_specs = [("router_hard", "naive_majority"), ("router_hard", best_fixed),
                      ("router_soft", "naive_majority"), ("router_soft", best_fixed),
                      ("oracle_router", best_fixed), ("router_hard", "oracle_router"),
                      ("router_soft", "oracle_router")]
        draws = {g: [] for g in gain_specs}
        for _ in range(args.n_boot):
            boot = []
            for r in REGIMES:
                pool_r = chosen_by_reg[r]
                if pool_r:
                    boot.extend(list(rng.choice(pool_r, size=len(pool_r), replace=True)))
            for a, b in gain_specs:
                pa = pooled(per_inst[a].loc[boot])
                pb = pooled(per_inst[b].loc[boot])
                draws[(a, b)].append(pa["precision"] - pb["precision"])
        for (a, b), dd in draws.items():
            point = pooled_by_method[a]["precision"] - pooled_by_method[b]["precision"]
            ci_rows.append({"mixture": mix_name, "comparison": f"{a} - {b}",
                            "gain_pts": round(100 * point, 2),
                            "ci95_low_pts": round(100 * float(np.quantile(dd, 0.025)), 2),
                            "ci95_high_pts": round(100 * float(np.quantile(dd, 0.975)), 2),
                            "significant": bool(np.quantile(dd, 0.025) > 0 or np.quantile(dd, 0.975) < 0)})

        # ---- router-oracle gap decomposition (hard router; exact identity) ----
        dsel = diag_df[diag_df["instance"].isin(chosen)]
        mis = dsel[dsel["pred_hard"] != dsel["true_regime"]]
        costs = []
        for inst_name in mis["instance"]:
            pr = per_inst["router_hard"].loc[inst_name, "precision"]
            po = per_inst["oracle_router"].loc[inst_name, "precision"]
            costs.append(po - pr)
        gap = pooled_by_method["oracle_router"]["precision"] - pooled_by_method["router_hard"]["precision"]
        headroom = pooled_by_method["oracle_router"]["precision"] - pooled_by_method[best_fixed]["precision"]
        decomp_rows.append({
            "mixture": mix_name, "best_fixed": best_fixed,
            "misroute_rate": round(len(mis) / len(chosen), 3),
            "mean_cost_per_misroute_pts": round(100 * float(np.mean(costs)), 2) if costs else 0.0,
            "router_oracle_gap_pts": round(100 * gap, 2),
            "oracle_headroom_over_best_fixed_pts": round(100 * headroom, 2),
        })

    mix_df = pd.DataFrame(mix_rows)
    ci_df = pd.DataFrame(ci_rows)
    dec_df = pd.DataFrame(decomp_rows)
    mix_df.to_csv(out_dir / "mixture_summary.csv", index=False)
    ci_df.to_csv(out_dir / "gains_ci.csv", index=False)
    dec_df.to_csv(out_dir / "gap_decomposition.csv", index=False)

    with open(out_dir / "summary.md", "w") as f:
        f.write("# Mixed-regime end-to-end routing benchmark\n\n")
        f.write(f"{len(instances)} synthetic instances (18 per regime; regime label = generator, "
                "by construction). Protocol identical to the regime/unified routers in scripts/routing_selector/: small_gold R from a "
                f"{SG_SIZE}-item labeled calibration split, held-out eval, matched retention at "
                "supermajority-0.75. Pooled metrics; instance-stratified bootstrap "
                f"({args.n_boot} draws, seed {BOOT_SEED}).\n\n")
        f.write("## Router accuracy on the instance pool\n\n")
        f.write(acc_df.to_markdown(index=False))
        f.write("\n\n## Pooled precision by mixture (main methods)\n\n")
        top = mix_df[~mix_df["mixture"].str.contains(":")]
        f.write(top.pivot(index="method", columns="mixture", values="precision").to_markdown())
        f.write("\n\n## Gains with 95% CIs (precision points)\n\n")
        f.write(ci_df.to_markdown(index=False))
        f.write("\n\n## Router-oracle gap decomposition (hard router)\n\n")
        f.write(dec_df.to_markdown(index=False))
        f.write("\n")
    print(f"[37] wrote {out_dir}")
    print(acc_df.to_string(index=False))
    print(dec_df.to_string(index=False))


if __name__ == "__main__":
    main()

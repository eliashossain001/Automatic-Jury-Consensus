#!/usr/bin/env python
"""Router-upgrade research: build an enlarged mixed-regime instance pool with
extended routing features and cached per-instance downstream filter metrics.

Research code only; the paper and the scripts/routing_selector/run_mixed_regime_benchmark.py benchmark are untouched. The
pool reuses the three paper generators (analysis-only vote replay):
  weak      content-poisoned UF, rates {5, 7.5, 10, 15, 20}% x 12 seeds  (60)
  subgroup  position-aligned UF, same grid                               (60)
  global    CFI biased banks, 3 mechanisms x ratios {0.5, 0.75, 1.0} x 7 seeds (63)
Each instance is evaluated with the scripts/routing_selector/run_mixed_regime_benchmark.py protocol (same filters, matched
retention, held-out eval after a 100-item small_gold calibration split), and a
set of label-free deployment features is extracted from the SAME eval split the
filters see. `config_id` identifies the generator parameterisation and is the
grouping unit for leakage-safe splits.

Usage: python experiments/router_upgrade/build_pool.py
Outputs -> outputs/router_upgrade/{pool_features.csv, pool_methods.csv}
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.linalg import eigh  # noqa: E402

from corrfilter.cfi.adaptive_r import disagreement_R  # noqa: E402
from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import majority_consensus  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.routing import (  # noqa: E402
    agree_features, load_votes_swaps, position_poison_mask,
    position_sensitivity_cluster)
from corrfilter.voting import VoteCache  # noqa: E402

spec37 = importlib.util.spec_from_file_location("mrb", ROOT / "scripts" / "routing_selector/run_mixed_regime_benchmark.py")
mrb = importlib.util.module_from_spec(spec37)
spec37.loader.exec_module(mrb)

BASE = 20260901           # distinct from scripts/routing_selector/run_mixed_regime_benchmark.py's instance seeds
RATES = [0.05, 0.075, 0.10, 0.15, 0.20]
N_SEEDS_UF = 8
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]
CFI_RATIOS = [0.5, 0.75, 1.0]
N_SEEDS_CFI = 5
UF_SIZE, CFI_SIZE = 300, 300


def extract_features(Oe, Me, label, feats, R_O, R_clean_uf, R_h1, ref, cpos_mask):
    """Label-free deployment features from the eval split (no gold anywhere)."""
    Mb = Me.astype(bool)
    n_j = Oe.shape[1]
    frac = np.where(Mb.sum(1) > 0, ((Oe == 1) & Mb).sum(1) / np.maximum(Mb.sum(1), 1), 0.5)
    margin = np.abs(frac - 0.5)
    pf = np.clip(frac, 1e-6, 1 - 1e-6)
    entropy = -(pf * np.log2(pf) + (1 - pf) * np.log2(1 - pf))

    # pairwise judge disagreement rates
    pdis = []
    for i in range(n_j):
        for j in range(i + 1, n_j):
            both = Mb[:, i] & Mb[:, j]
            if both.sum() >= 20:
                pdis.append(float((Oe[both, i] != Oe[both, j]).mean()))
    pdis = np.array(pdis) if pdis else np.array([0.0])

    off = R_O[~np.eye(n_j, dtype=bool)]
    ev = np.sort(eigh((R_O + R_O.T) / 2, eigvals_only=True))[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum() if ev.sum() > 0 else 1.0

    # spectral 2-partition of |R| graph: Fiedler value + cross/within contrast
    W = np.abs((R_O + R_O.T) / 2)
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    Lap = np.diag(d) - W
    lev, lvec = eigh(Lap)
    fiedler = float(lev[1]) if len(lev) > 1 else 0.0
    part = lvec[:, 1] >= 0
    if 0 < part.sum() < n_j:
        within = np.concatenate([R_O[np.ix_(part, part)][~np.eye(part.sum(), dtype=bool)],
                                 R_O[np.ix_(~part, ~part)][~np.eye((~part).sum(), dtype=bool)]])
        cross = R_O[np.ix_(part, ~part)].ravel()
        part_contrast = float(np.mean(within) - np.mean(cross)) if len(within) else 0.0
    else:
        part_contrast = 0.0

    # leave-one-judge-out majority flip rate, per judge -> max/mean
    lojo = []
    for j in range(n_j):
        keep = np.ones(n_j, bool)
        keep[j] = False
        Ok, Mk = Oe[:, keep], Me[:, keep]
        lab_k = majority_consensus(Ok, Mk)
        lojo.append(float((lab_k != label).mean()))
    lojo = np.array(lojo)

    return {
        # base (heuristic router's information set)
        "rho_bar": float(np.mean(off)),
        "n_eff": float(n_j / (1 + (n_j - 1) * np.mean(off))),
        "eig_overlap": float(mrb.eigenvector_overlap(R_O, R_clean_uf, 3)),
        "lco_mean": float(np.nanmean(feats["lco_flip"])),
        "conc_mean": float(np.nanmean(feats["conc"])),
        "rho_S_mean": float(np.nanmean(feats["rho"])),
        "disagree_mean": float(np.nanmean(feats["disagree"])),
        # vote-distribution features
        "margin_mean": float(margin.mean()), "margin_std": float(margin.std()),
        "margin_q25": float(np.quantile(margin, 0.25)),
        "entropy_mean": float(entropy.mean()),
        "near_tie_frac": float((margin < 0.1).mean()),
        "unanimous_frac": float((margin > 0.45).mean()),
        # pairwise disagreement
        "pdis_mean": float(pdis.mean()), "pdis_std": float(pdis.std()),
        "pdis_min": float(pdis.min()), "pdis_max": float(pdis.max()),
        # correlation-matrix shape
        "off_std": float(off.std()), "off_min": float(off.min()), "off_max": float(off.max()),
        "neg_frac": float((off < 0).mean()),
        "neg_mean": float(off[off < 0].mean()) if (off < 0).any() else 0.0,
        "top1_share": float(ev[0] / tot), "top2_share": float(ev[:2].sum() / tot),
        "spec_gap": float((ev[0] - ev[1]) / max(ev[0], 1e-9)) if len(ev) > 1 else 0.0,
        "eig_rank_ratio": float((tot ** 2) / max((ev ** 2).sum(), 1e-9) / n_j),
        "frob_drift_clean": float(np.linalg.norm(R_O - R_clean_uf)),
        "frob_drift_h1": float(np.linalg.norm(R_O - R_h1)),
        "rho_dev": float((np.mean(off) - ref["rho_bar"]) / max(ref["rho_bar"], 1e-6)),
        # graph
        "fiedler": fiedler, "part_contrast": part_contrast,
        # stability / cluster consistency
        "lojo_max": float(lojo.max()), "lojo_mean": float(lojo.mean()),
        "conc_std": float(np.nanstd(feats["conc"])),
        "lco_std": float(np.nanstd(feats["lco_flip"])),
    }


def main() -> None:
    out = ROOT / "outputs/router_upgrade"
    out.mkdir(exist_ok=True)

    bank_cfg = load_bank_config(ROOT / "configs/judge_bank.yaml")
    lids = [s.logical_id for s in bank_cfg.specs]
    man = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    item_ids = man["item_id"].astype(str).tolist()
    V, M, Sw = load_votes_swaps(VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"),
                                lids, item_ids)
    n_uf = len(item_ids)
    cpos_mask = position_sensitivity_cluster(V, M, Sw)
    R_h1 = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)["R"]
    R_clean_uf = disagreement_R(V, M)
    lab_clean = majority_consensus(V, M)
    feat_clean = agree_features(V, M, lab_clean, cpos_mask, R_h1)
    ref = {"rho_bar": float(np.mean(R_clean_uf[~np.eye(len(lids), dtype=bool)])),
           "rho_S": float(np.nanmean(feat_clean["rho"])),
           "lco_flip": float(np.nanmean(feat_clean["lco_flip"]))}

    poison20 = man["poisoned_20"].to_numpy(bool)
    pool_p, pool_c = np.where(poison20)[0], np.where(~poison20)[0]

    instances = []
    counter = 0

    def rng_for(k):
        return np.random.default_rng(BASE + 104729 * k)

    for s in range(N_SEEDS_UF):
        for rate in RATES:
            # weak: content-poisoned composition (poison pool limited to 400 items)
            rng = rng_for(counter)
            n_p = int(round(rate * UF_SIZE))
            pi = rng.choice(pool_p, size=n_p, replace=False)
            ci = rng.choice(pool_c, size=UF_SIZE - n_p, replace=False)
            idx = np.sort(np.concatenate([pi, ci]))
            pm = np.isin(idx, pi)
            O = V[idx].copy()
            O[pm] = 1 - O[pm]
            instances.append((f"weak_r{rate:g}_s{s}", "weak", f"weak_r{rate:g}", O, M[idx],
                              np.where(pm, 0, 1).astype(np.int8), True, counter))
            counter += 1
            # subgroup: position-aligned within a sampled subset
            rng = rng_for(counter)
            idx = np.sort(rng.choice(n_uf, size=UF_SIZE, replace=False))
            pm_full = position_poison_mask(V, M, lids, rate, subset_idx=idx)
            pm = pm_full[idx]
            O = V[idx].copy()
            O[pm] = 1 - O[pm]
            instances.append((f"subgroup_r{rate:g}_s{s}", "subgroup", f"subgroup_r{rate:g}", O, M[idx],
                              np.where(pm, 0, 1).astype(np.int8), True, counter))
            counter += 1

    cfg = yaml.safe_load((ROOT / "configs/cfi_bias_prompts.yaml").read_text())
    bias_bank = load_bias_bank(str(ROOT / "configs/cfi_bias_prompts.yaml"))
    cal = yaml.safe_load((ROOT / cfg["sources"]["h1_calibration_config"]).read_text())
    keep_ids = {l.strip() for l in (ROOT / "outputs/cfi/cfi_subset_items.txt").read_text().splitlines() if l.strip()}
    citems = [it for it in load_calibration_set(ROOT / cal["output"]["manifest_path"]) if it.item_id in keep_ids]
    clean_views = load_vote_views(ROOT / cfg["sources"]["h1_votes_dir"], lids)
    for mech in CFI_MECHS:
        bv = load_vote_views(ROOT / "outputs/cfi/votes" / mech, lids)
        for ratio in CFI_RATIOS:
            var = assemble_variant(citems, lids, bias_bank.by_name(mech), ratio, bv, clean_views, bias_bank.seed)
            gold_all = (majority_consensus(var.V, var.M) == 1).astype(np.int8)
            for s in range(N_SEEDS_CFI):
                rng = rng_for(counter)
                idx = np.sort(rng.choice(var.V.shape[0], size=min(CFI_SIZE, var.V.shape[0]), replace=False))
                instances.append((f"global_{mech[:8]}_x{int(ratio*100)}_s{s}", "global",
                                  f"global_{mech[:8]}_x{int(ratio*100)}", var.V[idx], var.M[idx],
                                  gold_all[idx], False, counter))
                counter += 1

    print(f"pool: {len(instances)} instances")
    feat_rows, method_rows = [], []
    for k, (name, regime, config_id, O, Md, gold, affirmed, cnt) in enumerate(instances):
        inst = (name, regime, O, Md, gold, affirmed, cnt)
        rows, diag = mrb.evaluate_instance(inst, lids, cpos_mask, R_h1, R_clean_uf, ref)
        for r in rows:
            method_rows.append({k2: r[k2] for k2 in ("instance", "true_regime", "method", "precision",
                                                     "n_kept", "n_kept_clean", "n_eval", "n_clean",
                                                     "recall", "false_retention_rate")})
        # recompute the eval-split observables for feature extraction (mirrors evaluate_instance)
        rng = np.random.default_rng(mrb.BASE_SEED + 104729 * (cnt + 1))
        cal_idx = rng.choice(len(gold), size=min(mrb.SG_SIZE, len(gold)), replace=False)
        ev_mask = np.ones(len(gold), bool)
        ev_mask[cal_idx] = False
        Oe, Me = O[ev_mask], Md[ev_mask]
        label = majority_consensus(Oe, Me)
        feats = agree_features(Oe, Me, label, cpos_mask, R_h1)
        R_O = disagreement_R(Oe, Me)
        f = extract_features(Oe, Me, label, feats, R_O, R_clean_uf, R_h1, ref, cpos_mask)
        f.update({"instance": name, "regime": regime, "config_id": config_id, "seed_rep": cnt,
                  "pred_hard": diag["pred_hard"], "pred_soft": diag["pred_soft"]})
        feat_rows.append(f)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(instances)}")

    pd.DataFrame(feat_rows).to_csv(out / "pool_features_eqsize.csv", index=False)
    pd.DataFrame(method_rows).to_csv(out / "pool_methods_eqsize.csv", index=False)
    print(f"wrote {out}/pool_features.csv ({len(feat_rows)} instances), pool_methods.csv")


if __name__ == "__main__":
    main()

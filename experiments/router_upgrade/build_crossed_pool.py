#!/usr/bin/env python
"""Redesigned routing benchmark: regime x source fully crossed (research only).

Fixes the source confound found in verification: every regime is now generated
from BOTH vote sources, so dataset identity carries no information about the
regime label.

Sources (cached votes, no new inference):
  uf  2,000 UltraFeedback pairs (judge_votes cache)
  rb  1,178 RewardBench calibration items (H1 vote cache)

Generators (identical construction on both sources; judge-level error processes
on a shared random set of truly-bad items, marginal-matched across regimes):
  Each instance samples 300 items uniformly from the source and marks a uniform
  random rate-fraction as truly bad (gold=0). Votes on good items are the
  source's real cached votes. On bad items the bank's error process is
  synthesised so that the EXPECTED wrong-vote fraction is the same 0.45 in all
  three regimes; only the JOINT dependence differs:
  weak      each judge affirms the bad item independently w.p. 0.45
  global    per-item shared shock z~Bern(4/9); all judges affirm w.p. 0.95 if
            z else 0.05 (bank-wide co-failure)
  subgroup  the position-sensitive cluster (cpos, the paper's vulnerable
            subgroup) co-fails via z~Bern(0.85) at 0.95/0.05; the remaining
            judges err independently at the rate that matches the 0.45 marginal
Because item selection and marginal error rates are identical across regimes,
neither dataset identity nor error rate carries regime information; only the
dependence structure does. The paper's CFI biased-bank instances (3 behavioural
mechanisms, ratio 1.0) are kept as extra rb-source global configs.

Grid: 3 regimes x 2 sources x rates {5,10,15,20}% x 8 seeds = 192, + 12 CFI
= 204 instances, all exactly 300 items (100-item small_gold calibration split,
200-item held-out eval; matched retention; same filters as scripts/37).
Fairness: the heuristic router gets a PER-SOURCE clean reference (rho/lco/rho_S
and clean R for eigenvector overlap), as a deployed heuristic would.

Construct validity is verified empirically per config (does each generator
produce its regime's diagnostic signature?) -> signature_check.csv.

Usage: python experiments/router_upgrade/build_crossed_pool.py
Outputs -> outputs/router_upgrade/crossed_{features,methods,signatures}.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP))

from build_pool import extract_features  # noqa: E402  (35-feature extractor)

from corrfilter.cfi.adaptive_r import disagreement_R, eigenvector_overlap, small_gold_R  # noqa: E402
from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import ABSTAIN, majority_consensus, supermajority_consensus  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.routing import (  # noqa: E402
    H1GAP, REGIME_FILTER, SG_SIZE, agree_features, classify_regime_hard,
    filter_metrics, filter_scores, keep_at_matched_retention, load_votes_swaps,
    position_sensitivity_cluster)
from corrfilter.voting import VoteCache  # noqa: E402

BASE = 20261001
RATES = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 8
INSTANCE_SIZE = 300
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]
N_SEEDS_CFI = 4


class Source:
    """A vote source: clean votes, swaps, per-source references and pools."""

    def __init__(self, name, V, M, Sw, len_true, len_worse, lids, R_h1):
        self.name, self.V, self.M, self.Sw = name, V, M, Sw
        self.n = V.shape[0]
        self.cpos = position_sensitivity_cluster(V, M, Sw)
        self.R_clean = disagreement_R(V, M)
        lab = majority_consensus(V, M)
        fc = agree_features(V, M, lab, self.cpos, R_h1)
        nj = V.shape[1]
        self.ref = {"rho_bar": float(np.mean(self.R_clean[~np.eye(nj, dtype=bool)])),
                    "rho_S": float(np.nanmean(fc["rho"])),
                    "lco_flip": float(np.nanmean(fc["lco_flip"]))}
        # weak pool: items where the LONGER response is the truly worse one
        self.weak_pool = np.where(len_worse > len_true)[0]
        w = np.array([H1GAP.get(l, 0.0) for l in lids]) / 100.0
        self.pos_score = ((1 - V) * M * w[None, :]).sum(1)                 # subgroup ranking
        self.glob_score = ((1 - V) * M).sum(1) / np.maximum(M.sum(1), 1)   # global ranking


WRONG_MARGINAL = 0.45  # expected wrong-vote fraction on bad items, ALL regimes
P_HI, P_LO = 0.95, 0.05  # co-failure error prob given shared shock z=1 / z=0
Z_SUBGROUP = 0.85        # subgroup-cluster shock rate


def marginal_params(nj: int, cl: np.ndarray):
    """(q_rest, p_z_global) that restore WRONG_MARGINAL under each generator."""
    n_cl = int(cl.sum())
    cl_marg = P_HI * Z_SUBGROUP + P_LO * (1 - Z_SUBGROUP)
    q_rest = (WRONG_MARGINAL * nj - cl_marg * n_cl) / max(nj - n_cl, 1)
    return float(np.clip(q_rest, 0.0, 1.0)), (WRONG_MARGINAL - P_LO) / (P_HI - P_LO)


def wrong_probs(regime: str, nj: int, cl: np.ndarray, rng, q_rest: float, p_z_global: float):
    """Per-judge wrong-vote probabilities for ONE bad item under a regime."""
    if regime == "global":
        z = rng.random() < p_z_global
        return np.full(nj, P_HI if z else P_LO)
    if regime == "subgroup":
        z = rng.random() < Z_SUBGROUP
        p = np.full(nj, q_rest)
        p[cl] = P_HI if z else P_LO
        return p
    return np.full(nj, WRONG_MARGINAL)  # weak: independent idiosyncratic errors


def make_instance(src: Source, regime: str, rate: float, seed_counter: int):
    """Sample items uniformly; inject regime-specific JOINT error structure.

    Item selection and the marginal wrong-vote rate on bad items are identical
    across regimes; only the dependence of the bank's errors differs.
    """
    rng = np.random.default_rng(BASE + 7919 * seed_counter)
    idx = np.sort(rng.choice(src.n, INSTANCE_SIZE, replace=False))
    O = src.V[idx].copy()
    Md = src.M[idx]
    nj = O.shape[1]
    n_bad = int(round(rate * INSTANCE_SIZE))
    bad = rng.choice(INSTANCE_SIZE, n_bad, replace=False)
    gold = np.ones(INSTANCE_SIZE, dtype=np.int8)
    gold[bad] = 0
    q_rest, p_z_global = marginal_params(nj, src.cpos)
    for t in bad:
        p = wrong_probs(regime, nj, src.cpos, rng, q_rest, p_z_global)
        wrong = rng.random(nj) < p
        # wrong vote on a truly-bad (gold=0) item = affirming it (vote 1)
        O[t] = np.where(Md[t] > 0, np.where(wrong, 1, 0), O[t])
    return O, Md, gold


def evaluate(src: Source, O, Md, gold, counter, R_h1, affirmed_only=True,
             retention_mode="nonabstain", selector_features_from_calibration=False):
    """scripts/37 protocol with PER-SOURCE references for heuristic fairness.

    retention_mode: "nonabstain" (default; the original protocol, matching the
    supermajority-0.75 non-abstain count) or "affirmed" (match the
    supermajority-0.75 AFFIRMED count). The default is degenerate on banks
    whose agreement is so concentrated that nearly every item clears the
    supermajority bar on one side (then n_match >= |affirmed pool| and every
    filter keeps everything); "affirmed" is the non-degenerate reading for
    such banks, applied identically to every method."""
    rng = np.random.default_rng(BASE + 104729 * (counter + 1))
    cal_idx = rng.choice(len(gold), size=SG_SIZE, replace=False)
    ev = np.ones(len(gold), bool)
    ev[cal_idx] = False
    R_sg = small_gold_R(O, Md, gold, cal_idx)
    Oe, Me, ge = O[ev], Md[ev], gold[ev]
    label = majority_consensus(Oe, Me)
    sm = supermajority_consensus(Oe, Me, 0.75)
    n_match = max(int((sm == 1).sum() if retention_mode == "affirmed"
                      else (sm != ABSTAIN).sum()), 1)
    pool = (label == 1) if affirmed_only else np.ones(len(ge), bool)
    feats = agree_features(Oe, Me, label, src.cpos, R_h1)
    scores = filter_scores(Oe, Me, label, R_h1, R_sg, src.cpos, feats)

    # The original routing benchmark is transductive: selector diagnostics are
    # computed from the same *unlabelled* deployment votes that are filtered.
    # For the stricter deployment-level selector experiment, compute every
    # diagnostic from the disjoint 100-item calibration split instead. Filter
    # outcomes remain measured only on the held-out evaluation split above.
    if selector_features_from_calibration:
        Of, Mf = O[cal_idx], Md[cal_idx]
        feature_label = majority_consensus(Of, Mf)
        feature_feats = agree_features(Of, Mf, feature_label, src.cpos, R_h1)
        R_O = disagreement_R(Of, Mf)
    else:
        Of, Mf = Oe, Me
        feature_label = label
        feature_feats = feats
        R_O = disagreement_R(Oe, Me)
    nj = Oe.shape[1]
    d = {"rho_bar": float(np.mean(R_O[~np.eye(nj, dtype=bool)])),
         "eig_overlap": float(eigenvector_overlap(R_O, src.R_clean, 3)),
         "lco_flip": float(np.nanmean(feature_feats["lco_flip"]))}
    pred_hard = classify_regime_hard(d, src.ref)
    keeps = {f: keep_at_matched_retention(s, pool, n_match) for f, s in scores.items()}
    keeps["oracle_filter"] = keep_at_matched_retention(np.where(ge == 1, 1.0, -np.inf), pool, n_match)
    metrics = {f: filter_metrics(k, ge) for f, k in keeps.items()}
    fvec = extract_features(Of, Mf, feature_label, feature_feats, R_O,
                            src.R_clean, R_h1, src.ref, src.cpos)
    if selector_features_from_calibration:
        # Trusted-label dependence summaries. These are available at the
        # proposed deployment interface and never use held-out labels.
        off = R_sg[~np.eye(R_sg.shape[0], dtype=bool)]
        eig = np.clip(np.linalg.eigvalsh((R_sg + R_sg.T) / 2), 0, None)[::-1]
        etot = max(float(eig.sum()), 1e-12)
        err = ((O[cal_idx] != gold[cal_idx, None]) & Md[cal_idx].astype(bool))
        denom = np.maximum(Md[cal_idx].sum(axis=0), 1)
        erate = err.sum(axis=0) / denom
        fvec.update({
            "gold_rho_bar": float(off.mean()),
            "gold_n_eff": float(R_sg.shape[0] /
                                (1 + (R_sg.shape[0] - 1) * off.mean())),
            "gold_top1_share": float(eig[0] / etot),
            "gold_top2_share": float(eig[:2].sum() / etot),
            "gold_eig_rank_ratio": float((etot ** 2) /
                                          max(float((eig ** 2).sum()), 1e-12) /
                                          R_sg.shape[0]),
            "gold_error_rate_mean": float(erate.mean()),
            "gold_error_rate_std": float(erate.std()),
            "gold_error_rate_min": float(erate.min()),
            "gold_error_rate_max": float(erate.max()),
        })
        # Direct calibration performance is also a legitimate deployment-level
        # signal: labels exist for these 100 items, while the outcome remains
        # the disjoint 200-item split. This provides both a transparent
        # choose-on-calibration baseline and inputs for a learned correction.
        cal_sm = supermajority_consensus(Of, Mf, 0.75)
        cal_n_match = max(int((cal_sm != ABSTAIN).sum()), 1)
        cal_pool = (feature_label == 1) if affirmed_only else np.ones(len(Of), bool)
        cal_scores = filter_scores(Of, Mf, feature_label, R_h1, R_sg,
                                   src.cpos, feature_feats)
        for method, score in cal_scores.items():
            keep = keep_at_matched_retention(score, cal_pool, cal_n_match)
            metric = filter_metrics(keep, gold[cal_idx])
            fvec[f"cal_precision__{method}"] = float(metric["precision"])
            fvec[f"cal_n_kept__{method}"] = float(metric["n_kept"])
    return metrics, fvec, pred_hard, d


def load_sources():
    """Load both vote sources and shared references. Returns (uf, rb, lids, R_h1)."""
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    lids = [s.logical_id for s in bank.specs]
    R_h1 = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)["R"]

    man_uf = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    uf_ids = man_uf.item_id.astype(str).tolist()
    Vu, Mu, Su = load_votes_swaps(VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"),
                                  lids, uf_ids)
    lt = np.where(man_uf.true_label.values == "A", man_uf.len_a.values, man_uf.len_b.values)
    lw = np.where(man_uf.true_label.values == "A", man_uf.len_b.values, man_uf.len_a.values)
    uf = Source("uf", Vu, Mu, Su, lt, lw, lids, R_h1)

    man_rb = pd.read_parquet(ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet")
    man_rb = man_rb.drop_duplicates("item_id", keep="last").reset_index(drop=True)
    rb_ids = man_rb.item_id.astype(str).tolist()
    Vr, Mr, Sr = load_votes_swaps(VoteCache(ROOT / "experiments/h1_measurement/votes"), lids, rb_ids)
    rb = Source("rb", Vr, Mr, Sr,
                man_rb.chosen.str.len().to_numpy(), man_rb.rejected.str.len().to_numpy(), lids, R_h1)
    return uf, rb, lids, R_h1


def main() -> None:
    out = ROOT / "outputs/router_upgrade"
    uf, rb, lids, R_h1 = load_sources()

    feat_rows, method_rows, sig_rows = [], [], []
    counter = 0
    for src in (uf, rb):
        for regime in ("weak", "subgroup", "global"):
            for rate in RATES:
                for s in range(N_SEEDS):
                    made = make_instance(src, regime, rate, counter)
                    counter += 1
                    if made is None:
                        continue
                    O, Md, gold = made
                    name = f"{src.name}_{regime}_r{rate:g}_s{s}"
                    cfg = f"{src.name}_{regime}_r{rate:g}"
                    metrics, fvec, pred_hard, d = evaluate(src, O, Md, gold, counter, R_h1)
                    for f, m in metrics.items():
                        method_rows.append({"instance": name, "true_regime": regime, "source": src.name,
                                            "method": f, **m})
                    fvec.update({"instance": name, "regime": regime, "source": src.name,
                                 "config_id": cfg, "pred_hard": pred_hard})
                    feat_rows.append(fvec)
                    sig_rows.append({"config": cfg, "regime": regime, "source": src.name,
                                     "rate": rate, **{k: round(v, 4) for k, v in d.items()}})

    # ---- extra global configs: the paper's CFI biased banks (rb source) ----
    cfg_y = yaml.safe_load((ROOT / "configs/cfi_bias_prompts.yaml").read_text())
    bb = load_bias_bank(str(ROOT / "configs/cfi_bias_prompts.yaml"))
    caly = yaml.safe_load((ROOT / cfg_y["sources"]["h1_calibration_config"]).read_text())
    keep = {l.strip() for l in (ROOT / "outputs/cfi/cfi_subset_items.txt").read_text().splitlines() if l.strip()}
    citems = [it for it in load_calibration_set(ROOT / caly["output"]["manifest_path"]) if it.item_id in keep]
    cviews = load_vote_views(ROOT / cfg_y["sources"]["h1_votes_dir"], lids)
    for mech in CFI_MECHS:
        bv = load_vote_views(ROOT / "outputs/cfi/votes" / mech, lids)
        var = assemble_variant(citems, lids, bb.by_name(mech), 1.0, bv, cviews, bb.seed)
        gold_all = (majority_consensus(var.V, var.M) == 1).astype(np.int8)
        for s in range(N_SEEDS_CFI):
            rng = np.random.default_rng(BASE + 7919 * counter)
            idx = np.sort(rng.choice(var.V.shape[0], INSTANCE_SIZE, replace=False))
            name = f"rb_globalCFI_{mech[:8]}_s{s}"
            cfg = f"rb_globalCFI_{mech[:8]}"
            metrics, fvec, pred_hard, d = evaluate(rb, var.V[idx], var.M[idx], gold_all[idx],
                                                   counter, R_h1, affirmed_only=False)
            for f, m in metrics.items():
                method_rows.append({"instance": name, "true_regime": "global", "source": "rb",
                                    "method": f, **m})
            fvec.update({"instance": name, "regime": "global", "source": "rb",
                         "config_id": cfg, "pred_hard": pred_hard})
            feat_rows.append(fvec)
            sig_rows.append({"config": cfg, "regime": "global", "source": "rb", "rate": 1.0,
                             **{k: round(v, 4) for k, v in d.items()}})
            counter += 1

    fdf = pd.DataFrame(feat_rows)
    fdf.to_csv(out / "crossed_features.csv", index=False)
    pd.DataFrame(method_rows).to_csv(out / "crossed_methods.csv", index=False)
    sig = pd.DataFrame(sig_rows).groupby(["config", "regime", "source"]).mean().reset_index()
    sig.to_csv(out / "crossed_signatures.csv", index=False)
    print(f"crossed pool: {len(fdf)} instances | regime x source counts:")
    print(fdf.groupby(["regime", "source"]).size())
    print("\nsignature check (per-config means; expectation: global rho_bar > clean ref, "
          "subgroup lco up / rho down, weak ~ clean):")
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()

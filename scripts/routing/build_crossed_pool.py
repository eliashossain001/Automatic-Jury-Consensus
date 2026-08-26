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
200-item held-out eval; matched retention; same filters as scripts/routing/superseded/mixed_regime_benchmark.py).
Fairness: the heuristic router gets a PER-SOURCE clean reference (rho/lco/rho_S
and clean R for eigenvector overlap), as a deployed heuristic would.

Construct validity is verified empirically per config (does each generator
produce its regime's diagnostic signature?) -> signature_check.csv.

Usage: python scripts/routing/build_crossed_pool.py
Outputs -> outputs/router_upgrade/crossed_{features,methods,signatures}.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from corrfilter.cfi.bias_bank import load_bias_bank
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views
from corrfilter.cfi.consensus import majority_consensus
from corrfilter.data import load_calibration_set
from corrfilter.judges import load_bank_config
from corrfilter.routing import load_votes_swaps
from corrfilter.screening.generators import BASE, INSTANCE_SIZE, Source, evaluate, make_instance
from corrfilter.voting import VoteCache

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]


RATES = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 8
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]
N_SEEDS_CFI = 4


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

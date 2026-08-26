#!/usr/bin/env python
"""Third routing source: PKU-SafeRLHF open-bank votes (800 items, 10 judges).

Builds (a) pure-regime crossed instances and (b) mixed-regime deployment
instances on the PKU source with the exact generators and evaluation protocol
of build_crossed_pool / build_mixed_pool. Counter offsets keep every RNG
stream disjoint from the uf/rb pools.

Usage: python scripts/routing/extend_pku.py
Outputs -> outputs/router_upgrade/pku_{features,methods}.csv (pure)
        -> outputs/router_upgrade/pku_mixed_{features,methods}.csv (mixed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from build_crossed_pool import N_SEEDS, RATES, Source, evaluate, make_instance

from corrfilter.judges import load_bank_config
from corrfilter.routing import load_votes_swaps
from corrfilter.screening.generators import majority_regime, make_mixed_instance, mixtures
from corrfilter.voting import VoteCache

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
MIX_RATES = [0.10, 0.20]  # = build_mixed_pool.RATES (name shadowed by crossed import)
MIX_SEEDS = 4             # = build_mixed_pool.N_SEEDS


PURE_OFFSET = 100_000   # counter offsets: disjoint RNG streams from uf/rb pools
MIX_OFFSET = 200_000


def load_pku_source(lids, R_h1) -> Source:
    man = pd.read_parquet(ROOT / "outputs/pku_saferlhf/pku_manifest.parquet")
    ids = man.item_id.astype(str).tolist()
    V, M, S = load_votes_swaps(
        VoteCache(ROOT / "results/strengthening_phase/pku_saferlhf/votes"), lids, ids)
    return Source("pku", V, M, S,
                  man.chosen.str.len().to_numpy(), man.rejected.str.len().to_numpy(),
                  lids, R_h1)


def main() -> None:
    out = ROOT / "outputs/router_upgrade"
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    lids = [s.logical_id for s in bank.specs]
    R_h1 = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz",
                   allow_pickle=True)["R"]
    pku = load_pku_source(lids, R_h1)
    print(f"pku source: {pku.n} items, cluster size {int(pku.cpos.sum())}")

    # ---- pure-regime instances (crossed-pool protocol) ----
    feat_rows, method_rows, sig_rows = [], [], []
    counter = PURE_OFFSET
    for regime in ("weak", "subgroup", "global"):
        for rate in RATES:
            for s in range(N_SEEDS):
                O, Md, gold = make_instance(pku, regime, rate, counter)
                name = f"pku_{regime}_r{rate:g}_s{s}"
                cfg = f"pku_{regime}_r{rate:g}"
                metrics, fvec, pred_hard, d = evaluate(pku, O, Md, gold, counter, R_h1)
                counter += 1
                for f, m in metrics.items():
                    method_rows.append({"instance": name, "true_regime": regime,
                                        "source": "pku", "method": f, **m})
                fvec.update({"instance": name, "regime": regime, "source": "pku",
                             "config_id": cfg, "pred_hard": pred_hard})
                feat_rows.append(fvec)
                sig_rows.append({"config": cfg, "regime": regime, "rate": rate,
                                 **{k: round(v, 4) for k, v in d.items()}})
    pd.DataFrame(feat_rows).to_csv(out / "pku_features.csv", index=False)
    pd.DataFrame(method_rows).to_csv(out / "pku_methods.csv", index=False)
    sig = pd.DataFrame(sig_rows).groupby(["config", "regime"]).mean().reset_index()
    print("\nsignature check (pku; expectation: global rho up, subgroup lco up, weak ~ clean):")
    print(sig.to_string(index=False))

    # ---- mixed-regime deployment instances ----
    feat_rows, method_rows = [], []
    counter = MIX_OFFSET
    for mix_name, weights in mixtures():
        for rate in MIX_RATES:
            for s in range(MIX_SEEDS):
                O, Md, gold, counts = make_mixed_instance(pku, weights, rate, counter)
                name = f"pku_{mix_name}_r{rate:g}_s{s}"
                cfg = f"pku_{mix_name}_r{rate:g}"
                metrics, fvec, pred_hard, d = evaluate(pku, O, Md, gold, counter, R_h1)
                counter += 1
                for f, m in metrics.items():
                    method_rows.append({"instance": name, "method": f, **m})
                fvec.update({"instance": name, "config_id": cfg, "source": "pku",
                             "mix_name": mix_name, "rate": rate,
                             "majority_regime": majority_regime(weights),
                             "pred_hard": pred_hard,
                             **{f"w_{r}": round(weights.get(r, 0.0), 4)
                                for r in ("weak", "global", "subgroup")},
                             **{f"n_bad_{r}": counts.get(r, 0)
                                for r in ("weak", "global", "subgroup")}})
                feat_rows.append(fvec)
    fdf = pd.DataFrame(feat_rows)
    fdf.to_csv(out / "pku_mixed_features.csv", index=False)
    pd.DataFrame(method_rows).to_csv(out / "pku_mixed_methods.csv", index=False)
    print(f"\npku pure: 96 instances; pku mixed: {len(fdf)} instances, "
          f"{fdf.config_id.nunique()} configs")


if __name__ == "__main__":
    main()

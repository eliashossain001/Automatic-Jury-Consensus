#!/usr/bin/env python
"""Mixed-regime deployment benchmark (research only; the paper is untouched).

Each deployment instance mixes MULTIPLE failure regimes: the uniform-random
bad-item set is partitioned among regimes according to mixture weights, and
each partition receives that regime's judge-level error process from
build_crossed_pool (matched 0.45 marginal wrong-vote rate; only the joint
dependence differs). Item selection, instance size (300), rates, and the
evaluation protocol are identical to the crossed pure-regime pool, so all
previously identified confounds (source, size, rate, marginal error rate)
remain controlled.

Grid: pairwise mixtures {weak+global, weak+subgroup, global+subgroup} at
second-component shares {0.10, 0.25, 0.50, 0.75, 0.90}, plus three-way
mixtures {equal, and 50/25/25 with each regime dominant}, x rates {10%, 20%}
x sources {uf, rb} x 4 seeds = (15 + 4) x 2 x 2 x 4 = 304 instances.

Routers are NEVER trained on this pool: it is deployment-only. The "majority
regime" (argmax mixture weight; 'none' for the equal mixture) is recorded for
secondary routing-accuracy analysis, not as a training target.

Usage: python scripts/routing/build_mixed_pool.py
Outputs -> outputs/router_upgrade/mixed_{features,methods}.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from build_crossed_pool import (
    evaluate,
    load_sources,
)

from corrfilter.screening.generators import majority_regime, make_mixed_instance, mixtures

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]


RATES = [0.10, 0.20]
N_SEEDS = 4


def main() -> None:
    out = ROOT / "outputs/router_upgrade"
    uf, rb, lids, R_h1 = load_sources()
    feat_rows, method_rows = [], []
    counter = 0
    for src in (uf, rb):
        for mix_name, weights in mixtures():
            for rate in RATES:
                for s in range(N_SEEDS):
                    O, Md, gold, counts = make_mixed_instance(src, weights, rate, counter)
                    name = f"{src.name}_{mix_name}_r{rate:g}_s{s}"
                    cfg = f"{src.name}_{mix_name}_r{rate:g}"
                    metrics, fvec, pred_hard, d = evaluate(src, O, Md, gold, counter, R_h1)
                    counter += 1
                    for fname, m in metrics.items():
                        method_rows.append({"instance": name, "method": fname, **m})
                    fvec.update({"instance": name, "config_id": cfg, "source": src.name,
                                 "mix_name": mix_name, "rate": rate,
                                 "majority_regime": majority_regime(weights),
                                 "pred_hard": pred_hard,
                                 **{f"w_{r}": round(weights.get(r, 0.0), 4)
                                    for r in ("weak", "global", "subgroup")},
                                 **{f"n_bad_{r}": counts.get(r, 0)
                                    for r in ("weak", "global", "subgroup")}})
                    feat_rows.append(fvec)
    fdf = pd.DataFrame(feat_rows)
    fdf.to_csv(out / "mixed_features.csv", index=False)
    pd.DataFrame(method_rows).to_csv(out / "mixed_methods.csv", index=False)
    print(f"mixed pool: {len(fdf)} instances, {fdf.config_id.nunique()} configs")
    print(fdf.groupby(["source", "majority_regime"]).size().to_string())


if __name__ == "__main__":
    main()

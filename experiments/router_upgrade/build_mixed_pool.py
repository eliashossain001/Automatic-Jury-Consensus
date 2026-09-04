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

Usage: python experiments/router_upgrade/build_mixed_pool.py
Outputs -> outputs/router_upgrade/mixed_{features,methods}.csv
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

from build_crossed_pool import (  # noqa: E402
    BASE, INSTANCE_SIZE, Source, evaluate, load_sources, marginal_params,
    wrong_probs)

MIX_BASE = 20261101
RATES = [0.10, 0.20]
N_SEEDS = 4
PAIR_SHARES = [0.10, 0.25, 0.50, 0.75, 0.90]
PAIRS = [("weak", "global"), ("weak", "subgroup"), ("global", "subgroup")]
THREEWAY = [
    {"weak": 1 / 3, "global": 1 / 3, "subgroup": 1 / 3},
    {"weak": 0.50, "global": 0.25, "subgroup": 0.25},
    {"weak": 0.25, "global": 0.50, "subgroup": 0.25},
    {"weak": 0.25, "global": 0.25, "subgroup": 0.50},
]


def mixtures():
    """Yield (mix_name, {regime: weight}) for the full grid."""
    for a, b in PAIRS:
        for s in PAIR_SHARES:
            yield f"{a[0]}{b[0]}_{int(round(100 * s))}", {a: 1 - s, b: s}
    for w in THREEWAY:
        tag = "".join(f"{r[0]}{int(round(100 * p))}" for r, p in sorted(w.items()))
        yield f"3way_{tag}", w


def majority_regime(weights: dict) -> str:
    mx = max(weights.values())
    top = [r for r, p in weights.items() if p == mx]
    return top[0] if len(top) == 1 else "none"


def make_mixed_instance(src: Source, weights: dict, rate: float, seed_counter: int):
    """Uniform item/bad-item sampling; bad items partitioned among regimes."""
    rng = np.random.default_rng(MIX_BASE + 7919 * seed_counter)
    idx = np.sort(rng.choice(src.n, INSTANCE_SIZE, replace=False))
    O = src.V[idx].copy()
    Md = src.M[idx]
    nj = O.shape[1]
    n_bad = int(round(rate * INSTANCE_SIZE))
    bad = rng.choice(INSTANCE_SIZE, n_bad, replace=False)
    gold = np.ones(INSTANCE_SIZE, dtype=np.int8)
    gold[bad] = 0
    # deterministic proportional partition (largest remainder), then shuffle
    regimes = sorted(weights)
    counts = {r: int(np.floor(weights[r] * n_bad)) for r in regimes}
    rem = n_bad - sum(counts.values())
    fracs = sorted(regimes, key=lambda r: -(weights[r] * n_bad - np.floor(weights[r] * n_bad)))
    for r in fracs[:rem]:
        counts[r] += 1
    perm = rng.permutation(bad)
    assign = {}
    pos = 0
    for r in regimes:
        for t in perm[pos:pos + counts[r]]:
            assign[t] = r
        pos += counts[r]
    q_rest, p_z_global = marginal_params(nj, src.cpos)
    for t in perm:
        p = wrong_probs(assign[t], nj, src.cpos, rng, q_rest, p_z_global)
        wrong = rng.random(nj) < p
        O[t] = np.where(Md[t] > 0, np.where(wrong, 1, 0), O[t])
    return O, Md, gold, counts


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

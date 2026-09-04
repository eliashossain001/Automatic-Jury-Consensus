#!/usr/bin/env python
"""Calibration-size study: small-gold label budget {25, 50, 100, 200}.

The filters are inference-time; the labelled calibration split per deployment
feeds small_gold_R (CorrFilter) and is the only per-deployment supervision.
This study rebuilds pure-regime instances at each label budget (uf and rb
sources, rates {10, 20}%, 4 seeds, all three regimes = 48 instances per size)
and reports, per budget: heuristic and frozen-logreg routing accuracy,
downstream pooled precision of routing vs the best fixed filter vs the
per-instance oracle. Router features do not use the calibration labels, so
the budget acts through filter quality and the evaluation split.

Usage: python experiments/router_upgrade/sg_size_study.py
Outputs -> outputs/router_upgrade/ext_sgsize.csv
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
from run_learned_router import (  # noqa: E402
    BASE_FEATS, FIXED, REGIME_FILTER, best_fixed_on, make_models,
    pooled_precision)
from run_mixed_eval import META  # noqa: E402

SG_SIZES = [25, 50, 100, 200]
RATES = [0.10, 0.20]
N_SEEDS = 4
OFFSET = 300_000


def main() -> None:
    out = ROOT / "outputs/router_upgrade"
    uf, rb, lids, R_h1 = bcp.load_sources()
    pure_f = pd.concat([pd.read_csv(out / "crossed_features.csv"),
                        pd.read_csv(out / "pku_features.csv")], ignore_index=True)
    cols = BASE_FEATS + [c for c in pure_f.columns if c not in BASE_FEATS + META]
    router = make_models(seed=0)["logreg"]
    router.fit(pure_f[cols].to_numpy(float), pure_f.regime.to_numpy())

    rows = []
    counter = OFFSET
    for sg in SG_SIZES:
        bcp.SG_SIZE_OVERRIDE = sg
        orig = bcp.SG_SIZE if hasattr(bcp, "SG_SIZE") else None
        bcp.SG_SIZE = sg
        feat_rows, method_rows = [], []
        for src in (uf, rb):
            for regime in ("weak", "subgroup", "global"):
                for rate in RATES:
                    for s in range(N_SEEDS):
                        O, Md, gold = bcp.make_instance(src, regime, rate, counter)
                        name = f"sg{sg}_{src.name}_{regime}_r{rate:g}_s{s}"
                        metrics, fvec, pred_hard, d = bcp.evaluate(src, O, Md, gold, counter, R_h1)
                        counter += 1
                        for f, m in metrics.items():
                            method_rows.append({"instance": name, "method": f, **m})
                        fvec.update({"instance": name, "regime": regime,
                                     "pred_hard": pred_hard})
                        feat_rows.append(fvec)
        if orig is not None:
            bcp.SG_SIZE = orig
        fdf = pd.DataFrame(feat_rows)
        md = pd.DataFrame(method_rows)
        insts = fdf.instance.to_numpy()
        y = fdf.regime.to_numpy()
        pred = router.predict(fdf[cols].to_numpy(float))
        dec_lr = {i: REGIME_FILTER[p] for i, p in zip(insts, pred)}
        dec_h = {i: REGIME_FILTER[p] for i, p in zip(insts, fdf.pred_hard)}
        bf = best_fixed_on(md, list(insts))
        dec_bf = dict.fromkeys(insts, bf)
        sub = md[md.method.isin(FIXED)].sort_values(["precision", "n_kept"], ascending=False)
        orc = sub.drop_duplicates("instance")
        dec_or = dict(zip(orc.instance, orc.method))
        rows.append({"sg_size": sg,
                     "heuristic_acc": round(float((fdf.pred_hard == y).mean()), 3),
                     "logreg_acc": round(float((pred == y).mean()), 3),
                     "ds_logreg": round(pooled_precision(md, dec_lr), 4),
                     "ds_heuristic": round(pooled_precision(md, dec_h), 4),
                     "ds_best_fixed": round(pooled_precision(md, dec_bf), 4),
                     "ds_perinstance_oracle": round(pooled_precision(md, dec_or), 4),
                     "ds_corrfilter": round(pooled_precision(
                         md, dict.fromkeys(insts, "corrfilter_small_gold_R")), 4)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out / "ext_sgsize.csv", index=False)


if __name__ == "__main__":
    main()

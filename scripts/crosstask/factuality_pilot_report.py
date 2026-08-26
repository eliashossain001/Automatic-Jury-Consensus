#!/usr/bin/env python
"""Judge-competence pilot gate for the pointwise factuality bank (Sec. 7).

Reads cached votes for a stage manifest and reports: per-judge committed-vote
accuracy, majority accuracy, coverage/abstention, class-conditioned accuracy,
mean error correlation (Ledoit-Wolf), effective ensemble size, jointly-wrong
counts, and parse-failure rate. Applies the non-binding recommended gate and
prints a verdict with diagnosis hints; does not hard-fail.

Usage: python scripts/crosstask/factuality_pilot_report.py --stage pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.cfi.adaptive_r import disagreement_R
from corrfilter.correlation import correlation_shrunk
from corrfilter.data.generic_task import batch_from_cache
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache

ROOT = Path(__file__).resolve().parents[2]


OUT = ROOT / "outputs/factuality_bank"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    args = ap.parse_args()
    man = pd.read_parquet(OUT / f"manifest_{args.stage}.parquet")
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    models = sorted({s.base_id for s in bank.specs})
    lids = [f"{m}::{st}" for m in models for st in ("pw_direct", "pw_analysis")]
    cache = VoteCache(OUT / "votes")
    batch = batch_from_cache(man, cache, lids)
    V, M, gold = batch.judge_votes, batch.availability, batch.gold_labels
    n, nj = V.shape

    rows = []
    for j, lid in enumerate(lids):
        m = M[:, j].astype(bool)
        acc = float((V[m, j] == gold[m]).mean()) if m.any() else np.nan
        rows.append({
            "judge": lid, "coverage": round(float(m.mean()), 3),
            "acc_committed": round(acc, 3),
            "acc_supported": round(float((V[m & (gold == 1), j] == 1).mean()), 3),
            "acc_unsupported": round(float((V[m & (gold == 0), j] == 0).mean()), 3),
            "affirm_rate": round(float(V[m, j].mean()), 3),
        })
    per = pd.DataFrame(rows)
    print(per.to_string(index=False))

    maj = (np.where(M, V, 0).sum(1) > (M.sum(1) / 2)).astype(int)
    maj_acc = float((maj == gold).mean())
    E = np.where(M.astype(bool), (V != gold[:, None]).astype(float), np.nan)
    Emask = ~np.isnan(E).any(1)
    R, _ = correlation_shrunk(np.nan_to_num(E[Emask]))
    off = ~np.eye(nj, dtype=bool)
    rho = float(np.nanmean(R[off]))
    neff = nj / (1 + (nj - 1) * rho) if rho > -1 / (nj - 1) else nj
    Rv = disagreement_R(V, M)
    jointly_wrong = int(((np.where(M, V != gold[:, None], False)).sum(1) >= nj * 0.5).sum())
    maj_wrong = int((maj != gold).sum())
    summary = {
        "n_items": n, "supported_share": round(float(gold.mean()), 3),
        "mean_coverage": round(float(M.mean()), 3),
        "majority_accuracy": round(maj_acc, 3),
        "mean_error_correlation": round(rho, 3),
        "n_eff": round(float(neff), 2),
        "vote_corr_rho": round(float(np.nanmean(Rv[off])), 3),
        "majority_wrong_items": maj_wrong,
        "items_half_bank_wrong": jointly_wrong,
        "parse_failure_rate": round(float(1 - M.mean()), 3),
    }
    print("\nSummary:", summary)
    pd.DataFrame([summary]).to_csv(OUT / f"pilot_summary_{args.stage}.csv", index=False)
    per.to_csv(OUT / f"pilot_perjudge_{args.stage}.csv", index=False)

    # non-binding gate
    chance = max(gold.mean(), 1 - gold.mean())
    gate = {
        "majority_above_chance": maj_acc > chance + 0.05,
        "enough_majority_wrong": maj_wrong >= 15,
        "no_judge_parse_dominated": bool((per.coverage >= 0.7).all()),
    }
    print("Gate:", gate, "-> PASS" if all(gate.values()) else "-> ATTENTION")
    if not all(gate.values()):
        print("Diagnosis hints: check per-judge affirm_rate extremes (prompt "
              "mismatch/label orientation), coverage (parser), class-conditioned "
              "accuracy (imbalance), and raw responses in the cache.")


if __name__ == "__main__":
    main()

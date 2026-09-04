"""Bucket 2 — evaluate a trained judge bank vs its matched base bank (CPU, after inference).

Given a trained bank (its votes already produced by scripts/02) and the matched base
sub-bank, computes the full base-vs-trained dependence comparison + CorrFilter gain and
writes table_<label>_vs_base.csv. Method-agnostic: works for DPO, ORPO, or GRPO judges.

Usage:
  python scripts/eval_judge_bank.py --trained-bank configs/dpo_judge.yaml --label dpo \
      --base-bank configs/judge_bank.yaml \
      --trained-id qwen-2.5-7b-dpo::pairwise ... --base-id qwen-2.5-7b::pairwise ... \
      --out outputs/dpo_judges/table_dpo_vs_base.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from bootstrap_ci import paired_frr_gain  # noqa: E402
from compute_dependence import (dependence_metrics, estimate_R,  # noqa: E402
                                item_ids_from_manifest, load_votes)
from corrfilter.cfi.consensus import consensus_level  # noqa: E402
from corrfilter.evaluation import evaluable_and_correct  # noqa: E402


def supermaj_frr(V, M, gold):
    """False retention at the 0.75 supermajority operating point.

    P0-0 corrected: ties abstain rather than resolving to a fixed value (this caller
    passes a constant gold vector). See src/corrfilter/evaluation.py.
    """
    _, lab, correct = evaluable_and_correct(V, M, gold, tie_policy="abstain")
    keep = lab & (consensus_level(V, M) >= 0.75)
    nk = int(keep.sum())
    return round(1 - float((keep & correct).sum()) / max(nk, 1), 4)


def one_bank(bank, ids, item_ids, label):
    V, M, lids = load_votes(bank, item_ids, ids)
    gold = np.ones(len(item_ids), dtype=np.int8)
    row, R = dependence_metrics(V, M, gold, lids, label)
    row["supermajority_false_retention"] = supermaj_frr(V, M, gold)
    gain = paired_frr_gain(V, M, R, gold, retention=0.60)
    row["corrfilter_gain_pts_at_0.60"] = gain["gain_pts"]
    row["corrfilter_gain_ci"] = f"[{gain['ci_low_pts']}, {gain['ci_high_pts']}]"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained-bank", required=True)
    ap.add_argument("--base-bank", required=True)
    ap.add_argument("--manifest", default="experiments/h1_measurement/results/calibration_manifest.parquet")
    ap.add_argument("--label", default="dpo")
    ap.add_argument("--trained-id", action="append", default=None)
    ap.add_argument("--base-id", action="append", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    item_ids = item_ids_from_manifest(args.manifest)
    base = one_bank(args.base_bank, args.base_id, item_ids, "base_matched")
    trained = one_bank(args.trained_bank, args.trained_id, item_ids, f"{args.label}_trained")

    cols = ["label", "n_judges", "rho_bar", "n_eff", "eigen_rank", "majority_cofailure",
            "supermajority_false_retention", "abstention_rate",
            "corrfilter_gain_pts_at_0.60", "corrfilter_gain_ci"]
    df = pd.DataFrame([base, trained])[cols]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nΔrho_bar = {trained['rho_bar'] - base['rho_bar']:+.4f}  "
          f"Δn_eff = {trained['n_eff'] - base['n_eff']:+.3f}  -> {out}")


if __name__ == "__main__":
    main()

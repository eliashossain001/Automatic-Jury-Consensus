"""Bucket 3 — combine policy eval outputs into table_policy_comparison.csv with paired CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from bootstrap_ci import paired_diff_ci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--majority-eval", required=True, help="per-item eval CSV for majority policy")
    ap.add_argument("--corrfilter-eval", required=True, help="per-item eval CSV for corrfilter policy")
    ap.add_argument("--dataset-stats", default=None, help="dataset_stats.csv with contamination")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = pd.read_csv(args.majority_eval).set_index("item_id")
    c = pd.read_csv(args.corrfilter_eval).set_index("item_id")
    common = m.index.intersection(c.index)
    m, c = m.loc[common], c.loc[common]

    # paired bootstrap on per-item correctness (corrfilter - majority)
    acc = paired_diff_ci(c["correct"].values, m["correct"].values)
    margin = paired_diff_ci(c["margin"].values, m["margin"].values)

    rows = [
        {"policy": "majority_filtered", "reward_accuracy": round(float(m["correct"].mean()), 4),
         "mean_margin": round(float(m["margin"].mean()), 4)},
        {"policy": "corrfilter_filtered", "reward_accuracy": round(float(c["correct"].mean()), 4),
         "mean_margin": round(float(c["margin"].mean()), 4)},
        {"policy": "corrfilter - majority (paired)",
         "reward_accuracy": acc["mean"], "acc_ci_low": acc["ci_low"], "acc_ci_high": acc["ci_high"],
         "acc_p_corrfilter_better": acc["p_pos"],
         "mean_margin": margin["mean"], "margin_ci_low": margin["ci_low"], "margin_ci_high": margin["ci_high"]},
    ]
    if args.dataset_stats and Path(args.dataset_stats).exists():
        ds = pd.read_csv(args.dataset_stats).set_index("method")["label_contamination_vs_gold"].to_dict()
        rows[0]["train_contamination"] = ds.get("majority")
        rows[1]["train_contamination"] = ds.get("corrfilter")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {out}")
    print("HEADLINE: CorrFilter-filtered policy reward-accuracy advantage = "
          f"{acc['mean']:+.4f} (95% CI [{acc['ci_low']:+.4f}, {acc['ci_high']:+.4f}], "
          f"P better={acc['p_pos']})")


if __name__ == "__main__":
    main()

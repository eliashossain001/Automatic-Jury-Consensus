"""Bucket 3 — build preference datasets from GRPO-judge votes under two filters.

Given the GRPO judge votes on a preference set, produce two DPO training files at a
MATCHED retention:
  A) majority/supermajority-filtered  (naive consensus)
  B) CorrFilter alpha_subset-filtered (decorrelated)
Each kept item becomes a DPO record {prompt, chosen, rejected} using the JUDGE-assigned
preference (the retained majority label), not gold. When gold is available we also log
the label contamination (fraction of kept pairs whose judge label disagrees with gold).

Deterministic; CPU-only. Runs on the already-cached RewardBench GRPO votes, so it needs
no GPU. Outputs go under --out-dir (default results/strengthening_phase/downstream/data).

Usage:
  python scripts/build_filtered_dpo_data.py --bank configs/grpo_judge_bank.yaml \
      --manifest experiments/h1_measurement/results/calibration_manifest.parquet \
      --retention 0.60 --out-dir results/strengthening_phase/downstream/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402
from compute_dependence import estimate_R, load_votes  # noqa: E402

GRPO_IDS = [f"{m}::{s}" for m in ("qwen-2.5-7b-grpo", "mistral-7b-grpo", "phi-3.5-mini-grpo")
            for s in ("pairwise", "likert")]


def _topk(score, k, n):
    keep = np.zeros(n, dtype=bool)
    if k > 0:
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
        keep[order[:min(k, n)]] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/grpo_judge_bank.yaml")
    ap.add_argument("--manifest", default="experiments/h1_measurement/results/calibration_manifest.parquet")
    ap.add_argument("--retention", type=float, default=0.60)
    ap.add_argument("--out-dir", default="results/strengthening_phase/downstream/data")
    ap.add_argument("--judge-id", action="append", default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.20,
                    help="fraction of items reserved as a CLEAN gold-labeled eval set, "
                         "excluded from BOTH filters' training data (deterministic by item_id hash)")
    args = ap.parse_args()

    p = Path(args.manifest)
    man = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    man["item_id"] = man["item_id"].astype(str)
    item_ids = man["item_id"].tolist()
    # chosen/rejected columns: RewardBench manifest uses chosen/rejected + prompt
    has_cr = {"chosen", "rejected", "prompt"} <= set(man.columns)
    if not has_cr and {"response_a", "response_b", "true_label"} <= set(man.columns):
        man["chosen"] = np.where(man["true_label"].astype(str).str.upper() == "A", man["response_a"], man["response_b"])
        man["rejected"] = np.where(man["true_label"].astype(str).str.upper() == "A", man["response_b"], man["response_a"])
    man = man.set_index("item_id")

    V, M, lids = load_votes(args.bank, item_ids, args.judge_id or GRPO_IDS)
    gold = np.ones(len(item_ids), dtype=np.int8)  # chosen>rejected by construction => gold=1

    # deterministic clean holdout by item_id hash — excluded from BOTH training filters
    import hashlib
    def _hf(iid):
        return int(hashlib.md5(iid.encode()).hexdigest(), 16) % 1000 / 1000.0
    is_holdout = np.array([_hf(iid) < args.holdout_frac for iid in item_ids])
    train_mask = ~is_holdout

    R, estimator, min_joint, _ = estimate_R(V[train_mask], M[train_mask], gold[train_mask])
    n = len(item_ids)
    k = int(round(args.retention * int(train_mask.sum())))

    maj = majority_consensus(V, M)
    level = consensus_level(V, M)
    labellable = maj != ABSTAIN
    trainable = labellable & train_mask   # never train on the clean holdout items
    sc = {
        "majority": np.where(trainable, level, -np.inf),
        "corrfilter": np.where(trainable, corrfilter_score(V, M, R, maj).score, -np.inf),
    }

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # clean gold-labeled held-out eval set (disjoint from both training filters)
    eval_recs = []
    for i, iid in enumerate(item_ids):
        if not is_holdout[i]:
            continue
        row = man.loc[iid]
        eval_recs.append({"prompt": str(row["prompt"]), "chosen": str(row["chosen"]),
                          "rejected": str(row["rejected"]), "item_id": iid})
    (out / "eval_clean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in eval_recs))
    print(f"holdout    clean eval items = {len(eval_recs)} (gold-labeled, disjoint)")

    stats = []
    for method, score in sc.items():
        keep = _topk(score, k, n) & labellable
        recs, wrong = [], 0
        for i, iid in enumerate(item_ids):
            if not keep[i]:
                continue
            lab = int(maj[i])                       # judge-assigned preference (1 => chosen preferred)
            row = man.loc[iid]
            chosen, rejected = (row["chosen"], row["rejected"]) if lab == 1 else (row["rejected"], row["chosen"])
            if lab != int(gold[i]):
                wrong += 1
            recs.append({"prompt": str(row["prompt"]), "chosen": str(chosen),
                         "rejected": str(rejected), "item_id": iid, "judge_label": lab})
        f = out / f"dpo_{method}.jsonl"
        f.write_text("".join(json.dumps(r) + "\n" for r in recs))
        contamination = round(wrong / max(len(recs), 1), 4)
        stats.append({"method": method, "n_kept": len(recs), "retention": round(len(recs) / n, 4),
                      "label_contamination_vs_gold": contamination, "file": str(f)})
        print(f"{method:11s} kept={len(recs):4d} retention={len(recs)/n:.3f} contamination={contamination}")

    meta = {"bank": args.bank, "manifest": args.manifest, "retention": args.retention,
            "n_items": n, "estimator": estimator, "min_joint_votes": min_joint,
            "judges": lids, "datasets": stats}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(stats).to_csv(out / "dataset_stats.csv", index=False)
    print(f"-> {out}  (dataset_stats.csv, build_meta.json)")


if __name__ == "__main__":
    main()

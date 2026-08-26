"""No-matched-retention prepare — each filter keeps its NATURAL set (no backfill).

Tests the matched-retention hypothesis: previous runs forced every method to keep the
same 790 items, so bias_cluster's poison removal was offset by clean-for-clean swaps
(net correction only 37 pairs). Here each filter keeps exactly the items it would
naturally trust, with NO size-matching and NO backfill:

  naive_consensus : every majority-affirmed item (label == 1).
  bias_cluster    : majority-affirmed items that REMAIN affirmed after removing the H1
                    position cluster (leave-cluster-out non-flip) -> drops the items
                    that are only affirmed because the biased subgroup co-voted. This
                    is the natural "trust support outside the cluster" rule.
  oracle          : every truly-clean item (gold == 1).

Uses the pos10 position-aligned regime. Same train/eval split and same held-out clean
eval (400 items) as the matched-retention pos10 run, so results are comparable. Writes
data/<method>.jsonl, data/eval_clean.jsonl, results/dataset_stats.csv. No training here.

Usage: python no_matched_retention/prepare_natural_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# reuse the exact vote-matrix builder + record builder from the Phase 1 prepare
from prepare_dpo_datasets import build_vote_matrix, clean_eval_records, dpo_records

from corrfilter import downstream as C
from corrfilter.cfi.consensus import majority_consensus
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "no_matched_retention"                                    # this experiment's outputs


H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}


def noncluster_affirm_frac(O, M, label, cb_mask):
    """Per-item fraction of NON-cluster judges that affirm the bank label."""
    Mb = M.astype(bool)
    noncl = (~cb_mask)[None, :]
    agree = (O == label[:, None]) & Mb & noncl
    present = Mb & noncl
    num = agree.sum(1); den = present.sum(1)
    return np.divide(num, den, out=np.full(O.shape[0], 0.0), where=den > 0)


def main() -> None:
    cfg = C.load_config(RUN_DIR / "config.yaml")
    C.set_run_dir(cfg["run_dir"])
    C.seed_everything(cfg["seed"])
    C.ensure_dirs(C.DATA_DIR, C.RESULTS_DIR)
    d = cfg["data"]; tag = str(d["contamination"]); regime = d["regime"]

    man = pd.read_csv(C.resolve(d["manifest"]))
    man["item_id"] = man["item_id"].astype(str)
    item_ids = man["item_id"].tolist()
    text_cols = ["prompt", "response_a", "response_b"]
    valid_text = (man[text_cols].notna().all(axis=1)
                  & (man[text_cols].apply(lambda c: c.astype(str).str.strip() != "")).all(axis=1)).to_numpy()

    bank = load_bank_config(str(C.resolve(d["bank"])))
    logical_ids = [s.logical_id for s in bank.specs]
    cache = VoteCache(C.resolve(d["votes_dir"]))
    V, M = build_vote_matrix(cache, logical_ids, item_ids)

    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids])
    cb_mask = np.zeros(len(logical_ids), bool); cb_mask[np.argsort(-gaps)[:5]] = True

    # identical split to the matched-retention pos10 run (same seed, same n)
    rng = np.random.default_rng(cfg["seed"]); n = len(item_ids)
    perm = rng.permutation(n); n_eval = int(round(d["eval_fraction"] * n))
    eval_idx = np.sort(perm[:n_eval]); train_idx = np.sort(perm[n_eval:])

    poisoned = man[f"poisoned_{tag}"].to_numpy(dtype=bool)
    gold_full = np.where(poisoned, 0, 1).astype(np.int8)
    O_full = V.copy(); O_full[poisoned] = 1 - V[poisoned]

    O, Mtr, gold = O_full[train_idx], M[train_idx], gold_full[train_idx]
    valid_tr = valid_text[train_idx]
    label = majority_consensus(O, Mtr)
    affirmed = (label == 1) & valid_tr
    noncl_frac = noncluster_affirm_frac(O, Mtr, label, cb_mask)

    # NATURAL keep sets (no matched retention, no backfill)
    masks = {
        "naive_consensus": affirmed,
        "bias_cluster": affirmed & (noncl_frac >= 0.5),
        "oracle": (gold == 1) & valid_tr,
    }
    masks = {k: v for k, v in masks.items() if k in cfg["filters"]}

    man_train = man.iloc[train_idx].reset_index(drop=True)
    man_eval = man.iloc[eval_idx[valid_text[eval_idx]]].reset_index(drop=True)
    stated_col = f"stated_label_{tag}"

    stats = []
    for method, keep in masks.items():
        recs = dpo_records(man_train, keep, stated_col, method, regime)
        n_written = C.write_jsonl(C.DATA_DIR / f"{method}.jsonl", recs)
        n_pois = sum(r["is_poisoned"] for r in recs)
        purity = 1.0 - n_pois / n_written if n_written else 0.0
        stats.append({"filter_method": method, "train_size": n_written,
                      "poisoned_kept": n_pois, "label_purity": round(purity, 4)})
        print(f"  {method:18s} n={n_written:4d}  poisoned_kept={n_pois:3d}  purity={purity:.3f}")

    n_eval_written = C.write_jsonl(C.DATA_DIR / "eval_clean.jsonl", clean_eval_records(man_eval, regime))
    print(f"held-out clean eval: {n_eval_written} pairs (identical to matched-retention pos10)")

    df = pd.DataFrame(stats)
    # net-correction accounting vs naive
    naive_n = df.loc[df.filter_method == "naive_consensus", "train_size"].iloc[0]
    naive_p = df.loc[df.filter_method == "naive_consensus", "poisoned_kept"].iloc[0]
    if "bias_cluster" in masks:
        bc_n = df.loc[df.filter_method == "bias_cluster", "train_size"].iloc[0]
        bc_p = df.loc[df.filter_method == "bias_cluster", "poisoned_kept"].iloc[0]
        removed = naive_n - bc_n
        pois_removed = naive_p - bc_p
        print(f"\nbias_cluster vs naive: removed {removed} examples, of which {pois_removed} poisoned "
              f"({100*pois_removed/removed:.1f}% removal precision); "
              f"net correction {100*pois_removed/naive_n:.1f}% of naive set")
    df.to_csv(C.RESULTS_DIR / "dataset_stats.csv", index=False)
    print(f"\nwrote {C.RESULTS_DIR/'dataset_stats.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the contamination-scaling DPO datasets (config_scaling.yaml).

Part 1 (dose-response): fixed-size training sets with controlled contamination
{0,20,30,40}% from position-aligned label flips, nested so every level shares the
same clean core and lower levels' poisoned items are subsets of higher levels'.
Part 2 (filtered arms): from the 40%-position-poisoned train pool, each filter
selects items at matched retention via the Phase-1 `compute_keep_masks`.

Reuses the Phase-1 split (same seed/eval_fraction), record schema, and eval set,
so results are directly comparable. CPU-only; run before train_dpo_by_filter.py.

Usage:
  python scripts/downstream/prepare_scaling_datasets.py \
      [--config experiments/downstream_dpo_validation/config_scaling.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_dpo_datasets import (
    H1_POSITION_GAP,
    build_vote_matrix,
    clean_eval_records,
    compute_keep_masks,
)

from corrfilter import downstream as C
from corrfilter.cfi.consensus import ABSTAIN, majority_consensus, supermajority_consensus
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models


def flip(label: str) -> str:
    return "B" if label == "A" else "A"


def records_for(man: pd.DataFrame, idx: np.ndarray, poisoned: np.ndarray,
                method: str, regime: str) -> list[dict]:
    recs = []
    for i in idx:
        r = man.iloc[i]
        stated = flip(r.true_label) if poisoned[i] else r.true_label
        chosen, rejected = (r.response_a, r.response_b) if stated == "A" else (r.response_b, r.response_a)
        recs.append({
            "item_id": r.item_id, "prompt": r.prompt, "chosen": chosen, "rejected": rejected,
            "filter_method": method, "regime": regime, "stated_label": stated,
            "true_label": r.true_label, "is_poisoned": bool(stated != r.true_label),
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(EXP_DIR / "config_scaling.yaml"))
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    C.set_run_dir(cfg.get("run_dir", "contamination_scaling"))
    C.seed_everything(cfg["seed"])
    C.ensure_dirs(C.DATA_DIR, C.RESULTS_DIR)

    d, sc = cfg["data"], cfg["scaling"]
    regime = d["regime"]

    man = pd.read_csv(C.resolve(d["manifest"]))
    man["item_id"] = man["item_id"].astype(str)
    item_ids = man["item_id"].tolist()
    n = len(item_ids)
    text_cols = ["prompt", "response_a", "response_b"]
    valid_text = (man[text_cols].notna().all(axis=1)
                  & (man[text_cols].apply(lambda c: c.astype(str).str.strip() != "")).all(axis=1)).to_numpy()

    bank = load_bank_config(str(C.resolve(d["bank"])))
    logical_ids = [s.logical_id for s in bank.specs]
    npz = np.load(C.resolve(d["correlation_npz"]), allow_pickle=True)
    R_clean = npz["R"]
    cache = VoteCache(C.resolve(d["votes_dir"]))
    V, M = build_vote_matrix(cache, logical_ids, item_ids)

    # Phase-1-identical split.
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(n)
    n_eval = int(round(d["eval_fraction"] * n))
    eval_idx = np.sort(perm[:n_eval])
    train_idx = np.sort(perm[n_eval:])
    in_train = np.zeros(n, bool)
    in_train[train_idx] = True

    # Position-aligned poison pool: top raw_rate by H1-weighted wrong-vote mass.
    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids]) / 100.0
    pos_score = ((1 - V) * M * gaps[None, :]).sum(1)
    order = np.argsort(-pos_score, kind="stable")
    n_poison_raw = int(round(float(sc["position_raw_rate"]) * n))
    pm = np.zeros(n, bool)
    pm[order[:n_poison_raw]] = True
    print(f"poison pool: top {sc['position_raw_rate']:.0%} by position mass = {pm.sum()} items "
          f"({(pm & in_train).sum()} in train pool)")

    # ---- Part 1: nested dose-response sets ----
    N = int(sc["train_size"])
    rng2 = np.random.default_rng(cfg["seed"] + 1)
    clean_pool = np.where(in_train & ~pm & valid_text)[0]
    pois_pool = np.where(in_train & pm & valid_text)[0]
    clean_order = rng2.permutation(clean_pool)
    pois_order = rng2.permutation(pois_pool)
    max_c = max(float(c) for c in sc["levels"])
    clean_core = clean_order[: N - int(round(max_c * N))]   # shared by every level

    stats = []
    poisoned_flag = pm.copy()
    for c in sc["levels"]:
        c = float(c)
        n_p = int(round(c * N))
        idx = np.concatenate([clean_order[: N - n_p], pois_order[:n_p]])
        assert len(idx) == N and len(set(idx)) == N
        name = f"contam_{int(round(c * 100)):02d}"
        recs = records_for(man, np.sort(idx), poisoned_flag, name, regime)
        n_written = C.write_jsonl(C.DATA_DIR / f"{name}.jsonl", recs)
        n_pois = sum(r["is_poisoned"] for r in recs)
        stats.append({"filter_method": name, "train_size": n_written, "poisoned_kept": n_pois,
                      "label_purity": round(1 - n_pois / n_written, 4), "router_choice": ""})
        print(f"  {name}: n={n_written} poisoned={n_pois} ({n_pois/n_written:.1%}) "
              f"clean-core overlap={len(set(idx) & set(clean_core))}")

    # ---- Part 2: filtered arms on the 40%-poisoned train pool ----
    O_full = V.copy()
    O_full[pm] = 1 - V[pm]
    gold_full = np.where(pm, 0, 1).astype(np.int8)
    O, Mtr, gold = O_full[train_idx], M[train_idx], gold_full[train_idx]
    valid_tr = valid_text[train_idx]
    label = majority_consensus(O, Mtr)
    affirmed = (label == 1) & valid_tr
    n_match = max(int(((supermajority_consensus(O, Mtr, 0.75) != ABSTAIN) & valid_tr).sum()), 1)
    cb_idx = set(np.argsort(-gaps)[:5].tolist())
    cb_mask = np.array([j in cb_idx for j in range(len(logical_ids))])
    masks, _ = compute_keep_masks(O, Mtr, R_clean, gold, label, affirmed, n_match,
                                  cb_mask, list(sc["filtered_arms"]))
    print(f"filtered arms: matched retention n={n_match} of {len(gold)} train-pool items")
    for method, keep in masks.items():
        name = f"pos40_{method}"
        kept_global = train_idx[np.where(keep)[0]]
        recs = records_for(man, kept_global, poisoned_flag, name, regime)
        n_written = C.write_jsonl(C.DATA_DIR / f"{name}.jsonl", recs)
        n_pois = sum(r["is_poisoned"] for r in recs)
        stats.append({"filter_method": name, "train_size": n_written, "poisoned_kept": n_pois,
                      "label_purity": round(1 - n_pois / n_written, 4), "router_choice": ""})
        print(f"  {name}: n={n_written} poisoned={n_pois} ({n_pois/max(n_written,1):.1%})")

    # ---- eval set: identical to Phase 1 ----
    man_eval = man.iloc[eval_idx[valid_text[eval_idx]]].reset_index(drop=True)
    eval_recs = clean_eval_records(man_eval, regime)
    n_eval_written = C.write_jsonl(C.DATA_DIR / "eval_clean.jsonl", eval_recs)
    print(f"eval_clean.jsonl: {n_eval_written} records")

    pd.DataFrame(stats).to_csv(C.RESULTS_DIR / "dataset_stats.csv", index=False)
    print(f"-> {C.DATA_DIR}")


if __name__ == "__main__":
    main()

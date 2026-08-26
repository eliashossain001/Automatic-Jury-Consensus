"""Phase 1, step 1 — build DPO-ready datasets, one per filter method.

Reuses the cached judge votes and the poisoned manifest from the synthetic-poisoned
UltraFeedback experiment (the only filtered set with recoverable preference text).
For a chosen contamination rate we:

  1. Reconstruct the per-judge vote matrix exactly as ``scripts/attacks/eval_poisoned_uf.py``
     does: ``V[i,j]=1`` iff judge j picked the truly-better response; the observable
     "affirm-stated-label" vote is ``O = V`` on clean items and ``O = 1-V`` on poisoned
     items; ``gold`` is 1 (clean) / 0 (poisoned).
  2. Split items ONCE into a held-out CLEAN eval set (true labels) and a train pool,
     so every method is judged on the same untouched preferences.
  3. On the train pool, reconstruct keep-masks for each filter at MATCHED RETENTION
     (every method keeps the same number of items — the count passing supermajority-
     0.75), so the only variable across methods is which items survive, i.e. label
     quality, not quantity.
  4. Emit one ``data/<method>.jsonl`` of {prompt, chosen, rejected, filter_method,
     regime, ...} using the STATED (possibly poisoned) label the filter let through,
     plus ``data/eval_clean.jsonl`` from the held-out true labels, and
     ``results/dataset_stats.csv`` (size + label purity per method — already a real,
     fabrication-free signal of filter quality).

No new model inference. Does not modify any existing script.

Usage: python scripts/downstream/prepare_dpo_datasets.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from corrfilter import downstream as C
from corrfilter.cfi.consensus import (
    ABSTAIN,
    consensus_level,
    majority_consensus,
    supermajority_consensus,
    vote_fraction,
)
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold
from corrfilter.correlation.effective_size import mean_off_diagonal
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache

# H1-measured position slot-gap per judge; the top-5 form the vulnerable cluster.
# Copied verbatim from scripts/filters/bias_cluster.py (calibration-time definition).
H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}


def build_vote_matrix(cache: VoteCache, logical_ids, item_ids):
    """V[i,j]=1 iff judge j picked the truly-better response; M=coverage mask."""
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), dtype=np.int8)
    M = np.zeros((n, m), dtype=np.int8)
    idx = {it: i for i, it in enumerate(item_ids)}
    for j, lid in enumerate(logical_ids):
        dfj = cache.load(lid)
        if dfj is None:
            continue
        for r in dfj.itertuples(index=False):
            i = idx.get(str(r.item_id))
            if i is None:
                continue
            v = int(r.vote)
            if v == ABSTAIN:
                continue
            V[i, j] = v
            M[i, j] = 1
    return V, M


def bias_cluster_score(O, M, label, cb_mask):
    """Independent support = #agreeing judges OUTSIDE the H1 position cluster.

    Mirrors the ``bias_cluster`` score in scripts/filters/bias_cluster.py.
    """
    n = O.shape[0]
    Mb = M.astype(bool)
    score = np.full(n, -np.inf)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        s = int(agree.sum())
        if s == 0:
            continue
        score[i] = s - int((agree & cb_mask).sum())
    return score


def keep_top_affirmed(score, affirmed, n_keep):
    """Keep the top-n_keep items by score among those the bank affirms (label==1)."""
    s = np.where(affirmed, score, -np.inf)
    keep, _ = retention_match_threshold(s, n_keep)
    return keep & affirmed


def regime_router_choice(O, M, R_clean, delta_threshold=0.03):
    """Lightweight regime detector -> name of the filter to apply.

    Replicates the paper's diagnostic at the dataset level: the SIGN of the shift in
    mean off-diagonal error correlation on the attacked votes ``O`` (all items) versus
    the clean baseline rho-bar. Rising correlation => globally correlated co-failure
    (route to CorrFilter); falling => vulnerable subgroup (route to bias-cluster);
    flat => weak dependence (route to supermajority). Returns the chosen filter key.
    """
    base_rho = float(mean_off_diagonal(R_clean))
    Ob = O.astype(np.float64)
    err = Ob - Ob.mean(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        R_att = np.nan_to_num(np.corrcoef(err, rowvar=False), nan=0.0)
    att_rho = float(mean_off_diagonal(R_att))
    delta = att_rho - base_rho
    if delta > delta_threshold:
        return "corrfilter", base_rho, att_rho
    if delta < -delta_threshold:
        return "bias_cluster", base_rho, att_rho
    return "supermajority_75", base_rho, att_rho


def compute_keep_masks(O, M, R_clean, gold, label, affirmed, n_match, cb_mask, filters):
    """Return (masks, router_choice) at matched retention for the requested filters.

    ``masks`` is {method: bool keep-mask}; ``router_choice`` names the filter the
    regime router selected (empty string if router not requested).
    """
    masks: dict[str, np.ndarray] = {}
    margin = np.abs(vote_fraction(O, M) - 0.5)
    cf_score = np.nan_to_num(corrfilter_score(O, M, R_clean, label).score, nan=-np.inf)
    bc_score = bias_cluster_score(O, M, label, cb_mask)

    base = {
        "naive_consensus": margin,
        "supermajority_75": consensus_level(O, M),
        "corrfilter": cf_score,
        "bias_cluster": bc_score,
        "oracle": np.where(gold == 1, 1.0, -np.inf),
    }
    for key, score in base.items():
        if key in filters:
            masks[key] = keep_top_affirmed(score, affirmed, n_match)

    router_choice = ""
    if "router" in filters:
        router_choice, base_rho, att_rho = regime_router_choice(O, M, R_clean)
        chosen_score = base.get(router_choice, base["supermajority_75"])
        masks["router"] = keep_top_affirmed(chosen_score, affirmed, n_match)
        print(f"  router: rho_base={base_rho:.3f} rho_attacked={att_rho:.3f} -> chose '{router_choice}'")
    return masks, router_choice


def dpo_records(man_sub, keep_mask, stated_col, method, regime):
    """Build DPO {prompt, chosen, rejected, ...} rows for kept items, stated label."""
    recs = []
    kept = man_sub[keep_mask]
    for r in kept.itertuples(index=False):
        stated = getattr(r, stated_col)            # 'A' or 'B' (possibly poisoned)
        if stated == "A":
            chosen, rejected = r.response_a, r.response_b
        else:
            chosen, rejected = r.response_b, r.response_a
        recs.append({
            "item_id": r.item_id,
            "prompt": r.prompt,
            "chosen": chosen,
            "rejected": rejected,
            "filter_method": method,
            "regime": regime,
            "stated_label": stated,
            "true_label": r.true_label,
            "is_poisoned": bool(stated != r.true_label),
        })
    return recs


def clean_eval_records(man_eval, regime):
    """Held-out evaluation rows using the TRUE (clean) preference label."""
    recs = []
    for r in man_eval.itertuples(index=False):
        if r.true_label == "A":
            chosen, rejected = r.response_a, r.response_b
        else:
            chosen, rejected = r.response_b, r.response_a
        recs.append({
            "item_id": r.item_id,
            "prompt": r.prompt,
            "chosen": chosen,
            "rejected": rejected,
            "true_label": r.true_label,
            "regime": regime,
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    C.set_run_dir(cfg.get("run_dir", "."))
    C.seed_everything(cfg["seed"])
    C.ensure_dirs(C.DATA_DIR, C.RESULTS_DIR)

    d = cfg["data"]
    tag = str(d["contamination"])
    regime = d["regime"]
    filters = [f for f in cfg["filters"]]

    man = pd.read_csv(C.resolve(d["manifest"]))
    man["item_id"] = man["item_id"].astype(str)
    item_ids = man["item_id"].tolist()
    # Mark rows with missing/empty preference text as INELIGIBLE rather than dropping
    # them: dropping would change n and reshuffle the train/eval split, breaking
    # cross-level comparability. Ineligible items are never selected by any filter and
    # never enter the eval set, so the split and held-out set stay identical everywhere.
    text_cols = ["prompt", "response_a", "response_b"]
    valid_text = (man[text_cols].notna().all(axis=1)
                  & (man[text_cols].apply(lambda c: c.astype(str).str.strip() != "")).all(axis=1)
                  ).to_numpy()
    if (~valid_text).any():
        print(f"marking {int((~valid_text).sum())} items ineligible (missing text): "
              f"{man.loc[~valid_text, 'item_id'].tolist()}")
    bank = load_bank_config(str(C.resolve(d["bank"])))
    logical_ids = [s.logical_id for s in bank.specs]
    npz = np.load(C.resolve(d["correlation_npz"]), allow_pickle=True)
    R_clean = npz["R"]
    if [str(x) for x in npz["logical_ids"]] != logical_ids:
        raise RuntimeError("correlation.npz judge order != bank order")

    cache = VoteCache(C.resolve(d["votes_dir"]))
    V, M = build_vote_matrix(cache, logical_ids, item_ids)
    print(f"vote matrix {V.shape}, coverage {M.mean():.1%}")

    # Vulnerable cluster = H1 top-5 position-gap judges (calibration-time only).
    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids])
    cb_idx = set(np.argsort(-gaps)[:5].tolist())
    cb_mask = np.array([j in cb_idx for j in range(len(logical_ids))])
    print("vulnerable cluster:", [logical_ids[j] for j in sorted(cb_idx)])

    # One fixed split: held-out clean eval vs train pool.
    rng = np.random.default_rng(cfg["seed"])
    n = len(item_ids)
    perm = rng.permutation(n)
    n_eval = int(round(d["eval_fraction"] * n))
    eval_idx = np.sort(perm[:n_eval])
    train_idx = np.sort(perm[n_eval:])
    is_train = np.zeros(n, bool); is_train[train_idx] = True
    print(f"split: {len(train_idx)} train pool, {len(eval_idx)} held-out clean eval")

    # Poisoning + bank verdict on the FULL set, then restrict to the train pool.
    poisoned = man[f"poisoned_{tag}"].to_numpy(dtype=bool)
    gold_full = np.where(poisoned, 0, 1).astype(np.int8)
    O_full = V.copy()
    O_full[poisoned] = 1 - V[poisoned]

    O, Mtr, gold = O_full[train_idx], M[train_idx], gold_full[train_idx]
    valid_tr = valid_text[train_idx]
    R = R_clean
    label = majority_consensus(O, Mtr)
    affirmed = (label == 1) & valid_tr            # ineligible-text items never selected
    n_match = max(int(((supermajority_consensus(O, Mtr, 0.75) != ABSTAIN) & valid_tr).sum()), 1)
    print(f"contamination {tag}%: train poison frac {1 - gold.mean():.1%}, "
          f"matched retention n={n_match}/{len(gold)}")

    masks, router_choice = compute_keep_masks(
        O, Mtr, R, gold, label, affirmed, n_match, cb_mask, filters)

    man_train = man.iloc[train_idx].reset_index(drop=True)
    man_eval = man.iloc[eval_idx[valid_text[eval_idx]]].reset_index(drop=True)
    stated_col = f"stated_label_{tag}"

    stats = []
    for method, keep in masks.items():
        recs = dpo_records(man_train, keep, stated_col, method, regime)
        n_written = C.write_jsonl(C.DATA_DIR / f"{method}.jsonl", recs)
        n_pois = sum(r["is_poisoned"] for r in recs)
        purity = 1.0 - n_pois / n_written if n_written else 0.0
        row = {
            "filter_method": method,
            "train_size": n_written,
            "poisoned_kept": n_pois,
            "label_purity": round(purity, 4),
            "router_choice": router_choice if method == "router" else "",
        }
        stats.append(row)
        print(f"  {method:18s} n={n_written:4d}  poisoned_kept={n_pois:3d}  purity={purity:.3f}")

    eval_recs = clean_eval_records(man_eval, regime)
    n_eval_written = C.write_jsonl(C.DATA_DIR / "eval_clean.jsonl", eval_recs)
    print(f"held-out clean eval: {n_eval_written} pairs (true labels)")

    stats_df = pd.DataFrame(stats)
    pct = int("".join(c for c in str(tag) if c.isdigit()) or 0)
    stats_df.insert(1, "contamination_pct", pct)
    stats_df.to_csv(C.RESULTS_DIR / "dataset_stats.csv", index=False)
    print(f"\nwrote datasets to {C.DATA_DIR} and stats to "
          f"{C.RESULTS_DIR / 'dataset_stats.csv'}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()

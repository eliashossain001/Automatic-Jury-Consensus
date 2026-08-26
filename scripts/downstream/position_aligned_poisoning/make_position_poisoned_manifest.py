"""Experiment C helper — build a POSITION-aligned poisoned manifest (no new inference).

Unlike the content-heuristic poisoning of the synthetic-UF manifest, this reproduces
the paper's vulnerable-subgroup attack (scripts/attacks/position_aligned_poisoning.py): items are
poisoned in proportion to the position-bias-weighted mass of judges that picked the
truly-worse response, so a strongly position-biased subgroup co-affirms the bad label.
This is the regime where the bias-cluster filter is supposed to win and CorrFilter is
not (global correlation falls, not rises).

Selection (verbatim from scripts/attacks/position_aligned_poisoning.py):
    w_j   = H1 position-bias gap of judge j / 100
    score_i = sum_j w_j * (1 - V[i,j]) * M[i,j]      # mass of biased judges that erred
    order   = argsort(-score)                         # poison the highest-mass items
Rates are nested (10% subset of 20% subset of 30%) because they are top-k prefixes
of the same ordering. Poisoned items get stated_label "B"; true_label stays "A".

Reuses the UF manifest (text) and the cached votes; touches no original file. Writes
  experiments/downstream_dpo_validation/position_aligned_poisoning/poisoned_manifest_position.csv
with poisoned_pos10/pos20/pos30, heuristic_pos* (="position"), stated_label_pos*.

Usage: python make_position_poisoned_manifest.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.cfi.consensus import ABSTAIN
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "position_aligned_poisoning"                                    # this experiment's outputs


# H1-measured per-judge position-bias magnitude (pp); verbatim from scripts/attacks/position_aligned_poisoning.py.
H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}
RATES = {"pos10": 0.10, "pos20": 0.20, "pos30": 0.30}
SRC_MANIFEST = ROOT / "outputs" / "synthetic_poisoned_ultrafeedback" / "poisoned_manifest.csv"
VOTES_DIR = ROOT / "outputs" / "synthetic_poisoned_ultrafeedback" / "judge_votes"
BANK = ROOT / "configs" / "judge_bank.yaml"
OUT = RUN_DIR / "poisoned_manifest_position.csv"


def build_vote_matrix(cache, logical_ids, item_ids):
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), np.int8); M = np.zeros((n, m), np.int8)
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
            V[i, j] = v; M[i, j] = 1
    return V, M


def main() -> None:
    df = pd.read_csv(SRC_MANIFEST)
    df["item_id"] = df["item_id"].astype(str)
    item_ids = df["item_id"].tolist()
    bank = load_bank_config(str(BANK))
    logical_ids = [s.logical_id for s in bank.specs]

    cache = VoteCache(VOTES_DIR)
    V, M = build_vote_matrix(cache, logical_ids, item_ids)
    print(f"vote matrix {V.shape}, coverage {M.mean():.1%}")

    w = np.array([H1_POSITION_GAP.get(lid, 0.0) / 100.0 for lid in logical_ids])
    score = ((1 - V) * M * w[None, :]).sum(axis=1)   # position-weighted mass of erring judges
    order = np.argsort(-score, kind="stable")
    n = len(item_ids)

    for tag, rate in RATES.items():
        cnt = int(round(rate * n))
        poison_idx = set(order[:cnt].tolist())
        poisoned = np.array([i in poison_idx for i in range(n)])
        df[f"poisoned_{tag}"] = poisoned
        df[f"heuristic_{tag}"] = np.where(poisoned, "position", "")
        df[f"stated_label_{tag}"] = np.where(poisoned, "B", "A")
        print(f"{tag}: {cnt} poisoned ({poisoned.mean():.1%}); "
              f"mean score among poisoned={score[list(poison_idx)].mean():.3f} "
              f"vs overall {score.mean():.3f}")

    # sanity: nesting
    assert set(np.where(df["poisoned_pos10"])[0]).issubset(set(np.where(df["poisoned_pos20"])[0]))
    assert set(np.where(df["poisoned_pos20"])[0]).issubset(set(np.where(df["poisoned_pos30"])[0]))
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  (original manifest untouched)")


if __name__ == "__main__":
    main()

"""Task 2: seed contrast — do independently GRPO-trained seeds of the SAME model co-fail?

Compares qwen-2.5-7b GRPO seed-0 vs seed-1 (same base model, same reward, different
training seed) on RewardBench v2, per prompt style. Reports, on items both seeds
committed a vote:
  - vote agreement
  - error correlation (Pearson of error indicators)
  - conditional co-failure Pr[s1 wrong | s0 wrong] vs the marginal (chance) baseline
  - overlap (Jaccard) of the two seeds' wrong-items, and within consensus-retained items
  - effect of ADDING seed-1 to the 6-judge bank on rho_bar / n_eff

Conservative rule: seed diversity is flagged as failing ONLY when error correlation is
clearly positive AND conditional co-failure exceeds the marginal (above chance).
Seeds fixed. Writes seed_contrast_summary.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.cfi.consensus import ABSTAIN, majority_consensus  # noqa: E402
from corrfilter.correlation import compute_error_matrix_pairwise, correlation_shrunk_pairwise  # noqa: E402
from corrfilter.correlation.effective_size import effective_size, mean_off_diagonal  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

VOTES = ROOT / "results/grpo_judges/rewardbench_votes"
OUT = ROOT / "results/grpo_judges/seed_contrast"
S0, S1 = "qwen-2.5-7b-grpo", "qwen-2.5-7b-grpo-s1"


def pair_metrics(style: str, item_ids, gold):
    cache = VoteCache(VOTES)
    id0, id1 = f"{S0}::{style}", f"{S1}::{style}"
    if not (cache.exists(id0) and cache.exists(id1)):
        return None
    V, M = load_vote_matrix(cache, [id0, id1], item_ids)
    both = M[:, 0].astype(bool) & M[:, 1].astype(bool)
    if both.sum() < 30:
        return None
    v0, v1 = V[both, 0], V[both, 1]
    e0, e1 = (v0 != gold[both]).astype(float), (v1 != gold[both]).astype(float)
    agree = float((v0 == v1).mean())
    err_corr = float(np.corrcoef(e0, e1)[0, 1]) if e0.std() > 0 and e1.std() > 0 else float("nan")
    p1_wrong = float(e1.mean())
    cond = float((e0 * e1).sum() / e0.sum()) if e0.sum() > 0 else float("nan")
    # Jaccard of wrong-item sets
    w0, w1 = e0 > 0, e1 > 0
    jac = float((w0 & w1).sum() / max((w0 | w1).sum(), 1))
    return {
        "prompt_style": style,
        "n_committed_both": int(both.sum()),
        "vote_agreement": round(agree, 4),
        "error_correlation": round(err_corr, 4),
        "marginal_err_s1": round(p1_wrong, 4),
        "cond_cofailure_s1_given_s0": round(cond, 4),
        "cofailure_lift_vs_chance": round(cond / p1_wrong, 3) if p1_wrong > 0 else float("nan"),
        "wrong_item_jaccard": round(jac, 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = yaml.safe_load((ROOT / "configs/calibration.yaml").read_text())
    items = load_calibration_set(cal["output"]["manifest_path"])
    item_ids = [it.item_id for it in items]
    gold = np.ones(len(items), dtype=np.int8)

    rows = [r for style in ("pairwise", "likert") if (r := pair_metrics(style, item_ids, gold))]
    if not rows:
        print("seed-1 votes not available yet; nothing to do")
        return

    # ---- overlap of consensus-retained WRONG items (6-judge bank) ----
    bank = load_bank_config(str(ROOT / "configs/grpo_judge_bank.yaml"))
    lids = [s.logical_id for s in bank.specs]
    Vb, Mb = load_vote_matrix(VoteCache(VOTES), lids, item_ids)
    maj = majority_consensus(Vb, Mb)
    retained_wrong = (maj != ABSTAIN) & (maj != gold)  # consensus keeps it but it's wrong
    cache = VoteCache(VOTES)
    overlap_row = {}
    if cache.exists(f"{S1}::pairwise"):
        Vp, Mp = load_vote_matrix(cache, [f"{S0}::pairwise", f"{S1}::pairwise"], item_ids)
        both = Mp[:, 0].astype(bool) & Mp[:, 1].astype(bool)
        rw = retained_wrong & both
        s0w = rw & (Vp[:, 0] != gold)
        s1w = rw & (Vp[:, 1] != gold)
        overlap_row = {
            "consensus_retained_wrong_items": int(retained_wrong.sum()),
            "retained_wrong_both_committed": int(rw.sum()),
            "both_seeds_wrong_on_retained_wrong": int((s0w & s1w).sum()),
            "frac_retained_wrong_where_both_seeds_wrong":
                round(float((s0w & s1w).sum() / max(rw.sum(), 1)), 4),
        }

    # ---- effect of adding seed-1 (pairwise) to the 6-judge bank ----
    aug = {}
    if cache.exists(f"{S1}::pairwise"):
        V6, M6 = Vb, Mb
        Vs1, Ms1 = load_vote_matrix(cache, [f"{S1}::pairwise"], item_ids)
        V7 = np.concatenate([V6, Vs1], axis=1)
        M7 = np.concatenate([M6, Ms1], axis=1)
        E6, a6 = compute_error_matrix_pairwise(V6, M6, gold)
        R6, _, _ = correlation_shrunk_pairwise(E6, a6)
        E7, a7 = compute_error_matrix_pairwise(V7, M7, gold)
        R7, _, _ = correlation_shrunk_pairwise(E7, a7)
        aug = {
            "bank6_rho_bar": round(mean_off_diagonal(R6), 4),
            "bank6_n_eff": round(effective_size(R6), 3),
            "bank7_with_s1_rho_bar": round(mean_off_diagonal(R7), 4),
            "bank7_with_s1_n_eff": round(effective_size(R7), 3),
            "delta_n_eff_from_adding_s1": round(effective_size(R7) - effective_size(R6), 3),
        }

    df = pd.DataFrame(rows)
    for k, v in {**overlap_row, **aug}.items():
        df[k] = v
    df.to_csv(OUT / "seed_contrast_summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / 'seed_contrast_summary.csv'}")


if __name__ == "__main__":
    main()

"""Task 1 analysis: dependence + CorrFilter metrics for a GRPO bank at a given
max_new_tokens, appended as one row to abstention_robustness_summary.csv.

Computes, on COMMITTED votes (abstentions excluded, pairwise-complete estimator):
abstention rate, min joint votes per pair, rho_bar, n_eff, eigen rank,
majority co-failure, supermajority-0.75 false-retention, and the paired-bootstrap
CorrFilter false-retention gain at retention 0.60 (same methodology as the
primary result). Seeds fixed (BOOT_SEED).

Usage:
  python scripts/training/abstention_robustness.py --bank <cfg> --setting <label> \
      --max-new-tokens <n> --out results/grpo_judges/abstention_robustness/abstention_robustness_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus
from corrfilter.cfi.corrfilter_score import corrfilter_score
from corrfilter.correlation import (
    compute_error_matrix_pairwise,
    correlation_shrunk_pairwise,
)
from corrfilter.correlation.effective_size import (
    effective_eig_rank,
    effective_size,
    mean_off_diagonal,
)
from corrfilter.data import load_calibration_set
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache, load_vote_matrix

ROOT = Path(__file__).resolve().parents[2]


BOOT_SEED = 20260706


def _topk(score, k, n):
    keep = np.zeros(n, dtype=bool)
    if k > 0:
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
        keep[order[:min(k, n)]] = True
    return keep


def _frr(keep, correct, labellable, idx):
    kk = keep[idx] & labellable[idx]
    nk = int(kk.sum())
    return np.nan if nk == 0 else 1.0 - float((kk & correct[idx]).sum()) / nk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--calibration", default="configs/calibration.yaml")
    ap.add_argument("--setting", required=True, help="row label, e.g. grpo_tok8 / grpo_tok32")
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retention", type=float, default=0.60)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    cal = yaml.safe_load(Path(args.calibration).read_text())
    items = load_calibration_set(cal["output"]["manifest_path"])
    gold = np.ones(len(items), dtype=np.int8)
    n = len(items)

    bank = load_bank_config(args.bank)
    lids = [s.logical_id for s in bank.specs]
    V, M = load_vote_matrix(VoteCache(bank.votes_dir), lids, [it.item_id for it in items])
    M_b = M.astype(bool)

    # dependence on committed votes (pairwise-complete + shrinkage)
    E, avail = compute_error_matrix_pairwise(V, M, gold)
    R, n_pairs, shrink = correlation_shrunk_pairwise(E, avail)
    m = len(lids)
    off = n_pairs[~np.eye(m, dtype=bool)]
    min_joint = int(off.min())

    abstention = float(1 - M_b.mean())
    # majority co-failure (committed), normalised per item
    W = ((V != gold[:, None]) & M_b).astype(np.float64)
    n_committed = M_b.sum(1)
    frac_wrong = np.divide(W.sum(1), np.maximum(n_committed, 1))
    has = n_committed > 0
    majority_cofailure = float((frac_wrong[has] > 0.5).mean())

    # consensus false-retention at supermajority 0.75
    maj = majority_consensus(V, M)
    level = consensus_level(V, M)
    labellable = maj != ABSTAIN
    correct = labellable & (maj == gold)
    sm = labellable & (level >= 0.75)
    supermaj_frr = float(1 - (sm & correct).sum() / max(sm.sum(), 1))

    # CorrFilter gain at target retention (paired bootstrap on FRR difference)
    cf = corrfilter_score(V, M, R, maj)
    sc_cons = np.where(labellable, level, -np.inf)
    sc_cf = cf.score
    k = int(round(args.retention * n))
    keep_cons, keep_cf = _topk(sc_cons, k, n), _topk(sc_cf, k, n)
    rng = np.random.default_rng(BOOT_SEED)
    diffs = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        d = _frr(keep_cons, correct, labellable, idx) - _frr(keep_cf, correct, labellable, idx)
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    gain = float(diffs.mean())            # consensus_FRR - corrfilter_FRR (positive = cf better)
    lo, hi = (float(x) for x in np.quantile(diffs, [0.025, 0.975]))

    row = {
        "setting": args.setting,
        "judge_count": m,
        "max_new_tokens": args.max_new_tokens,
        "abstention_rate": round(abstention, 4),
        "pairwise_complete_items_or_min_joint_votes": min_joint,
        "rho_bar": round(mean_off_diagonal(R), 4),
        "n_eff": round(effective_size(R), 3),
        "eigen_rank": round(effective_eig_rank(R), 3),
        "majority_cofailure": round(majority_cofailure, 4),
        "supermajority_false_retention": round(supermaj_frr, 4),
        "corrfilter_gain_at_0.60": round(gain, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        df = pd.read_csv(out)
        df = df[df["setting"] != args.setting]  # replace same-setting row idempotently
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out, index=False)
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"appended row '{args.setting}' -> {out}")


if __name__ == "__main__":
    main()

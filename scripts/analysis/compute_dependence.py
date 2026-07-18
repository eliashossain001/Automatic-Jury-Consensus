"""Dependence metrics for a judge bank — shared library + CLI (Bucket 1).

Reused by the reproduction pipeline and by the DPO-judge / extended-bank buckets.
Given a vote matrix on committed votes, computes the paper's headline dependence
quantities with the SAME estimator-selection rule as scripts/03 (listwise
Ledoit--Wolf when abstention is low, pairwise-complete + shrinkage otherwise), so
outputs match the published tables exactly.

CLI:
  python scripts/analysis/compute_dependence.py --bank configs/judge_bank.yaml \
      --manifest experiments/h1_measurement/results/calibration_manifest.parquet \
      --label rewardbench_base --out outputs/tables/dependence.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.correlation import (  # noqa: E402
    compute_error_matrix,
    compute_error_matrix_pairwise,
    correlation_shrunk,
    correlation_shrunk_pairwise,
)
from corrfilter.correlation.effective_size import (  # noqa: E402
    effective_eig_rank,
    effective_size,
    mean_off_diagonal,
)
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

LISTWISE_FLOOR = 0.70


def estimate_R(V, M, gold, listwise_floor=LISTWISE_FLOOR):
    """Return (R, estimator_name, min_joint, listwise_retention) matching scripts/03."""
    E_lw, complete = compute_error_matrix(V, M, gold)
    retention = float(complete.mean())
    n = V.shape[1]
    if retention >= listwise_floor and E_lw.shape[0] >= 2:
        R, _ = correlation_shrunk(E_lw)
        min_joint = int(complete.sum())
        return R, "listwise_ledoit_wolf", min_joint, retention
    E, avail = compute_error_matrix_pairwise(V, M, gold)
    R, n_pairs, _ = correlation_shrunk_pairwise(E, avail)
    off = n_pairs[~np.eye(n, dtype=bool)]
    return R, "pairwise_complete_shrunk", int(off.min()), retention


def cofailure(V, M, gold):
    """Co-failure metrics on committed votes (abstentions excluded)."""
    M_b = M.astype(bool)
    W = ((V != gold[:, None]) & M_b).astype(np.float64)   # wrong-and-committed
    n_c = M_b.sum(1)
    has = n_c > 0
    m = V.shape[1]
    frac_wrong = np.divide(W.sum(1), np.maximum(n_c, 1))
    p_err = np.array([float(W[M_b[:, j], j].mean()) if M_b[:, j].any() else np.nan for j in range(m)])
    cond = []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            both = M_b[:, i] & M_b[:, j]
            wi = W[both, i]
            if wi.sum() > 0:
                cond.append(float((W[both, i] * W[both, j]).sum() / wi.sum()))
    cond = np.nanmean(cond) if cond else np.nan
    marg = float(np.nanmean(p_err))
    return {
        "abstention_rate": round(float(1 - M_b.mean()), 4),
        "mean_per_judge_error": round(marg, 4),
        "majority_cofailure": round(float((frac_wrong[has] > 0.5).mean()), 4),
        "supermajority75_cofailure": round(float((frac_wrong[has] >= 0.75).mean()), 4),
        "conditional_cofailure": round(float(cond), 4),
        "conditional_lift": round(float(cond / marg), 3) if marg > 0 else float("nan"),
    }


def contrast(R, logical_ids, by):
    """Intra vs cross mean off-diagonal correlation for grouping `by` ('family'|'prompt')."""
    def key(lid):
        base, style = lid.split("::")
        return base.rsplit("-", 1)[0] if by == "family" else style
    groups = [key(l) for l in logical_ids]
    n = len(logical_ids)
    intra, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if groups[i] == groups[j] else cross).append(R[i, j])
    ri = float(np.mean(intra)) if intra else float("nan")
    rc = float(np.mean(cross)) if cross else float("nan")
    return round(ri, 4), round(rc, 4), round(ri - rc, 4)


def dependence_metrics(V, M, gold, logical_ids, label=""):
    R, estimator, min_joint, retention = estimate_R(V, M, gold)
    fi, fc, fd = contrast(R, logical_ids, "family")
    pi, pc, pd_ = contrast(R, logical_ids, "prompt")
    row = {
        "label": label,
        "n_judges": len(logical_ids),
        "estimator": estimator,
        "listwise_retention": round(retention, 4),
        "min_joint_votes": min_joint,
        "rho_bar": round(mean_off_diagonal(R), 4),
        "n_eff": round(effective_size(R), 3),
        "eigen_rank": round(effective_eig_rank(R), 3),
        "neff_ceiling_1_over_rho": round(1.0 / mean_off_diagonal(R), 3) if mean_off_diagonal(R) > 0 else float("inf"),
        **cofailure(V, M, gold),
        "family_contrast_delta": fd,
        "prompt_contrast_delta": pd_,
    }
    return row, R


def load_votes(bank_yaml, item_ids, judge_ids=None):
    bank = load_bank_config(bank_yaml)
    lids = [s.logical_id for s in bank.specs]
    if judge_ids:
        lids = [l for l in lids if l in set(judge_ids)]
    V, M = load_vote_matrix(VoteCache(bank.votes_dir), lids, item_ids)
    return V, M, lids


def item_ids_from_manifest(manifest):
    p = Path(manifest)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return df["item_id"].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-id", action="append", default=None,
                    help="restrict to these logical ids (repeatable)")
    args = ap.parse_args()

    item_ids = item_ids_from_manifest(args.manifest)
    V, M, lids = load_votes(args.bank, item_ids, args.judge_id)
    gold = np.ones(len(item_ids), dtype=np.int8)
    row, _ = dependence_metrics(V, M, gold, lids, args.label)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev["label"] != args.label]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""GRPO judges: co-failure metrics + CorrFilter-vs-consensus at matched retention.

Consumes the GRPO judge vote cache (produced by 02 on configs/grpo_judge_bank.yaml)
and the correlation.npz produced by 03, and answers Dr. Lim's three claims with
REAL GRPO judge outputs:

  1. dependence:  rho_bar, n_eff, eigen-rank, conditional & majority co-failure.
  2. consensus-fails: majority / supermajority accuracy under that dependence.
  3. CorrFilter-beats-consensus: at MATCHED retention, false-retention of the
     decorrelated alpha_subset filter vs consensus ranking, with bootstrap CIs.

Head-to-head is rank-based at matched retention: every method keeps the same
number of items, differing only in HOW it ranks them —
  * consensus  : rank by consensus level max(p1, 1-p1)              (naive k-of-n)
  * corrfilter : rank by alpha_subset = |S| / sqrt(1' R_S 1)        (decorrelated)
  * weighted   : rank by inverse-error-weighted margin              (independent-accuracy)
so any gap isolates the correlation correction. Fixed operating points
(naive majority, supermajority-0.75) are reported alongside.

Usage:
    python scripts/29_grpo_corrfilter_eval.py \
        --bank configs/grpo_judge_bank.yaml \
        --calibration configs/calibration.yaml \
        --corr results/grpo_judges/rb_corr/correlation.npz \
        --out results/grpo_judges/filtering
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus, vote_fraction  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402
from corrfilter.correlation.effective_size import (  # noqa: E402
    effective_eig_rank,
    effective_size,
    mean_off_diagonal,
)
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.evaluation import evaluable_and_correct, matched_k  # noqa: E402
from corrfilter.filtering import inverse_error_weights  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

SEED = 20260706


# ----------------------------- co-failure metrics -----------------------------
def cofailure_metrics(E: np.ndarray) -> dict:
    """Co-failure over COMMITTED votes only (abstentions excluded, not counted correct).

    Takes (V, M, gold): W[i,j]=1 iff judge j committed a vote on item i and it was
    wrong. Majority co-failure normalises by committed judges per item so a high
    abstention rate cannot masquerade as agreement.
    """
    V, M, gold = E  # packed tuple
    M_b = M.astype(bool)
    W = ((V != gold[:, None]) & M_b).astype(np.float64)   # wrong-and-committed
    n_committed = M_b.sum(1)
    n_wrong = W.sum(1)
    has = n_committed > 0
    m = V.shape[1]
    # per-judge marginal error over its own committed items
    p_err = np.array([
        float(W[M_b[:, j], j].mean()) if M_b[:, j].any() else np.nan for j in range(m)
    ])
    # conditional co-failure over items where BOTH committed
    cond = []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            both = M_b[:, i] & M_b[:, j]
            wi = W[both, i]
            if wi.sum() == 0:
                continue
            cond.append(float((W[both, i] * W[both, j]).sum() / wi.sum()))
    cond = np.array(cond) if cond else np.array([np.nan])
    frac_wrong = np.divide(n_wrong, np.maximum(n_committed, 1))
    return {
        "n_items_any_commit": int(has.sum()),
        "mean_abstention_rate": round(float(1 - M_b.mean()), 4),
        "mean_per_judge_error_committed": round(float(np.nanmean(p_err)), 4),
        "majority_cofailure": round(float((frac_wrong[has] > 0.5).mean()), 4),
        "supermajority75_cofailure": round(float((frac_wrong[has] >= 0.75).mean()), 4),
        "unanimous_cofailure": round(float(((n_wrong[has] == n_committed[has]) & (n_committed[has] >= 2)).mean()), 4),
        "conditional_cofailure": round(float(np.nanmean(cond)), 4),
        "marginal_error": round(float(np.nanmean(p_err)), 4),
        "conditional_lift": round(float(np.nanmean(cond) / np.nanmean(p_err)), 3),
    }


# ----------------------------- ranking-based filters -----------------------------
def _precision_frr(keep: np.ndarray, correct: np.ndarray, labellable: np.ndarray):
    kept = keep & labellable
    nk = int(kept.sum())
    if nk == 0:
        return np.nan, np.nan, 0
    prec = float((kept & correct).sum()) / nk
    return prec, 1.0 - prec, nk


def top_k_keep(score: np.ndarray, k: int) -> np.ndarray:
    n = score.shape[0]
    keep = np.zeros(n, dtype=bool)
    if k <= 0:
        return keep
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
    keep[order[:min(k, n)]] = True
    return keep


def bootstrap_ci(metric_fn, n_items: int, B: int = 1000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n_items, size=n_items)
        v = metric_fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan, np.nan)
    vals = np.array(vals)
    return (round(float(vals.mean()), 4),
            round(float(np.quantile(vals, 0.025)), 4),
            round(float(np.quantile(vals, 0.975)), 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/grpo_judge_bank.yaml")
    ap.add_argument("--calibration", default="configs/calibration.yaml")
    ap.add_argument("--corr", default="results/grpo_judges/rb_corr/correlation.npz")
    ap.add_argument("--out", default="results/grpo_judges/filtering")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cal_cfg = yaml.safe_load(Path(args.calibration).read_text())
    items = load_calibration_set(cal_cfg["output"]["manifest_path"])
    item_ids = [it.item_id for it in items]
    gold = np.ones(len(items), dtype=np.int8)

    bank_cfg = load_bank_config(args.bank)
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    V, M = load_vote_matrix(VoteCache(bank_cfg.votes_dir), logical_ids, item_ids)

    z = np.load(args.corr, allow_pickle=True)
    R = z["R"].astype(np.float64)
    E = z["E"].astype(np.float64)

    # ---- dependence + co-failure ----
    dep = {
        "bank": "grpo_judges",
        "n_judges": len(logical_ids),
        "logical_ids": logical_ids,
        "rho_bar": round(float(mean_off_diagonal(R)), 4),
        "n_eff": round(float(effective_size(R)), 3),
        "n_eff_eig": round(float(effective_eig_rank(R)), 3),
        "eigvals": [round(float(x), 3) for x in z["eigvals"]],
        **cofailure_metrics((V, M, gold)),
    }

    # ---- consensus accuracy (claim 2) ----
    # P0-0 corrected evaluation: ties abstain (gold is constant here), retention is
    # denominated on the evaluable pool. See src/corrfilter/evaluation.py.
    maj, labellable, correct = evaluable_and_correct(V, M, gold, tie_policy="abstain")
    p1 = vote_fraction(V, M)
    level = consensus_level(V, M)
    n = len(items)
    supermaj_keep = labellable & (level >= 0.75)
    consensus_stats = {
        "majority_retention": round(float(labellable.mean()), 4),
        "majority_accuracy": round(float(correct.sum() / max(labellable.sum(), 1)), 4),
        "supermajority75_retention": round(float(supermaj_keep.mean()), 4),
        "supermajority75_accuracy": round(
            float((supermaj_keep & correct).sum() / max(supermaj_keep.sum(), 1)), 4),
    }

    # ---- ranking scores ----
    cf = corrfilter_score(V, M, R, maj)
    score_corrfilter = np.where(labellable, cf.score, -np.inf)   # alpha_subset
    score_consensus = np.where(labellable, level, -np.inf)
    w = inverse_error_weights(V, M, gold)
    M_b = M.astype(bool)
    wsum = (w[None, :] * M_b).sum(1)
    wones = (V * (w[None, :] * M_b)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wp1 = np.where(wsum > 0, wones / np.maximum(wsum, 1e-12), 0.5)
    score_weighted = np.where(labellable, np.abs(wp1 - 0.5), -np.inf)

    methods = {
        "consensus": score_consensus,
        "corrfilter_alpha_subset": score_corrfilter,
        "independent_weighted": score_weighted,
    }

    # ---- retention curve ----
    grid = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    curve_rows = []
    for r in grid:
        k = matched_k(r, labellable)
        row = {"target_retention": r, "n_keep": k}
        for name, sc in methods.items():
            keep = top_k_keep(sc, k)
            prec, frr, nk = _precision_frr(keep, correct, labellable)
            row[f"precision_{name}"] = round(prec, 4) if not np.isnan(prec) else np.nan
            row[f"frr_{name}"] = round(frr, 4) if not np.isnan(frr) else np.nan
            row[f"actual_retention_{name}"] = round(nk / max(int(labellable.sum()), 1), 4)
        curve_rows.append(row)

    import pandas as pd
    pd.DataFrame(curve_rows).to_csv(out / "retention_curve.csv", index=False)

    # ---- headline table at matched retention points, with bootstrap CIs ----
    match_points = sorted({0.6, 0.7, 0.8, consensus_stats["supermajority75_retention"]})
    table = []
    for r in match_points:
        k = matched_k(r, labellable)
        for name, sc in methods.items():
            keep = top_k_keep(sc, k)
            prec, frr, nk = _precision_frr(keep, correct, labellable)

            def frr_on(idx, keep=keep):
                kk = keep[idx] & labellable[idx]
                nkk = int(kk.sum())
                if nkk == 0:
                    return np.nan
                return 1.0 - float((kk & correct[idx]).sum()) / nkk

            mean, lo, hi = bootstrap_ci(frr_on, n, B=args.bootstrap)
            table.append({
                "matched_retention": round(r, 4),
                "method": name,
                "n_keep": nk,
                "precision": round(prec, 4) if not np.isnan(prec) else np.nan,
                "false_retention": round(frr, 4) if not np.isnan(frr) else np.nan,
                "frr_ci_lower": lo,
                "frr_ci_upper": hi,
            })
    tbl = pd.DataFrame(table)
    tbl.to_csv(out / "method_comparison.csv", index=False)

    # ---- regime decision ----
    ev = np.array(dep["eigvals"], dtype=float)
    top_share = float(ev[0] / ev.sum())
    # crude cluster probe: judges with above-median mean-correlation to others
    offdiag = R.copy()
    np.fill_diagonal(offdiag, np.nan)
    per_judge_meanR = np.nanmean(offdiag, axis=1)
    cluster = int((per_judge_meanR > (np.nanmean(per_judge_meanR) + np.nanstd(per_judge_meanR))).sum())
    rho = dep["rho_bar"]
    if rho < 0.10:
        regime = "weak_dependence"
    elif cluster >= 2 and per_judge_meanR.max() > 1.6 * np.nanmedian(per_judge_meanR):
        regime = "vulnerable_subgroup"
    else:
        regime = "global_cofailure"
    regime_info = {
        "regime": regime,
        "rho_bar": rho,
        "top_eigen_share": round(top_share, 3),
        "n_high_corr_judges": cluster,
        "recommended_filter": {
            "weak_dependence": "supermajority may suffice",
            "global_cofailure": "corrfilter_alpha_subset",
            "vulnerable_subgroup": "bias_cluster filter",
        }[regime],
    }

    result = {"dependence": dep, "consensus": consensus_stats, "regime": regime_info}
    (out / "grpo_eval_summary.json").write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print("\n=== retention curve (false-retention by method) ===")
    print(pd.DataFrame(curve_rows)[
        ["target_retention", "frr_consensus", "frr_corrfilter_alpha_subset", "frr_independent_weighted"]
    ].to_string(index=False))
    print("\n=== matched-retention headline (with 95% bootstrap CI on FRR) ===")
    print(tbl.to_string(index=False))
    print(f"\nwrote → {out}")


if __name__ == "__main__":
    main()

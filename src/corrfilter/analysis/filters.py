"""Filter comparison at matched retention — shared library + CLI (Bucket 1).

Scores consensus (majority / supermajority-0.75), the decorrelated CorrFilter
alpha_subset, and an independent-accuracy-weighted baseline on a shared vote set /
R / gold, and reports retention / precision / false-retention. Also emits the
paired-bootstrap CorrFilter-vs-consensus false-retention gain (with 95% CI) at a
grid of matched retentions. Reuses corrfilter.filtering primitives.

CLI:
  python corrfilter.analysis.filters --bank configs/grpo_judge_bank.yaml \
      --manifest experiments/h1_measurement/results/calibration_manifest.parquet \
      --label grpo --out outputs/tables/filters.csv --gain-out outputs/tables/corrfilter_gain.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.analysis.bootstrap_ci import paired_frr_gain
from corrfilter.analysis.dependence import estimate_R, item_ids_from_manifest, load_votes
from corrfilter.cfi.consensus import consensus_level
from corrfilter.cfi.corrfilter_score import corrfilter_score
from corrfilter.evaluation import evaluable_and_correct, matched_k
from corrfilter.filtering import inverse_error_weights


def _prec_frr(keep, correct, labellable):
    kept = keep & labellable
    nk = int(kept.sum())
    if nk == 0:
        return np.nan, np.nan, 0
    prec = float((kept & correct).sum()) / nk
    return prec, 1.0 - prec, nk


def _topk(score, k, n):
    keep = np.zeros(n, dtype=bool)
    if k > 0:
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
        keep[order[:min(k, n)]] = True
    return keep


def filter_table(V, M, R, gold, label, retentions=(0.5, 0.6, 0.7, 0.8),
                 tie_policy="abstain", retention_mode="evaluable"):
    """Retention-matched filter comparison. Corrected by P0-0: ties abstain and retention
    is denominated on the evaluable pool (see corrfilter.evaluation)."""
    n = V.shape[0]
    maj, labellable, correct = evaluable_and_correct(V, M, gold, tie_policy=tie_policy)
    level = consensus_level(V, M)
    w = inverse_error_weights(V, M, gold)
    M_b = M.astype(bool)
    wsum = (w[None, :] * M_b).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wp1 = np.where(wsum > 0, (V * (w[None, :] * M_b)).sum(1) / np.maximum(wsum, 1e-12), 0.5)
    scores = {
        "consensus": np.where(labellable, level, -np.inf),
        "corrfilter_alpha_subset": np.where(labellable, corrfilter_score(V, M, R, maj).score, -np.inf),
        "independent_weighted": np.where(labellable, np.abs(wp1 - 0.5), -np.inf),
    }
    rows = []
    for r in retentions:
        k = matched_k(r, labellable, retention_mode=retention_mode)
        for name, sc in scores.items():
            keep = _topk(sc, k, n)
            prec, frr, nk = _prec_frr(keep, correct, labellable)
            rows.append({
                "label": label, "method": name, "matched_retention": r, "n_keep": nk,
                "n_evaluable": int(labellable.sum()),
                "precision": round(prec, 4) if not np.isnan(prec) else np.nan,
                "false_retention": round(frr, 4) if not np.isnan(frr) else np.nan,
            })
    # fixed operating points
    for name, keep in [
        ("naive_majority_all", labellable),
        ("supermajority_0.75", labellable & (level >= 0.75)),
    ]:
        prec, frr, nk = _prec_frr(keep, correct, labellable)
        rows.append({
            "label": label, "method": name,
            "matched_retention": round(nk / max(int(labellable.sum()), 1), 4), "n_keep": nk,
            "n_evaluable": int(labellable.sum()),
            "precision": round(prec, 4) if not np.isnan(prec) else np.nan,
            "false_retention": round(frr, 4) if not np.isnan(frr) else np.nan,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gain-out", default=None)
    ap.add_argument("--judge-id", action="append", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    item_ids = item_ids_from_manifest(args.manifest)
    V, M, lids = load_votes(args.bank, item_ids, args.judge_id)
    gold = np.ones(len(item_ids), dtype=np.int8)
    R, _, _, _ = estimate_R(V, M, gold)

    rows = filter_table(V, M, R, gold, args.label)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out); prev = prev[prev["label"] != args.label]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    if args.gain_out:
        grows = []
        for r in (0.5, 0.6, 0.7):
            g = paired_frr_gain(V, M, R, gold, retention=r, B=args.bootstrap)
            g = {"label": args.label, **g}
            grows.append(g)
        gout = Path(args.gain_out); gout.parent.mkdir(parents=True, exist_ok=True)
        gdf = pd.DataFrame(grows)
        if gout.exists():
            prev = pd.read_csv(gout); prev = prev[prev["label"] != args.label]
            gdf = pd.concat([prev, gdf], ignore_index=True)
        gdf.to_csv(gout, index=False)
        print("\nCorrFilter vs consensus (paired bootstrap):")
        print(gdf.to_string(index=False))
        print(f"-> {args.gain_out}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

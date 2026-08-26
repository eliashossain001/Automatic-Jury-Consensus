#!/usr/bin/env python
"""Explicit routing-accuracy evaluation for the regime routers (Prof. Lim: "How
accurate is the routing?").

Ground truth: the regime label of each deployment instance is assigned BY
CONSTRUCTION from the synthetic generator that produced it (position-aligned
poisoning -> subgroup, content poisoning -> weak, CFI prompt-injection ->
global). There are only 9 deployment instances (3 per regime); every number
here must be read against that sample size.

Routers evaluated:
  hard_unsupervised   scripts/routing/regime_router.py's clean-referenced threshold rule (dataset-level)
  supervised_loo      script train_router_dev's leave-one-out logistic regression
  soft_v2             scripts/routing/unified_router.py's softmax router, dataset-level argmax

Reports: overall accuracy + 95% Wilson CI, balanced accuracy, per-regime
precision/recall/F1, confusion matrices, per-instance predictions.

Analysis-only; reads outputs/regime_router/regime_diagnostics.csv and
outputs/unified_router_v2/routing_confidence_analysis.csv.

Usage: python scripts/routing/superseded/routing_accuracy.py
Outputs -> outputs/routing_accuracy/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REGIMES = ["weak", "global", "subgroup"]

# Constants copied from scripts/routing/regime_router.py (clean-bank-referenced).
THR = {"rho": 0.03, "lco": 0.05, "eig": 0.85}
REF = {"rho_bar": 0.217, "lco_flip": 0.11}

FEATS = ["rho_bar", "n_eff", "eig_overlap", "conc", "rho_S", "disagree", "lco_flip"]


def classify_hard(row: pd.Series) -> str:
    if row["rho_bar"] - REF["rho_bar"] > THR["rho"]:
        return "global"
    if (row["lco_flip"] - REF["lco_flip"] > THR["lco"]) or (row["eig_overlap"] < THR["eig"]):
        return "subgroup"
    return "weak"


def loo_predictions(diag: pd.DataFrame, seed: int = 20260708) -> list[str]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = diag[FEATS].to_numpy(float)
    y = diag["true_regime"].to_numpy()
    preds = []
    for k in range(len(diag)):
        tr = np.arange(len(diag)) != k
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        preds.append(str(clf.predict(sc.transform(X[k : k + 1]))[0]))
    return preds


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 3), round(centre + half, 3))


def score(y_true: list[str], y_pred: list[str], router: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y_true, y_pred = list(y_true), list(y_pred)
    n = len(y_true)
    k = sum(t == p for t, p in zip(y_true, y_pred))
    per_class = []
    recalls = []
    for r in REGIMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == r and p == r)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != r and p == r)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == r and p != r)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        recalls.append(rec)
        per_class.append({"router": router, "regime": r, "support": tp + fn,
                          "precision": round(prec, 3) if prec == prec else prec,
                          "recall": round(rec, 3), "f1": round(f1, 3) if f1 == f1 else f1})
    lo, hi = wilson_ci(k, n)
    conf = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred")).reindex(
        index=REGIMES, columns=REGIMES, fill_value=0)
    row = {"router": router, "n_instances": n, "n_correct": k,
           "accuracy": round(k / n, 3), "acc_ci95_low": lo, "acc_ci95_high": hi,
           "balanced_accuracy": round(float(np.nanmean(recalls)), 3)}
    return row, pd.DataFrame(per_class), conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.project_root)
    out_dir = root / "outputs" / "routing_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)

    diag = pd.read_csv(root / "outputs" / "regime_router" / "regime_diagnostics.csv")
    diag = diag.sort_values("dataset").reset_index(drop=True)
    y_true = diag["true_regime"].tolist()

    preds = {
        "hard_unsupervised": [classify_hard(r) for _, r in diag.iterrows()],
        "supervised_loo": loo_predictions(diag),
    }
    soft_path = root / "outputs" / "unified_router_v2" / "routing_confidence_analysis.csv"
    if soft_path.exists():
        soft = pd.read_csv(soft_path).sort_values("dataset").reset_index(drop=True)
        assert soft["dataset"].tolist() == diag["dataset"].tolist()
        preds["soft_v2"] = soft[["p_weak", "p_global", "p_subgroup"]].idxmax(axis=1).str[2:].tolist()

    # sanity check vs the persisted confusion matrix of scripts/routing/regime_router.py
    persisted = pd.read_csv(root / "outputs" / "regime_router" / "regime_confusion_matrix.csv", index_col=0)

    rows, pc_rows, inst = [], [], diag[["dataset", "true_regime"]].copy()
    for router, y_pred in preds.items():
        row, pc, conf = score(y_true, y_pred, router)
        rows.append(row)
        pc_rows.append(pc)
        conf.to_csv(out_dir / f"confusion_{router}.csv")
        inst[f"pred_{router}"] = y_pred

    acc_df = pd.DataFrame(rows)
    pc_df = pd.concat(pc_rows, ignore_index=True)
    acc_df.to_csv(out_dir / "routing_accuracy.csv", index=False)
    pc_df.to_csv(out_dir / "per_regime_prf.csv", index=False)
    inst.to_csv(out_dir / "instance_predictions.csv", index=False)

    with open(out_dir / "summary.md", "w") as f:
        f.write("# Routing accuracy (deployment instances)\n\n")
        f.write("Ground-truth regime labels are assigned by construction from the synthetic "
                "generator that produced each instance; they are not independent annotations. "
                f"Evaluation sample: n = {len(diag)} deployment instances "
                "(3 per regime), so every interval below is wide.\n\n")
        f.write(acc_df.to_markdown(index=False))
        f.write("\n\n## Per-regime precision / recall / F1\n\n")
        f.write(pc_df.to_markdown(index=False))
        f.write("\n\n## Per-instance predictions\n\n")
        f.write(inst.to_markdown(index=False))
        f.write("\n\nHard-router confusion matrix reproduced from scripts/routing/regime_router.py output: "
                f"match = {bool((pd.crosstab(inst.true_regime, inst.pred_hard_unsupervised).reindex(index=persisted.index, columns=persisted.columns, fill_value=0) == persisted).all().all())}\n")
    print(f"[36] wrote {out_dir}")
    print(acc_df.to_string(index=False))
    print(pc_df.to_string(index=False))


if __name__ == "__main__":
    main()

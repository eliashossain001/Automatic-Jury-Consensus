"""Run CorrFilter v1 and compare against naive baselines (Bucket 3, Part C).

Loads the empirical H1 error-correlation matrix R, a judge-vote set, and the
trusted gold labels, then scores five filtering methods on the same data:

  1. naive_majority
  2. naive_supermajority_75
  3. independent_accuracy_weighted   (inverse-error weights from clean H1 votes)
  4. corrfilter_subset               (α_subset = |S| / √(1ᵀ R_S 1), top-retention)
  5. corrfilter_subset_risk_tau      (α_subset ≥ a fixed risk floor)

By default it runs on the clean H1 bank (always available). Pass
``--mechanism <name>`` to run on that mechanism's all-biased CFI variant
(requires cached biased votes from scripts/cfi/run_bank_gpu.py) — the setting where correlated
failure makes naive consensus over-retain and CorrFilter should help most.

Outputs (under outputs/corrfilter/):
  filtered_items_corrfilter.csv, filtered_items_naive_majority.csv,
  filtering_comparison.csv, corrfilter_report.md

Usage:
    python scripts/cfi/run_corrfilter.py --project-root /home/elias/elias_projects/corrfilter
    python scripts/cfi/run_corrfilter.py --mechanism position_bias_stress_test --target-retention 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from corrfilter.cfi.bias_bank import load_bias_bank
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views
from corrfilter.data import load_calibration_set
from corrfilter.filtering import (
    evaluate_filter,
    filter_corrfilter_risk_tau,
    filter_corrfilter_subset,
    filter_independent_weighted,
    filter_naive_majority,
    filter_supermajority,
    inverse_error_weights,
)
from corrfilter.judges import load_bank_config


def _within_subset_rho(R: np.ndarray, V, M, label, keep) -> float:
    """Mean within-agreeing-subset off-diagonal correlation, averaged over kept items."""
    R_sym = (R + R.T) / 2.0
    M_b = M.astype(bool)
    vals = []
    for i in np.where(keep)[0]:
        agree = (V[i] == label[i]) & M_b[i]
        s = int(agree.sum())
        if s < 2:
            continue
        sub = R_sym[np.ix_(agree, agree)]
        off = sub[~np.eye(s, dtype=bool)]
        vals.append(float(off.mean()))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--mechanism", default=None,
                    help="run on this mechanism's all-biased variant; default = clean H1 bank")
    ap.add_argument("--target-retention", type=float, default=0.8)
    ap.add_argument("--alpha-floor", type=float, default=1.5)
    ap.add_argument("--r-mode",
                    choices=["clean", "variant", "oracle_variant", "disagreement", "identity"],
                    default="clean",
                    help="correlation matrix for CorrFilter scoring: clean H1 R (default); "
                         "oracle_variant (=variant) gold-derived biased R (upper bound); "
                         "disagreement = gold-free vote correlation; or identity")
    ap.add_argument("--out-dir", default="outputs/corrfilter")
    ap.add_argument("--items-file", default=None,
                    help="restrict to these item_ids (same subset as the CFI run)")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    cfg = yaml.safe_load(_p(args.config).read_text())
    bank = load_bias_bank(str(_p(args.config)))
    out_dir = _p(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_cfg = yaml.safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())
    items = load_calibration_set(_p(cal_cfg["output"]["manifest_path"]))
    if args.items_file:
        wanted = {ln.strip() for ln in _p(args.items_file).read_text().splitlines() if ln.strip()}
        items = [it for it in items if it.item_id in wanted]
    gold = np.ones(len(items), dtype=np.int8)

    bank_cfg = load_bank_config(str(_p(cfg["sources"]["h1_bank_config"])))
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    npz = np.load(_p(cfg["sources"]["h1_correlation_npz"]), allow_pickle=True)
    R = npz["R"]
    if [str(x) for x in npz["logical_ids"]] != logical_ids:
        raise RuntimeError("npz judge order != bank order")

    clean_views = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)

    if args.mechanism:
        cond = bank.by_name(args.mechanism)
        biased_views = load_vote_views(_p(cfg["runtime"]["votes_dir"]) / cond.name, logical_ids)
        if sum(len(v) for v in biased_views.values()) == 0:
            raise FileNotFoundError(
                f"no cached biased votes for '{args.mechanism}'; run scripts/cfi/run_bank_gpu.py first")
        var = assemble_variant(items, logical_ids, cond, 1.0, biased_views, clean_views, bank.seed)
        V, M = var.V, var.M
        source = f"all_biased variant of mechanism '{args.mechanism}'"
    else:
        # Clean H1 bank from the cached views.
        clean_cond = bank.by_name("clean")
        var = assemble_variant(items, logical_ids, clean_cond, 0.0, {}, clean_views, bank.seed)
        V, M = var.V, var.M
        source = "clean H1 bank"

    # Weights from clean H1 votes (deployment-realistic: estimated on calibration).
    clean_var = assemble_variant(items, logical_ids, bank.by_name("clean"), 0.0, {}, clean_views, bank.seed)
    weights = inverse_error_weights(clean_var.V, clean_var.M, gold)

    # CorrFilter scoring R. clean = H1 default; oracle_variant = gold-derived
    # biased R (upper bound); disagreement = gold-free vote correlation; identity.
    if args.r_mode in ("variant", "oracle_variant"):
        from corrfilter.cfi.adaptive_r import oracle_variant_R
        R = oracle_variant_R(V, M, gold)
        source += " [R=oracle_variant]"
    elif args.r_mode == "disagreement":
        from corrfilter.cfi.adaptive_r import disagreement_R
        R = disagreement_R(V, M)
        source += " [R=disagreement]"
    elif args.r_mode == "identity":
        R = np.eye(len(logical_ids))
        source += " [R=identity]"

    tr = args.target_retention
    results = [
        filter_naive_majority(V, M, gold),
        filter_supermajority(V, M, gold, threshold=0.75),
        filter_independent_weighted(V, M, gold, weights, target_retention=tr),
        filter_corrfilter_subset(V, M, R, gold, target_retention=tr),
        filter_corrfilter_risk_tau(V, M, R, gold, alpha_floor=args.alpha_floor),
    ]

    rows = []
    for res in results:
        metrics = evaluate_filter(res, gold)
        metrics["rho_S_within_subset"] = round(
            _within_subset_rho(R, V, M, res.retained_label, res.keep), 4)
        rows.append(metrics)
    comparison = pd.DataFrame(rows)
    comparison.insert(0, "source", source)
    comparison.to_csv(out_dir / "filtering_comparison.csv", index=False)

    # Detailed kept-item CSVs for CorrFilter and naive majority.
    _dump_kept(out_dir / "filtered_items_corrfilter.csv", results[3], items, V, M, R, gold)
    _dump_kept(out_dir / "filtered_items_naive_majority.csv", results[0], items, V, M, R, gold)

    _write_report(out_dir / "corrfilter_report.md", comparison, source, tr,
                  args.alpha_floor, len(items))
    print(f"CorrFilter comparison on {source}:")
    print(comparison.to_string(index=False))
    print(f"\nwrote → {out_dir}")


def _dump_kept(path, res, items, V, M, R, gold) -> None:
    R_sym = (R + R.T) / 2.0
    M_b = M.astype(bool)
    rows = []
    for i in np.where(res.keep)[0]:
        agree = (V[i] == res.retained_label[i]) & M_b[i]
        s = int(agree.sum())
        rho = float("nan")
        if s >= 2:
            sub = R_sym[np.ix_(agree, agree)]
            rho = float(sub[~np.eye(s, dtype=bool)].mean())
        rows.append({
            "item_id": items[i].item_id,
            "subset": items[i].subset,
            "retained_label": int(res.retained_label[i]),
            "gold_label": int(gold[i]),
            "correct": bool(res.retained_label[i] == gold[i]),
            "score": round(float(res.score[i]), 4) if res.score is not None and not np.isnan(res.score[i]) else None,
            "agreeing_set_size": s,
            "within_subset_rho": round(rho, 4) if not np.isnan(rho) else None,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_report(path, comparison, source, tr, alpha_floor, n_items) -> None:
    lines = ["# CorrFilter v1 Report (Bucket 3, Part C)\n"]
    lines.append(f"Vote source: **{source}**. Items: {n_items}. "
                 f"Target retention (ranked methods): {tr}. Risk floor α≥{alpha_floor}.\n")
    lines.append("## Filtering comparison\n")
    cols = ["method", "retention_rate", "false_retention_rate", "precision",
            "recall", "f1", "avg_score_kept", "rho_S_within_subset"]
    cols = [c for c in cols if c in comparison.columns]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in comparison.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("### How to read this\n")
    lines.append("- **false_retention_rate** = fraction of kept items whose preferred "
                 "label is wrong (lower is better).")
    lines.append("- **precision** = 1 − false_retention; **recall** = share of all "
                 "correctly-labellable items that survive.")
    lines.append("- **α_subset** (avg_score_kept for CorrFilter) discounts agreement "
                 "among correlated judges: 1ᵀR_S1 grows when the agreeing set is "
                 "internally correlated, shrinking α.")
    lines.append("- **rho_S_within_subset** = mean within-agreeing-subset error "
                 "correlation among kept items; CorrFilter should keep items with "
                 "*lower* ρ_S than naive majority.\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

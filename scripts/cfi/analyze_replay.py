"""Compute the CFI metrics from cached votes (replay or real).

For every (mechanism, biased_ratio) variant this script computes:

* per-item: consensus level, majority prediction + correctness, CorrFilter
  score, CorrFilter keep/discard at matched retention.
* per-variant: false retention rate at full retention, accuracy by consensus
  level, retention-matched CorrFilter comparison.
* per-mechanism: false-retention curves as a function of biased-judge ratio.
* per-judge × mechanism: conditional error rate (for the heatmap in scripts/cfi/analyze_gpu_bank.py).

Outputs (mirrored into both experiments/h2_cfi/results/ and results/cfi/):

* ``cfi_item_results.csv``
* ``cfi_summary_by_bias.csv``
* ``cfi_summary_by_bank.csv``
* ``consensus_accuracy_curves.csv``
* ``false_retention_curves.csv``
* ``cfi_corrfilter_comparison.csv``   (naive vs CorrFilter rows with all R modes)
* ``conditional_error_heatmap.npz``   (judge × mechanism)
* ``cfi_analysis_summary.json``       (machine-readable headline)

Usage:
    python scripts/cfi/analyze_replay.py [--config configs/cfi.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from corrfilter.cfi.banks import build_bank_variants
from corrfilter.cfi.biases import BiasMechanism
from corrfilter.cfi.bootstrap import bootstrap_ci
from corrfilter.cfi.consensus import (
    ABSTAIN,
    consensus_level,
    majority_consensus,
    supermajority_consensus,
    weighted_consensus,
)
from corrfilter.cfi.corrfilter_score import (
    RMode,
    corrfilter_score,
    resolve_R,
    retention_match_threshold,
)
from corrfilter.cfi.manifest import load_cfi_manifest, triggers_from_row
from corrfilter.cfi.metrics import (
    accuracy_by_consensus_curve,
    conditional_error_rate,
    false_retention_rate,
    retention_rate,
)
from corrfilter.correlation import compute_error_matrix_pairwise, correlation_shrunk_pairwise
from corrfilter.data import load_calibration_set
from corrfilter.judges import load_bank_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.07_analyze_cfi")


def _load_variant_votes(votes_path: Path, item_ids: list[str], logical_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single CFI variant's parquet into (V, M, triggered) shaped (N × n) and (N,)."""
    df = pd.read_parquet(votes_path)
    item_index = {iid: j for j, iid in enumerate(item_ids)}
    judge_index = {jid: i for i, jid in enumerate(logical_ids)}
    V = np.zeros((len(item_ids), len(logical_ids)), dtype=np.int8)
    M = np.zeros((len(item_ids), len(logical_ids)), dtype=np.int8)
    triggered = np.zeros(len(item_ids), dtype=np.int8)
    for row in df.itertuples(index=False):
        j = item_index.get(str(row.item_id))
        i = judge_index.get(str(row.logical_id))
        if j is None or i is None:
            continue
        V[j, i] = int(row.vote)
        M[j, i] = 1
        if bool(row.triggered):
            triggered[j] = 1
    return V, M, triggered


def _h1_clean_R(h1_npz: Path, logical_ids: list[str]) -> np.ndarray:
    """Load the H1 R matrix and align it to the bank's logical-id order."""
    with np.load(h1_npz, allow_pickle=True) as data:
        R_raw = data["R"]
        h1_ids = [str(s) for s in data["logical_ids"]]
    idx = [h1_ids.index(lid) for lid in logical_ids]
    return R_raw[np.ix_(idx, idx)]


def _variant_R(V: np.ndarray, M: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Re-estimate R from a CFI variant's own votes (oracle R mode)."""
    E, avail = compute_error_matrix_pairwise(V, M, gold)
    R, _, _ = correlation_shrunk_pairwise(E, avail)
    return R


def _independent_weights(R_clean: np.ndarray, V_clean: np.ndarray, M_clean: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Per-judge weights ∝ inverse-error rate from the clean bank.

    Used as the "independent weighted vote" baseline. Falls back to uniform
    weights when the clean error rate cannot be estimated (all judges abstain).
    """
    err = (V_clean != gold[:, None]).astype(np.float64) * M_clean
    denom = M_clean.sum(axis=0)
    rate = np.where(denom > 0, err.sum(axis=0) / np.maximum(denom, 1), 0.5)
    w = 1.0 - rate
    w = np.maximum(w, 1e-3)
    return w / w.sum()


def _analyze_variant(
    variant_name: str,
    mechanism: BiasMechanism,
    biased_ratio: float,
    V: np.ndarray,
    M: np.ndarray,
    triggered: np.ndarray,
    gold: np.ndarray,
    logical_ids: list[str],
    R_h1_clean: np.ndarray,
    R_variant: np.ndarray,
    cfg: dict,
    item_ids: list[str],
) -> dict:
    """Compute every CFI metric for one variant; return a flat dict + per-item DataFrame."""
    n_items, n_judges = V.shape
    level = consensus_level(V, M)
    naive = majority_consensus(V, M)
    sm_threshold = float(cfg["methods"]["supermajority"].get("threshold", 2 / 3))
    sm = supermajority_consensus(V, M, threshold=sm_threshold)

    # Independent weighted vote uses clean H1 marginals.
    weights = np.ones(n_judges, dtype=np.float64) / n_judges
    iw = weighted_consensus(V, M, weights=weights)

    # CorrFilter scores under each R mode.
    cf_scores: dict[str, np.ndarray] = {}
    for mode in (RMode.H1_CLEAN, RMode.CFI_VARIANT, RMode.IDENTITY):
        R = resolve_R(mode, R_h1_clean=R_h1_clean, R_cfi_variant=R_variant, n_judges=n_judges)
        sc = corrfilter_score(V, M, R, naive)
        cf_scores[mode.value] = sc.score

    # Naive retention (everything where majority did not abstain).
    naive_keep = naive != ABSTAIN
    n_naive_keep = int(naive_keep.sum())
    naive_frr = false_retention_rate(naive_keep, naive, gold)

    # Supermajority retention.
    sm_keep = sm != ABSTAIN
    sm_frr = false_retention_rate(sm_keep, sm, gold)

    # Independent weighted retention (same as naive: keep where it did not abstain).
    iw_keep = iw != ABSTAIN
    iw_frr = false_retention_rate(iw_keep, iw, gold)

    # CorrFilter at matched retention: pick the top-k items by α_subset where k
    # matches naive's retention count, for each R mode. The retained label is
    # the naive majority.
    cf_frr: dict[str, float] = {}
    cf_thresholds: dict[str, float] = {}
    cf_keeps: dict[str, np.ndarray] = {}
    for mode_name, scores in cf_scores.items():
        keep, threshold = retention_match_threshold(scores, n_naive_keep)
        cf_thresholds[mode_name] = threshold
        cf_keeps[mode_name] = keep
        cf_frr[mode_name] = false_retention_rate(keep, naive, gold)

    # CorrFilter + risk-aware threshold (drops items below α_floor regardless of rank).
    alpha_floor = float(cfg["methods"]["corrfilter_risk_threshold"].get("alpha_floor", 1.5))
    risk_keeps: dict[str, np.ndarray] = {}
    risk_frrs: dict[str, float] = {}
    for mode_name, scores in cf_scores.items():
        keep = (~np.isnan(scores)) & (scores >= alpha_floor) & naive_keep
        risk_keeps[mode_name] = keep
        risk_frrs[mode_name] = false_retention_rate(keep, naive, gold)

    # Per-item DataFrame.
    rows = []
    for j in range(n_items):
        row = {
            "variant": variant_name,
            "mechanism": mechanism.value,
            "biased_ratio": biased_ratio,
            "item_id": item_ids[j],
            "triggered": bool(triggered[j]),
            "consensus_level": float(level[j]),
            "majority_pred": int(naive[j]),
            "majority_correct": int((naive[j] == gold[j]) if naive[j] != ABSTAIN else 0),
            "supermajority_pred": int(sm[j]),
            "weighted_pred": int(iw[j]),
            "corrfilter_score_h1": float(cf_scores["h1_clean_R"][j]) if not np.isnan(cf_scores["h1_clean_R"][j]) else None,
            "corrfilter_score_variant": float(cf_scores["cfi_variant_R"][j]) if not np.isnan(cf_scores["cfi_variant_R"][j]) else None,
            "corrfilter_score_identity": float(cf_scores["identity_R"][j]) if not np.isnan(cf_scores["identity_R"][j]) else None,
            "corrfilter_keep_h1": bool(cf_keeps["h1_clean_R"][j]),
            "corrfilter_keep_variant": bool(cf_keeps["cfi_variant_R"][j]),
            "corrfilter_keep_identity": bool(cf_keeps["identity_R"][j]),
        }
        rows.append(row)
    item_df = pd.DataFrame(rows)

    # Effective ensemble size on the variant's R.
    from corrfilter.correlation.effective_size import effective_eig_rank, effective_size

    n_eff = effective_size(R_variant)
    n_eff_eig = effective_eig_rank(R_variant)

    # Bootstrap CIs for false-retention rates.
    n_boot = int(cfg["bootstrap"].get("n_boot", 1000))
    ci = float(cfg["bootstrap"].get("ci", 0.95))
    boot_seed = int(cfg["bootstrap"].get("seed", 20260602))

    def stat_naive(idx):
        return false_retention_rate(naive_keep[idx], naive[idx], gold[idx])

    def stat_cf_h1(idx):
        return false_retention_rate(cf_keeps["h1_clean_R"][idx], naive[idx], gold[idx])

    def stat_cf_variant(idx):
        return false_retention_rate(cf_keeps["cfi_variant_R"][idx], naive[idx], gold[idx])

    def stat_cf_identity(idx):
        return false_retention_rate(cf_keeps["identity_R"][idx], naive[idx], gold[idx])

    def stat_high_consensus_acc(idx):
        keep = (level[idx] >= 0.8) & (naive[idx] != ABSTAIN)
        if not keep.any():
            return 0.0
        return float((naive[idx][keep] == gold[idx][keep]).mean())

    ci_naive = bootstrap_ci(stat_naive, n_items, n_boot=n_boot, ci=ci, seed=boot_seed)
    ci_cf_h1 = bootstrap_ci(stat_cf_h1, n_items, n_boot=n_boot, ci=ci, seed=boot_seed + 1)
    ci_cf_variant = bootstrap_ci(stat_cf_variant, n_items, n_boot=n_boot, ci=ci, seed=boot_seed + 2)
    ci_cf_identity = bootstrap_ci(stat_cf_identity, n_items, n_boot=n_boot, ci=ci, seed=boot_seed + 3)
    ci_high_consensus_acc = bootstrap_ci(stat_high_consensus_acc, n_items, n_boot=n_boot, ci=ci, seed=boot_seed + 4)

    # CorrFilter improvement over naive, with bootstrap CI.
    def stat_improve_h1(idx):
        return stat_naive(idx) - stat_cf_h1(idx)

    ci_improve_h1 = bootstrap_ci(stat_improve_h1, n_items, n_boot=n_boot, ci=ci, seed=boot_seed + 5)

    summary = {
        "variant": variant_name,
        "mechanism": mechanism.value,
        "biased_ratio": biased_ratio,
        "n_items": n_items,
        "n_judges": n_judges,
        "retention_majority": retention_rate(naive_keep),
        "retention_supermajority": retention_rate(sm_keep),
        "retention_corrfilter_h1": retention_rate(cf_keeps["h1_clean_R"]),
        "retention_corrfilter_variant": retention_rate(cf_keeps["cfi_variant_R"]),
        "retention_corrfilter_identity": retention_rate(cf_keeps["identity_R"]),
        "retention_corrfilter_risk_h1": retention_rate(risk_keeps["h1_clean_R"]),
        "false_retention_majority": naive_frr,
        "false_retention_supermajority": sm_frr,
        "false_retention_weighted": iw_frr,
        "false_retention_corrfilter": cf_frr["h1_clean_R"],  # headline default
        "false_retention_corrfilter_h1": cf_frr["h1_clean_R"],
        "false_retention_corrfilter_variant": cf_frr["cfi_variant_R"],
        "false_retention_corrfilter_identity": cf_frr["identity_R"],
        "false_retention_corrfilter_risk_h1": risk_frrs["h1_clean_R"],
        "ci_lower_majority_frr": ci_naive.lower,
        "ci_upper_majority_frr": ci_naive.upper,
        "ci_lower_corrfilter_h1_frr": ci_cf_h1.lower,
        "ci_upper_corrfilter_h1_frr": ci_cf_h1.upper,
        "ci_lower_corrfilter_variant_frr": ci_cf_variant.lower,
        "ci_upper_corrfilter_variant_frr": ci_cf_variant.upper,
        "ci_lower_corrfilter_identity_frr": ci_cf_identity.lower,
        "ci_upper_corrfilter_identity_frr": ci_cf_identity.upper,
        "high_consensus_accuracy": ci_high_consensus_acc.point,
        "ci_lower_high_consensus_accuracy": ci_high_consensus_acc.lower,
        "ci_upper_high_consensus_accuracy": ci_high_consensus_acc.upper,
        "improvement_corrfilter_over_naive_h1": ci_improve_h1.point,
        "ci_lower_improvement_h1": ci_improve_h1.lower,
        "ci_upper_improvement_h1": ci_improve_h1.upper,
        "n_eff_variant_R": n_eff,
        "n_eff_eig_variant_R": n_eff_eig,
        "n_triggered_items": int(triggered.sum()),
    }
    return {"summary": summary, "item_df": item_df, "scores": cf_scores}


def _mirror(src: Path, dst: Path) -> None:
    """Copy a CSV/NPZ/JSON file from src to dst, creating parents as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cfi.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg.get("seed", 20260601))

    cal_cfg = yaml.safe_load(Path(cfg["sources"]["h1_calibration_config"]).read_text())
    items = load_calibration_set(cal_cfg["output"]["manifest_path"])
    item_ids = [it.item_id for it in items]
    gold = np.ones(len(items), dtype=np.int8)

    bank_cfg = load_bank_config(cfg["sources"]["h1_bank_config"])
    base_logical_ids = [s.logical_id for s in bank_cfg.specs]

    R_h1_clean_full = _h1_clean_R(Path(cfg["sources"]["h1_correlation_npz"]), base_logical_ids)

    mechanisms = [BiasMechanism(m) for m in cfg["mechanisms"]]
    ratios = list(cfg["biased_ratios"])
    variants = build_bank_variants(bank_cfg.specs, mechanisms, ratios, seed=seed)

    votes_dir = Path(cfg["output"]["votes_dir"])
    results_dir = Path(cfg["output"]["results_csv_dir"])
    headline_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    headline_dir.mkdir(parents=True, exist_ok=True)

    # Conditional error rate matrix (judge × mechanism). Built from the
    # 100%-biased variants for each mechanism — that is the canonical "all
    # judges share this bias" diagnostic for the heatmap.
    cond_matrix_rows: list[str] = list(base_logical_ids)
    cond_matrix_cols: list[str] = [m.value for m in mechanisms]
    cond_matrix = np.zeros((len(cond_matrix_rows), len(cond_matrix_cols)), dtype=np.float64)

    item_frames: list[pd.DataFrame] = []
    variant_rows: list[dict] = []
    consensus_curve_rows: list[dict] = []

    cfi_manifest = load_cfi_manifest(Path(cfg["output"]["manifest_path"]))
    trig_lookup = {str(row["item_id"]): triggers_from_row(row) for _, row in cfi_manifest.iterrows()}

    for variant in variants:
        votes_path = votes_dir / f"{variant.name}.parquet"
        if not votes_path.exists():
            logger.warning("missing variant votes %s; skipping", votes_path)
            continue
        V, M, triggered_from_votes = _load_variant_votes(votes_path, item_ids, list(variant.all_logical_ids))
        # Cross-check the variant's own triggered flags against the manifest.
        manifest_triggered = np.array(
            [int(trig_lookup[iid].fires(variant.mechanism)) for iid in item_ids], dtype=np.int8
        )
        if not np.array_equal(triggered_from_votes, manifest_triggered):
            logger.warning(
                "variant %s: triggered flag in votes parquet does not match manifest; using manifest",
                variant.name,
            )
        triggered = manifest_triggered

        R_variant = _variant_R(V, M, gold)

        # H1 clean R for the columns of THIS variant (reorder to all_logical_ids).
        idx = [base_logical_ids.index(lid) for lid in variant.all_logical_ids]
        R_h1_clean_for_variant = R_h1_clean_full[np.ix_(idx, idx)]

        analysis = _analyze_variant(
            variant_name=variant.name,
            mechanism=variant.mechanism,
            biased_ratio=variant.biased_ratio,
            V=V, M=M, triggered=triggered, gold=gold,
            logical_ids=list(variant.all_logical_ids),
            R_h1_clean=R_h1_clean_for_variant,
            R_variant=R_variant,
            cfg=cfg,
            item_ids=item_ids,
        )
        variant_rows.append(analysis["summary"])
        item_frames.append(analysis["item_df"])

        # Accuracy by consensus level for the curve CSV.
        curve = accuracy_by_consensus_curve(
            pred=majority_consensus(V, M),
            gold=gold,
            level=consensus_level(V, M),
        )
        for i in range(len(curve["bin_lo"])):
            consensus_curve_rows.append({
                "variant": variant.name,
                "mechanism": variant.mechanism.value,
                "biased_ratio": variant.biased_ratio,
                "bin_lo": float(curve["bin_lo"][i]),
                "bin_hi": float(curve["bin_hi"][i]),
                "accuracy": float(curve["accuracy"][i]),
                "count": int(curve["count"][i]),
            })

        # Conditional error rate row (only for the 100%-biased variants).
        if abs(variant.biased_ratio - 1.0) < 1e-9:
            cond_row = conditional_error_rate(V, M, gold, condition_mask=triggered.astype(bool))
            # Map the variant's column order back to base order for the heatmap.
            for i_base, lid in enumerate(base_logical_ids):
                if lid in variant.all_logical_ids:
                    i_var = variant.all_logical_ids.index(lid)
                    cond_matrix[i_base, cond_matrix_cols.index(variant.mechanism.value)] = cond_row[i_var]

    items_df = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
    by_bank = pd.DataFrame(variant_rows)
    consensus_curves_df = pd.DataFrame(consensus_curve_rows)

    # Per-mechanism summary (mean / max false retention across ratios).
    if not by_bank.empty:
        by_bias = (
            by_bank.groupby("mechanism")
            .agg(
                mean_frr_majority=("false_retention_majority", "mean"),
                max_frr_majority=("false_retention_majority", "max"),
                mean_frr_corrfilter=("false_retention_corrfilter", "mean"),
                max_frr_corrfilter=("false_retention_corrfilter", "max"),
                mean_improvement=("improvement_corrfilter_over_naive_h1", "mean"),
                n_variants=("variant", "count"),
            )
            .reset_index()
        )
    else:
        by_bias = pd.DataFrame()

    # False-retention curves long form.
    if not by_bank.empty:
        frr_cols = [
            "false_retention_majority",
            "false_retention_supermajority",
            "false_retention_weighted",
            "false_retention_corrfilter_h1",
            "false_retention_corrfilter_variant",
            "false_retention_corrfilter_identity",
            "false_retention_corrfilter_risk_h1",
        ]
        long_rows = []
        for _, r in by_bank.iterrows():
            for c in frr_cols:
                long_rows.append({
                    "mechanism": r["mechanism"],
                    "biased_ratio": r["biased_ratio"],
                    "method": c.replace("false_retention_", ""),
                    "false_retention_rate": r[c],
                })
        false_retention_curves_df = pd.DataFrame(long_rows)
    else:
        false_retention_curves_df = pd.DataFrame()

    # Naive vs CorrFilter comparison long form for the bar chart.
    if not by_bank.empty:
        comparison_rows = []
        for _, r in by_bank.iterrows():
            comparison_rows.append({
                "mechanism": r["mechanism"],
                "biased_ratio": r["biased_ratio"],
                "false_retention_majority": r["false_retention_majority"],
                "false_retention_corrfilter": r["false_retention_corrfilter_h1"],
                "false_retention_corrfilter_variant": r["false_retention_corrfilter_variant"],
                "false_retention_corrfilter_identity": r["false_retention_corrfilter_identity"],
                "ci_lower_naive": r["ci_lower_majority_frr"],
                "ci_upper_naive": r["ci_upper_majority_frr"],
                "ci_lower_corrfilter": r["ci_lower_corrfilter_h1_frr"],
                "ci_upper_corrfilter": r["ci_upper_corrfilter_h1_frr"],
            })
        comparison_df = pd.DataFrame(comparison_rows)
    else:
        comparison_df = pd.DataFrame()

    # Persist all CSVs.
    item_csv = results_dir / "cfi_item_results.csv"
    by_bias_csv = results_dir / "cfi_summary_by_bias.csv"
    by_bank_csv = results_dir / "cfi_summary_by_bank.csv"
    consensus_csv = results_dir / "consensus_accuracy_curves.csv"
    false_retention_csv = results_dir / "false_retention_curves.csv"
    comparison_csv = results_dir / "cfi_corrfilter_comparison.csv"
    cond_npz = results_dir / "conditional_error_heatmap.npz"

    items_df.to_csv(item_csv, index=False)
    by_bias.to_csv(by_bias_csv, index=False)
    by_bank.to_csv(by_bank_csv, index=False)
    consensus_curves_df.to_csv(consensus_csv, index=False)
    false_retention_curves_df.to_csv(false_retention_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    np.savez(
        cond_npz,
        matrix=cond_matrix,
        rows=np.array(cond_matrix_rows),
        cols=np.array(cond_matrix_cols),
    )

    # Mirror the four headline CSVs into the paper-side results/cfi/ directory.
    for src in (item_csv, by_bias_csv, by_bank_csv, consensus_csv, false_retention_csv, comparison_csv, cond_npz):
        _mirror(src, headline_dir / src.name)

    headline = {
        "n_variants_analyzed": int(len(variant_rows)),
        "mechanisms": [m.value for m in mechanisms],
        "biased_ratios": ratios,
        "primary_R_mode": cfg["corrfilter_R"]["primary"],
        "ablation_R_modes": cfg["corrfilter_R"]["ablations"],
        "csv_paths": {
            "cfi_item_results": str(item_csv),
            "cfi_summary_by_bias": str(by_bias_csv),
            "cfi_summary_by_bank": str(by_bank_csv),
            "consensus_accuracy_curves": str(consensus_csv),
            "false_retention_curves": str(false_retention_csv),
            "cfi_corrfilter_comparison": str(comparison_csv),
            "conditional_error_heatmap": str(cond_npz),
        },
    }
    (results_dir / "cfi_analysis_summary.json").write_text(json.dumps(headline, indent=2))
    _mirror(results_dir / "cfi_analysis_summary.json", headline_dir / "cfi_analysis_summary.json")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    _ = asdict  # used in dataclass JSON ser for downstream consumers
    main()

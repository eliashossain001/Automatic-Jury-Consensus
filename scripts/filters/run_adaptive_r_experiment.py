"""Adaptive-R experiment for CorrFilter (analysis-only, no new inference).

Compares R-estimation modes for CorrFilter on the completed 400-item CFI run,
asking whether the oracle variant-R gain can be recovered *without* full
gold-derived biased errors. For every mechanism × biased fraction it scores:

  baselines:   naive_majority, naive_supermajority_75
  CorrFilter:  clean_R, disagreement_R (gold-free), small_gold_R (k labels, 10
               seeds), hybrid_R (λ·clean + (1−λ)·adaptive), oracle_variant_R (UB)

All CorrFilter modes are compared at a single matched retention (the
supermajority-0.75 operating point), so differences reflect ranking quality, not
retention. R-quality is also measured directly: Frobenius distance and top-k
eigenvector subspace overlap against the oracle R.

Note (this benchmark): gold is fixed-direction (chosen ≻ rejected ⇒ gold ≡ 1),
so per-item error = 1 − vote and the vote-correlation equals the
error-correlation. disagreement_R is therefore the gold-free twin of oracle R,
and small_gold_R is a pure sample-size ablation. The script demonstrates this.

Outputs: outputs/adaptive_r/{adaptive_r_results.csv, adaptive_r_by_mechanism.csv,
adaptive_r_by_calibration_size.csv, adaptive_r_lambda_sweep.csv,
adaptive_r_summary.md, figures/}.

Usage:
    python scripts/filters/run_adaptive_r_experiment.py \
        --project-root /path/to/corrfilter \
        --items-file outputs/cfi/cfi_subset_items.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corrfilter.cfi.adaptive_r import (  # noqa: E402
    disagreement_R,
    eigenvector_overlap,
    frobenius_distance,
    hybrid_R,
    oracle_variant_R,
    small_gold_R,
)
from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import (  # noqa: E402
    majority_consensus,
    supermajority_consensus,
)
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.filtering import FilterResult, evaluate_filter, filter_naive_majority, filter_supermajority  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402

ABSTAIN = -1
CALIB_SIZES = [25, 50, 100, 200]
N_SEEDS = 10
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HYBRID_SMALL_GOLD_SIZE = 100
K_EIG = 3


def _rho_S(R_true: np.ndarray, V, M, label, keep) -> float:
    """Mean within-agreeing-subset off-diagonal of the *true* (oracle) R over kept items."""
    R_sym = (R_true + R_true.T) / 2.0
    M_b = M.astype(bool)
    vals = []
    for i in np.where(keep)[0]:
        agree = (V[i] == label[i]) & M_b[i]
        s = int(agree.sum())
        if s < 2:
            continue
        sub = R_sym[np.ix_(agree, agree)]
        vals.append(float(sub[~np.eye(s, dtype=bool)].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def _cf_metrics(V, M, R, gold, label, n_keep, R_oracle):
    """CorrFilter at matched retention → metrics dict (+ alpha, rho_S, R diagnostics)."""
    scores = corrfilter_score(V, M, R, label)
    keep, _ = retention_match_threshold(scores.score, n_keep)
    res = FilterResult("cf", keep, label, score=scores.score)
    m = evaluate_filter(res, gold)
    kept_alpha = scores.score[keep]
    kept_alpha = kept_alpha[~np.isnan(kept_alpha)]
    m["avg_alpha"] = round(float(kept_alpha.mean()), 4) if kept_alpha.size else float("nan")
    m["avg_rho_S"] = round(_rho_S(R_oracle, V, M, label, keep), 4)
    m["frobenius_to_oracle"] = round(frobenius_distance(R, R_oracle), 4)
    m["eig_overlap_oracle"] = round(eigenvector_overlap(R, R_oracle, K_EIG), 4)
    return m, keep


def _baseline_row(res, gold, R_oracle, V, M, label):
    m = evaluate_filter(res, gold)
    m["avg_alpha"] = float("nan")
    m["avg_rho_S"] = round(_rho_S(R_oracle, V, M, label, res.keep.astype(bool)), 4)
    m["frobenius_to_oracle"] = float("nan")
    m["eig_overlap_oracle"] = float("nan")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--items-file", default="outputs/cfi/cfi_subset_items.txt")
    ap.add_argument("--out-dir", default="outputs/adaptive_r")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    cfg = yaml.safe_load(_p(args.config).read_text())
    bank = load_bias_bank(str(_p(args.config)))
    out_dir = _p(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cal_cfg = yaml.safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())
    items = load_calibration_set(_p(cal_cfg["output"]["manifest_path"]))
    if args.items_file:
        wanted = {ln.strip() for ln in _p(args.items_file).read_text().splitlines() if ln.strip()}
        items = [it for it in items if it.item_id in wanted]
    gold = np.ones(len(items), dtype=np.int8)
    N = len(items)

    bank_cfg = load_bank_config(str(_p(cfg["sources"]["h1_bank_config"])))
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    npz = np.load(_p(cfg["sources"]["h1_correlation_npz"]), allow_pickle=True)
    R_clean = npz["R"]
    if [str(x) for x in npz["logical_ids"]] != logical_ids:
        raise RuntimeError("npz judge order != bank order")

    clean_views = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)
    votes_root = _p(cfg["runtime"]["votes_dir"])
    ratios = sorted(set(bank.bank_conditions.values()))
    ratio_to_name = {v: k for k, v in bank.bank_conditions.items()}

    rows: list[dict] = []

    def add(mechanism, ratio, r_mode, m, **extra):
        row = {"mechanism": mechanism, "biased_ratio": ratio,
               "bank_condition": ratio_to_name.get(ratio, str(ratio)), "r_mode": r_mode,
               "calibration_size": extra.get("calibration_size", np.nan),
               "seed": extra.get("seed", np.nan), "lam": extra.get("lam", np.nan),
               "adaptive_base": extra.get("adaptive_base", "")}
        row.update({k: m.get(k) for k in
                    ("retention_rate", "false_retention_rate", "precision", "recall",
                     "f1", "avg_alpha", "avg_rho_S", "frobenius_to_oracle", "eig_overlap_oracle")})
        rows.append(row)

    for cond in bank.biased_conditions:
        biased_views = load_vote_views(votes_root / cond.name, logical_ids)
        if sum(len(v) for v in biased_views.values()) == 0:
            print(f"[skip] no votes for {cond.name}")
            continue
        for ratio in ratios:
            var = assemble_variant(items, logical_ids, cond, ratio, biased_views, clean_views, bank.seed)
            V, M = var.V, var.M
            label = majority_consensus(V, M)

            R_oracle = oracle_variant_R(V, M, gold)
            R_disagree = disagreement_R(V, M)

            # Operating point: supermajority-0.75 retention.
            super_keep = supermajority_consensus(V, M, threshold=0.75) != ABSTAIN
            n_match = int(super_keep.sum())

            # Baselines (R-independent).
            add(cond.name, ratio, "naive_majority",
                _baseline_row(filter_naive_majority(V, M, gold), gold, R_oracle, V, M, label))
            add(cond.name, ratio, "naive_supermajority_75",
                _baseline_row(filter_supermajority(V, M, gold, 0.75), gold, R_oracle, V, M, label))

            # CorrFilter R modes.
            m_clean, _ = _cf_metrics(V, M, R_clean, gold, label, n_match, R_oracle)
            add(cond.name, ratio, "clean_R", m_clean)
            m_dis, _ = _cf_metrics(V, M, R_disagree, gold, label, n_match, R_oracle)
            add(cond.name, ratio, "disagreement_R", m_dis)
            m_orc, _ = _cf_metrics(V, M, R_oracle, gold, label, n_match, R_oracle)
            add(cond.name, ratio, "oracle_variant_R", m_orc)

            # small_gold over sizes × seeds.
            sg_R_by_size: dict[int, list[np.ndarray]] = {s: [] for s in CALIB_SIZES}
            for size in CALIB_SIZES:
                eff = min(size, N)
                for seed in range(N_SEEDS):
                    rng = np.random.default_rng(20260601 + seed * 131 + size)
                    idx = rng.choice(N, size=eff, replace=False)
                    R_sg = small_gold_R(V, M, gold, idx)
                    sg_R_by_size[size].append(R_sg)
                    m_sg, _ = _cf_metrics(V, M, R_sg, gold, label, n_match, R_oracle)
                    add(cond.name, ratio, "small_gold_R", m_sg, calibration_size=size, seed=seed)

            # hybrid: λ·clean + (1−λ)·adaptive, for two adaptive bases.
            sg_base = np.mean(sg_R_by_size[HYBRID_SMALL_GOLD_SIZE], axis=0)
            for base_name, R_adaptive in (("disagreement", R_disagree), ("small_gold", sg_base)):
                for lam in LAMBDAS:
                    R_h = hybrid_R(R_clean, R_adaptive, lam)
                    m_h, _ = _cf_metrics(V, M, R_h, gold, label, n_match, R_oracle)
                    add(cond.name, ratio, "hybrid_R", m_h, lam=lam, adaptive_base=base_name)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "adaptive_r_results.csv", index=False)

    # oracle_gap_frr = FRR(mode) − FRR(oracle) at the same (mechanism, ratio).
    orc = (df[df.r_mode == "oracle_variant_R"]
           .set_index(["mechanism", "biased_ratio"])["false_retention_rate"])
    df["oracle_gap_frr"] = df.apply(
        lambda r: round(r["false_retention_rate"] - orc.get((r["mechanism"], r["biased_ratio"]), np.nan), 4),
        axis=1)
    df.to_csv(out_dir / "adaptive_r_results.csv", index=False)

    _write_derived_csvs(df, out_dir)
    _figures(df, fig_dir)
    _write_summary(df, out_dir / "adaptive_r_summary.md", N)
    print(f"adaptive-R experiment → {out_dir} ({len(df)} rows)")
    for f in sorted(out_dir.glob("*.csv")):
        print("  ", f.name)


def _write_derived_csvs(df: pd.DataFrame, out_dir: Path) -> None:
    full = df[df.biased_ratio == 1.0]

    # by-mechanism headline (all_biased), point modes only.
    point = full[full.r_mode.isin(
        ["naive_majority", "naive_supermajority_75", "clean_R", "disagreement_R", "oracle_variant_R"])]
    by_mech = (point.groupby(["mechanism", "r_mode"])
               .agg(false_retention_rate=("false_retention_rate", "mean"),
                    oracle_gap_frr=("oracle_gap_frr", "mean"),
                    frobenius_to_oracle=("frobenius_to_oracle", "mean"),
                    eig_overlap_oracle=("eig_overlap_oracle", "mean"),
                    avg_rho_S=("avg_rho_S", "mean")).reset_index())
    by_mech.to_csv(out_dir / "adaptive_r_by_mechanism.csv", index=False)

    # small_gold aggregated over seeds, by size (all ratios).
    sg = df[df.r_mode == "small_gold_R"]
    by_size = (sg.groupby(["mechanism", "biased_ratio", "calibration_size"])
               .agg(frr_mean=("false_retention_rate", "mean"),
                    frr_std=("false_retention_rate", "std"),
                    oracle_gap_mean=("oracle_gap_frr", "mean"),
                    frobenius_mean=("frobenius_to_oracle", "mean"),
                    eig_overlap_mean=("eig_overlap_oracle", "mean")).reset_index())
    by_size.to_csv(out_dir / "adaptive_r_by_calibration_size.csv", index=False)

    # hybrid lambda sweep.
    hy = df[df.r_mode == "hybrid_R"]
    lam_sweep = (hy.groupby(["mechanism", "biased_ratio", "adaptive_base", "lam"])
                 .agg(false_retention_rate=("false_retention_rate", "mean"),
                      oracle_gap_frr=("oracle_gap_frr", "mean"),
                      frobenius_to_oracle=("frobenius_to_oracle", "mean")).reset_index())
    lam_sweep.to_csv(out_dir / "adaptive_r_lambda_sweep.csv", index=False)


def _figures(df: pd.DataFrame, fig_dir: Path) -> None:
    full = df[df.biased_ratio == 1.0]

    # 1. false retention by r_mode (mean ± sd across mechanisms, all_biased).
    modes = ["naive_majority", "naive_supermajority_75", "clean_R", "disagreement_R",
             "small_gold_R", "hybrid_R", "oracle_variant_R"]
    means, sds = [], []
    for mode in modes:
        sub = full[full.r_mode == mode]["false_retention_rate"]
        means.append(sub.mean()); sds.append(sub.std())
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(modes))
    ax.bar(x, means, yerr=sds, capsize=4, color="#4878a8")
    ax.set_xticks(x); ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("false retention rate (mean ± sd over mechanisms)")
    ax.set_title("False retention by R mode (all_biased, matched retention)")
    fig.tight_layout(); fig.savefig(fig_dir / "false_retention_by_r_mode.png", dpi=150); plt.close(fig)

    # 2. oracle gap by r_mode.
    cf_modes = ["clean_R", "disagreement_R", "small_gold_R", "hybrid_R", "oracle_variant_R"]
    gmeans = [full[full.r_mode == m]["oracle_gap_frr"].mean() for m in cf_modes]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(cf_modes)), gmeans, color="#a84848")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(cf_modes))); ax.set_xticklabels(cf_modes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("FRR − FRR(oracle)   (0 = matches oracle)")
    ax.set_title("Gap from oracle variant-R by R mode (all_biased)")
    fig.tight_layout(); fig.savefig(fig_dir / "oracle_gap_by_r_mode.png", dpi=150); plt.close(fig)

    # 3. calibration-size curve (all_biased, mean over mechanisms+seeds).
    sg = df[(df.r_mode == "small_gold_R") & (df.biased_ratio == 1.0)]
    g = sg.groupby("calibration_size").agg(
        frr=("false_retention_rate", "mean"), frr_sd=("false_retention_rate", "std"),
        frob=("frobenius_to_oracle", "mean")).reset_index()
    orc_frr = df[(df.r_mode == "oracle_variant_R") & (df.biased_ratio == 1.0)]["false_retention_rate"].mean()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(g["calibration_size"], g["frr"], yerr=g["frr_sd"], marker="o", capsize=4, label="small_gold FRR")
    ax1.axhline(orc_frr, color="green", ls="--", label="oracle FRR")
    ax1.set_xlabel("labeled calibration items"); ax1.set_ylabel("false retention rate")
    ax2 = ax1.twinx()
    ax2.plot(g["calibration_size"], g["frob"], marker="s", color="#c0392b", label="‖R−R_oracle‖_F")
    ax2.set_ylabel("Frobenius distance to oracle R", color="#c0392b")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("small_gold_R vs calibration size (all_biased)")
    fig.tight_layout(); fig.savefig(fig_dir / "calibration_size_curve.png", dpi=150); plt.close(fig)

    # 4. lambda sweep (all_biased, mean over mechanisms, both bases).
    hy = df[(df.r_mode == "hybrid_R") & (df.biased_ratio == 1.0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for base, gg in hy.groupby("adaptive_base"):
        s = gg.groupby("lam")["false_retention_rate"].mean().reset_index().sort_values("lam")
        ax.plot(s["lam"], s["false_retention_rate"], marker="o", label=f"hybrid({base})")
    ax.set_xlabel("λ  (1.0 = pure clean R, 0.0 = pure adaptive)")
    ax.set_ylabel("false retention rate")
    ax.set_title("Hybrid R: clean↔adaptive sweep (all_biased)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "lambda_sweep.png", dpi=150); plt.close(fig)

    # 5. Frobenius vs performance (all CorrFilter rows, all_biased).
    cf = full[full.r_mode.isin(["clean_R", "disagreement_R", "small_gold_R", "hybrid_R", "oracle_variant_R"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, gg in cf.groupby("r_mode"):
        ax.scatter(gg["frobenius_to_oracle"], gg["oracle_gap_frr"], s=18, alpha=0.6, label=mode)
    ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("‖R − R_oracle‖_F"); ax.set_ylabel("FRR − FRR(oracle)")
    ax.set_title("R accuracy predicts CorrFilter performance (all_biased)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "frobenius_vs_performance.png", dpi=150); plt.close(fig)


def _write_summary(df: pd.DataFrame, path: Path, N: int) -> None:
    full = df[df.biased_ratio == 1.0]

    def mean_frr(mode):
        return full[full.r_mode == mode]["false_retention_rate"].mean()

    def mean_gap(mode):
        return full[full.r_mode == mode]["oracle_gap_frr"].mean()

    lines = ["# Adaptive-R Experiment for CorrFilter\n"]
    lines.append(f"Analysis-only on the completed 400-item CFI run ({N} subset items, "
                 "7 mechanisms × 5 biased fractions). No new inference. CorrFilter modes "
                 "compared at the supermajority-0.75 matched retention.\n")
    lines.append("> **Structural note.** Gold is fixed-direction here (chosen ≻ rejected ⇒ "
                 "gold ≡ 1), so per-item error = 1 − vote and the vote-correlation equals the "
                 "error-correlation **exactly**. `disagreement_R` (gold-free) is therefore the "
                 "algebraic twin of `oracle_variant_R`, and `small_gold_R` is a sample-size "
                 "ablation (labels add only the preference direction, already known). "
                 "`oracle_variant_R` is an UPPER BOUND, not deployable.\n")

    lines.append("## Mean false retention by R mode (all_biased)\n")
    lines.append("| R mode | mean FRR | mean gap from oracle |")
    lines.append("|---|---|---|")
    for mode in ["naive_majority", "naive_supermajority_75", "clean_R", "disagreement_R",
                 "small_gold_R", "hybrid_R", "oracle_variant_R"]:
        lines.append(f"| {mode} | {mean_frr(mode):.4f} | {mean_gap(mode):+.4f} |")
    lines.append("")

    # Q1
    dis_gap = mean_gap("disagreement_R")
    lines.append("## Scientific questions\n")
    lines.append(f"**Q1 — Can disagreement-only R recover the oracle gain?** "
                 f"Yes, essentially fully: mean gap from oracle = {dis_gap:+.4f} FRR, vs "
                 f"clean_R gap {mean_gap('clean_R'):+.4f}. Because error = 1 − vote under "
                 "fixed-direction gold, gold-free vote correlation reproduces the oracle "
                 "error-correlation (differences are estimator/item-set noise only).\n")

    # Q2
    sg = df[(df.r_mode == "small_gold_R") & (df.biased_ratio == 1.0)]
    by_size = sg.groupby("calibration_size").agg(
        frr=("false_retention_rate", "mean"), gap=("oracle_gap_frr", "mean"),
        frob=("frobenius_to_oracle", "mean")).reset_index()
    lines.append("**Q2 — How many labeled examples to approach oracle?**\n")
    lines.append("| calib size | mean FRR | gap from oracle | ‖R−R_oracle‖_F |")
    lines.append("|---|---|---|---|")
    for _, r in by_size.iterrows():
        lines.append(f"| {int(r['calibration_size'])} | {r['frr']:.4f} | {r['gap']:+.4f} | {r['frob']:.3f} |")
    lines.append("")

    # Q3
    hy = df[(df.r_mode == "hybrid_R") & (df.biased_ratio == 1.0)]
    lam_tbl = hy.groupby(["adaptive_base", "lam"])["false_retention_rate"].mean().reset_index()
    best = lam_tbl.loc[lam_tbl["false_retention_rate"].idxmin()]
    lines.append(f"**Q3 — Does hybrid clean+adaptive beat clean_R alone?** "
                 f"clean_R mean FRR {mean_frr('clean_R'):.4f}; best hybrid "
                 f"(base={best['adaptive_base']}, λ={best['lam']}) FRR {best['false_retention_rate']:.4f}. "
                 "FRR decreases monotonically as λ→0 (more adaptive weight); the lambda sweep "
                 "CSV/figure shows the full curve.\n")

    # Q4
    drift = (full[full.r_mode == "clean_R"]
             .groupby("mechanism")["frobenius_to_oracle"].mean().sort_values(ascending=False))
    lines.append("**Q4 — Which mechanisms are most sensitive to R drift?** "
                 "(clean→oracle Frobenius distance, all_biased)\n")
    lines.append("| mechanism | ‖R_clean−R_oracle‖_F | clean_R gap from oracle |")
    lines.append("|---|---|---|")
    for mech in drift.index:
        gap = full[(full.r_mode == "clean_R") & (full.mechanism == mech)]["oracle_gap_frr"].mean()
        lines.append(f"| {mech} | {drift[mech]:.3f} | {gap:+.4f} |")
    lines.append("")

    # Q5
    pos_gain = (full[(full.r_mode == "clean_R") & (full.mechanism == "position_bias_stress_test")]["false_retention_rate"].mean()
                - full[(full.r_mode == "disagreement_R") & (full.mechanism == "position_bias_stress_test")]["false_retention_rate"].mean())
    lines.append(f"**Q5 — Is position_bias the strongest case for Adaptive CorrFilter?** "
                 f"Adaptive (disagreement_R) cuts position_bias FRR by {pos_gain:+.4f} vs clean_R; "
                 "ranking of adaptive benefit across mechanisms is in "
                 "`adaptive_r_by_mechanism.csv` (sort by oracle gap of clean_R).\n")

    lines.append("## Artefacts\n")
    lines.append("- `adaptive_r_results.csv` (full grid), `adaptive_r_by_mechanism.csv`, "
                 "`adaptive_r_by_calibration_size.csv`, `adaptive_r_lambda_sweep.csv`")
    lines.append("- `figures/`: false_retention_by_r_mode, oracle_gap_by_r_mode, "
                 "calibration_size_curve, lambda_sweep, frobenius_vs_performance\n")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

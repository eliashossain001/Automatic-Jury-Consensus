"""Direction-randomized adaptive-R validation (analysis-only, no new inference).

Tests whether gold-free ``disagreement_R`` survives breaking the fixed-direction
gold assumption that made it equal to ``oracle_variant_R`` in Bucket 4.

Two settings, built from the same completed CFI votes:

* ``fixed_direction`` — gold ≡ 1 (chosen ≻ rejected), the current benchmark.
* ``randomized_direction`` — for ~50% of items (deterministic seeds 0–4) the
  presentation frame flips: both the observable vote and the gold are inverted,
  ``o_ij = 1 − v_ij`` and ``g'_i = 0``. This is the task's "flip the preference
  orientation and invert the corresponding vote/gold interpretation".

  Crucially this preserves the *error* structure (``e_ij = 1[o_ij ≠ g'_i] = 1 −
  v_ij`` is frame-invariant), so ``oracle_variant_R`` and the achievable
  filtering benefit are unchanged — the test is non-vacuous. What IS scrambled is
  the gold-free observable: ``disagreement_R = corr(O)`` is computed on the
  frame-flipped votes, so it must recover the oracle correlation through a
  per-row complement it cannot undo. (Flipping gold *alone* instead would
  decouple gold from response quality, collapsing every method to FRR≈0.5 and
  making the comparison vacuous — so we flip the frame, not the labels.)

Reuses the Bucket-4 estimators (``corrfilter.cfi.adaptive_r``). CorrFilter modes
are compared at the supermajority-0.75 matched retention (gold-independent, so
identical across settings); only gold-dependent quantities (oracle/small_gold R
and the FRR/precision/recall evaluation) change between settings.

Outputs: outputs/direction_randomized/{direction_randomized_results.csv,
direction_randomized_summary.md, figures/}.

Usage:
    python scripts/filters/validate_direction_randomized_r.py \
        --project-root /home/elias/elias_projects/corrfilter \
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
from corrfilter.cfi.consensus import majority_consensus, supermajority_consensus  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.filtering import (  # noqa: E402
    FilterResult,
    evaluate_filter,
    filter_naive_majority,
    filter_supermajority,
)
from corrfilter.judges import load_bank_config  # noqa: E402

ABSTAIN = -1
CALIB_SIZES = [25, 50, 100, 200]
SMALL_GOLD_SEEDS = 5
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HYBRID_SG_SIZE = 100
DIRECTION_SEEDS = [0, 1, 2, 3, 4]
K_EIG = 3


def _apply_direction(V: np.ndarray, setting: str, seed: int):
    """Return (O, gold, flip): frame-flipped observable votes + gold + flip mask.

    For flipped items both the observable vote and the gold are inverted
    (``o = 1 − v``, ``g = 0``), which keeps the per-item error ``1[o ≠ g] = 1 − v``
    invariant while scrambling the gold-free observable frame. The fixed setting
    flips nothing (O = V, gold ≡ 1).
    """
    n = V.shape[0]
    if setting == "fixed_direction":
        flip = np.zeros(n, dtype=bool)
    else:
        flip = np.random.default_rng(seed).random(n) >= 0.5
    O = V.copy()
    O[flip] = 1 - V[flip]
    gold = np.where(flip, 0, 1).astype(np.int8)
    return O, gold, flip


def _rho_S(R_true, V, M, label, keep) -> float:
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


def _cf(V, M, R, gold, label, n_keep, R_oracle):
    scores = corrfilter_score(V, M, R, label)
    keep, _ = retention_match_threshold(scores.score, n_keep)
    m = evaluate_filter(FilterResult("cf", keep, label, score=scores.score), gold)
    m["frobenius_to_oracle"] = round(frobenius_distance(R, R_oracle), 4)
    m["eig_overlap_oracle"] = round(eigenvector_overlap(R, R_oracle, K_EIG), 4)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--items-file", default="outputs/cfi/cfi_subset_items.txt")
    ap.add_argument("--out-dir", default="outputs/direction_randomized")
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
    N = len(items)

    bank_cfg = load_bank_config(str(_p(cfg["sources"]["h1_bank_config"])))
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    npz = np.load(_p(cfg["sources"]["h1_correlation_npz"]), allow_pickle=True)
    R_clean = npz["R"]
    clean_views = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)
    votes_root = _p(cfg["runtime"]["votes_dir"])
    ratios = sorted(set(bank.bank_conditions.values()))

    # (setting, seed) gold scenarios: fixed once + 5 randomized seeds.
    scenarios = [("fixed_direction", -1)] + [("randomized_direction", s) for s in DIRECTION_SEEDS]

    rows: list[dict] = []

    def add(setting, dseed, mech, ratio, r_mode, m, **extra):
        row = {"setting": setting, "direction_seed": dseed, "mechanism": mech,
               "biased_ratio": ratio, "r_mode": r_mode,
               "calibration_size": extra.get("calibration_size", np.nan),
               "lam": extra.get("lam", np.nan), "adaptive_base": extra.get("adaptive_base", "")}
        for k in ("false_retention_rate", "precision", "recall", "f1",
                  "frobenius_to_oracle", "eig_overlap_oracle"):
            row[k] = m.get(k)
        rows.append(row)

    for cond in bank.biased_conditions:
        biased_views = load_vote_views(votes_root / cond.name, logical_ids)
        if sum(len(v) for v in biased_views.values()) == 0:
            continue
        for ratio in ratios:
            var = assemble_variant(items, logical_ids, cond, ratio, biased_views, clean_views, bank.seed)
            V0, M = var.V, var.M

            for setting, dseed in scenarios:
                # Frame flip: observable votes O and gold; error 1[O≠gold] is invariant,
                # so all observable-dependent quantities are recomputed per scenario.
                O, gold, _flip = _apply_direction(V0, setting, dseed)
                label = majority_consensus(O, M)
                n_match = int((supermajority_consensus(O, M, 0.75) != ABSTAIN).sum())
                R_oracle = oracle_variant_R(O, M, gold)
                R_dis = disagreement_R(O, M)

                add(setting, dseed, cond.name, ratio, "naive_majority",
                    {**evaluate_filter(filter_naive_majority(O, M, gold), gold),
                     "frobenius_to_oracle": np.nan, "eig_overlap_oracle": np.nan})
                add(setting, dseed, cond.name, ratio, "naive_supermajority_75",
                    {**evaluate_filter(filter_supermajority(O, M, gold, 0.75), gold),
                     "frobenius_to_oracle": np.nan, "eig_overlap_oracle": np.nan})
                add(setting, dseed, cond.name, ratio, "clean_R",
                    _cf(O, M, R_clean, gold, label, n_match, R_oracle))
                add(setting, dseed, cond.name, ratio, "disagreement_R",
                    _cf(O, M, R_dis, gold, label, n_match, R_oracle))
                add(setting, dseed, cond.name, ratio, "oracle_variant_R",
                    _cf(O, M, R_oracle, gold, label, n_match, R_oracle))

                sg_for_hybrid = []
                for size in CALIB_SIZES:
                    eff = min(size, N)
                    for sgseed in range(SMALL_GOLD_SEEDS):
                        rng = np.random.default_rng(1009 * (dseed + 2) + 131 * sgseed + size)
                        idx = rng.choice(N, size=eff, replace=False)
                        R_sg = small_gold_R(O, M, gold, idx)
                        add(setting, dseed, cond.name, ratio, "small_gold_R",
                            _cf(O, M, R_sg, gold, label, n_match, R_oracle), calibration_size=size)
                        if size == HYBRID_SG_SIZE:
                            sg_for_hybrid.append(R_sg)

                sg_base = np.mean(sg_for_hybrid, axis=0)
                for base_name, R_adaptive in (("disagreement", R_dis), ("small_gold", sg_base)):
                    for lam in LAMBDAS:
                        R_h = hybrid_R(R_clean, R_adaptive, lam)
                        add(setting, dseed, cond.name, ratio, "hybrid_R",
                            _cf(O, M, R_h, gold, label, n_match, R_oracle),
                            lam=lam, adaptive_base=base_name)

    df = pd.DataFrame(rows)
    orc = (df[df.r_mode == "oracle_variant_R"]
           .groupby(["setting", "direction_seed", "mechanism", "biased_ratio"])["false_retention_rate"].mean())
    df["oracle_gap"] = df.apply(
        lambda r: round(r["false_retention_rate"]
                        - orc.get((r["setting"], r["direction_seed"], r["mechanism"], r["biased_ratio"]), np.nan), 4),
        axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "direction_randomized_results.csv", index=False)

    _figures(df, fig_dir)
    _summary(df, out_dir / "direction_randomized_summary.md", N)
    print(f"direction-randomized experiment → {out_dir} ({len(df)} rows)")


def _mean_frr(df, setting, r_mode, ratio=1.0, **filt):
    sub = df[(df.setting == setting) & (df.r_mode == r_mode) & (df.biased_ratio == ratio)]
    for k, v in filt.items():
        sub = sub[sub[k] == v]
    return sub["false_retention_rate"].mean()


def _mean_gap(df, setting, r_mode, ratio=1.0):
    sub = df[(df.setting == setting) & (df.r_mode == r_mode) & (df.biased_ratio == ratio)]
    return sub["oracle_gap"].mean()


def _figures(df, fig_dir: Path) -> None:
    modes = ["naive_majority", "naive_supermajority_75", "clean_R",
             "disagreement_R", "small_gold_R", "hybrid_R", "oracle_variant_R"]

    # 1. fixed vs randomized FRR by mode (all_biased).
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(modes)); w = 0.38
    fixed = [_mean_frr(df, "fixed_direction", m) for m in modes]
    rand = [_mean_frr(df, "randomized_direction", m) for m in modes]
    ax.bar(x - w / 2, fixed, w, label="fixed_direction")
    ax.bar(x + w / 2, rand, w, label="randomized_direction")
    ax.set_xticks(x); ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean false retention rate"); ax.legend()
    ax.set_title("Fixed vs randomized direction: FRR by R mode (all_biased)")
    fig.tight_layout(); fig.savefig(fig_dir / "fixed_vs_randomized_false_retention.png", dpi=150); plt.close(fig)

    # 2. disagreement oracle gap, fixed vs randomized (+ frobenius/overlap annotation).
    fig, ax = plt.subplots(figsize=(7, 5))
    g_fixed = _mean_gap(df, "fixed_direction", "disagreement_R")
    g_rand = _mean_gap(df, "randomized_direction", "disagreement_R")
    fr = df[(df.r_mode == "disagreement_R") & (df.biased_ratio == 1.0)]
    frob_fix = fr[fr.setting == "fixed_direction"]["frobenius_to_oracle"].mean()
    frob_rand = fr[fr.setting == "randomized_direction"]["frobenius_to_oracle"].mean()
    ov_fix = fr[fr.setting == "fixed_direction"]["eig_overlap_oracle"].mean()
    ov_rand = fr[fr.setting == "randomized_direction"]["eig_overlap_oracle"].mean()
    bars = ax.bar(["fixed", "randomized"], [g_fixed, g_rand], color=["#4878a8", "#a84848"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("disagreement_R oracle gap (FRR − oracle)")
    ax.set_title("disagreement_R gap from oracle: fixed vs randomized")
    for b, frob, ov in zip(bars, [frob_fix, frob_rand], [ov_fix, ov_rand]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"‖ΔR‖={frob:.2f}\noverlap={ov:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "disagreement_oracle_gap.png", dpi=150); plt.close(fig)

    # 3. small_gold recovery curve, fixed vs randomized.
    fig, ax = plt.subplots(figsize=(8, 5))
    for setting, color in (("fixed_direction", "#4878a8"), ("randomized_direction", "#a84848")):
        sg = df[(df.r_mode == "small_gold_R") & (df.setting == setting) & (df.biased_ratio == 1.0)]
        g = sg.groupby("calibration_size")["false_retention_rate"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["calibration_size"], g["mean"], yerr=g["std"], marker="o", capsize=4,
                    color=color, label=f"small_gold ({setting})")
        orc = _mean_frr(df, setting, "oracle_variant_R")
        ax.axhline(orc, color=color, ls="--", alpha=0.7, label=f"oracle ({setting})")
        dis = _mean_frr(df, setting, "disagreement_R")
        ax.axhline(dis, color=color, ls=":", alpha=0.7, label=f"disagreement ({setting})")
    ax.set_xlabel("labeled calibration items"); ax.set_ylabel("false retention rate")
    ax.set_title("small_gold_R recovery vs calibration size")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(fig_dir / "small_gold_recovery_curve.png", dpi=150); plt.close(fig)

    # 4. hybrid lambda curve (disagreement base), fixed vs randomized.
    fig, ax = plt.subplots(figsize=(8, 5))
    for setting, color in (("fixed_direction", "#4878a8"), ("randomized_direction", "#a84848")):
        hy = df[(df.r_mode == "hybrid_R") & (df.setting == setting)
                & (df.adaptive_base == "disagreement") & (df.biased_ratio == 1.0)]
        s = hy.groupby("lam")["false_retention_rate"].mean().reset_index().sort_values("lam")
        ax.plot(s["lam"], s["false_retention_rate"], marker="o", color=color, label=setting)
    ax.set_xlabel("λ  (1.0 = clean R, 0.0 = adaptive)"); ax.set_ylabel("false retention rate")
    ax.set_title("Hybrid R (disagreement base): clean↔adaptive, fixed vs randomized")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(fig_dir / "hybrid_lambda_curve.png", dpi=150); plt.close(fig)


def _summary(df, path: Path, N: int) -> None:
    modes = ["naive_majority", "naive_supermajority_75", "clean_R",
             "disagreement_R", "small_gold_R", "hybrid_R", "oracle_variant_R"]
    lines = ["# Direction-Randomized Adaptive-R Validation\n"]
    lines.append(f"Analysis-only on the completed CFI run ({N} subset items, 7 mechanisms × 5 "
                 "biased fractions). Randomized setting averaged over direction seeds 0–4. "
                 "CorrFilter compared at supermajority-0.75 matched retention.\n")
    lines.append("> Randomization flips the presentation frame on ~50% of items (invert vote AND "
                 "gold). This keeps the per-item error 1[O≠gold]=1−v invariant — so oracle R and "
                 "the achievable filtering benefit are preserved — while scrambling the gold-free "
                 "observable `disagreement_R = corr(O)`. `oracle_variant_R` is an UPPER BOUND.\n")

    lines.append("## Mean FRR by R mode: fixed vs randomized (all_biased)\n")
    lines.append("| R mode | FRR fixed | FRR randomized | gap fixed | gap randomized |")
    lines.append("|---|---|---|---|---|")
    for m in modes:
        lines.append(f"| {m} | {_mean_frr(df,'fixed_direction',m):.4f} | "
                     f"{_mean_frr(df,'randomized_direction',m):.4f} | "
                     f"{_mean_gap(df,'fixed_direction',m):+.4f} | "
                     f"{_mean_gap(df,'randomized_direction',m):+.4f} |")
    lines.append("")

    fr = df[(df.r_mode == "disagreement_R") & (df.biased_ratio == 1.0)]
    frob = fr.groupby("setting")["frobenius_to_oracle"].mean()
    ov = fr.groupby("setting")["eig_overlap_oracle"].mean()
    lines.append("## R-quality of disagreement_R vs oracle\n")
    lines.append("| setting | ‖R_dis − R_oracle‖_F | top-3 eigenvector overlap |")
    lines.append("|---|---|---|")
    for s in ("fixed_direction", "randomized_direction"):
        lines.append(f"| {s} | {frob.get(s, float('nan')):.3f} | {ov.get(s, float('nan')):.3f} |")
    lines.append("")

    g_dis = _mean_gap(df, "randomized_direction", "disagreement_R")
    g_clean = _mean_gap(df, "randomized_direction", "clean_R")
    lines.append("## Scientific questions\n")
    lines.append(f"**Q1 — Does disagreement_R still match oracle after randomization?** "
                 f"Randomized oracle gap = {g_dis:+.4f} FRR (vs clean_R {g_clean:+.4f}). "
                 f"Pearson entries drift (Frobenius {frob.get('fixed_direction',0):.2f}→"
                 f"{frob.get('randomized_direction',0):.2f}) but the top-3 eigenvector overlap "
                 f"stays {ov.get('randomized_direction',0):.2f}: α_subset uses the quadratic form "
                 "dominated by the leading eigenstructure, which row-flips preserve.\n")

    sg = df[(df.r_mode == "small_gold_R") & (df.setting == "randomized_direction") & (df.biased_ratio == 1.0)]
    by_size = sg.groupby("calibration_size").agg(frr=("false_retention_rate", "mean"),
                                                 gap=("oracle_gap", "mean")).reset_index()
    lines.append("**Q2 — If not, how much small_gold calibration is needed (randomized)?**\n")
    lines.append("| calib size | mean FRR | gap from oracle |")
    lines.append("|---|---|---|")
    for _, r in by_size.iterrows():
        lines.append(f"| {int(r['calibration_size'])} | {r['frr']:.4f} | {r['gap']:+.4f} |")
    lines.append("")

    hy = df[(df.r_mode == "hybrid_R") & (df.setting == "randomized_direction")
            & (df.adaptive_base == "disagreement") & (df.biased_ratio == 1.0)]
    lam_tbl = hy.groupby("lam")["false_retention_rate"].mean()
    best_lam = lam_tbl.idxmin()
    lines.append(f"**Q3 — Does hybrid beat both clean_R and disagreement_R (randomized)?** "
                 f"clean_R {_mean_frr(df,'randomized_direction','clean_R'):.4f}, "
                 f"disagreement_R {_mean_frr(df,'randomized_direction','disagreement_R'):.4f}, "
                 f"best hybrid (disagreement base) FRR {lam_tbl.min():.4f} at λ={best_lam}. "
                 "See `figures/hybrid_lambda_curve.png`.\n")

    full = df[(df.setting == "randomized_direction") & (df.r_mode == "clean_R") & (df.biased_ratio == 1.0)]
    drift = full.groupby("mechanism")["frobenius_to_oracle"].mean().sort_values(ascending=False)
    dis_full = df[(df.setting == "randomized_direction") & (df.r_mode == "disagreement_R") & (df.biased_ratio == 1.0)]
    dis_gap_by_mech = dis_full.groupby("mechanism")["oracle_gap"].mean()
    lines.append("**Q4 — Which mechanisms degrade most under randomization?** "
                 "(clean_R Frobenius drift + disagreement_R residual gap, randomized all_biased)\n")
    lines.append("| mechanism | clean ‖ΔR‖_F | disagreement_R oracle gap |")
    lines.append("|---|---|---|")
    for mech in drift.index:
        lines.append(f"| {mech} | {drift[mech]:.3f} | {dis_gap_by_mech.get(mech, float('nan')):+.4f} |")
    lines.append("")

    # Verdict line for the paper claim.
    claim = ("gold-free Adaptive CorrFilter" if abs(g_dis) <= 0.01
             else "small-calibration Adaptive CorrFilter")
    lines.append(f"## Verdict\n\nRandomized disagreement_R oracle gap {g_dis:+.4f} FRR ⇒ the paper "
                 f"can plausibly claim **{claim}**. (Threshold: |gap| ≤ 0.01 FRR for a gold-free claim.)\n")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

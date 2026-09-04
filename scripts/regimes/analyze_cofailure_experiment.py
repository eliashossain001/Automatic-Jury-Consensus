"""Analyze the CFI experiment (Bucket 3, Part B).

Reads the cached biased votes (scripts/regimes/run_cofailure_votes_gpu.py) + clean H1 votes + H1 clean R, then
for every (mechanism, bank-ratio) variant computes:

  1. agreement-vs-accuracy curves      (triggered vs non-triggered items)
  2. false-retention curves            (consensus thresholds 0.50–0.90)
  3. biased-ratio curves               (all_clean → all_biased; majority vs CorrFilter)
  4. failure-mode heatmap              (judges × mechanisms conditional error rate)

Outputs (under outputs/cfi/):
  cfi_agreement_accuracy.csv, cfi_false_retention.csv, cfi_biased_ratio.csv,
  cfi_failure_heatmap.csv, cfi_report.md, figures/

Usage:
    python scripts/regimes/analyze_cofailure_experiment.py --config configs/cfi_bias_prompts.yaml \
        --project-root /home/elias/elias_projects/corrfilter
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

from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import (  # noqa: E402
    consensus_level,
    majority_consensus,
    supermajority_consensus,
)
from corrfilter.cfi.corrfilter_score import (  # noqa: E402
    corrfilter_score,
    retention_match_threshold,
)
from corrfilter.cfi.metrics import (  # noqa: E402
    accuracy_by_consensus_curve,
    conditional_error_rate,
    false_retention_rate,
    retention_rate,
)
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402

ABSTAIN = -1
FRR_THRESHOLDS = [0.50, 0.70, 0.75, 0.80, 0.90]
CONSENSUS_BINS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])


def _frr_and_acc(label, gold, keep):
    keep = keep.astype(bool) & (label != ABSTAIN)
    rr = retention_rate(keep)
    frr = false_retention_rate(keep, label, gold)
    return rr, frr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--out-dir", default="outputs/cfi")
    ap.add_argument("--items-file", default=None,
                    help="restrict analysis to these item_ids (same subset as the run)")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    cfg = yaml.safe_load(_p(args.config).read_text())
    bank = load_bias_bank(str(_p(args.config)))
    seed = bank.seed
    out_dir = _p(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cal_cfg = yaml.safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())
    items = load_calibration_set(_p(cal_cfg["output"]["manifest_path"]))
    if args.items_file:
        wanted = {ln.strip() for ln in _p(args.items_file).read_text().splitlines() if ln.strip()}
        items = [it for it in items if it.item_id in wanted]
        print(f"restricted analysis to {len(items)} items from {args.items_file}")
    gold = np.ones(len(items), dtype=np.int8)

    bank_cfg = load_bank_config(str(_p(cfg["sources"]["h1_bank_config"])))
    logical_ids = [s.logical_id for s in bank_cfg.specs]

    npz = np.load(_p(cfg["sources"]["h1_correlation_npz"]), allow_pickle=True)
    R_clean = npz["R"]
    if [str(x) for x in npz["logical_ids"]] != logical_ids:
        raise RuntimeError("npz judge order != bank order; R columns would misalign")

    clean_views = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)
    votes_root = _p(cfg["runtime"]["votes_dir"])
    bank_conditions = bank.bank_conditions
    biased = bank.biased_conditions

    agreement_rows: list[dict] = []
    frr_rows: list[dict] = []
    ratio_rows: list[dict] = []
    heatmap_rows: list[dict] = []

    available_mechs: list[str] = []

    for cond in biased:
        biased_views = load_vote_views(votes_root / cond.name, logical_ids)
        n_biased_votes = sum(len(v) for v in biased_views.values())
        if n_biased_votes == 0:
            print(f"[skip] no cached biased votes for mechanism '{cond.name}'")
            continue
        available_mechs.append(cond.name)

        for cond_name, ratio in bank_conditions.items():
            var = assemble_variant(items, logical_ids, cond, ratio, biased_views, clean_views, seed)
            V, M = var.V, var.M
            trig = var.triggered_flag.astype(bool)
            label = majority_consensus(V, M)
            level = consensus_level(V, M)

            # (1) agreement-vs-accuracy, triggered vs non-triggered
            for split_name, mask in (("all", np.ones(len(items), bool)),
                                     ("triggered", trig), ("non_triggered", ~trig)):
                if mask.sum() == 0:
                    continue
                curve = accuracy_by_consensus_curve(label[mask], gold[mask], level[mask], CONSENSUS_BINS)
                for b in range(len(curve["accuracy"])):
                    if curve["count"][b] == 0:
                        continue
                    agreement_rows.append({
                        "mechanism": cond.name, "bank_condition": cond_name, "biased_ratio": ratio,
                        "split": split_name, "consensus_lo": round(float(curve["bin_lo"][b]), 2),
                        "consensus_hi": round(float(min(curve["bin_hi"][b], 1.0)), 2),
                        "accuracy": round(float(curve["accuracy"][b]), 4),
                        "count": int(curve["count"][b]),
                    })

            # (2) false-retention curves across consensus thresholds
            for tau in FRR_THRESHOLDS:
                pred = supermajority_consensus(V, M, threshold=tau)
                rr, frr = _frr_and_acc(pred, gold, pred != ABSTAIN)
                frr_rows.append({
                    "mechanism": cond.name, "bank_condition": cond_name, "biased_ratio": ratio,
                    "threshold": tau, "retention_rate": round(rr, 4),
                    "false_retention_rate": round(frr, 4),
                })

            # (3) biased-ratio curve. Naive majority keeps every non-abstaining
            # item (retention ≈ 1.0), so "CorrFilter at majority's retention" is
            # degenerate (both keep everything). The informative head-to-head is
            # at a retention < 1.0: we take supermajority-0.75's natural retention
            # as the operating point and compare which *ranking* — top-consensus
            # (supermajority) vs top-α_subset (CorrFilter) — drops more wrong
            # items at that same retention.
            maj_keep = (M.astype(bool).any(axis=1)) & (label != ABSTAIN)
            rr_maj, frr_maj = _frr_and_acc(label, gold, maj_keep)
            super75 = supermajority_consensus(V, M, threshold=0.75)
            super_keep = super75 != ABSTAIN
            rr_super, frr_super = _frr_and_acc(label, gold, super_keep)
            n_match = int(super_keep.sum())
            cf = corrfilter_score(V, M, R_clean, label)
            cf_keep, _ = retention_match_threshold(cf.score, n_match)
            _, frr_cf = _frr_and_acc(label, gold, cf_keep)
            valid_alpha = cf.score[~np.isnan(cf.score)]
            ratio_rows.append({
                "mechanism": cond.name, "bank_condition": cond_name, "biased_ratio": ratio,
                "retention_majority": round(rr_maj, 4),
                "false_retention_majority": round(frr_maj, 4),
                "matched_retention": round(rr_super, 4),
                "false_retention_supermajority": round(frr_super, 4),
                "false_retention_corrfilter": round(frr_cf, 4),
                "frr_reduction_vs_supermajority": round(frr_super - frr_cf, 4),
                "mean_alpha_subset": round(float(valid_alpha.mean()), 4) if valid_alpha.size else float("nan"),
            })

        # (4) failure heatmap: conditional error on triggered items, all_biased variant
        var_full = assemble_variant(items, logical_ids, cond, 1.0, biased_views, clean_views, seed)
        trig_full = var_full.triggered_flag.astype(bool)
        cond_err = conditional_error_rate(var_full.V, var_full.M, gold, trig_full)
        for i, lid in enumerate(logical_ids):
            heatmap_rows.append({
                "judge": lid, "mechanism": cond.name,
                "conditional_error_rate": round(float(cond_err[i]), 4),
                "n_triggered": int(trig_full.sum()),
            })

    # --- Persist CSVs ------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    df_agree = pd.DataFrame(agreement_rows)
    df_frr = pd.DataFrame(frr_rows)
    df_ratio = pd.DataFrame(ratio_rows)
    df_heat = pd.DataFrame(heatmap_rows)
    df_agree.to_csv(out_dir / "cfi_agreement_accuracy.csv", index=False)
    df_frr.to_csv(out_dir / "cfi_false_retention.csv", index=False)
    df_ratio.to_csv(out_dir / "cfi_biased_ratio.csv", index=False)
    df_heat.to_csv(out_dir / "cfi_failure_heatmap.csv", index=False)

    if not available_mechs:
        msg = ("No cached biased votes found under "
               f"{votes_root}. Run scripts/regimes/run_cofailure_votes_gpu.py first.")
        (out_dir / "cfi_report.md").write_text(f"# CFI analysis\n\n{msg}\n")
        print(msg)
        return

    # --- Figures -----------------------------------------------------------
    _fig_biased_ratio(df_ratio, fig_dir)
    _fig_false_retention(df_frr, available_mechs, fig_dir)
    _fig_failure_heatmap(df_heat, logical_ids, available_mechs, fig_dir)
    _fig_agreement_accuracy(df_agree, available_mechs, fig_dir)

    _write_report(out_dir / "cfi_report.md", df_ratio, df_frr, df_heat,
                  available_mechs, bank_conditions, len(items))
    print(f"analyzed {len(available_mechs)} mechanisms → {out_dir}")
    for f in sorted(out_dir.glob("*.csv")):
        print("  ", f.name)


def _fig_biased_ratio(df_ratio: pd.DataFrame, fig_dir: Path) -> None:
    if df_ratio.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech, g in df_ratio.groupby("mechanism"):
        g = g.sort_values("biased_ratio")
        ax.plot(g["biased_ratio"], g["false_retention_majority"], "-o", label=f"{mech} (maj)")
    ax.set_xlabel("biased ratio")
    ax.set_ylabel("false retention rate (majority)")
    ax.set_title("False retention vs biased ratio, by mechanism")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "cfi_false_retention_vs_ratio.png", dpi=150)
    plt.close(fig)

    # supermajority vs corrfilter at matched retention, all_biased
    full = df_ratio[df_ratio["biased_ratio"] == 1.0].sort_values("mechanism")
    if not full.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(full))
        ax.bar(x - 0.27, full["false_retention_majority"], 0.27, label="majority (ret≈1.0)")
        ax.bar(x, full["false_retention_supermajority"], 0.27, label="supermajority-0.75")
        ax.bar(x + 0.27, full["false_retention_corrfilter"], 0.27, label="CorrFilter (matched ret)")
        ax.set_xticks(x)
        ax.set_xticklabels(full["mechanism"], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("false retention rate")
        ax.set_title("False retention at matched retention (all_biased)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "cfi_majority_vs_corrfilter.png", dpi=150)
        plt.close(fig)


def _fig_false_retention(df_frr: pd.DataFrame, mechs: list[str], fig_dir: Path) -> None:
    full = df_frr[df_frr["biased_ratio"] == 1.0]
    if full.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech, g in full.groupby("mechanism"):
        g = g.sort_values("threshold")
        ax.plot(g["threshold"], g["false_retention_rate"], "-o", label=mech)
    ax.set_xlabel("consensus threshold k/n")
    ax.set_ylabel("false retention rate")
    ax.set_title("False-retention curves (all_biased)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "cfi_false_retention_curves.png", dpi=150)
    plt.close(fig)


def _fig_failure_heatmap(df_heat: pd.DataFrame, judges: list[str], mechs: list[str], fig_dir: Path) -> None:
    if df_heat.empty:
        return
    pivot = df_heat.pivot(index="judge", columns="mechanism", values="conditional_error_rate")
    pivot = pivot.reindex(index=judges, columns=mechs)
    fig, ax = plt.subplots(figsize=(1.2 * len(mechs) + 3, 0.5 * len(judges) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(mechs)))
    ax.set_xticklabels(mechs, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges, fontsize=7)
    for i in range(len(judges)):
        for j in range(len(mechs)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=6)
    ax.set_title("Conditional error rate on triggered items (judge × mechanism)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "cfi_failure_heatmap.png", dpi=150)
    plt.close(fig)


def _fig_agreement_accuracy(df_agree: pd.DataFrame, mechs: list[str], fig_dir: Path) -> None:
    full = df_agree[(df_agree["biased_ratio"] == 1.0) & (df_agree["split"] != "all")]
    if full.empty:
        return
    n = len(mechs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for idx, mech in enumerate(mechs):
        ax = axes[idx // ncol][idx % ncol]
        g = full[full["mechanism"] == mech]
        for split, gg in g.groupby("split"):
            gg = gg.sort_values("consensus_lo")
            ax.plot(gg["consensus_lo"], gg["accuracy"], "-o", label=split, markersize=4)
        ax.set_title(mech, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("consensus", fontsize=7)
        ax.set_ylabel("accuracy", fontsize=7)
        ax.legend(fontsize=6)
    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    fig.suptitle("Agreement vs accuracy (all_biased): triggered vs non-triggered")
    fig.tight_layout()
    fig.savefig(fig_dir / "cfi_agreement_accuracy.png", dpi=150)
    plt.close(fig)


def _write_report(path, df_ratio, df_frr, df_heat, mechs, bank_conditions, n_items) -> None:
    lines = ["# CFI Experiment Report (Bucket 3, Part B)\n"]
    lines.append(f"Mechanisms with cached biased votes: {', '.join(mechs)}.")
    lines.append(f"Calibration items: {n_items}. Bank conditions: "
                 f"{', '.join(f'{k}={v}' for k, v in bank_conditions.items())}.\n")

    lines.append("## Headline: supermajority vs CorrFilter at matched retention (all_biased)\n")
    full = df_ratio[df_ratio["biased_ratio"] == 1.0].sort_values("mechanism")
    lines.append("| mechanism | FRR majority (ret≈1.0) | matched ret | FRR supermajority | "
                 "FRR CorrFilter | Δ (super−CF) | mean α_subset |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in full.iterrows():
        lines.append(f"| {r['mechanism']} | {r['false_retention_majority']:.3f} | "
                     f"{r['matched_retention']:.2f} | {r['false_retention_supermajority']:.3f} | "
                     f"{r['false_retention_corrfilter']:.3f} | "
                     f"{r['frr_reduction_vs_supermajority']:+.3f} | {r['mean_alpha_subset']:.3f} |")
    lines.append("")
    lines.append("_At the supermajority-0.75 retention, both supermajority (keep top-consensus) "
                 "and CorrFilter (keep top-α_subset) drop the same number of items; positive Δ "
                 "means CorrFilter's ranking retains fewer wrong items. Naive majority keeps "
                 "everything (retention≈1.0) and is the unfiltered baseline._\n")

    lines.append("## False retention vs biased ratio (majority consensus)\n")
    pivot = df_ratio.pivot_table(index="mechanism", columns="biased_ratio",
                                 values="false_retention_majority")
    lines.append("| mechanism | " + " | ".join(f"ratio={c}" for c in pivot.columns) + " |")
    lines.append("|" + "---|" * (len(pivot.columns) + 1))
    for mech, row in pivot.iterrows():
        lines.append(f"| {mech} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
    lines.append("")

    lines.append("## Failure-mode heatmap summary (mean conditional error on triggered items)\n")
    if not df_heat.empty:
        mech_mean = df_heat.groupby("mechanism")["conditional_error_rate"].mean().sort_values(ascending=False)
        lines.append("| mechanism | mean judge error on triggered items |")
        lines.append("|---|---|")
        for mech, v in mech_mean.items():
            lines.append(f"| {mech} | {v:.3f} |")
        lines.append("")

    lines.append("## Artefacts\n")
    lines.append("- `cfi_agreement_accuracy.csv`, `cfi_false_retention.csv`, "
                 "`cfi_biased_ratio.csv`, `cfi_failure_heatmap.csv`")
    lines.append("- `figures/cfi_false_retention_vs_ratio.png`, "
                 "`figures/cfi_majority_vs_corrfilter.png`, "
                 "`figures/cfi_false_retention_curves.png`, "
                 "`figures/cfi_failure_heatmap.png`, "
                 "`figures/cfi_agreement_accuracy.png`\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

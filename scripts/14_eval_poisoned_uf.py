"""Evaluate filtering on Synthetic-Poisoned UltraFeedback (H2b, analysis-only).

Loads the poisoned manifest + cached judge votes, then for each corruption rate
(5/10/20%) compares filtering methods at matched retention. Judge inference is
content-based: vote V[i,j] = 1 iff judge picked the truly-better response. For a
given corruption rate the observable "affirm the stated label" vote is
  O = V on clean items,  O = 1 - V on poisoned items,
and the label-reliability gold is g = 1 (clean) / 0 (poisoned). A filter keeps
items it judges clean; we score how many kept items are actually clean.

Methods: no_filter, random_same_retention, naive_majority, naive_supermajority_75,
and CorrFilter under clean_R / disagreement_R / small_gold_R / hybrid_R, plus an
oracle_filter (labels known) and oracle_variant_R CorrFilter (upper bounds).

Reuses corrfilter.cfi.adaptive_r (R estimators + diagnostics),
corrfilter.cfi.corrfilter_score, and corrfilter.cfi.consensus.

Outputs: outputs/synthetic_poisoned_ultrafeedback/{filtering_results.csv,
calibration_size_results.csv, method_comparison.csv, summary.md, figures/}.

Usage:
    python scripts/14_eval_poisoned_uf.py --project-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corrfilter.cfi.adaptive_r import (  # noqa: E402
    disagreement_R, eigenvector_overlap, frobenius_distance, hybrid_R,
    oracle_variant_R, small_gold_R,
)
from corrfilter.cfi.consensus import (  # noqa: E402
    consensus_level, majority_consensus, supermajority_consensus, vote_fraction,
)
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

ABSTAIN = -1
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
CALIB_SIZES = [25, 50, 100, 200]
N_SEEDS = 10
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HYBRID_SG_SIZE = 100


def _vote_matrix(cache: VoteCache, logical_ids, item_ids):
    """V[i,j] = 1 iff judge j picked the truly-better response (vote==1)."""
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), dtype=np.int8)
    M = np.zeros((n, m), dtype=np.int8)
    idx = {it: i for i, it in enumerate(item_ids)}
    for j, lid in enumerate(logical_ids):
        dfj = cache.load(lid)
        if dfj is None:
            continue
        for r in dfj.itertuples(index=False):
            i = idx.get(str(getattr(r, "item_id")))
            if i is None:
                continue
            v = int(getattr(r, "vote"))
            if v == ABSTAIN:
                continue
            V[i, j] = v
            M[i, j] = 1
    return V, M


def _metrics(keep, gold, label):
    """Poisoning-filter metrics. gold: 1=clean, 0=poisoned. label: bank verdict."""
    keep = keep.astype(bool)
    clean = gold == 1
    poisoned = gold == 0
    n = len(gold)
    n_kept = int(keep.sum())
    kept_clean = int((keep & clean).sum())
    kept_pois = int((keep & poisoned).sum())
    precision = kept_clean / n_kept if n_kept else 0.0
    recall = kept_clean / int(clean.sum()) if clean.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    removal = (int(poisoned.sum()) - kept_pois) / int(poisoned.sum()) if poisoned.sum() else 0.0
    return {
        "retention_rate": round(n_kept / n, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_retention_rate": round(1 - precision if n_kept else 0.0, 4),
        "corrupted_removal_rate": round(removal, 4),
        "n_kept": n_kept,
    }


def _keep_topk_affirmed(score, label, n_keep):
    """Keep top-n_keep items among those the bank affirms clean (label==1)."""
    affirm = label == 1
    s = np.where(affirm, score, -np.inf)
    keep, _ = retention_match_threshold(s, n_keep)
    return keep & affirm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--manifest", default="outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    ap.add_argument("--votes-dir", default="outputs/synthetic_poisoned_ultrafeedback/judge_votes")
    ap.add_argument("--npz", default="experiments/h1_measurement/results/correlation.npz")
    ap.add_argument("--out-dir", default="outputs/synthetic_poisoned_ultrafeedback")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    out_dir = _p(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(_p(args.manifest))
    item_ids = man["item_id"].astype(str).tolist()
    bank_cfg = load_bank_config(str(_p(args.bank)))
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    npz = np.load(_p(args.npz), allow_pickle=True)
    R_clean = npz["R"]
    if [str(x) for x in npz["logical_ids"]] != logical_ids:
        raise RuntimeError("npz judge order != bank order")

    cache = VoteCache(_p(args.votes_dir))
    V, M = _vote_matrix(cache, logical_ids, item_ids)
    coverage = M.mean()
    print(f"vote matrix {V.shape}, coverage {coverage:.1%}")
    if coverage < 0.5:
        print("WARNING: <50% vote coverage; run scripts/13 to completion first.")

    rng_global = np.random.default_rng(20260601)
    rows = []
    calib_rows = []

    for tag, rate in RATES.items():
        poisoned = man[f"poisoned_{tag}"].to_numpy(dtype=bool)
        gold = np.where(poisoned, 0, 1).astype(np.int8)         # 1 clean, 0 poisoned
        O = V.copy()
        O[poisoned] = 1 - V[poisoned]                            # affirm-stated-label vote
        label = majority_consensus(O, M)                         # bank verdict (1 affirm)
        n_match = int((supermajority_consensus(O, M, 0.75) != ABSTAIN).sum())
        n_match = max(n_match, 1)

        R_oracle = oracle_variant_R(O, M, gold)
        R_dis = disagreement_R(O, M)

        def cf_keep(R):
            sc = corrfilter_score(O, M, R, label)
            return _keep_topk_affirmed(np.nan_to_num(sc.score, nan=-np.inf), label, n_match), sc

        def add(method, keep, **extra):
            m = _metrics(keep, gold, label)
            m.update({"corruption_rate": rate, "method": method, **extra})
            rows.append(m)

        # Baselines.
        add("no_filter", np.ones(len(gold), bool))
        rand = rng_global.permutation(len(gold))[:n_match]
        rmask = np.zeros(len(gold), bool); rmask[rand] = True
        add("random_same_retention", rmask)
        # naive majority: keep affirmed items, ranked by affirm margin to n_match.
        margin = np.abs(vote_fraction(O, M) - 0.5)
        add("naive_majority", _keep_topk_affirmed(margin, label, n_match))
        add("naive_supermajority_75", _keep_topk_affirmed(consensus_level(O, M), label, n_match))
        # oracle filter (labels known): keep clean items up to n_match.
        oracle_score = np.where(gold == 1, 1.0, -np.inf)
        add("oracle_filter", _keep_topk_affirmed(oracle_score + 1.0, np.ones_like(label), n_match))

        # CorrFilter R modes.
        k_clean, _ = cf_keep(R_clean); add("corrfilter_clean_R", k_clean)
        k_dis, _ = cf_keep(R_dis); add("corrfilter_disagreement_R", k_dis)
        k_orc, _ = cf_keep(R_oracle); add("corrfilter_oracle_variant_R", k_orc)

        # small_gold over sizes x seeds.
        sg_for_hybrid = []
        for size in CALIB_SIZES:
            eff = min(size, len(gold))
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(7919 * (int(rate * 100) + 1) + 131 * seed + size)
                idx = rng.choice(len(gold), size=eff, replace=False)
                R_sg = small_gold_R(O, M, gold, idx)
                ksg, _ = cf_keep(R_sg)
                m = _metrics(ksg, gold, label)
                m.update({"corruption_rate": rate, "method": "corrfilter_small_gold_R",
                          "calibration_size": size, "seed": seed,
                          "frobenius_to_oracle": round(frobenius_distance(R_sg, R_oracle), 4)})
                calib_rows.append(m)
                if size == HYBRID_SG_SIZE:
                    sg_for_hybrid.append(R_sg)
        # representative small_gold (mean over seeds at size 100) for the headline table.
        add("corrfilter_small_gold_R", cf_keep(np.mean(sg_for_hybrid, axis=0))[0],
            calibration_size=HYBRID_SG_SIZE)

        # hybrid sweep (disagreement base).
        for lam in LAMBDAS:
            kh, _ = cf_keep(hybrid_R(R_clean, R_dis, lam))
            add("corrfilter_hybrid_R", kh, lam=lam)

        print(f"rate {rate:.0%}: n_poison={int(poisoned.sum())}, matched retention={n_match}/{len(gold)}, "
              f"R_dis eig-overlap w/ oracle={eigenvector_overlap(R_dis, R_oracle, 3):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "filtering_results.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out_dir / "calibration_size_results.csv", index=False)

    # method_comparison: headline point methods at each rate (best/representative).
    point = df[~df.method.isin(["corrfilter_hybrid_R"]) | (df.get("lam") == 0.5)]
    method_cmp = point.pivot_table(index="method", columns="corruption_rate",
                                   values="precision", aggfunc="mean")
    method_cmp.to_csv(out_dir / "method_comparison.csv")

    _figures(df, pd.DataFrame(calib_rows), fig_dir)
    _summary(df, pd.DataFrame(calib_rows), out_dir / "summary.md", len(item_ids), coverage)
    print(f"wrote results to {out_dir}")


def _figures(df, calib, fig_dir):
    point_methods = ["no_filter", "random_same_retention", "naive_majority",
                     "naive_supermajority_75", "corrfilter_clean_R",
                     "corrfilter_disagreement_R", "corrfilter_small_gold_R",
                     "corrfilter_oracle_variant_R", "oracle_filter"]
    r20 = df[(df.corruption_rate == 0.20) & (df.method.isin(point_methods))].set_index("method").reindex(point_methods)

    # precision / recall / f1 grouped bars at 20%.
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(point_methods)); w = 0.27
    ax.bar(x - w, r20["precision"], w, label="precision")
    ax.bar(x, r20["recall"], w, label="recall")
    ax.bar(x + w, r20["f1"], w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(point_methods, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("score"); ax.set_title("Filtering precision/recall/F1 (20% corruption, matched retention)")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "precision_recall_f1.png", dpi=150); plt.close(fig)

    # false retention by method (20%).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, r20["false_retention_rate"], color="#a84848")
    ax.set_xticks(x); ax.set_xticklabels(point_methods, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("false retention rate"); ax.set_title("False retention by method (20% corruption)")
    fig.tight_layout(); fig.savefig(fig_dir / "false_retention_by_method.png", dpi=150); plt.close(fig)

    # calibration-size curve (precision vs size, per rate).
    fig, ax = plt.subplots(figsize=(8, 5))
    for rate, g in calib.groupby("corruption_rate"):
        s = g.groupby("calibration_size")["precision"].agg(["mean", "std"]).reset_index()
        ax.errorbar(s["calibration_size"], s["mean"], yerr=s["std"], marker="o", capsize=4,
                    label=f"{rate:.0%} corruption")
    ax.set_xlabel("labeled calibration items"); ax.set_ylabel("precision")
    ax.set_title("small_gold Adaptive CorrFilter: precision vs calibration size")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "calibration_size_curve.png", dpi=150); plt.close(fig)

    # corruption-rate curve (precision vs rate, key methods).
    key = ["naive_supermajority_75", "corrfilter_clean_R", "corrfilter_small_gold_R", "corrfilter_oracle_variant_R"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in key:
        g = df[(df.method == method)].groupby("corruption_rate")["precision"].mean().reset_index()
        ax.plot(g["corruption_rate"], g["precision"], marker="o", label=method)
    ax.set_xlabel("corruption rate"); ax.set_ylabel("precision")
    ax.set_title("Precision vs corruption rate"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "corruption_rate_curve.png", dpi=150); plt.close(fig)

    # adaptive vs static corrfilter (precision, all rates).
    fig, ax = plt.subplots(figsize=(8, 5))
    comp = ["corrfilter_clean_R", "corrfilter_disagreement_R", "corrfilter_small_gold_R", "corrfilter_oracle_variant_R"]
    rates = sorted(df.corruption_rate.unique()); xr = np.arange(len(rates)); w = 0.2
    for i, method in enumerate(comp):
        vals = [df[(df.method == method) & (df.corruption_rate == r)]["precision"].mean() for r in rates]
        ax.bar(xr + (i - 1.5) * w, vals, w, label=method.replace("corrfilter_", ""))
    ax.set_xticks(xr); ax.set_xticklabels([f"{r:.0%}" for r in rates])
    ax.set_xlabel("corruption rate"); ax.set_ylabel("precision")
    ax.set_title("Adaptive vs static CorrFilter precision"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "adaptive_vs_static_corrfilter.png", dpi=150); plt.close(fig)


def _summary(df, calib, path, n_items, coverage):
    lines = ["# Synthetic-Poisoned UltraFeedback (H2b)\n"]
    lines.append(f"{n_items} preference pairs, vote coverage {coverage:.1%}. Corruption is "
                 "structured (heuristic-aligned: verbosity / length / polite / style). Methods "
                 "compared at the supermajority-0.75 matched retention; CorrFilter keeps the "
                 "top-$\\alpha$ items among those the bank affirms clean.\n")

    lines.append("## Precision by method and corruption rate\n")
    methods = ["no_filter", "random_same_retention", "naive_majority", "naive_supermajority_75",
               "corrfilter_clean_R", "corrfilter_disagreement_R", "corrfilter_small_gold_R",
               "corrfilter_hybrid_R", "corrfilter_oracle_variant_R", "oracle_filter"]
    rates = sorted(df.corruption_rate.unique())
    lines.append("| method | " + " | ".join(f"{r:.0%}" for r in rates) + " |")
    lines.append("|" + "---|" * (len(rates) + 1))
    for m in methods:
        sub = df[df.method == m]
        if m == "corrfilter_hybrid_R":
            sub = sub[sub.get("lam") == 0.5]
        cells = []
        for r in rates:
            v = sub[sub.corruption_rate == r]["precision"]
            cells.append(f"{v.mean():.3f}" if len(v) else "--")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")

    # Headline question: adaptive vs naive k-of-n improvement.
    lines.append("## Main question: does Small-Calibrated Adaptive CorrFilter beat naive k-of-n by >= 5 points?\n")
    lines.append("| corruption | naive_supermajority_75 | small_gold Adaptive | improvement (pts) | oracle |")
    lines.append("|---|---|---|---|---|")
    verdicts = []
    for r in rates:
        naive = df[(df.method == "naive_supermajority_75") & (df.corruption_rate == r)]["precision"].mean()
        adapt = df[(df.method == "corrfilter_small_gold_R") & (df.corruption_rate == r)]["precision"].mean()
        orc = df[(df.method == "corrfilter_oracle_variant_R") & (df.corruption_rate == r)]["precision"].mean()
        gain = 100 * (adapt - naive)
        verdicts.append(gain)
        lines.append(f"| {r:.0%} | {naive:.3f} | {adapt:.3f} | {gain:+.1f} | {orc:.3f} |")
    lines.append("")
    max_gain = max(verdicts) if verdicts else 0.0
    verdict = "CONFIRMED" if max_gain >= 5.0 else "NOT CONFIRMED"
    lines.append(f"**H2b ({'>= 5 pt precision gain over naive k-of-n at matched retention'}): {verdict}** "
                 f"(best improvement {max_gain:+.1f} points).\n")

    # calibration-size sensitivity.
    lines.append("## Calibration-size sensitivity (precision, mean over seeds)\n")
    cs = calib.groupby(["corruption_rate", "calibration_size"])["precision"].mean().unstack()
    lines.append("| corruption | " + " | ".join(f"{int(c)} labels" for c in cs.columns) + " |")
    lines.append("|" + "---|" * (len(cs.columns) + 1))
    for r, row in cs.iterrows():
        lines.append(f"| {r:.0%} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
    lines.append("")
    lines.append("## Artefacts\n- `filtering_results.csv`, `calibration_size_results.csv`, "
                 "`method_comparison.csv`\n- `figures/`: precision_recall_f1, false_retention_by_method, "
                 "calibration_size_curve, corruption_rate_curve, adaptive_vs_static_corrfilter\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

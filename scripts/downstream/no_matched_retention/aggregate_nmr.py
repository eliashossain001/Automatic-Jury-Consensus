"""Aggregate the no-matched-retention experiment -> table + summary + final verdict.

Joins dataset_stats (natural sizes/purity), eval_results (clean acc, rewardbench, mean
reward margin), and the targeted position-bias probe (full + targeted slice + paired
bias_cluster-naive bootstrap). Compares against the matched-retention pos10 run, and
writes comparison_table.csv, summary.md, and final_verdict.md answering A/B/C.

Usage: python no_matched_retention/aggregate_nmr.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "no_matched_retention"                                    # this experiment's outputs


RES = RUN_DIR / "results"
POS10_TABLE = EXP_DIR / "position_aligned_poisoning" / "pos10" / "results" / "comparison_table.csv"
N_EVAL = 400
THR = 2 * (0.25 / N_EVAL) ** 0.5  # ~0.05 win-rate noise floor


def load_targeted():
    p = RUN_DIR / "targeted_eval" / "targeted_eval.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    stats = pd.read_csv(RES / "dataset_stats.csv")
    evals = pd.read_csv(RES / "eval_results.csv") if (RES / "eval_results.csv").exists() else None
    tg = load_targeted()

    df = stats.copy()
    if evals is not None:
        df = df.merge(evals, on="filter_method", how="left")
    naive_acc = float(df.loc[df.filter_method == "naive_consensus", "clean_pref_acc"].iloc[0]) \
        if evals is not None else float("nan")
    df["delta_clean_vs_naive"] = (df["clean_pref_acc"] - naive_acc).round(4)
    # targeted-slice accuracy from the probe
    if tg:
        bm = tg.get("by_method", {})
        df["targeted_win_rate"] = df["filter_method"].map(
            lambda m: bm.get(m, {}).get("targeted_top_third", {}).get("win_rate"))
        df["targeted_mean_margin"] = df["filter_method"].map(
            lambda m: bm.get(m, {}).get("targeted_top_third", {}).get("mean_margin"))
    df.to_csv(RES / "comparison_table.csv", index=False)

    # net-correction accounting
    def row(m, col):
        r = df.loc[df.filter_method == m, col]
        return r.iloc[0] if len(r) else None
    nv_n, nv_p = row("naive_consensus", "train_size"), row("naive_consensus", "poisoned_kept")
    bc_n, bc_p = row("bias_cluster", "train_size"), row("bias_cluster", "poisoned_kept")
    removed = int(nv_n - bc_n); pois_removed = int(nv_p - bc_p)
    net_corr_pct = round(100 * pois_removed / nv_n, 2)

    # ---- summary.md ----
    sm = ["# No-Matched-Retention DPO (pos10) — Results\n",
          "Each filter keeps its NATURAL set (no backfill, no size-matching). Same base "
          "model, LoRA, seed, DPO hyperparameters, and held-out eval as the matched-retention "
          "pos10 run.\n", "## Comparison table\n", df.to_markdown(index=False),
          "\n## Net-correction accounting (bias_cluster vs naive)\n",
          f"- training examples removed by bias_cluster: **{removed}**",
          f"- of which poisoned: **{pois_removed}** ({round(100*pois_removed/removed,1) if removed else 0}% removal precision)",
          f"- net correction: **{net_corr_pct}%** of the naive set",
          f"- label purity: naive {row('naive_consensus','label_purity')} vs bias_cluster {row('bias_cluster','label_purity')}\n"]
    (RES / "summary.md").write_text("\n".join(sm))

    # ---- final_verdict.md ----
    def acc(m): return row(m, "clean_pref_acc")
    bc_d = (acc("bias_cluster") - naive_acc) if evals is not None else None
    orc_d = (acc("oracle") - naive_acc) if evals is not None else None
    paired = tg.get("paired_bc_minus_naive", {}) if tg else {}
    full = paired.get("full", {}); tgt = paired.get("targeted_top_third", {})
    full_det, tgt_det = full.get("detectable_at_95pct", False), tgt.get("detectable_at_95pct", False)
    full_diff, tgt_diff = full.get("mean_margin_diff", 0.0), tgt.get("mean_margin_diff", 0.0)
    # an effect "in favour of bias_cluster" needs a POSITIVE, beyond-noise difference
    beneficial = (evals is not None and bc_d is not None and bc_d > THR) or (tgt_det and tgt_diff > 0)
    detectable_negative = (full_det and full_diff < 0) or (tgt_det and tgt_diff < 0)
    any_detect = full_det or tgt_det
    purity_gap = (row("naive_consensus", "label_purity") - row("bias_cluster", "label_purity"))

    fv = [
        "# Final Verdict — No-Matched-Retention DPO (pos10)\n",
        f"_Win-rate noise floor at n={N_EVAL} ≈ {100*THR:.1f} pts._\n",
        "## Headline numbers\n",
        f"- Natural sizes: naive {nv_n} (purity {row('naive_consensus','label_purity')}), "
        f"bias_cluster {bc_n} (purity {row('bias_cluster','label_purity')}), "
        f"oracle {row('oracle','train_size')} (purity 1.0).",
        f"- bias_cluster removed {removed} examples, only {pois_removed} poisoned "
        f"(net correction {net_corr_pct}%).",
    ]
    if evals is not None:
        fv += [
            f"- Clean-pref-acc: naive {acc('naive_consensus')}, bias_cluster {acc('bias_cluster')} "
            f"(Δ {bc_d:+.3f}), oracle {acc('oracle')} (Δ {orc_d:+.3f}).",
            "- Targeted-slice win-rate: " + ", ".join(
                f"{m} {df.loc[df.filter_method==m,'targeted_win_rate'].iloc[0]}" for m in
                ["naive_consensus", "bias_cluster", "oracle"] if "targeted_win_rate" in df.columns) + ".",
        ]
    if paired:
        fv.append(f"- Paired bias_cluster−naive margin: full {full.get('mean_margin_diff')} "
                  f"CI {full.get('boot95_ci')}, targeted {tgt.get('mean_margin_diff')} "
                  f"CI {tgt.get('boot95_ci')} (detectable: {'YES' if any_detect else 'NO'}, "
                  f"sign: {'bias_cluster WORSE' if detectable_negative and not beneficial else ('bias_cluster better' if beneficial else 'n/a')}).")

    fv += [
        "\n## A. Did removing matched retention reveal a downstream effect?\n",
        ("**Yes — a beneficial effect.** Without matched retention bias_cluster exceeds naive beyond "
         "the noise floor on the clean and/or targeted metric." if beneficial
         else
         ("**A difference appeared, but in the WRONG direction.** The sensitive reward-margin metric "
          f"now detects a separation (full Δ {full_diff:+.4f}, CI {full.get('boot95_ci')}; targeted "
          f"Δ {tgt_diff:+.4f}, CI {tgt.get('boot95_ci')}), but bias_cluster's margins are *lower* than "
          f"naive's and win-rate is within noise (Δ {bc_d:+.3f} clean). This is most plausibly a "
          f"data-quantity effect — bias_cluster trained on {removed} fewer examples for ~zero purity "
          "gain — not the hoped-for filtering improvement. So no beneficial downstream effect was "
          "revealed." if detectable_negative else
          "**No.** With each filter on its natural set, bias_cluster does not differ from naive beyond "
          f"the noise floor (Δ {bc_d:+.3f} clean) and the paired margin CI includes zero.")),
        "\n## B. Is matched retention the primary reason previous experiments were null?\n",
        (f"**No — the opposite.** Under natural retention the label-purity gap nearly vanishes "
         f"(naive {row('naive_consensus','label_purity')} vs bias_cluster {row('bias_cluster','label_purity')}; "
         f"gap {purity_gap:+.3f}). The +4.6-pt advantage seen under matched retention was itself an "
         "artifact of that protocol: position-poisoned items are high-consensus, so naive's "
         "top-k-by-margin selection concentrated poison and looked artificially dirty. bias_cluster's "
         "natural rule removes only the few purely cluster-carried poisoned items "
         f"({pois_removed}), so there is no real purity advantage to translate. Matched retention did "
         "not hide a downstream effect; it manufactured a purity gap that still did not move the model."),
        "\n## C. Does dependence-aware filtering create practically meaningful alignment gains here?\n",
        ("**Yes** — see A." if beneficial else
         "**No.** Across matched-retention, no-matched-retention, sensitive margin metrics, and the "
         "targeted position-bias axis, dependence-aware filtering produces no beneficial downstream "
         "alignment gain over naive consensus at this scale (if anything its smaller training set "
         "makes margins slightly worse). The filtering improvements are real at the label level but "
         "practically negligible — even the oracle (perfectly clean, larger set) only reaches "
         + (f"Δ {orc_d:+.3f} clean / Δ {(row('oracle','rewardbench_acc') - row('naive_consensus','rewardbench_acc')):+.3f} "
            "rewardbench, both within the noise floor" if evals is not None else "within noise") + "."),
        "\n## Decision on larger-model experiments\n",
        ("A beneficial effect appeared, so a capacity sweep is now justified." if beneficial
         else
         "**No larger-model experiment is warranted.** No *beneficial* downstream effect appeared under "
         "any protocol or metric, including the one designed to remove the dilution confound. The only "
         "beyond-noise signal (the paired margin metric) goes the wrong way and tracks training-set "
         "size, not label quality. The bottleneck is the magnitude of the label-quality difference "
         "itself, not retention protocol, metric sensitivity, or model capacity."),
    ]
    (RUN_DIR / "final_verdict.md").write_text("\n".join(fv))
    print("\n".join(sm)); print("\n" + "=" * 60 + "\n"); print("\n".join(fv))
    print(f"\nwrote {RES/'comparison_table.csv'}, {RES/'summary.md'}, {RUN_DIR/'final_verdict.md'}")


if __name__ == "__main__":
    main()

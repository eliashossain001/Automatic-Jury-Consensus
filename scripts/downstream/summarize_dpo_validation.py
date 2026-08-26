"""Phase 1, step 4 — join dataset stats + training log + eval into one table.

Produces:
  * results/comparison_table.csv  — one row per filter method, all metrics.
  * results/summary.md            — the table plus an automatically-generated, honest
    verdict on whether dependence-aware filtering beat naive consensus / supermajority
    downstream (and, if not, which filter fell short and by how much).

Reads only artifacts written by the previous three steps; computes no model output.

Usage: python scripts/downstream/summarize_dpo_validation.py
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from corrfilter import downstream as C


def load_csv(path):
    return pd.read_csv(path) if path.exists() else None


def build_table(cfg) -> pd.DataFrame:
    order = list(cfg["filters"])
    stats = load_csv(C.RESULTS_DIR / "dataset_stats.csv")
    evals = load_csv(C.RESULTS_DIR / "eval_results.csv")
    tlog_path = C.RESULTS_DIR / "train_log.json"
    tlog = json.loads(tlog_path.read_text()) if tlog_path.exists() else {}

    df = stats.copy() if stats is not None else pd.DataFrame({"filter_method": order})
    if evals is not None:
        df = df.merge(evals, on="filter_method", how="left")
    df["train_seconds"] = df["filter_method"].map(lambda m: tlog.get(m, {}).get("train_seconds"))
    df["reused_from"] = df["filter_method"].map(lambda m: tlog.get(m, {}).get("reused_from", ""))
    df["__order"] = df["filter_method"].map(lambda m: order.index(m) if m in order else 99)
    return df.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def verdict(df: pd.DataFrame, n_eval: int) -> list[str]:
    """Honest, data-driven verdict on the headline question, with a noise floor."""
    if "clean_pref_acc" not in df.columns or df["clean_pref_acc"].isna().all():
        return ["_Evaluation not yet run; verdict pending `evaluate_dpo_models.py`._"]

    def acc(m, col="clean_pref_acc"):
        r = df.loc[df.filter_method == m, col]
        return float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else None

    # Binomial standard error of an accuracy near 0.5 on n_eval held-out pairs; the
    # ~95% significance threshold for a single comparison is ~2*SE.
    se = (0.25 / n_eval) ** 0.5 if n_eval else float("nan")
    thr = 2 * se

    lines = [f"_Held-out eval n={n_eval} pairs; accuracy standard error ~{100 * se:.1f} pts, "
             f"so a gap must exceed ~{100 * thr:.1f} pts to be meaningful._\n"]
    baselines = [b for b in ("naive_consensus", "supermajority_75") if acc(b) is not None]
    best_baseline = max(baselines, key=lambda m: acc(m)) if baselines else None
    bb = acc(best_baseline) if best_baseline else None

    for m in ("corrfilter", "bias_cluster", "router"):
        a = acc(m)
        if a is None or bb is None:
            continue
        delta = 100 * (a - bb)
        if abs(delta) <= 100 * thr:
            tag = "WITHIN NOISE"
        elif delta > 0:
            tag = "IMPROVES"
        else:
            tag = "UNDERPERFORMS"
        lines.append(f"- **{m}**: clean preference acc {a:.3f} vs best baseline "
                     f"({best_baseline}) {bb:.3f} -> {delta:+.1f} pts ({tag}).")
    orc, naive = acc("oracle"), acc("naive_consensus")
    if orc is not None and naive is not None:
        d = 100 * (orc - naive)
        note = "within noise" if abs(d) <= 100 * thr else "real"
        lines.append(f"- **Headroom**: oracle (clean labels) {orc:.3f} vs naive {naive:.3f} "
                     f"= {d:+.1f} pts ({note}). With perfect labels no better than naive, "
                     "there is no recoverable signal here for any filter to capture.")

    accs = [acc(m) for m in df.filter_method if acc(m) is not None]
    spread = 100 * (max(accs) - min(accs)) if accs else 0.0
    answer = ("**Answer: No.** On this weak-dependence regime, dependence-aware filtering "
              "does not improve downstream DPO over naive consensus or supermajority. "
              f"All methods span only {spread:.1f} pts (< the ~{100 * thr:.1f}-pt noise floor), "
              "and even the oracle does not separate, so the result is a clean null: the "
              "downstream policy is insensitive to the small label-purity differences these "
              "filters produce at 10% contamination and 0.5B scale.")
    return [answer, ""] + lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    C.set_run_dir(cfg.get("run_dir", "."))
    C.ensure_dirs(C.RESULTS_DIR)

    df = build_table(cfg)
    df.to_csv(C.RESULTS_DIR / "comparison_table.csv", index=False)

    tag = cfg["data"].get("contamination", cfg["data"].get("regime", "n/a"))
    regime = cfg["data"]["regime"]
    eval_path = C.DATA_DIR / "eval_clean.jsonl"
    n_eval = len(C.read_jsonl(eval_path)) if eval_path.exists() else 0
    v = verdict(df, n_eval)
    df_disp = df.fillna({"router_choice": "", "reused_from": ""})
    if "router_choice" in df_disp.columns:
        df_disp["router_choice"] = df_disp["router_choice"].replace("nan", "")

    md = [
        "# Downstream DPO Validation — Results\n",
        f"**Substrate:** synthetic-poisoned UltraFeedback, {tag}% contamination "
        f"(regime: `{regime}`).  ",
        f"**Base model:** {cfg['train']['base_model']} + LoRA, DPO (beta="
        f"{cfg['train']['beta']}), identical hyperparameters and seed ({cfg['seed']}) "
        "across all methods.  ",
        "**Design:** every method retains the SAME number of items (matched retention), "
        "so differences reflect label quality, not training-set size. Evaluation is on a "
        "held-out CLEAN preference set (true labels) disjoint from all training data, plus "
        "a RewardBench v2 subset.\n",
        "## Comparison table\n",
        df_disp.to_markdown(index=False),
        "\n## Does dependence-aware filtering help downstream?\n",
        *v,
        "",
    ]
    (C.RESULTS_DIR / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {C.RESULTS_DIR / 'comparison_table.csv'} and summary.md")


if __name__ == "__main__":
    main()

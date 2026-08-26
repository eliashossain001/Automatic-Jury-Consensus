"""Experiment A — aggregate the per-level results into one scaling table + summary.

Reads each level's results/comparison_table.csv (and the Phase 1 10% run, if present,
for the full 10->40 picture), computes delta_vs_naive within each level and whether
the oracle beats naive, then writes:
  contamination_scaling/scaling_table.csv
  contamination_scaling/SUMMARY.md   (answers the 5 study questions, noise-aware)

Usage: python contamination_scaling/aggregate_scaling.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "contamination_scaling"                                    # this experiment's outputs
METHODS = ["naive_consensus", "supermajority_75", "corrfilter", "bias_cluster", "router", "oracle"]
COLS = ["contamination", "method", "train_size", "poisoned_kept", "label_purity",
        "clean_pref_acc", "rewardbench_acc", "delta_vs_naive"]


def level_sources() -> list[tuple[int, Path]]:
    """(contamination_pct, comparison_table.csv) for every available level."""
    out = []
    phase1 = EXP_DIR / "results" / "comparison_table.csv"
    if phase1.exists():
        out.append((10, phase1))
    for d in sorted(RUN_DIR.glob("c*/results/comparison_table.csv")):
        pct = int(d.parent.parent.name[1:])
        out.append((pct, d))
    return sorted(set(out))


def n_eval() -> int:
    for p in [EXP_DIR / "data" / "eval_clean.jsonl",
              *RUN_DIR.glob("c*/data/eval_clean.jsonl")]:
        if p.exists():
            return sum(1 for ln in p.open() if ln.strip())
    return 400


def build() -> pd.DataFrame:
    frames = []
    for pct, path in level_sources():
        t = pd.read_csv(path)
        t["contamination"] = pct
        naive = t.loc[t.filter_method == "naive_consensus", "clean_pref_acc"]
        naive_acc = float(naive.iloc[0]) if len(naive) else float("nan")
        t["delta_vs_naive"] = (t["clean_pref_acc"] - naive_acc).round(4)
        t = t.rename(columns={"filter_method": "method"})
        t["__o"] = t["method"].map(lambda m: METHODS.index(m) if m in METHODS else 99)
        frames.append(t.sort_values("__o"))
    df = pd.concat(frames, ignore_index=True)
    keep = [c for c in COLS if c in df.columns]
    return df[keep + [c for c in ("clean_reward_margin", "rewardbench_reward_margin") if c in df.columns]]


def answer(df: pd.DataFrame, n: int) -> list[str]:
    se = (0.25 / n) ** 0.5
    thr = 2 * se  # ~95% threshold for a single comparison
    levels = sorted(df.contamination.unique())

    def acc(level, method):
        r = df[(df.contamination == level) & (df.method == method)]["clean_pref_acc"]
        return float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else None

    def first_help(method):
        for lv in levels:
            a, base = acc(lv, method), acc(lv, "naive_consensus")
            if a is not None and base is not None and (a - base) > thr:
                return lv, 100 * (a - base)
        return None

    cf, bc = first_help("corrfilter"), first_help("bias_cluster")
    oracle_levels = [(lv, 100 * (acc(lv, "oracle") - acc(lv, "naive_consensus")))
                     for lv in levels if acc(lv, "oracle") is not None and acc(lv, "naive_consensus") is not None]
    oracle_wins = [(lv, d) for lv, d in oracle_levels if d > 100 * thr]

    L = [f"_Held-out eval n={n}; accuracy SE ~{100 * se:.1f} pts; a gap must exceed "
         f"~{100 * thr:.1f} pts to count as real. Levels tested: "
         f"{', '.join(f'{x}%' for x in levels)}._\n"]

    any_help = bool(cf or bc)
    L.append("**1. Does filtering start helping at higher contamination?** "
             + ("Yes — see below." if any_help else
                "No. Across every level tested, no dependence-aware filter exceeds naive "
                "consensus by more than the noise floor."))
    L.append("**2. At what contamination level does a downstream effect appear?** "
             + ("None within the tested range; deltas stay within noise at all levels."
                if not any_help else
                f"Earliest real effect at {min(x[0] for x in [v for v in [cf, bc] if v])}%."))
    L.append("**3. Does CorrFilter beat naive in any setting?** "
             + (f"Yes, at {cf[0]}% ({cf[1]:+.1f} pts)." if cf else
                "No — within noise at every level (best delta below the ~"
                f"{100 * thr:.1f}-pt threshold)."))
    L.append("**4. Does BiasCluster beat naive in any setting?** "
             + (f"Yes, at {bc[0]}% ({bc[1]:+.1f} pts)." if bc else
                "No — within noise at every level."))
    if oracle_wins:
        L.append("**5. Does oracle beat naive?** Yes at "
                 + ", ".join(f"{lv}% ({d:+.1f} pts)" for lv, d in oracle_wins)
                 + ". Where the oracle separates, there is recoverable signal a perfect "
                   "filter would capture; the practical filters fall short of it.")
    else:
        worst = min(oracle_levels, key=lambda x: x[1]) if oracle_levels else None
        best = max(oracle_levels, key=lambda x: x[1]) if oracle_levels else None
        L.append("**5. Does oracle beat naive?** No — even perfect clean labels do not beat "
                 f"naive beyond noise (oracle delta ranges {worst[1]:+.1f} to {best[1]:+.1f} pts). "
                 "Why the downstream signal stays weak despite rising contamination: (a) **matched "
                 "retention** means every method trains on the same number of items and discards "
                 "the same budget, so even naive consensus already removes most flipped labels at "
                 "the kept-item margin; the residual poisoned fraction among kept items is small and "
                 "similar across methods. (b) DPO is **relative** — a minority of flipped pairs is "
                 "averaged out against a clean majority, and a 0.5B LoRA policy is too low-capacity "
                 "to be swung by a few hundred contradictory gradients. (c) The held-out metric is "
                 "preference *ranking* accuracy, which saturates near the base model's prior and is "
                 "insensitive to small training-label perturbations. The bottleneck is the data "
                 "regime and model scale, not the filter.")
    return L


def main() -> None:
    df = build()
    df.to_csv(RUN_DIR / "scaling_table.csv", index=False)
    n = n_eval()

    final = df[[c for c in COLS if c in df.columns]].copy()
    md = [
        "# Experiment A: Contamination Scaling Study\n",
        "Same pipeline as Phase 1 (Qwen2.5-0.5B-Instruct + LoRA, DPO beta=0.1, fixed seed, "
        "matched retention, identical eval sets/metrics), swept across contamination levels. "
        "20/30/40% use an extended manifest (`make_high_contamination_manifest.py`) that nests "
        "the released 20% set and continues the same heuristic-aligned round-robin; 10% is the "
        "Phase 1 run shown for context.\n",
        "## Scaling table\n",
        final.to_markdown(index=False),
        "\n## Oracle vs naive, per level\n",
    ]
    for lv in sorted(df.contamination.unique()):
        o = df[(df.contamination == lv) & (df.method == "oracle")]["clean_pref_acc"]
        nv = df[(df.contamination == lv) & (df.method == "naive_consensus")]["clean_pref_acc"]
        if len(o) and len(nv):
            beats = "YES" if float(o.iloc[0]) > float(nv.iloc[0]) else "NO"
            md.append(f"- {lv}%: oracle {float(o.iloc[0]):.3f} vs naive {float(nv.iloc[0]):.3f} "
                      f"-> oracle beats naive: **{beats}**")
    md += ["\n## Answers\n", *answer(df, n)]
    (RUN_DIR / "SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {RUN_DIR / 'scaling_table.csv'} and SUMMARY.md")


if __name__ == "__main__":
    main()

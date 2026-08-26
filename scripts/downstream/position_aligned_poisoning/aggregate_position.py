"""Experiment C aggregator — position-poisoning scaling table + summary.

Reads each pos-level results/comparison_table.csv, computes delta_vs_naive within
each level, whether bias_cluster improves downstream over naive, and whether the
oracle beats naive. Writes:
  position_aligned_poisoning/position_table.csv
  position_aligned_poisoning/SUMMARY.md

Main question answered: does the bias-cluster filter's label-quality advantage in
the vulnerable-subgroup regime translate into downstream DPO improvement?

Usage: python position_aligned_poisoning/aggregate_position.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "position_aligned_poisoning"                                    # this experiment's outputs
METHODS = ["naive_consensus", "supermajority_75", "corrfilter", "bias_cluster", "router", "oracle"]
COLS = ["contamination", "method", "train_size", "poisoned_kept", "label_purity",
        "clean_pref_acc", "rewardbench_acc", "delta_vs_naive"]


def n_eval() -> int:
    for p in RUN_DIR.glob("*/data/eval_clean.jsonl"):
        return sum(1 for ln in p.open() if ln.strip())
    return 400


def build() -> pd.DataFrame:
    frames = []
    for path in sorted(RUN_DIR.glob("pos*/results/comparison_table.csv")):
        tag = path.parent.parent.name             # e.g. pos20
        pct = int(tag.replace("pos", ""))
        t = pd.read_csv(path)
        t["contamination"] = pct
        naive = t.loc[t.filter_method == "naive_consensus", "clean_pref_acc"]
        naive_acc = float(naive.iloc[0]) if len(naive) else float("nan")
        t["delta_vs_naive"] = (t["clean_pref_acc"] - naive_acc).round(4)
        t = t.rename(columns={"filter_method": "method"})
        t["__o"] = t["method"].map(lambda m: METHODS.index(m) if m in METHODS else 99)
        frames.append(t.sort_values("__o"))
    if not frames:
        raise SystemExit("no pos*/results/comparison_table.csv found — run the pipeline first")
    df = pd.concat(frames, ignore_index=True)
    extra = [c for c in ("clean_reward_margin", "rewardbench_reward_margin", "router_choice") if c in df.columns]
    return df[[c for c in COLS if c in df.columns] + extra]


def answer(df: pd.DataFrame, n: int) -> list[str]:
    se = (0.25 / n) ** 0.5
    thr = 2 * se
    levels = sorted(df.contamination.unique())

    def val(level, method, col="clean_pref_acc"):
        r = df[(df.contamination == level) & (df.method == method)][col]
        return float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else None

    def purity_gap(level, method):
        a, b = val(level, method, "label_purity"), val(level, "naive_consensus", "label_purity")
        return None if a is None or b is None else 100 * (a - b)

    def acc_gap(level, method):
        a, b = val(level, method), val(level, "naive_consensus")
        return None if a is None or b is None else 100 * (a - b)

    L = [f"_Held-out eval n={n}; accuracy SE ~{100 * se:.1f} pts; gap must exceed "
         f"~{100 * thr:.1f} pts to be real. Levels: {', '.join(f'{x}%' for x in levels)}._\n"]

    # bias_cluster: purity advantage vs downstream advantage, per level.
    L.append("**Does the bias-cluster label-quality advantage translate downstream?**")
    translated = False
    for lv in levels:
        pg, ag = purity_gap(lv, "bias_cluster"), acc_gap(lv, "bias_cluster")
        if pg is None:
            continue
        real = ag is not None and ag > 100 * thr
        translated = translated or real
        L.append(f"- {lv}%: bias_cluster label-purity {pg:+.1f} pts vs naive, "
                 f"downstream clean-acc {ag:+.1f} pts -> "
                 + ("**REAL downstream gain**" if real else "within noise downstream"))
    L.append("")
    L.append("**Headline:** "
             + ("Yes — bias_cluster's label-quality edge produces a real downstream DPO "
                "improvement in the position/vulnerable-subgroup regime."
                if translated else
                "Not measurably — bias_cluster recovers more label purity here than in the "
                "content regime, but the downstream clean-accuracy gain stays within the "
                f"~{100 * thr:.1f}-pt noise floor at the tested scale."))
    L.append("")

    # CorrFilter and oracle context.
    for m in ("corrfilter", "router"):
        gains = [acc_gap(lv, m) for lv in levels if acc_gap(lv, m) is not None]
        if gains:
            best = max(gains)
            L.append(f"- **{m}** vs naive: best {best:+.1f} pts "
                     + ("(real)" if best > 100 * thr else "(within noise)") + ".")
    obeats = []
    for lv in levels:
        og = acc_gap(lv, "oracle")
        if og is not None:
            obeats.append(f"{lv}%: {og:+.1f} pts" + (" (real)" if og > 100 * thr else " (noise)"))
    L.append("- **Oracle vs naive**: " + "; ".join(obeats) + ".")
    return L


def main() -> None:
    df = build()
    df.to_csv(RUN_DIR / "position_table.csv", index=False)
    n = n_eval()
    final = df[[c for c in COLS if c in df.columns]].copy()
    md = [
        "# Experiment C: Position-Aligned Poisoning + Downstream DPO\n",
        "Same pipeline as Phase 1 (Qwen2.5-0.5B-Instruct + LoRA, DPO beta=0.1, fixed seed, "
        "matched retention, identical eval sets/metrics), on the position-aligned "
        "vulnerable-subgroup attack (`make_position_poisoned_manifest.py`, replicating "
        "scripts/attacks/position_aligned_poisoning.py). This is the regime where the paper claims a filter advantage.\n",
        "## Results table\n",
        final.to_markdown(index=False),
        "\n## Bias-cluster vs naive, and oracle vs naive\n",
    ]
    for lv in sorted(df.contamination.unique()):
        for m in ("bias_cluster", "oracle"):
            a = df[(df.contamination == lv) & (df.method == m)]["clean_pref_acc"]
            nv = df[(df.contamination == lv) & (df.method == "naive_consensus")]["clean_pref_acc"]
            if len(a) and len(nv):
                beats = "YES" if float(a.iloc[0]) > float(nv.iloc[0]) else "NO"
                md.append(f"- {lv}% {m}: {float(a.iloc[0]):.3f} vs naive {float(nv.iloc[0]):.3f} "
                          f"-> beats naive: **{beats}**")
    md += ["\n## Answer to the main question\n", *answer(df, n)]
    (RUN_DIR / "SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {RUN_DIR / 'position_table.csv'} and SUMMARY.md")


if __name__ == "__main__":
    main()

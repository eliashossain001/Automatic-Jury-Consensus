"""Experiment A helper — extend the poisoned manifest to 30% and 40% contamination.

The original manifest (scripts/attacks/build_poisoned_ultrafeedback.py) only carries nested poison sets at 5/10/20%.
Higher rates were never generated, so this script EXTENDS the released set without
touching any original file or any vote cache (the votes encode the truly-better
response and are contamination-independent — only the stated label changes).

Method (identical to scripts/attacks/build_poisoned_ultrafeedback.py, just continued):
  * Recompute, per heuristic, the items whose surface-feature pick is B (the truly
    worse response) — flipping the label to that pick is a genuine corruption.
  * Seed the poison set with the EXISTING 20% items (preserving 20% ⊂ 30% ⊂ 40%),
    then continue the same round-robin across heuristics, drawing distinct
    poisonable items deterministically (seed 20260601) until 30% and 40% are filled.
  * Poisoned items get stated_label = "B"; true_label stays "A".

Writes a NEW manifest (original untouched):
  experiments/downstream_dpo_validation/contamination_scaling/poisoned_manifest_ext.csv
with all original columns plus poisoned_30/40, heuristic_30/40, stated_label_30/40.

Usage: python make_high_contamination_manifest.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Heuristic definitions copied verbatim from scripts/attacks/build_poisoned_ultrafeedback.py.
CONFIDENT_RE = re.compile(
    r"\b(definitely|certainly|clearly|obviously|surely|undoubtedly|absolutely|"
    r"in fact|indeed|without (?:a )?doubt|of course|essentially|fundamentally)\b",
    re.IGNORECASE,
)
FORMAT_RE = re.compile(r"(^|\n)\s*(#{1,6}\s|[-*]\s|\d+\.\s)|\n\s*\n", re.MULTILINE)
HEURISTICS = ["verbosity", "length", "polite", "style"]
LENGTH_TARGET = 1200
SEED = 20260601
NEW_RATES = {"30": 0.30, "40": 0.40}

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "contamination_scaling"                                    # this experiment's outputs
SRC_MANIFEST = ROOT / "outputs" / "synthetic_poisoned_ultrafeedback" / "poisoned_manifest.csv"
OUT_MANIFEST = RUN_DIR / "poisoned_manifest_ext.csv"


def heuristic_pick(a: str, b: str, heuristic: str):
    if heuristic == "verbosity":
        return None if len(a) == len(b) else ("A" if len(a) > len(b) else "B")
    if heuristic == "length":
        da, db = abs(len(a) - LENGTH_TARGET), abs(len(b) - LENGTH_TARGET)
        return None if da == db else ("A" if da < db else "B")
    if heuristic == "polite":
        ca, cb = len(CONFIDENT_RE.findall(a)), len(CONFIDENT_RE.findall(b))
        return None if ca == cb else ("A" if ca > cb else "B")
    if heuristic == "style":
        sa, sb = len(FORMAT_RE.findall(a)), len(FORMAT_RE.findall(b))
        return None if sa == sb else ("A" if sa > sb else "B")
    raise ValueError(heuristic)


def main() -> None:
    df = pd.read_csv(SRC_MANIFEST)
    n = len(df)
    a, b = df["response_a"].astype(str), df["response_b"].astype(str)

    poisonable = {h: [i for i in df.index if heuristic_pick(a[i], b[i], h) == "B"]
                  for h in HEURISTICS}
    rng = np.random.default_rng(SEED)
    for h in HEURISTICS:
        rng.shuffle(poisonable[h])

    # Preserve the released 20% set exactly, then continue round-robin from there.
    existing20 = [i for i in df.index if bool(df.loc[i, "poisoned_20"])]
    heur20 = {i: df.loc[i, "heuristic_20"] for i in existing20}
    chosen = list(existing20)
    chosen_heur = dict(heur20)
    seen = set(existing20)

    target_max = int(round(max(NEW_RATES.values()) * n))
    cursors = {h: 0 for h in HEURISTICS}
    hi, stalls = 0, 0
    while len(chosen) < target_max and stalls < len(HEURISTICS):
        h = HEURISTICS[hi % len(HEURISTICS)]
        hi += 1
        pool = poisonable[h]
        advanced = False
        while cursors[h] < len(pool):
            cand = pool[cursors[h]]
            cursors[h] += 1
            if cand not in seen:
                seen.add(cand)
                chosen.append(cand)
                chosen_heur[cand] = h
                advanced = True
                break
        stalls = 0 if advanced else stalls + 1

    if len(chosen) < target_max:
        raise RuntimeError(f"poisonable pool exhausted at {len(chosen)} < target {target_max}")

    for tag, rate in NEW_RATES.items():
        cnt = int(round(rate * n))
        poison_set = set(chosen[:cnt])
        df[f"poisoned_{tag}"] = [i in poison_set for i in df.index]
        df[f"heuristic_{tag}"] = [chosen_heur.get(i, "") if i in poison_set else "" for i in df.index]
        df[f"stated_label_{tag}"] = ["B" if i in poison_set else "A" for i in df.index]
        assert set(existing20).issubset(poison_set), "nesting broken: 20% not subset"
        print(f"{tag}%: {cnt} poisoned ({df[f'poisoned_{tag}'].mean():.1%}); "
              f"heuristics={pd.Series([chosen_heur[i] for i in chosen[:cnt]]).value_counts().to_dict()}")

    df.to_csv(OUT_MANIFEST, index=False)
    print(f"\nwrote {OUT_MANIFEST}  (original manifest untouched)")


if __name__ == "__main__":
    main()

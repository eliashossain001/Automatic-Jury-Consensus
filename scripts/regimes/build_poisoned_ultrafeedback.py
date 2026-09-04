"""Build a Synthetic-Poisoned UltraFeedback manifest for H2b (no inference).

Loads UltraFeedback binarized preference pairs, keeps the canonical pair
(A = truly-better = score-chosen, B = truly-worse = score-rejected), and plants
*structured* label corruption: on a fraction of items the stated preference is
flipped toward the response a weak heuristic prefers. Corruption is therefore
correlated with surface features that biased judges share, not random.

Heuristics (each flips an item only where its pick disagrees with the true
better response, so every flip is a genuine corruption):
  verbosity   -> prefer the longer response
  length      -> prefer the response closer to a moderate target length
  polite      -> prefer the response with more confident / assertive lexicon
  style       -> prefer the response with more markdown structure
(Position preference is intrinsically a presentation bias, not a content-label
flip; it is already universal in the bank, so it is exercised judge-side at
inference rather than as a corruption heuristic here.)

Corruption rates 5/10/20% are nested (5% ⊂ 10% ⊂ 20%) for comparability. The
judge run (scripts/regimes/run_ultrafeedback_judges.py) votes on the content-canonical pair and is independent of
the label, so a single vote cache serves all corruption rates; corruption is
applied at evaluation time (scripts/regimes/evaluate_poisoned_ultrafeedback.py).

Output: outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv
        (+ poisoned_manifest_summary.json)

Usage:
    python scripts/regimes/build_poisoned_ultrafeedback.py --n 2000 --project-root .
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CONFIDENT_RE = re.compile(
    r"\b(definitely|certainly|clearly|obviously|surely|undoubtedly|absolutely|"
    r"in fact|indeed|without (?:a )?doubt|of course|essentially|fundamentally)\b",
    re.IGNORECASE,
)
FORMAT_RE = re.compile(r"(^|\n)\s*(#{1,6}\s|[-*]\s|\d+\.\s)|\n\s*\n", re.MULTILINE)
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
HEURISTICS = ["verbosity", "length", "polite", "style"]
LENGTH_TARGET = 1200  # chars; "length" heuristic prefers the closer response


def _text(field) -> str:
    """Extract assistant text from a UF chosen/rejected field (list or str)."""
    if isinstance(field, list):
        for msg in reversed(field):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        if field and isinstance(field[-1], dict):
            return str(field[-1].get("content", ""))
        return str(field[-1]) if field else ""
    return str(field)


def _prompt_text(row) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    ch = row.get("chosen")
    if isinstance(ch, list):
        for msg in ch:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    return ""


def _heuristic_pick(a: str, b: str, heuristic: str) -> str:
    """Return 'A' or 'B' for which response the heuristic prefers (None if tie)."""
    if heuristic == "verbosity":
        if len(a) == len(b):
            return None
        return "A" if len(a) > len(b) else "B"
    if heuristic == "length":
        da, db = abs(len(a) - LENGTH_TARGET), abs(len(b) - LENGTH_TARGET)
        if da == db:
            return None
        return "A" if da < db else "B"
    if heuristic == "polite":
        ca, cb = len(CONFIDENT_RE.findall(a)), len(CONFIDENT_RE.findall(b))
        if ca == cb:
            return None
        return "A" if ca > cb else "B"
    if heuristic == "style":
        sa, sb = len(FORMAT_RE.findall(a)), len(FORMAT_RE.findall(b))
        if sa == sb:
            return None
        return "A" if sa > sb else "B"
    raise ValueError(heuristic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--n", type=int, default=2000, help="number of preference pairs")
    ap.add_argument("--repo", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", default="train_prefs")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--max-chars", type=int, default=4000, help="truncate responses")
    ap.add_argument("--out-dir", default="outputs/synthetic_poisoned_ultrafeedback")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = (root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print(f"loading {args.repo}:{args.split} ...")
    ds = load_dataset(args.repo, split=args.split)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ds))

    rows = []
    for k in order:
        if len(rows) >= args.n:
            break
        row = ds[int(k)]
        a = _text(row.get("chosen"))[: args.max_chars].strip()   # truly-better
        b = _text(row.get("rejected"))[: args.max_chars].strip()  # truly-worse
        prompt = _prompt_text(row)[: args.max_chars].strip()
        if not a or not b or not prompt or a == b:
            continue
        sc, sr = row.get("score_chosen"), row.get("score_rejected")
        rows.append({
            "item_id": f"uf{int(k)}",
            "prompt": prompt,
            "response_a": a,            # canonical: A = truly better
            "response_b": b,            # B = truly worse
            "true_label": "A",
            "len_a": len(a), "len_b": len(b),
            "score_chosen": float(sc) if sc is not None else np.nan,
            "score_rejected": float(sr) if sr is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"kept {n} usable pairs")

    # Poisonable heuristics per item: a heuristic can poison item i iff its pick
    # is B (the truly-worse response) -> flipping the label to that pick is a
    # genuine corruption. Round-robin assign poisoned items across heuristics.
    poisonable = {h: [] for h in HEURISTICS}
    for i, r in df.iterrows():
        for h in HEURISTICS:
            if _heuristic_pick(r["response_a"], r["response_b"], h) == "B":
                poisonable[h].append(i)
    for h in HEURISTICS:
        rng.shuffle(poisonable[h])

    # Build nested poison sets. Greedy round-robin draw distinct items.
    sorted_rates = sorted(RATES.items(), key=lambda kv: kv[1])
    max_rate = sorted_rates[-1][1]
    target_max = int(round(max_rate * n))
    chosen_items: list[int] = []
    chosen_heur: dict[int, str] = {}
    seen: set[int] = set()
    cursors = {h: 0 for h in HEURISTICS}
    hi = 0
    stalls = 0
    while len(chosen_items) < target_max and stalls < len(HEURISTICS):
        h = HEURISTICS[hi % len(HEURISTICS)]
        hi += 1
        pool = poisonable[h]
        advanced = False
        while cursors[h] < len(pool):
            cand = pool[cursors[h]]
            cursors[h] += 1
            if cand not in seen:
                seen.add(cand)
                chosen_items.append(cand)
                chosen_heur[cand] = h
                advanced = True
                break
        stalls = 0 if advanced else stalls + 1

    # Nested rate assignment: first 5% ⊂ 10% ⊂ 20%.
    for tag, rate in RATES.items():
        cnt = int(round(rate * n))
        poison_set = set(chosen_items[:cnt])
        df[f"poisoned_{tag}"] = [i in poison_set for i in df.index]
        df[f"heuristic_{tag}"] = [chosen_heur.get(i, "") if i in poison_set else "" for i in df.index]
        # stated_label after corruption: flip A->B on poisoned items.
        df[f"stated_label_{tag}"] = ["B" if i in poison_set else "A" for i in df.index]

    manifest_path = out_dir / "poisoned_manifest.csv"
    df.to_csv(manifest_path, index=False)

    summary = {
        "repo": args.repo, "split": args.split, "n_pairs": n, "seed": args.seed,
        "rates": {tag: int(df[f"poisoned_{tag}"].sum()) for tag in RATES},
        "heuristic_composition_20pct": df[df["poisoned_20"]]["heuristic_20"].value_counts().to_dict(),
        "n_poisonable_per_heuristic": {h: len(poisonable[h]) for h in HEURISTICS},
        "mean_len_a": float(df["len_a"].mean()), "mean_len_b": float(df["len_b"].mean()),
    }
    (out_dir / "poisoned_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

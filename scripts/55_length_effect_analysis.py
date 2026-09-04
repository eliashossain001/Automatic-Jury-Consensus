#!/usr/bin/env python
"""Exploratory: does prompt length independently predict judge-error dependence?

Analysis-only, CPU, no inference. Uses the published base-bank cache (never truncated,
max_length 4096), so the length effect studied here is a property of the data, not of the
truncation defect.

Motivating observation (P0-1): on the published base bank, rho_bar is 0.251 on items whose
prompt is <= 1024 tokens and 0.054 above, with n_eff 2.66 vs 4.72.

Leading alternative explanation tested here: long items are HARDER, judges sit closer to
chance on them, and near-chance judges guess close to independently, so low correlation
would be a consequence of low accuracy rather than of length. Distinguishing these matters
because they imply opposite things for the paper's shared-difficulty story.

Outputs -> outputs/length_effect/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from compute_dependence import estimate_R, load_votes  # noqa: E402
from corrfilter.correlation.effective_size import (effective_size,  # noqa: E402
                                                   mean_off_diagonal)
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges.prompts import get_prompt_template  # noqa: E402

MANIFEST = ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet"
OUT = ROOT / "outputs/length_effect"
BANK = ROOT / "configs/judge_bank.yaml"
SEED = 20260706


def token_lengths(items):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tmpl = get_prompt_template("pairwise")
    out = []
    for it in items:
        raw = tmpl.render(it, False).text
        sysp = raw.split("<|user|>", 1)[0].replace("<|system|>", "").strip()
        usr = raw.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0].strip()
        s = tok.apply_chat_template(
            [{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
            tokenize=False, add_generation_prompt=True)
        out.append(len(tok.encode(s, add_special_tokens=False)))
    return np.array(out)


def dep(V, M, gold, mask):
    if mask.sum() < 30:
        return np.nan, np.nan, int(mask.sum()), np.nan
    R, _, _, _ = estimate_R(V[mask], M[mask], gold[mask])
    Mb = M[mask].astype(bool)
    acc = float(np.nanmean([
        (V[mask][Mb[:, j], j] == gold[mask][Mb[:, j]]).mean() if Mb[:, j].any() else np.nan
        for j in range(V.shape[1])]))
    return float(mean_off_diagonal(R)), float(effective_size(R)), int(mask.sum()), acc


def boot_rho(V, M, gold, mask, B=1000, seed=SEED):
    idx_pool = np.where(mask)[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.choice(idx_pool, len(idx_pool), replace=True)
        try:
            R, _, _, _ = estimate_R(V[idx], M[idx], gold[idx])
            vals.append(float(mean_off_diagonal(R)))
        except Exception:
            continue
    return (np.nan, np.nan) if not vals else tuple(np.quantile(vals, [0.025, 0.975]))


def main():
    items = load_calibration_set(MANIFEST)
    ids = [str(i.item_id) for i in items]
    gold = np.ones(len(ids), dtype=np.int8)
    V, M, lids = load_votes(BANK, ids, None)
    L = token_lengths(items)
    man = pd.read_parquet(MANIFEST).drop_duplicates("item_id", keep="first").set_index("item_id")
    subset = np.array([str(man.loc[i, "subset"]) if i in man.index else "?" for i in ids])
    n_wrong = ((V != gold[:, None]) & M.astype(bool)).sum(1)   # per-item difficulty
    n_comm = M.astype(bool).sum(1)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(group, level, mask, note=""):
        r, ne, n, acc = dep(V, M, gold, mask)
        rows.append({"grouping": group, "level": level, "n": n, "acc": None if np.isnan(acc) else round(acc, 4),
                     "rho_bar": None if np.isnan(r) else round(r, 4),
                     "n_eff": None if np.isnan(ne) else round(ne, 3), "note": note})

    # 1. the raw effect, by length quartile
    q = np.quantile(L, [0.25, 0.5, 0.75])
    for lab, mask in [("Q1 shortest", L <= q[0]), ("Q2", (L > q[0]) & (L <= q[1])),
                      ("Q3", (L > q[1]) & (L <= q[2])), ("Q4 longest", L > q[2])]:
        add("length quartile", lab, mask)

    # 2. accuracy as the competing explanation: bin by per-item difficulty
    for k in range(0, 11, 2):
        m = (n_wrong >= k) & (n_wrong < k + 2) & (n_comm >= 5)
        add("difficulty (n judges wrong)", f"{k}-{k+1}", m)

    # 3. length effect WITHIN difficulty strata (the decisive test)
    med = np.median(L)
    for k in range(0, 11, 2):
        base_m = (n_wrong >= k) & (n_wrong < k + 2) & (n_comm >= 5)
        add("within-difficulty length", f"wrong {k}-{k+1} | short", base_m & (L <= med))
        add("within-difficulty length", f"wrong {k}-{k+1} | long", base_m & (L > med))

    # 4. length effect WITHIN subset (source/category confound)
    for s in sorted(set(subset)):
        m = subset == s
        if m.sum() < 80:
            continue
        add("within-subset length", f"{s} | short", m & (L <= med))
        add("within-subset length", f"{s} | long", m & (L > med))

    # 5. subset composition by length (is length a proxy for source?)
    comp = pd.crosstab(pd.Series(subset, name="subset"),
                       pd.Series(np.where(L <= med, "short", "long"), name="length"))
    comp.to_csv(OUT / "subset_by_length.csv")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "length_strata.csv", index=False)
    print("\n=== 1. raw length effect ===")
    print(df[df.grouping == "length quartile"].to_string(index=False))
    print("\n=== 2. dependence vs item difficulty ===")
    print(df[df.grouping == "difficulty (n judges wrong)"].to_string(index=False))
    print("\n=== 3. length effect WITHIN difficulty strata ===")
    print(df[df.grouping == "within-difficulty length"].to_string(index=False))
    print("\n=== 4. length effect WITHIN subset ===")
    print(df[df.grouping == "within-subset length"].to_string(index=False))
    print("\n=== 5. subset composition by length ===")
    print(comp.to_string())

    # 6. headline CIs
    lo_s, hi_s = boot_rho(V, M, gold, L <= 1024)
    lo_l, hi_l = boot_rho(V, M, gold, L > 1024)
    r_s, ne_s, n_s, a_s = dep(V, M, gold, L <= 1024)
    r_l, ne_l, n_l, a_l = dep(V, M, gold, L > 1024)
    print(f"\n=== 6. headline (base bank, 1024-token split) ===")
    print(f"  short: n={n_s} acc={a_s:.4f} rho_bar={r_s:.4f} [{lo_s:.4f},{hi_s:.4f}] n_eff={ne_s:.3f}")
    print(f"  long : n={n_l} acc={a_l:.4f} rho_bar={r_l:.4f} [{lo_l:.4f},{hi_l:.4f}] n_eff={ne_l:.3f}")
    pd.DataFrame([
        {"stratum": "<=1024", "n": n_s, "acc": round(a_s, 4), "rho_bar": round(r_s, 4),
         "ci_low": round(lo_s, 4), "ci_high": round(hi_s, 4), "n_eff": round(ne_s, 3)},
        {"stratum": ">1024", "n": n_l, "acc": round(a_l, 4), "rho_bar": round(r_l, 4),
         "ci_low": round(lo_l, 4), "ci_high": round(hi_l, 4), "n_eff": round(ne_l, 3)},
    ]).to_csv(OUT / "headline.csv", index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage-P frontier pointwise-code pilot (author-approved; Stage P ONLY).

60 pointwise code items (the first 30 pairs of the existing frozen
CodeJudgeBench manifest, by stable hash order -> 60 items, gold from
unit-test execution) x 6 pinned frontier judges (3 Gemini on the author's
key; 3 OpenRouter). Hard budget ceiling $2.50 enforced after every judge;
402 aborts; dry-run default.

Preregistered gate: majority accuracy >= 0.65 (chance 0.50); >= 15
majority-wrong items; < 5% abstention per judge; both classes represented.
On failure: STOP. On pass: report only (no Stage F launch).

Usage: python scripts/52_frontier_code_pilot.py [--live]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/router_upgrade"))

from corrfilter.correlation import correlation_shrunk  # noqa: E402
from corrfilter.data.generic_task import batch_from_cache, manifest_to_items  # noqa: E402
from corrfilter.judges.base import JudgeSpec  # noqa: E402
from corrfilter.judges.gemini_judge import GeminiJudge  # noqa: E402
from corrfilter.judges.openrouter_judge import OpenRouterJudge  # noqa: E402
from corrfilter.judges.prompts_pointwise import CODE_TEMPLATES  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

OUT = ROOT / "outputs/screening_bank"
BUDGET = 2.50
GEMINI = [("gemini-3.1-pro", "gemini-3.1-pro-preview", "low", 512),
          ("gemini-3.6-flash", "gemini-3.6-flash", "minimal", 32),
          ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite", "minimal", 32)]
OPENROUTER = [("gpt-5.6-sol", "openai/gpt-5.6-sol", False),
              ("claude-opus-5", "anthropic/claude-opus-5", True),
              ("grok-4.5", "x-ai/grok-4.5", True)]


def load_dotenv_keys(env_path: Path) -> None:
    import os
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def pilot_manifest() -> pd.DataFrame:
    path = OUT / "manifest_frontier_code_pilot.parquet"
    if path.exists():
        return pd.read_parquet(path)
    man = pd.read_parquet(OUT / "manifest_cjb_point.parquet")
    pairs = sorted(man.pair_hash.unique())[:30]
    sub = man[man.pair_hash.isin(pairs)].reset_index(drop=True)
    sub.to_parquet(path)
    return sub


def make_spec(base_id: str) -> JudgeSpec:
    return JudgeSpec(base_id=base_id, family="frontier", scale="api",
                     prompt_style="pwc_direct", hf_model="", quantization="none",
                     dtype="bfloat16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    load_dotenv_keys(ROOT / ".env")
    # register the pointwise-code template for API judges
    from corrfilter.judges.prompts import PROMPT_TEMPLATES
    from corrfilter.judges.prompts_pointwise import CodePointwiseDirectTemplate
    PROMPT_TEMPLATES.setdefault("pwc_direct", CodePointwiseDirectTemplate)

    man = pilot_manifest()
    items = manifest_to_items(man)
    cache = VoteCache(OUT / "votes/frontier_code")
    lids = []
    print(f"Stage-P pilot: {len(items)} items (supported {man.gold_supported.mean():.2f}) "
          f"x 6 frontier judges; ceiling ${BUDGET:.2f}; live={args.live}")
    if not args.live:
        return

    spent = 0.0
    log = []
    judges = []
    for bid, model, thinking, mot in GEMINI:
        judges.append((bid, GeminiJudge(make_spec(bid), model=model,
                                        thinking_level=thinking,
                                        max_output_tokens=mot, temperature=0.0,
                                        shuffle_position=False, dry_run=False,
                                        workers=4)))
    for bid, model, sup_temp in OPENROUTER:
        judges.append((bid, OpenRouterJudge(make_spec(bid), model=model,
                                            reasoning_effort="minimal",
                                            supports_temperature=sup_temp,
                                            max_output_tokens=2048, temperature=0.0,
                                            shuffle_position=False, dry_run=False,
                                            workers=4)))
    for bid, judge in judges:
        lids.append(judge.spec.logical_id)
        done = cache.cached_item_ids(judge.spec.logical_id)
        todo = [it for it in items if it.item_id not in done]
        if not todo:
            print(f"  {bid}: cached")
            continue
        judge.load()
        votes = judge.vote_batch(todo)
        cache.write(judge.spec.logical_id, votes)
        cost = getattr(judge.usage, "cost_usd", None)
        if cost is None:
            cost = (judge.usage.input_tokens * judge.price_in
                    + judge.usage.output_tokens * judge.price_out) / 1e6
        spent += float(cost or 0.0)
        v = np.array([x.vote for x in votes])
        log.append({"judge": bid, "n": len(votes), "abstain": float((v == -1).mean()),
                    "cost_usd": round(float(cost or 0.0), 4),
                    "in_tok": judge.usage.input_tokens,
                    "out_tok": judge.usage.output_tokens})
        print(f"  {bid}: {len(votes)} votes, ${cost:.3f} (total ${spent:.3f})", flush=True)
        if spent >= BUDGET:
            print("BUDGET CEILING REACHED; stopping")
            break
    pd.DataFrame(log).to_csv(OUT / "frontier_code_pilot_runlog.csv", index=False)

    # ---- report + gate ----
    b = batch_from_cache(man, cache, lids)
    V, M, gold = b.judge_votes, b.availability, b.gold_labels
    nj = V.shape[1]
    Mb = M.astype(bool)
    per = {lids[j]: round(float((V[Mb[:, j], j] == gold[Mb[:, j]]).mean()), 3)
           for j in range(nj)}
    abst = {lids[j]: round(float(1 - Mb[:, j].mean()), 3) for j in range(nj)}
    maj = (np.where(Mb, V, 0).sum(1) > (M.sum(1) / 2)).astype(int)
    maj_acc = float((maj == gold).mean())
    maj_wrong = int((maj != gold).sum())
    E = (V != gold[:, None]).astype(float)
    rows_ok = Mb.all(1)
    R, _ = correlation_shrunk(E[rows_ok]) if rows_ok.sum() >= 2 else (np.eye(nj), 1)
    off = ~np.eye(nj, dtype=bool)
    rho = float(np.nanmean(R[off]))
    neff = nj / (1 + (nj - 1) * rho) if rho > -1 / (nj - 1) else float(nj)
    patterns = len({tuple(r) for r in np.where(Mb, V, -9)})
    # preliminary headroom: fixed-filter menu on the 60-item set (cal 20)
    import build_crossed_pool as bcp
    from feasibility_screen import _mk_source
    src, Rv = _mk_source("fcode", V, M, gold, lids, cluster_k=3)
    sg_orig = bcp.SG_SIZE
    bcp.SG_SIZE = 20
    metrics, _, _, _ = bcp.evaluate(src, V, M, gold, 0, Rv, retention_mode="affirmed")
    bcp.SG_SIZE = sg_orig
    FIXED = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
    prec = {f: metrics[f]["precision"] for f in FIXED}
    headroom = 100 * (max(prec.values()) - prec["naive_majority"])
    gate = {"majority_acc>=0.65": maj_acc >= 0.65,
            "majority_wrong>=15": maj_wrong >= 15,
            "abstain<5% per judge": all(a < 0.05 for a in abst.values()),
            "both classes": len(set(gold.tolist())) == 2}
    print("\n== Stage-P report ==")
    print("per-judge acc:", per)
    print("abstention:", abst)
    print(f"majority acc {maj_acc:.3f} vs 0.50 | majority-wrong {maj_wrong}/60")
    print(f"rho_err {rho:.3f} | n_eff {neff:.2f}/6 | vote patterns {patterns}/60")
    print(f"filter precisions: { {k: round(v,4) for k,v in prec.items()} }")
    print(f"preliminary headroom (single-set): {headroom:.2f} pts")
    print(f"total cost ${spent:.3f} of ${BUDGET:.2f}")
    print("GATE:", gate, "->", "PASS" if all(gate.values()) else "FAIL - STOP")
    pd.DataFrame([{**{f"acc_{k}": v for k, v in per.items()}, "majority_acc": maj_acc,
                   "majority_wrong": maj_wrong, "rho_err": rho, "n_eff": neff,
                   "patterns": patterns, "headroom_prelim": headroom,
                   "cost_usd": spent, "gate_pass": all(gate.values())}]).to_csv(
        OUT / "frontier_code_pilot_result.csv", index=False)


if __name__ == "__main__":
    main()

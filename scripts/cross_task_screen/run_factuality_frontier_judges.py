#!/usr/bin/env python
"""Frontier factuality bank: 6 API judges pointwise on the VitaminC manifest.

Judges (pinned IDs, one pw_direct style each; internal reasoning per model
config exactly as the paper's frontier runs):
  Gemini API:  gemini-3.1-pro-preview (thinking low), gemini-3.6-flash,
               gemini-3.5-flash-lite (thinking minimal)
  OpenRouter:  openai/gpt-5.6-sol, anthropic/claude-opus-5, x-ai/grok-4.5
               (reasoning effort minimal; hard stop on HTTP 402)

Safety rails identical to scripts/frontier/run_gemini_judges.py and scripts/frontier/run_openrouter_judges.py: dry-run by default (prints cost
projection), checkpointed via the vote cache, bounded retries, abstain on
failure, per-run usage/cost logs, and a HARD --budget ceiling checked after
every judge (run aborts before starting a judge that would exceed it).
Spend requires BOTH --live and an explicit --budget.

Usage:
  python scripts/cross_task_screen/run_factuality_frontier_judges.py --stage pilot            # dry-run
  python scripts/cross_task_screen/run_factuality_frontier_judges.py --stage pilot --live --budget 0.50
  python scripts/cross_task_screen/run_factuality_frontier_judges.py --stage main  --live --budget 9.00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data.generic_task import manifest_to_items  # noqa: E402
from corrfilter.judges.base import JudgeSpec  # noqa: E402
from corrfilter.judges.gemini_judge import GeminiJudge  # noqa: E402
from corrfilter.judges.openrouter_judge import OpenRouterJudge  # noqa: E402
from corrfilter.judges.prompts_pointwise import register_pointwise_templates  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

OUT = ROOT / "outputs/factuality_bank"

GEMINI = [
    {"id": "gemini-3.1-pro", "model": "gemini-3.1-pro-preview", "thinking_level": "low",
     "max_output_tokens": 1024},
    {"id": "gemini-3.6-flash", "model": "gemini-3.6-flash", "thinking_level": "minimal",
     "max_output_tokens": 64},
    {"id": "gemini-3.5-flash-lite", "model": "gemini-3.5-flash-lite",
     "thinking_level": "minimal", "max_output_tokens": 64},
]
OPENROUTER = [
    {"id": "gpt-5.6-sol", "model": "openai/gpt-5.6-sol", "reasoning_effort": "minimal",
     "supports_temperature": False, "max_output_tokens": 2048},
    {"id": "claude-opus-5", "model": "anthropic/claude-opus-5",
     "reasoning_effort": "minimal", "max_output_tokens": 2048},
    {"id": "grok-4.5", "model": "x-ai/grok-4.5", "reasoning_effort": "minimal",
     "max_output_tokens": 2048},
]


def make_spec(base_id: str) -> JudgeSpec:
    return JudgeSpec(base_id=base_id, family="frontier", scale="api",
                     prompt_style="pw_direct", hf_model="", quantization="none",
                     dtype="bfloat16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="hard USD ceiling for this invocation (required with --live)")
    ap.add_argument("--provider", choices=["gemini", "openrouter", "both"], default="both")
    args = ap.parse_args()
    if args.live and args.budget <= 0:
        raise SystemExit("--live requires an explicit --budget ceiling")
    register_pointwise_templates()

    man = pd.read_parquet(OUT / f"manifest_{'pilot' if args.stage == 'pilot' else 'main'}.parquet")
    items = manifest_to_items(man)
    cache = VoteCache(OUT / "votes_frontier")
    entries = ([("gemini", e) for e in GEMINI] if args.provider in ("gemini", "both") else []) + \
              ([("openrouter", e) for e in OPENROUTER] if args.provider in ("openrouter", "both") else [])
    print(f"stage={args.stage}: {len(items)} items x {len(entries)} frontier judges "
          f"(pw_direct); budget ${args.budget:.2f}; live={args.live}")
    if not args.live:
        print("dry-run: rough projection ~"
              f"${0.0006 * len(items) * len(entries):.2f}-{0.004 * len(items) * len(entries):.2f} "
              "(short prompts; grok reasoning overhead dominates)")
        return

    spent = 0.0
    logrows = []
    for provider, e in entries:
        spec = make_spec(e["id"])
        done = cache.cached_item_ids(spec.logical_id)
        todo = [it for it in items if it.item_id not in done]
        if not todo:
            print(f"  {spec.logical_id}: all cached")
            continue
        if provider == "gemini":
            judge = GeminiJudge(spec, model=e["model"], thinking_level=e["thinking_level"],
                                max_output_tokens=e["max_output_tokens"], temperature=0.0,
                                shuffle_position=False, dry_run=False, workers=4)
        else:
            judge = OpenRouterJudge(spec, model=e["model"],
                                    reasoning_effort=e.get("reasoning_effort"),
                                    supports_temperature=e.get("supports_temperature", True),
                                    max_output_tokens=e["max_output_tokens"], temperature=0.0,
                                    shuffle_position=False, dry_run=False, workers=4)
        judge.load()
        votes = judge.vote_batch(todo)
        cache.write(spec.logical_id, votes)
        cost = getattr(judge.usage, "cost_usd", None)
        if cost is None:
            cost = (judge.usage.input_tokens * judge.price_in
                    + judge.usage.output_tokens * judge.price_out) / 1e6
        spent += float(cost or 0.0)
        v = np.array([x.vote for x in votes])
        logrows.append({"judge": spec.logical_id, "n": len(votes),
                        "abstain": float((v == -1).mean()),
                        "cost_usd": round(float(cost or 0.0), 4),
                        "in_tok": judge.usage.input_tokens,
                        "out_tok": judge.usage.output_tokens})
        print(f"  {spec.logical_id}: {len(votes)} votes, ${cost:.3f} "
              f"(total ${spent:.3f} of ${args.budget:.2f})")
        if spent >= args.budget:
            print("BUDGET CEILING REACHED; stopping before next judge")
            break
    pd.DataFrame(logrows).to_csv(OUT / f"frontier_runlog_{args.stage}.csv", index=False)
    print(f"total spent this run: ${spent:.3f}")


if __name__ == "__main__":
    main()

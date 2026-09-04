#!/usr/bin/env python
"""Stage-driven pointwise factuality vote collection (LLM-AggreFact).

Runs the five open-weight base judges with TWO pointwise prompt styles
(pw_direct, pw_analysis) = 10 logical judges, one model resident on GPU at a
time (nf4). Votes cached one parquet per logical judge under
outputs/factuality_bank/votes/ in the standard cache schema
(item_id, vote, position_swapped, raw_response); position_swapped is always
False because pointwise judging performs no swap (shuffle_position=False;
no fake metadata is generated) and the swap matrix must never be consumed
for this task.

Stages (deterministic item manifests, disjoint by construction):
  smoke  ~15 items   pipeline check
  pilot  ~150 items  judge-competence gate (scripts/44 reports it)
  main   ~1000 items clean vote bank (includes pilot items: same clean
                     distribution, votes reused from cache when present)

Usage:
  python scripts/43_run_factuality_judges.py --stage smoke --live
  python scripts/43_run_factuality_judges.py --stage pilot --live
  python scripts/43_run_factuality_judges.py --stage main  --live
Dry-run (default) prints the plan and exits. CPU/GPU only; $0 API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data.generic_task import (  # noqa: E402
    load_aggrefact, load_vitaminc, manifest_to_items, stratified_sample)
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.judges.base import JudgeSpec  # noqa: E402
from corrfilter.judges.hf_judge import HFJudge  # noqa: E402
from corrfilter.judges.prompts_pointwise import POINTWISE_TEMPLATES  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

OUT = ROOT / "outputs/factuality_bank"
SEEDS = {"smoke": 20260810, "pilot": 20260811, "main": 20260812}
SIZES = {"smoke": 15, "pilot": 150, "main": 1000}
MAX_DOC_CHARS = 2000


DATASET = "vitaminc"  # primary ungated source; switch to "aggrefact" once
                       # Hub access to lytang/LLM-AggreFact is granted


def stage_manifest(stage: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"manifest_{stage}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    full = load_vitaminc("test") if DATASET == "vitaminc" else load_aggrefact("dev")
    (OUT / f"{DATASET}_audit.json").write_text(json.dumps(full.attrs["audit"], indent=2))
    exclude: set[str] = set()
    if stage == "pilot":
        exclude |= set(stage_manifest("smoke").item_id)
    if stage == "main":
        # main INCLUDES pilot items (same clean distribution; cached votes
        # reused) but excludes the smoke scratch items.
        exclude |= set(stage_manifest("smoke").item_id)
        pilot = stage_manifest("pilot")
        extra = stratified_sample(full, SIZES["main"] - len(pilot), SEEDS["main"],
                                  MAX_DOC_CHARS, exclude | set(pilot.item_id))
        man = pd.concat([pilot, extra]).sort_values("item_id").reset_index(drop=True)
        man.to_parquet(path)
        return man
    man = stratified_sample(full, SIZES[stage], SEEDS[stage], MAX_DOC_CHARS, exclude)
    man.to_parquet(path)
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "pilot", "main"], required=True)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--models", default=None, help="comma list to restrict base models")
    args = ap.parse_args()

    man = stage_manifest(args.stage)
    items = manifest_to_items(man)
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    base_specs = {s.base_id: s for s in bank.specs}  # one per model
    models = sorted(base_specs)
    if args.models:
        models = [m for m in models if m in set(args.models.split(","))]
    cache = VoteCache(OUT / "votes")
    print(f"stage={args.stage}: {len(items)} items "
          f"(supported {man.gold_supported.mean():.2f}), models={models}, "
          f"styles={list(POINTWISE_TEMPLATES)}")
    if not args.live:
        print("dry-run; pass --live to spend GPU time")
        return

    runlog = []
    for base_id in models:
        ref = base_specs[base_id]
        for style, tmpl in POINTWISE_TEMPLATES.items():
            spec = JudgeSpec(base_id=base_id, family=ref.family, scale=ref.scale,
                             prompt_style=style, hf_model=ref.hf_model,
                             hf_revision=ref.hf_revision, quantization=ref.quantization,
                             dtype=ref.dtype, trust_remote_code=ref.trust_remote_code)
            done = cache.cached_item_ids(spec.logical_id)
            todo = [it for it in items if it.item_id not in done]
            if not todo:
                print(f"  {spec.logical_id}: all {len(items)} cached")
                continue
            judge = HFJudge(spec, prompt_template=tmpl, batch_size=4,
                            max_new_tokens=tmpl.max_new_tokens,
                            shuffle_position=False, max_length=4096)
            t0 = time.time()
            judge.load()
            votes = judge.vote_batch(todo)
            judge.unload()
            dt = time.time() - t0
            cache.write(spec.logical_id, votes)
            v = np.array([x.vote for x in votes])
            runlog.append({"logical_id": spec.logical_id, "n": len(votes),
                           "abstain": float((v == -1).mean()),
                           "affirm_rate": float((v == 1).mean()),
                           "seconds": round(dt, 1)})
            print(f"  {spec.logical_id}: {len(votes)} votes in {dt:.0f}s, "
                  f"abstain {(v==-1).mean():.3f}")
    pd.DataFrame(runlog).to_csv(OUT / f"runlog_{args.stage}.csv", index=False)


if __name__ == "__main__":
    main()

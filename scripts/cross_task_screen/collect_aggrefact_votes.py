#!/usr/bin/env python
"""AggreFact component-level screening: vote collection (Priority 1).

Eight eligible components (>=100 items with doc <= 2,000 chars), ~150 items
each, label-stratified, original gold preserved; existing open bank first
(5 models x pw_direct/pw_analysis = 10 judges). One shared cache; per-item
component recorded in the manifest. Checkpointed; $0 API.

Usage: python scripts/cross_task_screen/collect_aggrefact_votes.py [--live]
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data.generic_task import load_aggrefact, manifest_to_items
from corrfilter.judges import load_bank_config
from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.hf_judge import HFJudge
from corrfilter.judges.prompts_pointwise import POINTWISE_TEMPLATES
from corrfilter.voting import VoteCache

OUT = ROOT / "outputs/screening_bank"
SEED = 20270501
COMPONENTS = ["RAGTruth", "ExpertQA", "Lfqa", "Reveal", "FactCheck-GPT",
              "AggreFact-XSum", "AggreFact-CNN", "ClaimVerify"]
N_PER = 150
MAX_DOC = 2000


def build_manifest():
    path = OUT / "manifest_aggrefact_screen.parquet"
    if path.exists():
        return pd.read_parquet(path)
    full = load_aggrefact("dev")
    rng = np.random.default_rng(SEED)
    parts = []
    for comp in COMPONENTS:
        sub = full[(full.source_dataset == comp) & (full.doc_chars <= MAX_DOC)]
        take = []
        for g in (0, 1):
            gg = sub[sub.gold_supported == g]
            k = min(N_PER // 2, len(gg))
            take.append(gg.iloc[np.sort(rng.choice(len(gg), k, replace=False))])
        got = pd.concat(take)
        if len(got) < min(N_PER, len(sub)):  # top up from majority class
            rest = sub[~sub.item_id.isin(got.item_id)]
            k = min(N_PER - len(got), len(rest))
            got = pd.concat([got, rest.iloc[np.sort(rng.choice(len(rest), k, replace=False))]])
        parts.append(got)
        print(f"  {comp}: {len(got)} items, supported {got.gold_supported.mean():.2f}")
    man = pd.concat(parts).reset_index(drop=True)
    man.to_parquet(path)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    man = build_manifest()
    items = manifest_to_items(man)
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    refs = {s.base_id: s for s in bank.specs}
    cache = VoteCache(OUT / "votes/aggrefact")
    print(f"aggrefact screen: {len(items)} items x 10 judges; live={args.live}")
    if not args.live:
        return
    for m in sorted(refs):
        for st in ("pw_direct", "pw_analysis"):
            spec = JudgeSpec(base_id=m, family=refs[m].family, scale=refs[m].scale,
                             prompt_style=st, hf_model=refs[m].hf_model,
                             quantization="nf4", dtype="bfloat16",
                             trust_remote_code=refs[m].trust_remote_code)
            done = cache.cached_item_ids(spec.logical_id)
            todo = [it for it in items if it.item_id not in done]
            if not todo:
                print(f"  {spec.logical_id}: cached"); continue
            tmpl = POINTWISE_TEMPLATES[st]
            j = HFJudge(spec, prompt_template=tmpl, batch_size=4,
                        max_new_tokens=tmpl.max_new_tokens,
                        shuffle_position=False, max_length=4096)
            t0 = time.time(); j.load()
            votes = j.vote_batch(todo); j.unload()
            cache.write(spec.logical_id, votes)
            v = np.array([x.vote for x in votes])
            print(f"  {spec.logical_id}: {len(votes)} in {time.time()-t0:.0f}s, "
                  f"abstain {(v==-1).mean():.3f}", flush=True)
    print("aggrefact collection complete")


if __name__ == "__main__":
    main()

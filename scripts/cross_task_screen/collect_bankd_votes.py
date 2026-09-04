#!/usr/bin/env python
"""Full bankD vote collection for the cross-task routing benchmark.

Runs ONLY if the NEI mechanism pilot validated (checks nei_pilot_result.csv).
Collects: (a) pw_decompose + pw_contradict votes for the 5 open models on the
1,000-item VitaminC main manifest (two-class pool; ~800 uncached items per
judge after the 200-item bank-screen sample); (b) 150 additional NEI items
(total mechanism pool 400). Checkpointed; est. 8-11 GPU-h; $0 API.

Usage: python scripts/cross_task_screen/collect_bankd_votes.py --live
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data.generic_task import manifest_to_items
from corrfilter.judges import load_bank_config
from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.hf_judge import HFJudge
from corrfilter.judges.prompts_pointwise import POINTWISE_TEMPLATES
from corrfilter.voting import VoteCache

OUT = ROOT / "outputs/factuality_bank"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    res = pd.read_csv(OUT / "nei_pilot_result.csv").iloc[0]
    if "VALIDATED" not in str(res.verdict):
        raise SystemExit(f"mechanism not validated ({res.verdict}); refusing full collection")

    # extend NEI pool to 400 deterministically (superset sample, same seed chain)
    nei_pilot = pd.read_parquet(OUT / "manifest_nei_pilot.parquet")
    ext_path = OUT / "manifest_nei_full.parquet"
    if ext_path.exists():
        nei_full = pd.read_parquet(ext_path)
    else:
        from datasets import load_dataset
        from corrfilter.data.generic_task import _item_hash
        df = load_dataset("tals/vitaminc", split="test").to_pandas()
        df = df[(df.label == "NOT ENOUGH INFO") & (df.evidence.str.len() <= 2000)].copy()
        df["item_hash"] = [_item_hash(d, c) for d, c in zip(df.evidence, df.claim)]
        df = df.drop_duplicates("item_hash")
        df = df[~df.item_hash.isin(set(nei_pilot.item_hash))].reset_index(drop=True)
        rng = np.random.default_rng(20270402)
        take = df.iloc[np.sort(rng.choice(len(df), 150, replace=False))]
        extra = pd.DataFrame({"item_id": "nei_" + take.item_hash,
                              "document": take.evidence, "claim": take.claim,
                              "gold_supported": 0, "source_dataset": "vitaminc-nei",
                              "doc_chars": take.evidence.str.len(),
                              "claim_chars": take.claim.str.len(),
                              "item_hash": take.item_hash})
        nei_full = pd.concat([nei_pilot, extra]).reset_index(drop=True)
        nei_full.to_parquet(ext_path)

    main_man = pd.read_parquet(OUT / "manifest_main.parquet")
    jobs = [(manifest_to_items(main_man), VoteCache(OUT / "votes"), "main-1000"),
            (manifest_to_items(nei_full), VoteCache(ROOT / "outputs/screening_bank/votes/nei"), "nei-400")]
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    refs = {s.base_id: s for s in bank.specs}
    print("full bankD collection plan:",
          [(tag, len(items)) for items, _, tag in jobs], "| live =", args.live)
    if not args.live:
        return
    for items, cache, tag in jobs:
        for m in sorted(refs):
            for st in ("pw_decompose", "pw_contradict"):
                spec = JudgeSpec(base_id=m, family=refs[m].family, scale=refs[m].scale,
                                 prompt_style=st, hf_model=refs[m].hf_model,
                                 quantization="nf4", dtype="bfloat16",
                                 trust_remote_code=refs[m].trust_remote_code)
                done = cache.cached_item_ids(spec.logical_id)
                todo = [it for it in items if it.item_id not in done]
                if not todo:
                    print(f"  [{tag}] {spec.logical_id}: cached"); continue
                tmpl = POINTWISE_TEMPLATES[st]
                j = HFJudge(spec, prompt_template=tmpl, batch_size=4,
                            max_new_tokens=tmpl.max_new_tokens,
                            shuffle_position=False, max_length=4096)
                t0 = time.time(); j.load()
                votes = j.vote_batch(todo); j.unload()
                cache.write(spec.logical_id, votes)
                v = np.array([x.vote for x in votes])
                print(f"  [{tag}] {spec.logical_id}: {len(votes)} in {time.time()-t0:.0f}s, "
                      f"abstain {(v==-1).mean():.3f}", flush=True)
    print("collection complete")


if __name__ == "__main__":
    main()

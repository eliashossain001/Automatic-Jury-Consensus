#!/usr/bin/env python
"""NEI mechanism pilot (preregistered: SUBGROUP_MECHANISM_PREREG.md).

Collects bankD votes (5 models x pw_decompose/pw_contradict) on 250
NOT-ENOUGH-INFO VitaminC items (gold_supported=0 per the preregistered
semantics) and reports H-M1..H-M3 with the stop rule.

Usage: python scripts/cross_task_screen/run_nei_mechanism_pilot.py [--live] [--n 250]
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data.generic_task import _item_hash, manifest_to_items
from corrfilter.judges import load_bank_config
from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.hf_judge import HFJudge
from corrfilter.judges.prompts_pointwise import POINTWISE_TEMPLATES
from corrfilter.voting import VoteCache

OUT = ROOT / "outputs/factuality_bank"
SEED = 20270401


def nei_manifest(n):
    path = OUT / "manifest_nei_pilot.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from datasets import load_dataset
    df = load_dataset("tals/vitaminc", split="test").to_pandas()
    df = df[(df.label == "NOT ENOUGH INFO") & (df.evidence.str.len() <= 2000)].copy()
    df["item_hash"] = [_item_hash(d, c) for d, c in zip(df.evidence, df.claim)]
    df = df.drop_duplicates("item_hash").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    take = df.iloc[np.sort(rng.choice(len(df), n, replace=False))].reset_index(drop=True)
    man = pd.DataFrame({"item_id": "nei_" + take.item_hash,
                        "document": take.evidence, "claim": take.claim,
                        "gold_supported": 0, "source_dataset": "vitaminc-nei",
                        "doc_chars": take.evidence.str.len(),
                        "claim_chars": take.claim.str.len(),
                        "item_hash": take.item_hash})
    man.to_parquet(path)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()
    man = nei_manifest(args.n)
    items = manifest_to_items(man)
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    refs = {s.base_id: s for s in bank.specs}
    models = sorted(refs)
    cache = VoteCache(ROOT / "outputs/screening_bank/votes/nei")
    print(f"NEI pilot: {len(items)} items x 10 bankD judges; live={args.live}")
    if not args.live:
        return
    for m in models:
        for st in ("pw_decompose", "pw_contradict"):
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
            print(f"  {spec.logical_id}: {len(votes)} votes in {time.time()-t0:.0f}s, "
                  f"abstain {(v==-1).mean():.3f}, false-affirm {(v==1).mean():.3f}", flush=True)

    # analysis: H-M1..H-M3
    from corrfilter.data.generic_task import batch_from_cache
    lids = [f"{m}::{st}" for m in models for st in ("pw_decompose", "pw_contradict")]
    b = batch_from_cache(man, cache, lids)
    V, M = b.judge_votes, b.availability
    dec = np.array([("pw_decompose" in l) for l in lids])
    con = ~dec
    fa = lambda g: float(np.where(M[:, g], V[:, g], 0).sum() / max(M[:, g].sum(), 1))
    fa_d, fa_c = fa(dec), fa(con)
    ab_d = 1 - M[:, dec].mean(); ab_c = 1 - M[:, con].mean()
    cmaj = np.where(M[:, con], V[:, con], 0).sum(1) > (M[:, con].sum(1) / 2)
    dmaj = np.where(M[:, dec], V[:, dec], 0).sum(1) > (M[:, dec].sum(1) / 2)
    split = float((cmaj & ~dmaj).mean())
    E = np.where(M.astype(bool), V, 0).astype(float)  # wrong = affirm (gold 0)
    from corrfilter.correlation import correlation_shrunk
    R, _ = correlation_shrunk(E)
    within = np.mean([R[i, j] for i in np.where(con)[0] for j in np.where(con)[0] if i < j])
    cross = np.mean([R[i, j] for i in np.where(con)[0] for j in np.where(dec)[0]])
    hm1 = (fa_c - fa_d) >= 0.15
    hm2 = within > cross
    hm3 = split >= 0.25
    abst_ok = abs(ab_c - ab_d) < 0.05
    print(f"\nfalse-affirm: contradict {fa_c:.3f} vs decompose {fa_d:.3f} (gap {fa_c-fa_d:+.3f})")
    print(f"abstention: contradict {ab_c:.3f} vs decompose {ab_d:.3f}")
    print(f"H-M1 (gap>=.15): {hm1} | H-M2 (within {within:.3f} > cross {cross:.3f}): {hm2} "
          f"| H-M3 (split-majority {split:.3f} >= .25): {hm3} | abstention-clean: {abst_ok}")
    verdict = "MECHANISM VALIDATED" if (hm1 and hm2 and hm3 and abst_ok) else "MECHANISM REJECTED (stop rule)"
    print(verdict)
    pd.DataFrame([{"fa_contradict": fa_c, "fa_decompose": fa_d, "gap": fa_c-fa_d,
                   "ab_c": ab_c, "ab_d": ab_d, "within": within, "cross": cross,
                   "split_majority": split, "HM1": hm1, "HM2": hm2, "HM3": hm3,
                   "abstention_clean": abst_ok, "verdict": verdict}]).to_csv(
        OUT / "nei_pilot_result.csv", index=False)


if __name__ == "__main__":
    main()

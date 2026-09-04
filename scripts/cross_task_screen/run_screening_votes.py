#!/usr/bin/env python
"""GPU vote collection for the routing-feasibility screening campaign.

Priority order (per instruction): factuality sources -> alternative banks ->
code pointwise -> code pairwise. Each phase is checkpointed via the vote
cache; the script can be re-run and skips cached work. $0 API.

Phases:
  halueval   HaluEval-QA pointwise, 200 items, base bank (5 models x
             pw_direct/pw_analysis)
  bankD      VitaminC bank-screen sample (200 items), 5 models x
             {pw_decompose, pw_contradict} (reasoning-mode diversity)
  bankB      same sample, {qwen-2.5-14b, qwen-2.5-1.5b} x
             {pw_direct, pw_analysis} (capability-tier diversity)
  cjb_point  CodeJudgeBench codegen pointwise, 50 pairs -> 100 items,
             5 models x {pwc_direct, pwc_trace}
  cjb_pair   same 50 pairs pairwise, 5 models x {pairwise, likert}

Usage: python scripts/cross_task_screen/run_screening_votes.py [--phases halueval,bankD,...] --live
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/router_upgrade"))

from screening_sources import (  # noqa: E402
    build_cjb_pairwise_items, build_cjb_pointwise_manifest, build_halueval_manifest)

from corrfilter.data.generic_task import manifest_to_items  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.judges.base import JudgeSpec  # noqa: E402
from corrfilter.judges.hf_judge import HFJudge  # noqa: E402
from corrfilter.judges.prompts_pointwise import CODE_TEMPLATES, POINTWISE_TEMPLATES  # noqa: E402

OUT = ROOT / "outputs/screening_bank"
VITC_SAMPLE_SEED = 20270220
EXTENDED = {"qwen-2.5-14b": ("qwen", "14B", "Qwen/Qwen2.5-14B-Instruct"),
            "qwen-2.5-1.5b": ("qwen", "1.5B", "Qwen/Qwen2.5-1.5B-Instruct")}


def vitc_bank_sample() -> pd.DataFrame:
    path = OUT / "manifest_vitc_bankscreen.parquet"
    if path.exists():
        return pd.read_parquet(path)
    man = pd.read_parquet(ROOT / "outputs/factuality_bank/manifest_main.parquet")
    rng = np.random.default_rng(VITC_SAMPLE_SEED)
    take = man.iloc[np.sort(rng.choice(len(man), 200, replace=False))].reset_index(drop=True)
    take.to_parquet(path)
    return take


def run_bank(tag, items, spec_rows, cache_dir, batch_size=4):
    from corrfilter.voting import VoteCache

    cache = VoteCache(cache_dir)
    log = []
    for spec, tmpl in spec_rows:
        done = cache.cached_item_ids(spec.logical_id)
        todo = [it for it in items if it.item_id not in done]
        if not todo:
            print(f"  [{tag}] {spec.logical_id}: cached")
            continue
        judge = HFJudge(spec, prompt_template=tmpl, batch_size=batch_size,
                        max_new_tokens=getattr(tmpl, "max_new_tokens", 8),
                        shuffle_position=(tmpl is None), max_length=4096)
        t0 = time.time()
        judge.load()
        votes = judge.vote_batch(todo)
        judge.unload()
        cache.write(spec.logical_id, votes)
        v = np.array([x.vote for x in votes])
        print(f"  [{tag}] {spec.logical_id}: {len(votes)} votes in "
              f"{time.time()-t0:.0f}s, abstain {(v==-1).mean():.3f}", flush=True)
        log.append({"phase": tag, "judge": spec.logical_id, "n": len(votes),
                    "abstain": float((v == -1).mean()), "sec": round(time.time() - t0)})
    return log


def base_model_specs():
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    return {s.base_id: s for s in bank.specs}


def spec_for(base_id, family, scale, hf_model, style, ref=None):
    return JudgeSpec(base_id=base_id, family=family, scale=scale, prompt_style=style,
                     hf_model=hf_model,
                     hf_revision=getattr(ref, "hf_revision", None),
                     quantization="nf4", dtype="bfloat16",
                     trust_remote_code=bool(getattr(ref, "trust_remote_code", False)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="halueval,bankD,bankB,cjb_point,cjb_pair")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    phases = args.phases.split(",")
    refs = base_model_specs()
    models = sorted(refs)
    all_log = []

    if "halueval" in phases:
        man = build_halueval_manifest(200)
        man.to_parquet(OUT / "manifest_halueval.parquet")
        items = manifest_to_items(man)
        rows = [(spec_for(m, refs[m].family, refs[m].scale, refs[m].hf_model, st, refs[m]),
                 POINTWISE_TEMPLATES[st])
                for m in models for st in ("pw_direct", "pw_analysis")]
        print(f"phase halueval: {len(items)} items x {len(rows)} judges")
        if args.live:
            all_log += run_bank("halueval", items, rows, OUT / "votes/halueval")

    if "bankD" in phases:
        man = vitc_bank_sample()
        items = manifest_to_items(man)
        rows = [(spec_for(m, refs[m].family, refs[m].scale, refs[m].hf_model, st, refs[m]),
                 POINTWISE_TEMPLATES[st])
                for m in models for st in ("pw_decompose", "pw_contradict")]
        print(f"phase bankD: {len(items)} items x {len(rows)} judges")
        if args.live:
            all_log += run_bank("bankD", items, rows, OUT / "votes/vitc_extra")

    if "bankB" in phases:
        man = vitc_bank_sample()
        items = manifest_to_items(man)
        rows = [(spec_for(mid, fam, sc, hf, st), POINTWISE_TEMPLATES[st])
                for mid, (fam, sc, hf) in EXTENDED.items()
                for st in ("pw_direct", "pw_analysis")]
        print(f"phase bankB: {len(items)} items x {len(rows)} judges")
        if args.live:
            all_log += run_bank("bankB", items, rows, OUT / "votes/vitc_extra",
                                batch_size=2)

    if "cjb_point" in phases:
        man = build_cjb_pointwise_manifest(50)
        man.to_parquet(OUT / "manifest_cjb_point.parquet")
        items = manifest_to_items(man)
        rows = [(spec_for(m, refs[m].family, refs[m].scale, refs[m].hf_model, st, refs[m]),
                 CODE_TEMPLATES[st])
                for m in models for st in ("pwc_direct", "pwc_trace")]
        print(f"phase cjb_point: {len(items)} items x {len(rows)} judges")
        if args.live:
            all_log += run_bank("cjb_point", items, rows, OUT / "votes/cjb_point",
                                batch_size=2)

    if "cjb_pair" in phases:
        items, meta = build_cjb_pairwise_items(50)
        meta.to_parquet(OUT / "manifest_cjb_pair.parquet")
        rows = [(spec_for(m, refs[m].family, refs[m].scale, refs[m].hf_model, st, refs[m]),
                 None)  # None -> default pairwise/likert template by style
                for m in models for st in ("pairwise", "likert")]
        print(f"phase cjb_pair: {len(items)} pairs x {len(rows)} judges")
        if args.live:
            all_log += run_bank("cjb_pair", items, rows, OUT / "votes/cjb_pair",
                                batch_size=2)

    pd.DataFrame(all_log).to_csv(OUT / "screening_runlog.csv", index=False)
    print("campaign complete" if args.live else "dry-run complete")


if __name__ == "__main__":
    main()

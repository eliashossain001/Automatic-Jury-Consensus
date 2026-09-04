#!/usr/bin/env python
"""P0-1: run the base / GRPO / DPO banks under forced-choice verdict scoring.

Loads each base model ONCE (4-bit nf4) and sweeps its three adapter states (none, GRPO,
DPO) x two prompt styles, so nine logical-bank columns come from three model loads. Votes
are written to a separate cache so no published cache is overwritten. Resumable: a
(bank, logical_id) whose parquet already covers every requested item is skipped.

Usage:
  python scripts/53_forced_choice_bank.py                          # all models, all banks
  python scripts/53_forced_choice_bank.py --base-id qwen-2.5-7b    # one model
  python scripts/53_forced_choice_bank.py --max-items 40 --smoke   # quick check

See experiments/abstention_control/PREREG.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges.base import JudgeSpec  # noqa: E402
from corrfilter.judges.forced_choice import ForcedChoiceJudge  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("p0-1")

MANIFEST = ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet"
OUT = ROOT / "outputs/forced_choice/votes"

MODELS = {
    "qwen-2.5-7b":      dict(hf_model="Qwen/Qwen2.5-7B-Instruct",           family="qwen",    scale="7B"),
    "mistral-7b-v0.3":  dict(hf_model="mistralai/Mistral-7B-Instruct-v0.3", family="mistral", scale="7B"),
    "phi-3.5-mini":     dict(hf_model="microsoft/Phi-3.5-mini-instruct",    family="phi",     scale="3.8B"),
}

# Adapter directories per (bank, base_id). None => plain base weights.
ADAPTERS = {
    "base": {b: None for b in MODELS},
    "grpo": {
        "qwen-2.5-7b":     "results/grpo_judges/adapters/qwen-2.5-7b_seed0",
        "mistral-7b-v0.3": "results/grpo_judges/adapters/mistral-7b-v0.3_seed0",
        "phi-3.5-mini":    "results/grpo_judges/adapters/phi-3.5-mini_seed0",
    },
    "dpo": {
        "qwen-2.5-7b":     "results/strengthening_phase/dpo_judges/adapters/qwen-2.5-7b_seed0",
        "mistral-7b-v0.3": "results/strengthening_phase/dpo_judges/adapters/mistral-7b-v0.3_seed0",
        "phi-3.5-mini":    "results/strengthening_phase/dpo_judges/adapters/phi-3.5-mini_seed0",
    },
}
STYLES = ("pairwise", "likert")


def cache_for(bank: str) -> VoteCache:
    return VoteCache(OUT / bank)


def published_positions(base_id: str, style: str) -> dict[str, bool]:
    """Slot bits from the PUBLISHED base-bank cache for this (model, style).

    Applied to all three banks so base / GRPO / DPO see identical items in identical
    slots, and so the forced-choice-vs-cached-base validation is like-for-like. See
    ForcedChoiceJudge.__init__ for why this cannot be recomputed from hash().
    """
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    df = VoteCache(bank.votes_dir).load(f"{base_id}::{style}")
    if df is None:
        raise FileNotFoundError(f"no published base cache for {base_id}::{style}")
    df = df.drop_duplicates(subset="item_id", keep="first")
    return {str(r.item_id): bool(r.position_swapped) for r in df.itertuples()}


def already_done(bank: str, logical_id: str, item_ids: list[str]) -> bool:
    have = cache_for(bank).cached_item_ids(logical_id)
    return set(item_ids).issubset(have)


def run_model(base_id: str, items, item_ids, args) -> list[dict]:
    """Load one base model, sweep its adapter states and prompt styles."""
    meta = MODELS[base_id]
    rows = []
    judge = None
    for bank in args.banks:
        adapter = ADAPTERS[bank][base_id]
        if adapter is not None and not (ROOT / adapter).exists():
            log.warning("adapter missing, skipping %s/%s: %s", bank, base_id, adapter)
            continue
        for style in STYLES:
            spec = JudgeSpec(
                base_id=base_id if bank == "base" else f"{base_id}-{bank}",
                family=meta["family"], scale=meta["scale"], prompt_style=style,
                hf_model=meta["hf_model"], quantization="nf4", dtype="bfloat16",
                adapter_path=str(ROOT / adapter) if adapter else None,
                trust_remote_code=False,
            )
            lid = spec.logical_id
            if not args.force and already_done(bank, lid, item_ids):
                log.info("cached, skipping %s / %s", bank, lid)
                continue
            judge = ForcedChoiceJudge(
                spec, batch_size=args.batch_size, device="cuda",
                position_seed=20260601, shuffle_position=True, max_length=args.max_length,
                position_override=published_positions(base_id, style),
            )
            t0 = time.time()
            log.info("loading %s / %s", bank, lid)
            judge.load()
            votes = judge.vote_batch(items)
            cache_for(bank).write(lid, votes)
            dt = time.time() - t0
            rows.append({
                "bank": bank, "logical_id": lid, "base_id": base_id, "style": style,
                "n_items": len(votes), "n_exact_ties": judge.stats.n_exact_ties,
                "n_degenerate": judge.stats.n_degenerate,
                "mean_mass": round(judge.stats.mean_mass(), 6),
                "max_length": args.max_length,
                "positions": "pinned_to_published_base_cache",
                "mean_margin": round(judge.stats.mean_margin(), 6),
                "seconds": round(dt, 1),
            })
            log.info("done %s / %s in %.1fs (ties=%d, degenerate=%d, mean mass=%.4g)",
                     bank, lid, dt, judge.stats.n_exact_ties, judge.stats.n_degenerate,
                     judge.stats.mean_mass())
            judge.unload()
            judge = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-id", action="append", default=None,
                    help="restrict to these base models (repeatable)")
    ap.add_argument("--banks", nargs="+", default=["base", "grpo", "dpo"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=4096,
                    help="4096 = no truncation on this manifest; see PREREG Deviation 1")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="write to a scratch cache")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global OUT
    if args.smoke:
        OUT = ROOT / "outputs/forced_choice/smoke_votes"

    items = load_calibration_set(MANIFEST)
    if args.max_items:
        items = items[: args.max_items]
    item_ids = [str(it.item_id) for it in items]
    log.info("items: %d", len(items))

    targets = args.base_id or list(MODELS)
    rows = []
    for base_id in targets:
        rows += run_model(base_id, items, item_ids, args)

    if rows:
        run_log = OUT.parent / "run_log.jsonl"
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        log.info("appended %d rows -> %s", len(rows), run_log)


if __name__ == "__main__":
    main()

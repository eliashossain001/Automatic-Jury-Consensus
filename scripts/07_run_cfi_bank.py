"""Run the CFI bias bank with real GPU inference (Bucket 3, Part A).

Generates biased judge votes by re-running each base judge under a
prompt-level bias injection, for every non-clean mechanism in
``configs/cfi_bias_prompts.yaml``. The clean control reuses the cached H1
votes (no inference).

Efficiency: a biased judge's votes depend only on (base model, prompt method,
mechanism) — not on the bank ratio — so we load each of the five base models
exactly once and, while it is resident, sweep both prompt methods × all biased
mechanisms. The bank ratio conditions (all_clean / mixed_25/50/75 / all_biased)
are assembled later, at analysis time, by selecting biased vs clean votes per
judge (scripts/08_analyze_cfi.py). Votes are cached per mechanism so reruns
are free and the job is resumable.

Outputs:
    outputs/cfi/votes/<mechanism>/<logical_id>.parquet   (biased votes)
    outputs/cfi/votes/cfi_run_summary.json

Usage:
    export CUDA_VISIBLE_DEVICES=0
    python scripts/07_run_cfi_bank.py --config configs/cfi_bias_prompts.yaml
    # smoke test:
    python scripts/07_run_cfi_bank.py --max-items 24 --mechanism verbosity_bias \
        --base-id phi-3.5-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from corrfilter.cfi.bias_bank import load_bias_bank
from corrfilter.cfi.prompts import BiasedPrompt
from corrfilter.cfi.replay import variant_position_swap
from corrfilter.data import load_calibration_set
from corrfilter.judges import load_bank_config
from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.prompts import get_prompt_template
from corrfilter.voting import VoteCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.07_run_cfi_bank")


def _bank_entries(bank_cfg) -> list[dict]:
    """One entry per base model (deduplicated across prompt styles)."""
    seen: dict[str, JudgeSpec] = {}
    for spec in bank_cfg.specs:
        seen.setdefault(spec.base_id, spec)
    return [{"base_id": bid, "spec": spec} for bid, spec in seen.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cfi_bias_prompts.yaml")
    parser.add_argument("--mechanism", action="append", default=None,
                        help="subset of biased mechanisms (repeatable)")
    parser.add_argument("--base-id", action="append", default=None,
                        help="subset of base models, e.g. phi-3.5-mini (repeatable)")
    parser.add_argument("--method", action="append", default=None,
                        help="subset of prompt methods: pairwise / likert")
    parser.add_argument("--max-items", type=int, default=None,
                        help="cap items for a smoke test")
    parser.add_argument("--subset", action="append", default=None,
                        help="restrict to RewardBench subsets (repeatable)")
    parser.add_argument("--items-file", default=None,
                        help="newline-separated item_ids to restrict to (stratified subset)")
    parser.add_argument("--triggered-only", action="store_true",
                        help="only run inference on items where the mechanism triggers; "
                             "non-triggered items fall back to the clean H1 vote at analysis time")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    bank = load_bias_bank(args.config)
    seed = bank.seed
    rt = cfg["runtime"]
    votes_root = Path(rt["votes_dir"])
    votes_root.mkdir(parents=True, exist_ok=True)

    cal_cfg = yaml.safe_load(Path(cfg["sources"]["h1_calibration_config"]).read_text())
    items = load_calibration_set(cal_cfg["output"]["manifest_path"])
    if args.subset:
        keep = set(args.subset)
        items = [it for it in items if it.subset in keep]
    if args.items_file:
        wanted = {ln.strip() for ln in Path(args.items_file).read_text().splitlines() if ln.strip()}
        items = [it for it in items if it.item_id in wanted]
    if args.max_items:
        items = items[: args.max_items]
    logger.info("loaded %d calibration items", len(items))

    bank_cfg = load_bank_config(cfg["sources"]["h1_bank_config"])
    entries = _bank_entries(bank_cfg)
    if args.base_id:
        keep_b = set(args.base_id)
        entries = [e for e in entries if e["base_id"] in keep_b]

    methods = args.method or bank_cfg_styles(cfg)
    biased = bank.biased_conditions
    if args.mechanism:
        keep_m = set(args.mechanism)
        biased = [c for c in biased if c.name in keep_m]

    logger.info(
        "real CFI run: %d base models × %d methods × %d biased mechanisms over %d items%s",
        len(entries), len(methods), len(biased), len(items),
        " (triggered-only)" if args.triggered_only else "",
    )

    from corrfilter.judges.hf_judge import HFJudge

    summary = {
        "mode": "real",
        "n_items": len(items),
        "methods": methods,
        "mechanisms": [c.name for c in biased],
        "base_ids": [e["base_id"] for e in entries],
        "triggered_only": bool(args.triggered_only),
        "passes": [],
    }

    for entry in entries:
        base_id = entry["base_id"]
        base_spec: JudgeSpec = entry["spec"]
        judge = HFJudge(
            spec=base_spec,
            batch_size=bank_cfg.batch_size,
            max_new_tokens=bank_cfg.max_new_tokens,
            device=rt.get("device", "cuda"),
            position_seed=int(rt.get("position_seed", seed)),
            shuffle_position=bool(rt.get("shuffle_position", True)),
        )
        t_load = time.perf_counter()
        judge.load()
        logger.info("loaded %s in %.1fs", base_id, time.perf_counter() - t_load)
        try:
            for method in methods:
                base_template = get_prompt_template(method)
                logical_id = f"{base_id}::{method}"
                for cond in biased:
                    cache = VoteCache(votes_root / cond.name)
                    already = cache.cached_item_ids(logical_id)
                    pending = [it for it in items if it.item_id not in already]
                    if args.triggered_only:
                        pending = [it for it in pending if cond.fires(it)]
                    if not pending:
                        logger.info("  %s / %s fully cached, skipping", logical_id, cond.name)
                        continue
                    # Shared per-(mechanism, item) position swap so biased judges
                    # see the same A/B layout (correlated position bias).
                    swaps = {
                        it.item_id: variant_position_swap(cond.name, it.item_id, seed)
                        for it in pending
                    }
                    judge.prompt = BiasedPrompt.from_instruction(
                        base_template, cond.name, cond.instruction
                    )
                    judge._position_swap = lambda iid, _s=swaps: _s.get(iid, False)  # type: ignore[method-assign]
                    t0 = time.perf_counter()
                    n_done = 0
                    for start in range(0, len(pending), judge.batch_size):
                        batch = pending[start : start + judge.batch_size]
                        votes = judge.vote_batch(batch)
                        cache.write(logical_id, votes)
                        n_done += len(votes)
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "  %s / %s: %d votes in %.1fs (%.2fs/item)",
                        logical_id, cond.name, n_done, elapsed,
                        elapsed / max(n_done, 1),
                    )
                    summary["passes"].append({
                        "logical_id": logical_id,
                        "mechanism": cond.name,
                        "n_votes": n_done,
                        "seconds": round(elapsed, 1),
                    })
        finally:
            judge.unload()
            logger.info("unloaded %s", base_id)

    summary_path = votes_root / "cfi_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote run summary → %s (%d passes)", summary_path, len(summary["passes"]))


def bank_cfg_styles(cfg: dict) -> list[str]:
    """Prompt methods from the H1 bank config referenced by the CFI config."""
    h1 = yaml.safe_load(Path(cfg["sources"]["h1_bank_config"]).read_text())
    return list(h1["prompt_styles"])


if __name__ == "__main__":
    main()

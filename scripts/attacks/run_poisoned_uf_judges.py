"""Run the clean judge bank on the poisoned-UltraFeedback pairs (real inference).

Votes are content-based: each judge compares response_a (truly better) vs
response_b (truly worse) and we record vote = 1 iff the judge prefers the truly
better response. This is independent of the corruption labels, so one vote cache
serves all corruption rates (corruption is applied at evaluation time, scripts/attacks/eval_poisoned_uf.py).

Efficiency mirrors scripts/cfi/run_bank_gpu.py: load each of the five base models once and sweep
both prompt methods while resident. Votes cache to a single VoteCache dir and the
run is resumable (re-running skips cached judge/item pairs).

Outputs: outputs/synthetic_poisoned_ultrafeedback/judge_votes/<logical_id>.parquet
         + uf_run_summary.json

Usage:
    export CUDA_VISIBLE_DEVICES=0
    python scripts/attacks/run_poisoned_uf_judges.py --project-root .
    # smoke: python scripts/attacks/run_poisoned_uf_judges.py --base-id phi-3.5-mini --max-items 24
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from corrfilter.data import CalibrationItem
from corrfilter.judges import load_bank_config
from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.prompts import get_prompt_template
from corrfilter.voting import VoteCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.13_run_uf_judges")


def _load_items(manifest_path: Path, max_items: int | None) -> list[CalibrationItem]:
    df = pd.read_csv(manifest_path)
    if max_items:
        df = df.head(max_items)
    return [
        CalibrationItem(
            item_id=str(r["item_id"]), prompt=str(r["prompt"]),
            chosen=str(r["response_a"]), rejected=str(r["response_b"]),
            subset="uf", category="uf", gold_label=1,
        )
        for _, r in df.iterrows()
    ]


def _bank_entries(bank_cfg):
    seen: dict[str, JudgeSpec] = {}
    for spec in bank_cfg.specs:
        seen.setdefault(spec.base_id, spec)
    return list(seen.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--manifest", default="outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    ap.add_argument("--out-dir", default="outputs/synthetic_poisoned_ultrafeedback/judge_votes")
    ap.add_argument("--base-id", action="append", default=None)
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    items = _load_items(_p(args.manifest), args.max_items)
    logger.info("loaded %d UF pairs", len(items))

    bank_cfg = load_bank_config(str(_p(args.bank)))
    entries = _bank_entries(bank_cfg)
    if args.base_id:
        keep = set(args.base_id)
        entries = [e for e in entries if e[0] in keep]
    methods = args.method or _bank_styles(_p(args.bank))
    votes_dir = _p(args.out_dir)
    votes_dir.mkdir(parents=True, exist_ok=True)
    cache = VoteCache(votes_dir)

    from corrfilter.judges.hf_judge import HFJudge

    summary = {"n_items": len(items), "methods": methods,
               "base_ids": [b for b, _ in entries], "passes": []}

    for base_id, base_spec in entries:
        judge = HFJudge(
            spec=base_spec, batch_size=bank_cfg.batch_size,
            max_new_tokens=bank_cfg.max_new_tokens, device=bank_cfg.device,
            position_seed=bank_cfg.position_seed, shuffle_position=bank_cfg.shuffle_position,
        )
        t_load = time.perf_counter()
        judge.load()
        logger.info("loaded %s in %.1fs", base_id, time.perf_counter() - t_load)
        try:
            for method in methods:
                logical_id = f"{base_id}::{method}"
                judge.prompt = get_prompt_template(method)
                already = cache.cached_item_ids(logical_id)
                pending = [it for it in items if it.item_id not in already]
                if not pending:
                    logger.info("  %s fully cached, skip", logical_id)
                    continue
                t0 = time.perf_counter()
                n_done = 0
                for start in range(0, len(pending), judge.batch_size):
                    batch = pending[start: start + judge.batch_size]
                    votes = judge.vote_batch(batch)
                    # vote_batch tags votes with the judge's own spec.logical_id;
                    # write under the method-correct logical_id explicitly.
                    cache.write(logical_id, votes)
                    n_done += len(votes)
                dt = time.perf_counter() - t0
                logger.info("  %s: %d votes in %.1fs (%.2fs/item)", logical_id, n_done, dt, dt / max(n_done, 1))
                summary["passes"].append({"logical_id": logical_id, "n_votes": n_done, "seconds": round(dt, 1)})
        finally:
            judge.unload()
            logger.info("unloaded %s", base_id)

    (votes_dir / "uf_run_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote summary (%d passes)", len(summary["passes"]))


def _bank_styles(bank_path: Path) -> list[str]:
    import yaml
    return list(yaml.safe_load(bank_path.read_text())["prompt_styles"])


if __name__ == "__main__":
    main()

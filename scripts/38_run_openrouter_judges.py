#!/usr/bin/env python
"""Run the OpenRouter multi-provider frontier bank (OpenAI / Anthropic / xAI).

Mirror of scripts/33_run_gemini_judges.py for OpenRouter, writing to the fully
separate outputs/openrouter_bank/ tree. Safe by default (dry-run); ``--live``
makes paid calls; ``--smoke-bank`` targets the free-tier pipeline-validation
model instead of the paid bank. A 402 (insufficient credits) aborts the run
immediately and is never retried.

Usage:
  python scripts/38_run_openrouter_judges.py --dataset rewardbench --limit 20 \
      [--live] [--smoke-bank] [--judge gpt-5.6-sol] [--workers 6]
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.judges.base import JudgeSpec  # noqa: E402
from corrfilter.judges.openrouter_judge import KEY_ENV_VARS, OpenRouterJudge  # noqa: E402
from corrfilter.voting.cache import VoteCache  # noqa: E402

# Reuse item loading + .env parsing from the Gemini runner (same datasets/splits).
_spec33 = importlib.util.spec_from_file_location("run_gemini", ROOT / "scripts" / "33_run_gemini_judges.py")
_s33 = importlib.util.module_from_spec(_spec33)
_spec33.loader.exec_module(_s33)


def load_dotenv_keys(env_path: Path) -> None:
    import os

    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if name in KEY_ENV_VARS and value:
            os.environ.setdefault(name, value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/judge_bank_openrouter.yaml")
    ap.add_argument("--dataset", choices=["rewardbench", "pku"], required=True)
    ap.add_argument("--live", action="store_true", help="make API calls (default: dry-run)")
    ap.add_argument("--smoke-bank", action="store_true",
                    help="use the free-tier smoke_bank entries instead of the paid bank")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--items-file", default=None,
                    help="path to an explicit item-id list overriding the dataset default")
    ap.add_argument("--judge", action="append", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()

    root = Path(args.project_root)
    cfg = yaml.safe_load((root / args.config).read_text())
    rt, gen = cfg["runtime"], cfg["generation"]
    out_root = root / cfg["output"]["root"]
    sub = "smoke" if args.smoke_bank else "frontier"
    votes_dir = out_root / "votes" / sub / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    if args.live:
        load_dotenv_keys(root / ".env")

    if args.items_file:
        import copy
        cfg_items = copy.deepcopy(cfg)
        cfg_items["datasets"][args.dataset]["items_file"] = args.items_file
        cfg_items["datasets"][args.dataset].pop("subsample", None)
        items = _s33.load_items(cfg_items, args.dataset, root)
    else:
        items = _s33.load_items(cfg, args.dataset, root)
    if args.limit:
        items = items[: args.limit]
    if args.limit is None and not args.items_file:
        (out_root / f"items_{args.dataset}.txt").write_text("\n".join(it.item_id for it in items) + "\n")
    print(f"[38] dataset={args.dataset} bank={sub} n_items={len(items)} live={args.live}")

    cache = VoteCache(votes_dir)
    checkpoint = int(rt.get("checkpoint_every", 25))
    run_records = []

    for entry in cfg["smoke_bank" if args.smoke_bank else "bank"]:
        if args.judge and entry["id"] not in args.judge:
            continue
        spec = JudgeSpec(base_id=entry["id"], family=entry["family"], scale=str(entry["scale"]),
                         prompt_style=cfg.get("prompt_style", "pairwise"), hf_model=entry["model"])
        judge = OpenRouterJudge(
            spec, model=entry["model"],
            reasoning_effort=entry.get("reasoning_effort"),
            reasoning_exclude=bool(entry.get("reasoning_exclude", False)),
            supports_temperature=bool(entry.get("supports_temperature", True)),
            max_output_tokens=int(entry.get("max_output_tokens", gen["max_output_tokens"])),
            temperature=float(gen["temperature"]),
            position_seed=int(rt["position_seed"]),
            shuffle_position=bool(rt["shuffle_position"]),
            dry_run=not args.live,
            price_in=float(entry.get("price_in", 0.0)),
            price_out=float(entry.get("price_out", 0.0)),
            max_retries=int(rt["max_retries"]),
            backoff_base_s=float(rt["backoff_base_s"]),
            backoff_max_s=float(rt["backoff_max_s"]),
            request_timeout_s=float(rt["request_timeout_s"]),
            workers=args.workers or int(rt["workers"]),
        )

        cached = cache.cached_item_ids(judge.logical_id)
        prior = cache.load(judge.logical_id)
        healed: set[str] = set()
        if prior is not None:
            bad = prior[prior["raw_response"].astype(str).str.startswith("<API_ERROR")]
            healed = set(bad["item_id"].astype(str))
        pending = [it for it in items if it.item_id not in cached or it.item_id in healed]
        print(f"[38] {judge.logical_id}: {len(pending)} pending ({len(cached)} cached)")
        if not pending:
            continue

        judge.load()
        t0 = time.perf_counter()
        all_failures: list[dict] = []
        try:
            for start in range(0, len(pending), checkpoint):
                block = pending[start : start + checkpoint]
                votes = judge.vote_batch(block)
                if args.live:
                    cache.write(judge.logical_id, votes)
                all_failures.extend(judge.failures)
                done = min(start + checkpoint, len(pending))
                print(f"[38]   {judge.logical_id}: {done}/{len(pending)} "
                      f"(abstain {judge.usage.abstentions}, cost ${judge.cost_so_far():.4f})")
        finally:
            judge.unload()
        elapsed = time.perf_counter() - t0

        run_records.append({
            "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "bank": sub,
            "bank_id": entry["id"],
            "model_id": entry["model"],
            "logical_id": judge.logical_id,
            "live": bool(args.live),
            "n_items_requested": len(pending),
            "settings": {
                "temperature": float(gen["temperature"]) if judge.supports_temperature else None,
                "max_output_tokens": judge.max_new_tokens,
                "reasoning_effort": entry.get("reasoning_effort"),
                "reasoning_exclude": bool(entry.get("reasoning_exclude", False)),
                "position_seed": int(rt["position_seed"]),
                "prompt_style": cfg.get("prompt_style", "pairwise"),
                "workers": judge.workers,
                "max_retries": judge.max_retries,
            },
            "usage": {
                "calls": judge.usage.calls,
                "input_tokens": judge.usage.input_tokens,
                "output_tokens": judge.usage.output_tokens,
                "reasoning_tokens": judge.reasoning_tokens,
                "abstentions": judge.usage.abstentions,
            },
            "providers_seen": judge.providers_seen,
            "price_per_1M_in_out": [judge.price_in, judge.price_out],
            "cost_usd": round(judge.cost_so_far(), 4),
            "seconds": round(elapsed, 1),
            "n_failures": len(all_failures),
            "failures": all_failures[:200],
        })

    log_path = out_root / f"run_log_{sub}_{args.dataset}.json"
    existing = json.loads(log_path.read_text()) if log_path.exists() else []
    existing.extend(run_records)
    log_path.write_text(json.dumps(existing, indent=2))
    total = sum(r["cost_usd"] for r in run_records)
    print(f"[38] wrote {log_path} | this invocation cost ${total:.4f} "
          f"({'LIVE' if args.live else 'dry-run estimate'})")


if __name__ == "__main__":
    main()

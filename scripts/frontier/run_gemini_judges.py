#!/usr/bin/env python
"""Run the Gemini frontier-judge pilot bank over the RQ1 item sets.

Safe by default: without ``--live`` every judge runs in dry-run mode (prompts
rendered, tokens estimated, zero paid calls). With ``--live`` the script reads
the API key from the environment or ./.env (never printed, never written to
outputs) and votes only on items missing from the parquet cache, checkpointing
every ``checkpoint_every`` items so partial failures never lose completed work.

Usage:
  python scripts/frontier/run_gemini_judges.py --config configs/judge_bank_gemini.yaml \
      --dataset rewardbench [--live] [--limit 5] [--judge gemini-3.6-flash] [--workers 6]

Outputs (under outputs/gemini_bank/):
  items_<dataset>.txt                 exact item ids evaluated (deterministic)
  votes/<dataset>/<judge>__pairwise.parquet   vote cache (item_id, vote, position_swapped, raw_response)
  run_log_<dataset>.json              per-run model IDs, settings, usage, cost, failures, coverage
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data import CalibrationItem  # noqa: E402
from corrfilter.judges.base import JudgeSpec  # noqa: E402
from corrfilter.judges.gemini_judge import KEY_ENV_VARS, GeminiJudge  # noqa: E402
from corrfilter.voting.cache import VoteCache  # noqa: E402


def load_dotenv_keys(env_path: Path) -> None:
    """Export key-bearing vars from .env into os.environ (no printing)."""
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


def load_items(cfg: dict, dataset: str, root: Path) -> list[CalibrationItem]:
    ds_cfg = cfg["datasets"][dataset]
    df = pd.read_parquet(root / ds_cfg["manifest"])
    if "items_file" in ds_cfg:
        keep = [ln.strip() for ln in (root / ds_cfg["items_file"]).read_text().splitlines() if ln.strip()]
        df = df[df["item_id"].astype(str).isin(set(keep))]
        # RewardBench v2 reuses integer ids across subsets; the open-weight vote
        # cache dedups on item_id keeping the LAST manifest occurrence, so match
        # that convention for a like-for-like comparison.
        df = df.drop_duplicates(subset="item_id", keep="last")
        # preserve the items-file order for determinism
        order = {iid: k for k, iid in enumerate(keep)}
        df = df.sort_values(by="item_id", key=lambda s: s.map(order))
    elif "subsample" in ds_cfg:
        n_take = int(ds_cfg["subsample"])
        rng = np.random.default_rng(int(ds_cfg["seed"]))
        parts = []
        for subset, grp in df.groupby("subset", sort=True):
            k = int(round(n_take * len(grp) / len(df)))
            idx = rng.permutation(len(grp))[:k]
            parts.append(grp.iloc[np.sort(idx)])
        df = pd.concat(parts).sort_values("item_id", key=lambda s: s.str.extract(r"(\d+)", expand=False).astype(int))
    return [CalibrationItem(**row) for row in df.to_dict(orient="records")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/judge_bank_gemini.yaml")
    ap.add_argument("--dataset", choices=["rewardbench", "pku"], required=True)
    ap.add_argument("--live", action="store_true", help="make paid API calls (default: dry-run)")
    ap.add_argument("--limit", type=int, default=None, help="cap item count (smoke tests)")
    ap.add_argument("--items-file", default=None,
                    help="path to an explicit item-id list overriding the dataset default")
    ap.add_argument("--judge", action="append", default=None, help="run only these bank ids")
    ap.add_argument("--workers", type=int, default=None, help="override runtime.workers")
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()

    root = Path(args.project_root)
    cfg = yaml.safe_load((root / args.config).read_text())
    rt = cfg["runtime"]
    gen = cfg["generation"]
    out_root = root / cfg["output"]["root"]
    votes_dir = out_root / "votes" / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    if args.live:
        load_dotenv_keys(root / ".env")

    if args.items_file:
        import copy
        cfg_items = copy.deepcopy(cfg)
        cfg_items["datasets"][args.dataset]["items_file"] = args.items_file
        cfg_items["datasets"][args.dataset].pop("subsample", None)
        items = load_items(cfg_items, args.dataset, root)
    else:
        items = load_items(cfg, args.dataset, root)
    if args.limit:
        items = items[: args.limit]
    items_file = out_root / f"items_{args.dataset}.txt"
    if args.limit is None and not args.items_file:
        items_file.write_text("\n".join(it.item_id for it in items) + "\n")
    print(f"[33] dataset={args.dataset} n_items={len(items)} live={args.live}")

    cache = VoteCache(votes_dir)
    checkpoint = int(rt.get("checkpoint_every", 25))
    run_records = []

    for entry in cfg["bank"]:
        if args.judge and entry["id"] not in args.judge:
            continue
        spec = JudgeSpec(
            base_id=entry["id"],
            family=entry["family"],
            scale=str(entry["scale"]),
            prompt_style=cfg.get("prompt_style", "pairwise"),
            hf_model=entry["model"],   # vendor id recorded in the spec slot
        )
        judge = GeminiJudge(
            spec,
            model=entry["model"],
            thinking_level=entry.get("thinking_level"),
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
        # Re-vote items whose cached raw response is an error sentinel so a
        # resumed run heals transient failures instead of freezing them.
        prior = cache.load(judge.logical_id)
        healed: set[str] = set()
        if prior is not None:
            bad = prior[prior["raw_response"].astype(str).str.startswith("<API_ERROR")]
            healed = set(bad["item_id"].astype(str))
        pending = [it for it in items if it.item_id not in cached or it.item_id in healed]
        print(f"[33] {judge.logical_id}: {len(pending)} pending ({len(cached)} cached)")
        if not pending:
            continue

        judge.load()
        t0 = time.perf_counter()
        all_failures: list[dict] = []
        try:
            for start in range(0, len(pending), checkpoint):
                block = pending[start : start + checkpoint]
                votes = judge.vote_batch(block)
                if args.live:  # never cache dry-run placeholder votes
                    cache.write(judge.logical_id, votes)
                all_failures.extend(judge.failures)
                done = min(start + checkpoint, len(pending))
                print(f"[33]   {judge.logical_id}: {done}/{len(pending)} "
                      f"(abstain so far {judge.usage.abstentions}, cost ${judge.cost_so_far():.3f})")
        finally:
            judge.unload()
        elapsed = time.perf_counter() - t0

        run_records.append({
            "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "bank_id": entry["id"],
            "model_id": entry["model"],
            "logical_id": judge.logical_id,
            "live": bool(args.live),
            "n_items_requested": len(pending),
            "settings": {
                "temperature": float(gen["temperature"]),
                "max_output_tokens": judge.max_new_tokens,
                "thinking_level": entry.get("thinking_level"),
                "position_seed": int(rt["position_seed"]),
                "prompt_style": cfg.get("prompt_style", "pairwise"),
                "workers": judge.workers,
                "max_retries": judge.max_retries,
            },
            "usage": {
                "calls": judge.usage.calls,
                "input_tokens": judge.usage.input_tokens,
                "output_tokens": judge.usage.output_tokens,
                "thought_tokens": judge.thought_tokens,
                "abstentions": judge.usage.abstentions,
            },
            "price_per_1M_in_out": [judge.price_in, judge.price_out],
            "cost_usd": round(judge.cost_so_far()
                              + judge.thought_tokens / 1e6 * judge.price_out, 4),
            "seconds": round(elapsed, 1),
            "n_failures": len(all_failures),
            "failures": all_failures[:200],
        })

    log_path = out_root / f"run_log_{args.dataset}.json"
    existing = json.loads(log_path.read_text()) if log_path.exists() else []
    existing.extend(run_records)
    log_path.write_text(json.dumps(existing, indent=2))
    total = sum(r["cost_usd"] for r in run_records)
    print(f"[33] wrote {log_path} | this invocation cost ${total:.3f} "
          f"({'LIVE' if args.live else 'dry-run estimate'})")


if __name__ == "__main__":
    main()

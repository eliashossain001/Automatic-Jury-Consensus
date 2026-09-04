"""Run every logical judge in the bank over the calibration set, caching votes.

Usage:
    python scripts/calibration/run_judge_bank.py
        --bank configs/judge_bank.yaml
        --calibration configs/calibration.yaml
        [--judge-id qwen-2.5-7b::pairwise ...]
        [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from corrfilter.data import load_calibration_set
from corrfilter.judges import build_bank, load_bank_config
from corrfilter.voting import run_calibration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="configs/judge_bank.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument(
        "--judge-id",
        action="append",
        default=None,
        help="Run only the listed logical judges (e.g. 'qwen-2.5-7b::pairwise'). Repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate calibration set to first N items (smoke-test convenience).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-vote items already in the cache instead of skipping them.",
    )
    parser.add_argument(
        "--sort-by-length",
        action="store_true",
        help="Batch items of similar length together (ascending) to cut padding "
        "waste during generation. Vote assignment is by item_id so ordering is "
        "irrelevant to results; this only affects throughput.",
    )
    args = parser.parse_args()

    cal_cfg = yaml.safe_load(Path(args.calibration).read_text())
    manifest_path = cal_cfg["output"]["manifest_path"]
    items = load_calibration_set(manifest_path)
    if args.sort_by_length:
        items = sorted(items, key=lambda it: len(it.prompt) + len(it.chosen) + len(it.rejected))
    if args.limit is not None:
        items = items[: args.limit]
    print(f"loaded {len(items)} calibration items from {manifest_path}")

    bank_cfg = load_bank_config(args.bank)
    bank = list(build_bank(args.bank))
    if args.judge_id:
        wanted = set(args.judge_id)
        bank = [(spec, fac) for spec, fac in bank if spec.logical_id in wanted]
        print(f"filtered bank → {len(bank)} judges: {[s.logical_id for s, _ in bank]}")

    summary = run_calibration(
        items=items,
        bank=bank,
        cache_dir=bank_cfg.votes_dir,
        skip_cached=not args.no_cache,
    )
    out_path = Path("experiments/h1_measurement/results/voting_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"voting summary → {out_path}")


if __name__ == "__main__":
    main()

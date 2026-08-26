"""Run the CFI banks across all bias mechanisms and ratios.

Two execution modes:

* ``replay`` (default): deterministic CPU simulation of biased judge votes
  starting from the cached H1 votes. No GPU needed.
* ``real``: re-invokes the HuggingFace judge with the bias-injected prompt.
  Requires the same GPU posture as the H1 voting pass.

Outputs:
    experiments/h2_cfi/votes/{variant}.parquet
        Per-variant long-form parquet with one row per (item, judge).
    experiments/h2_cfi/results/cfi_voting_summary.json
        Summary of variant sizes + biased-judge identities.

Usage:
    python scripts/cfi/run_bank_replay.py [--config configs/cfi.yaml] [--mode replay|real]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from corrfilter.cfi.banks import build_bank_variants
from corrfilter.cfi.biases import BIAS_REGISTRY, BiasMechanism, get_bias
from corrfilter.cfi.manifest import load_cfi_manifest, triggers_from_row
from corrfilter.cfi.prompts import BiasedPrompt
from corrfilter.cfi.replay import (
    load_clean_votes_from_cache,
    replay_to_dataframe,
    simulate_bank_votes,
    variant_position_swap,
)
from corrfilter.data import load_calibration_set
from corrfilter.judges import build_bank, load_bank_config
from corrfilter.judges.prompts import get_prompt_template
from corrfilter.voting import VoteCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.06_run_cfi_votes")


def _load_triggers(manifest_path: Path) -> dict:
    df = load_cfi_manifest(manifest_path)
    return {str(row["item_id"]): triggers_from_row(row) for _, row in df.iterrows()}


def _replay_one_variant(variant, items, triggers, clean_votes, votes_dir: Path, seed: int) -> dict:
    result = simulate_bank_votes(variant, items, triggers, clean_votes, seed=seed)
    df = replay_to_dataframe(result)
    out_path = votes_dir / f"{variant.name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return {
        "variant": variant.name,
        "mechanism": variant.mechanism.value,
        "biased_ratio": variant.biased_ratio,
        "biased_logical_ids": list(variant.biased_logical_ids),
        "clean_logical_ids": list(variant.clean_logical_ids),
        "n_rows": int(len(df)),
        "n_triggered_items": int(result.triggered_flag.sum()),
        "votes_path": str(out_path),
    }


def _real_one_variant(
    variant, items, triggers, bank_cfg, biased_factories, votes_dir: Path, seed: int
) -> dict:
    """Re-run the bank under bias injection (real GPU mode).

    Writes per-logical-judge parquet votes via the standard VoteCache layout
    inside ``votes_dir / variant.name`` so each variant has its own subtree.
    """
    from corrfilter.judges.hf_judge import HFJudge

    variant_votes_dir = votes_dir / variant.name
    cache = VoteCache(variant_votes_dir)
    biased_set = set(variant.biased_logical_ids)
    bias_spec = get_bias(variant.mechanism)
    rows = []

    for spec, factory in biased_factories:
        is_biased = spec.logical_id in biased_set
        # Use the shared per-variant swap for biased judges so position bias is correlated.
        swaps_by_item = {it.item_id: variant_position_swap(variant.name, it.item_id, seed) for it in items}
        triggered = {it.item_id for it in items if triggers[it.item_id].fires(variant.mechanism)}

        judge: HFJudge = factory()
        if is_biased:
            judge.prompt = BiasedPrompt(judge.prompt, bias_spec)

        if is_biased:
            # Override per-item swap to the variant-level shared swap on triggered items
            # so the position-bias mechanism produces correlated errors.
            orig_swap = judge._position_swap

            def _swap(item_id, _orig=orig_swap, _swaps=swaps_by_item, _trig=triggered):
                if item_id in _trig:
                    return _swaps[item_id]
                return _orig(item_id)

            judge._position_swap = _swap  # type: ignore[method-assign]

        try:
            judge.load()
            pending = [it for it in items if it.item_id not in cache.cached_item_ids(spec.logical_id)]
            for start in range(0, len(pending), judge.batch_size):
                batch = pending[start : start + judge.batch_size]
                votes = judge.vote_batch(batch)
                cache.write(spec.logical_id, votes)
                rows.append({"variant": variant.name, "judge": spec.logical_id, "batch": start, "votes": len(votes)})
        finally:
            judge.unload()

    return {
        "variant": variant.name,
        "mechanism": variant.mechanism.value,
        "biased_ratio": variant.biased_ratio,
        "biased_logical_ids": list(variant.biased_logical_ids),
        "clean_logical_ids": list(variant.clean_logical_ids),
        "votes_path": str(variant_votes_dir),
        "n_writes": int(len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cfi.yaml")
    parser.add_argument("--mode", choices=["replay", "real"], default=None)
    parser.add_argument("--mechanism", action="append", default=None, help="Subset of mechanisms (repeatable).")
    parser.add_argument("--ratio", type=float, action="append", default=None, help="Subset of biased ratios (repeatable).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = args.mode or cfg.get("mode", "replay")
    seed = int(cfg.get("seed", 20260601))

    cal_cfg = yaml.safe_load(Path(cfg["sources"]["h1_calibration_config"]).read_text())
    items = load_calibration_set(cal_cfg["output"]["manifest_path"])
    manifest_path = Path(cfg["output"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"CFI manifest not found at {manifest_path}; run scripts/cfi/build_manifest.py first"
        )
    triggers = _load_triggers(manifest_path)

    bank_cfg = load_bank_config(cfg["sources"]["h1_bank_config"])
    base_specs = bank_cfg.specs

    if args.mechanism:
        mechanism_names = list(args.mechanism)
    else:
        mechanism_names = cfg["mechanisms"]
    mechanisms = [BiasMechanism(m) for m in mechanism_names]
    ratios = list(args.ratio) if args.ratio else list(cfg["biased_ratios"])
    variants = build_bank_variants(base_specs, mechanisms, ratios, seed=seed)
    logger.info(
        "CFI: mode=%s, mechanisms=%s, ratios=%s, n_variants=%d",
        mode,
        [m.value for m in mechanisms],
        ratios,
        len(variants),
    )

    out_dir = Path(cfg["output"]["votes_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"mode": mode, "mechanisms": [m.value for m in mechanisms], "ratios": ratios, "variants": []}

    if mode == "replay":
        clean_votes = load_clean_votes_from_cache(
            VoteCache(Path(cfg["sources"]["h1_votes_dir"])),
            [s.logical_id for s in base_specs],
        )
        for variant in variants:
            row = _replay_one_variant(variant, items, triggers, clean_votes, out_dir, seed)
            logger.info(
                "replay variant %s: %d triggered items, %d rows",
                row["variant"], row["n_triggered_items"], row["n_rows"],
            )
            summary["variants"].append(row)
    elif mode == "real":
        bank_pairs = list(build_bank(cfg["sources"]["h1_bank_config"]))
        for variant in variants:
            row = _real_one_variant(variant, items, triggers, bank_cfg, bank_pairs, out_dir, seed)
            logger.info("real variant %s: %s", row["variant"], row["votes_path"])
            summary["variants"].append(row)

    results_dir = Path(cfg["output"]["results_csv_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cfi_voting_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(summary['variants'])} variant rows → {results_dir / 'cfi_voting_summary.json'}")


if __name__ == "__main__":
    # Touch the registry / templates so unused imports stay visible to ruff
    # in dev: each is part of the public path callers use to extend behaviour.
    _ = BIAS_REGISTRY, get_prompt_template, pd  # noqa: F841
    main()

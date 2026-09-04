"""Run the H1 contrast (intra-family vs cross-family, intra-prompt vs cross-prompt).

Usage:
    python scripts/dependence/test_diversification_contrasts.py
        --bank configs/judge_bank.yaml
        --calibration configs/calibration.yaml
        [--bootstrap 1000]
        [--threshold 0.10]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from corrfilter.analysis import H1Result, family_contrast, prompt_contrast
from corrfilter.correlation import compute_error_matrix, compute_error_matrix_pairwise
from corrfilter.data import load_calibration_set
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache, load_vote_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _h1_to_dict(result: H1Result) -> dict:
    return asdict(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="configs/judge_bank.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--results-dir", default="experiments/h1_measurement/results")
    parser.add_argument("--listwise-retention-floor", type=float, default=0.70)
    args = parser.parse_args()

    cal_cfg = yaml.safe_load(Path(args.calibration).read_text())
    items = load_calibration_set(cal_cfg["output"]["manifest_path"])
    item_ids = [it.item_id for it in items]
    gold = np.ones(len(items), dtype=np.int8)

    bank_cfg = load_bank_config(args.bank)
    specs = bank_cfg.specs
    logical_ids = [s.logical_id for s in specs]
    cache = VoteCache(bank_cfg.votes_dir)
    V, M = load_vote_matrix(cache, logical_ids, item_ids)

    E_listwise, complete = compute_error_matrix(V, M, gold)
    n_items_total = int(complete.size)
    n_items_kept = int(complete.sum())
    retention = n_items_kept / max(n_items_total, 1)
    if retention >= args.listwise_retention_floor:
        estimator = "listwise_ledoit_wolf"
        E = E_listwise
        avail = None
    else:
        estimator = "pairwise_complete_shrunk"
        E_full, avail = compute_error_matrix_pairwise(V, M, gold)
        E = E_full
    print(
        f"H1 estimator: {estimator}; listwise retention "
        f"{n_items_kept}/{n_items_total} = {retention:.1%}; n_judges={len(specs)}"
    )

    fam = family_contrast(
        E, specs, availability=avail, n_boot=args.bootstrap, threshold=args.threshold, seed=20260601
    )
    pmt = prompt_contrast(
        E, specs, availability=avail, n_boot=args.bootstrap, threshold=args.threshold, seed=20260602
    )

    out = {
        "estimator": estimator,
        "listwise_retention": retention,
        "n_items_total": n_items_total,
        "n_items_kept_listwise": n_items_kept,
        "n_judges": len(specs),
        "threshold": args.threshold,
        "family_contrast": _h1_to_dict(fam),
        "prompt_contrast": _h1_to_dict(pmt),
        "accepts_h1_either": fam.accepts_h1 or pmt.accepts_h1,
    }
    out_path = Path(args.results_dir) / "h1_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\nFamily contrast:")
    print(json.dumps(_h1_to_dict(fam), indent=2))
    print("\nPrompt-style contrast:")
    print(json.dumps(_h1_to_dict(pmt), indent=2))
    decision = "ACCEPT H1" if out["accepts_h1_either"] else "REJECT H1"
    print(f"\nDecision (Δ ≥ {args.threshold}, p < 0.01): {decision}")


if __name__ == "__main__":
    main()

"""Build the CFI evaluation manifest (per-item trigger annotations).

Usage:
    python scripts/cfi/build_manifest.py [--config configs/cfi.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from corrfilter.cfi.manifest import build_cfi_manifest, save_cfi_manifest
from corrfilter.cfi.triggers import TriggerConfig
from corrfilter.data import load_calibration_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cfi.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cal_cfg = yaml.safe_load(Path(cfg["sources"]["h1_calibration_config"]).read_text())
    items = load_calibration_set(cal_cfg["output"]["manifest_path"])
    trig_cfg = TriggerConfig(
        verbosity_ratio=float(cfg["triggers"]["verbosity_ratio"]),
        polish_subsets=tuple(cfg["triggers"]["polish_subsets"]),
        polish_categories=tuple(cfg["triggers"]["polish_categories"]),
        refusal_subsets=tuple(cfg["triggers"]["refusal_subsets"]),
        refusal_categories=tuple(cfg["triggers"]["refusal_categories"]),
    )

    df = build_cfi_manifest(items, trig_cfg)
    out_path = Path(cfg["output"]["manifest_path"])
    save_cfi_manifest(df, out_path)
    sidecar = out_path.with_suffix(".summary.json")
    print(f"wrote {len(df)} CFI-manifest rows → {out_path}")
    print(json.dumps(json.loads(sidecar.read_text()), indent=2))


if __name__ == "__main__":
    main()

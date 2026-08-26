"""Build and persist the stratified RewardBench v2 calibration set.

Usage:
    python scripts/calibration/build_calibration_set.py [--config configs/calibration.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from corrfilter.data import build_calibration_set
from corrfilter.data.rewardbench import save_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/calibration.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out_path = Path(config["output"]["manifest_path"])

    items = build_calibration_set(args.config)
    save_manifest(items, out_path)

    summary_path = out_path.with_suffix(".summary.json")
    print(f"wrote {len(items)} calibration items → {out_path}")
    print(json.dumps(json.loads(summary_path.read_text()), indent=2))


if __name__ == "__main__":
    main()

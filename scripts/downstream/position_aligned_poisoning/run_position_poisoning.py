"""Experiment C driver — position-aligned poisoning + downstream DPO (10/20/30%).

Identical to the contamination-scaling driver except it points the Phase 1 scripts
at the position-poisoned manifest and the pos10/pos20/pos30 tags. Everything else
(model, LoRA, seed, matched retention, eval sets, metrics) is inherited from the
base Phase 1 config. Runs prepare -> train -> eval -> summarize per level as
subprocesses; outputs land under position_aligned_poisoning/<tag>/.

Add `--prepare-only` to build datasets + purity tables WITHOUT training (used to
validate the regime before committing GPU time). Default levels: pos10 pos20 pos30.

Usage:
    python position_aligned_poisoning/run_position_poisoning.py --prepare-only
    python position_aligned_poisoning/run_position_poisoning.py            # full run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "position_aligned_poisoning"                                    # this experiment's outputs
POS_MANIFEST = "experiments/downstream_dpo_validation/position_aligned_poisoning/poisoned_manifest_position.csv"
ALL_STEPS = ["prepare_dpo_datasets.py", "train_dpo_by_filter.py",
             "evaluate_dpo_models.py", "summarize_downstream_results.py"]


def derived_config(tag: str) -> Path:
    base = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    base["data"]["contamination"] = tag
    base["data"]["manifest"] = POS_MANIFEST
    base["data"]["regime"] = f"position_subgroup_{tag}"
    base["run_dir"] = f"position_aligned_poisoning/{tag}"
    out_dir = RUN_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False))
    return cfg_path


def run_step(script: str, cfg_path: Path) -> None:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [sys.executable, "-u", str(CODE_DIR / script), "--config", str(cfg_path)]
    print(f"\n>>> {script}  (config={cfg_path.relative_to(ROOT)})", flush=True)
    subprocess.run(cmd, check=True, cwd=str(EXP_DIR), env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", nargs="+", default=["pos10", "pos20", "pos30"])
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    steps = ALL_STEPS[:1] if args.prepare_only else ALL_STEPS
    for tag in args.levels:
        print(f"\n========== POSITION POISONING {tag} ==========", flush=True)
        cfg_path = derived_config(tag)
        for script in steps:
            run_step(script, cfg_path)
    print("\n========== POSITION RUN COMPLETE ==========", flush=True)


if __name__ == "__main__":
    main()

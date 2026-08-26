"""Experiment A — contamination scaling driver (20% / 30% / 40%).

Reuses the Phase 1 scripts unchanged. For each contamination level it writes a
derived config (Phase 1 config + that contamination rate + a per-level run_dir +
the extended manifest) and runs prepare -> train -> eval -> summarize as
subprocesses, so GPU memory is released between stages. Everything else (model,
seed, matched-retention logic, DPO hyperparameters, eval sets, metrics) is
inherited verbatim from the base config.

Outputs land under contamination_scaling/c20, c30, c40 (each with data/ models/
results/). Run `aggregate_scaling.py` afterwards for the combined table + summary.

Usage:
    python contamination_scaling/run_contamination_scaling.py            # 20,30,40
    python contamination_scaling/run_contamination_scaling.py --levels 30
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent                       # scripts/downstream
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "contamination_scaling"                                    # this experiment's outputs
EXT_MANIFEST = "experiments/downstream_dpo_validation/contamination_scaling/poisoned_manifest_ext.csv"
STEPS = ["prepare_dpo_datasets.py", "train_dpo_by_filter.py",
         "evaluate_dpo_models.py", "summarize_downstream_results.py"]


def derived_config(level: str) -> Path:
    base = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    base["data"]["contamination"] = level
    base["data"]["manifest"] = EXT_MANIFEST
    base["data"]["regime"] = f"synthetic_uf_{level}pct"
    base["run_dir"] = f"contamination_scaling/c{level}"
    out_dir = RUN_DIR / f"c{level}"
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
    ap.add_argument("--levels", nargs="+", default=["20", "30", "40"])
    args = ap.parse_args()
    for level in args.levels:
        print(f"\n========== CONTAMINATION {level}% ==========", flush=True)
        cfg_path = derived_config(level)
        for script in STEPS:
            run_step(script, cfg_path)
    print("\n========== ALL LEVELS COMPLETE ==========", flush=True)


if __name__ == "__main__":
    main()

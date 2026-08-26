"""Shared helpers for the downstream DPO validation pipeline.

Pure utility layer: config loading, deterministic paths, seeding, and JSONL I/O.
No experiment logic lives here, so the entry-point scripts under
``scripts/downstream/`` stay readable and every path and seed is defined exactly
once. Artefacts (configs, datasets, checkpoints, results) live under
``experiments/downstream_dpo_validation/``; ``set_run_dir`` redirects them so
several studies can share the same scripts.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"
# data/models/results live under a run directory. Default is the Phase 1 root
# (run_dir="."); set_run_dir() redirects them so multiple studies coexist.
DATA_DIR = EXP_DIR / "data"
MODELS_DIR = EXP_DIR / "models"
RESULTS_DIR = EXP_DIR / "results"


def set_run_dir(subdir: str | os.PathLike) -> Path:
    """Point DATA_DIR/MODELS_DIR/RESULTS_DIR under EXP_DIR/<subdir>.

    Called at the top of each script's main() with cfg.get("run_dir", "."), so the
    same scripts serve Phase 1 (run_dir=".") and the contamination-scaling study
    (run_dir="contamination_scaling/c30", ...) with no code duplication.
    """
    global DATA_DIR, MODELS_DIR, RESULTS_DIR
    base = (EXP_DIR / subdir).resolve()
    DATA_DIR, MODELS_DIR, RESULTS_DIR = base / "data", base / "models", base / "results"
    return base


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else EXP_DIR / "config.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def resolve(rel: str | os.PathLike) -> Path:
    """Resolve a config-relative path against the repo root (absolutes pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


@dataclass(frozen=True)
class FilterSpec:
    """Display/order metadata for a filter method in tables and plots."""

    key: str
    label: str
    family: str  # naive | dependence_aware | vulnerability_aware | router | oracle


FILTER_SPECS: dict[str, FilterSpec] = {
    "naive_consensus": FilterSpec("naive_consensus", "Naive consensus", "naive"),
    "supermajority_75": FilterSpec("supermajority_75", "Supermajority-0.75", "naive"),
    "corrfilter": FilterSpec("corrfilter", "CorrFilter (dependence-aware)", "dependence_aware"),
    "bias_cluster": FilterSpec("bias_cluster", "Bias-cluster (vulnerability-aware)", "vulnerability_aware"),
    "router": FilterSpec("router", "Regime router", "router"),
    "oracle": FilterSpec("oracle", "Oracle (clean labels)", "oracle"),
}

"""RewardBench v2 calibration set construction for H1 measurement.

Each calibration item is a single preference pair ``(prompt, chosen, rejected)``
with a trusted gold label of ``chosen > rejected``. Judges that disagree with
the gold label produce error indicators ``e_{ij}`` from which the empirical
error-correlation matrix R is estimated (proposal §3.1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class CalibrationItem:
    """A single calibration preference pair with trusted gold label."""

    item_id: str
    prompt: str
    chosen: str
    rejected: str
    subset: str
    category: str
    gold_label: int = 1  # 1 ⇒ chosen ≻ rejected (always true by construction)

    def to_dict(self) -> dict:
        return asdict(self)


def _invert_category_map(category_map: dict[str, list[str]]) -> dict[str, str]:
    """Subset name (RewardBench) → proposal category (chat / reasoning / ...)."""
    inverted: dict[str, str] = {}
    for category, subsets in category_map.items():
        for subset in subsets:
            inverted[subset] = category
    return inverted


def build_calibration_set(config_path: str | Path) -> list[CalibrationItem]:
    """Build the stratified calibration set from a YAML config.

    Stratification is per-RewardBench-subset: each subset contributes up to
    ``per_category`` items, sampled without replacement using a seeded
    permutation. Smaller subsets (notably Ties at 102 items) contribute every
    available item. The total count is therefore the sum of ``min(per_category,
    subset_size)`` across subsets, which lands near ~850 with per_category=250
    rather than exactly 1000.
    """
    from datasets import load_dataset

    config = yaml.safe_load(Path(config_path).read_text())
    ds_cfg = config["dataset"]
    sample_cfg = config["sampling"]

    raw = load_dataset(
        ds_cfg["hf_repo"],
        split=ds_cfg["split"],
        cache_dir=config["output"].get("cache_dir"),
    )

    subset_to_category = _invert_category_map(ds_cfg["category_map"])
    per_category = int(sample_cfg["per_category"])
    seed = int(sample_cfg["seed"])
    chosen_idx = int(sample_cfg.get("chosen_index", 0))
    rejected_idx = int(sample_cfg.get("rejected_index", 0))

    rng = np.random.default_rng(seed)
    items: list[CalibrationItem] = []

    for subset_name in subset_to_category:
        subset_rows = [r for r in raw if r["subset"] == subset_name]
        if not subset_rows:
            continue
        order = rng.permutation(len(subset_rows))
        take = min(per_category, len(subset_rows))
        for k in order[:take]:
            row = subset_rows[int(k)]
            chosen_list = row["chosen"]
            rejected_list = row["rejected"]
            if not chosen_list or not rejected_list:
                continue
            ci = chosen_idx if chosen_idx < len(chosen_list) else 0
            ri = rejected_idx if rejected_idx < len(rejected_list) else 0
            items.append(
                CalibrationItem(
                    item_id=str(row["id"]),
                    prompt=row["prompt"],
                    chosen=chosen_list[ci],
                    rejected=rejected_list[ri],
                    subset=subset_name,
                    category=subset_to_category[subset_name],
                )
            )

    return items


def save_manifest(items: Iterable[CalibrationItem], path: str | Path) -> Path:
    """Persist the calibration set as a parquet manifest plus a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([it.to_dict() for it in items])
    df.to_parquet(path, index=False)
    sidecar = path.with_suffix(".summary.json")
    summary = {
        "n_items": len(df),
        "per_category": df["category"].value_counts().to_dict(),
        "per_subset": df["subset"].value_counts().to_dict(),
    }
    sidecar.write_text(json.dumps(summary, indent=2))
    return path


def load_calibration_set(manifest_path: str | Path) -> list[CalibrationItem]:
    """Reload a previously built calibration manifest."""
    df = pd.read_parquet(manifest_path)
    return [CalibrationItem(**row) for row in df.to_dict(orient="records")]

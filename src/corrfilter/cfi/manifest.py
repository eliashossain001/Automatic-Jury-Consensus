"""CFI evaluation manifest: per-item trigger annotations.

The manifest lives next to the H1 calibration manifest and identifies which
items plausibly exercise each bias mechanism. It is materialised as a parquet
file so downstream scripts can read it without re-running heuristics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from corrfilter.cfi.biases import BiasMechanism
from corrfilter.cfi.triggers import (
    TriggerConfig,
    detect_polish_trigger,
    detect_position_trigger,
    detect_refusal_trigger,
    detect_sycophancy_trigger,
    detect_verbosity_trigger,
)
from corrfilter.data import CalibrationItem

TRIGGER_FUNCS = {
    BiasMechanism.VERBOSITY: detect_verbosity_trigger,
    BiasMechanism.SYCOPHANCY: detect_sycophancy_trigger,
    BiasMechanism.POSITION: detect_position_trigger,
    BiasMechanism.POLISH: detect_polish_trigger,
    BiasMechanism.REFUSAL: detect_refusal_trigger,
}


@dataclass(frozen=True)
class CFITriggers:
    """Boolean trigger flags for a single calibration item."""

    item_id: str
    verbosity: bool
    sycophancy: bool
    position: bool
    polish: bool
    refusal: bool

    def fires(self, mechanism: BiasMechanism) -> bool:
        return bool(getattr(self, mechanism.value))


def detect_triggers(
    item: CalibrationItem, cfg: TriggerConfig | None = None
) -> CFITriggers:
    """Compute all five trigger flags for one item."""
    return CFITriggers(
        item_id=item.item_id,
        verbosity=detect_verbosity_trigger(item, cfg),
        sycophancy=detect_sycophancy_trigger(item, cfg),
        position=detect_position_trigger(item, cfg),
        polish=detect_polish_trigger(item, cfg),
        refusal=detect_refusal_trigger(item, cfg),
    )


def build_cfi_manifest(
    items: Iterable[CalibrationItem], cfg: TriggerConfig | None = None
) -> pd.DataFrame:
    """Build the CFI manifest as a DataFrame keyed by item_id."""
    rows = []
    for it in items:
        trig = detect_triggers(it, cfg)
        rows.append({
            "item_id": it.item_id,
            "subset": it.subset,
            "category": it.category,
            "len_chosen": len(it.chosen),
            "len_rejected": len(it.rejected),
            "verbosity": trig.verbosity,
            "sycophancy": trig.sycophancy,
            "position": trig.position,
            "polish": trig.polish,
            "refusal": trig.refusal,
        })
    return pd.DataFrame(rows)


def save_cfi_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist the manifest with a side JSON summary of trigger rates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    summary = {
        "n_items": int(len(df)),
        "trigger_rate": {
            mech.value: float(df[mech.value].mean()) for mech in BiasMechanism
        },
        "trigger_count": {
            mech.value: int(df[mech.value].sum()) for mech in BiasMechanism
        },
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return path


def load_cfi_manifest(path: str | Path) -> pd.DataFrame:
    """Reload a previously built CFI manifest."""
    return pd.read_parquet(path)


def triggers_from_row(row: pd.Series | dict) -> CFITriggers:
    """Reconstruct a CFITriggers dataclass from a manifest row."""
    if isinstance(row, dict):
        getter = row.__getitem__
    else:
        getter = row.__getitem__
    return CFITriggers(
        item_id=str(getter("item_id")),
        verbosity=bool(getter("verbosity")),
        sycophancy=bool(getter("sycophancy")),
        position=bool(getter("position")),
        polish=bool(getter("polish")),
        refusal=bool(getter("refusal")),
    )


__all__ = [
    "TRIGGER_FUNCS",
    "CFITriggers",
    "detect_triggers",
    "build_cfi_manifest",
    "save_cfi_manifest",
    "load_cfi_manifest",
    "triggers_from_row",
]


# Suppress the unused import warning in flat re-export style — asdict is part
# of the public dataclass surface for downstream JSON serialisation.
_ = asdict

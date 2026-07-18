"""Config-driven bias bank for the 8-condition CFI experiment (Bucket 3).

This is an additive layer over the enum-based :mod:`corrfilter.cfi.biases`
registry. Where ``biases.py`` hard-codes five mechanisms with replay rules,
this module loads an arbitrary set of *prompt-level* bias conditions from
``configs/cfi_bias_prompts.yaml`` for the real-inference experiment, which only
needs an instruction string plus a heuristic trigger per mechanism.

A :class:`BiasCondition` carries everything the runner and analysis need:

* ``instruction`` — the system-prompt addition (empty for the clean control);
* ``trigger`` — a feature key (``length`` / ``formatting`` / ``opinion`` /
  ``safety`` / ``factuality`` / ``all`` / ``none``) evaluated per item so the
  analysis can split triggered vs non-triggered items.

The trigger heuristics reuse the lexicons in :mod:`corrfilter.cfi.triggers`
so the clean/biased detection stays consistent with the H1 manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from corrfilter.cfi.triggers import SYCOPHANCY_RE, _length_ratio
from corrfilter.data import CalibrationItem

CLEAN = "clean"

# Markdown / structure markers for the style_template_preference trigger.
_FORMAT_RE = re.compile(r"(^|\n)\s*(#{1,6}\s|[-*]\s|\d+\.\s)|\n\s*\n", re.MULTILINE)

VALID_TRIGGERS = {"length", "formatting", "opinion", "safety", "factuality", "all", "none"}


@dataclass(frozen=True)
class BiasCondition:
    """One CFI bias condition: a prompt-level instruction plus a trigger rule."""

    name: str
    instruction: str
    description: str
    trigger: str

    @property
    def is_clean(self) -> bool:
        return self.name == CLEAN or not self.instruction.strip()

    def fires(self, item: CalibrationItem) -> bool:
        """Whether this mechanism plausibly exercises the bias on ``item``."""
        t = self.trigger
        if t == "none":
            return False
        if t == "all":
            return True
        if t == "length":
            return _length_ratio(item) >= 0.5
        if t == "formatting":
            ca = bool(_FORMAT_RE.search(item.chosen or ""))
            cb = bool(_FORMAT_RE.search(item.rejected or ""))
            return ca != cb
        if t == "opinion":
            return bool(SYCOPHANCY_RE.search(item.prompt or ""))
        if t == "safety":
            return item.subset == "Safety" or item.category == "safety"
        if t == "factuality":
            return item.subset in ("Factuality", "Math") or item.category in ("chat", "reasoning")
        raise ValueError(f"unknown trigger {t!r}; expected one of {sorted(VALID_TRIGGERS)}")


@dataclass(frozen=True)
class BiasBank:
    """The full set of CFI bias conditions plus bank-ratio definitions."""

    conditions: list[BiasCondition]
    bank_conditions: dict[str, float]
    seed: int

    @property
    def biased_conditions(self) -> list[BiasCondition]:
        """Non-clean mechanisms — the ones that require real biased inference."""
        return [c for c in self.conditions if not c.is_clean]

    def by_name(self, name: str) -> BiasCondition:
        for c in self.conditions:
            if c.name == name:
                return c
        raise KeyError(f"no bias condition named {name!r}")


def load_bias_bank(path: str | Path) -> BiasBank:
    """Load the bias conditions + bank ratios from a YAML config."""
    cfg = yaml.safe_load(Path(path).read_text())
    conditions: list[BiasCondition] = []
    for entry in cfg["mechanisms"]:
        trigger = str(entry.get("trigger", "none"))
        if trigger not in VALID_TRIGGERS:
            raise ValueError(
                f"mechanism {entry['name']!r} has invalid trigger {trigger!r}"
            )
        conditions.append(
            BiasCondition(
                name=str(entry["name"]),
                instruction=str(entry.get("instruction", "") or "").strip(),
                description=str(entry.get("description", "")),
                trigger=trigger,
            )
        )
    names = [c.name for c in conditions]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate mechanism names in {path}: {names}")
    bank_conditions = {str(k): float(v) for k, v in cfg.get("bank_conditions", {}).items()}
    return BiasBank(
        conditions=conditions,
        bank_conditions=bank_conditions,
        seed=int(cfg.get("seed", 20260601)),
    )


__all__ = ["BiasCondition", "BiasBank", "load_bias_bank", "CLEAN"]

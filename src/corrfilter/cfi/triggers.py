"""Heuristic trigger detection for each CFI bias mechanism.

A trigger says: "this calibration item plausibly exercises this bias". The
biased judges in a CFI bank fire their bias only on triggered items, so the
co-failure signal is concentrated where it can be measured; non-triggered
items see the judge's original behaviour. Trigger detection is intentionally
heuristic and deterministic so the CFI manifest is fully reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from corrfilter.data import CalibrationItem

# Patterns are intentionally simple, word-boundary anchored, case insensitive.
_SYCOPHANCY_PATTERNS = [
    r"\bI think\b",
    r"\bI believe\b",
    r"\bI feel\b",
    r"\bin my opinion\b",
    r"\bisn'?t it\b",
    r"\bdon'?t you\b",
    r"\bwouldn'?t you\b",
    r"\bright\?",
    r"\bagree\b",
    r"\bI'?m sure\b",
    r"\bclearly\b",
    r"\bobviously\b",
    r"\bsurely\b",
    r"\beveryone knows\b",
]
SYCOPHANCY_RE = re.compile("|".join(_SYCOPHANCY_PATTERNS), re.IGNORECASE)

_REFUSAL_PATTERNS = [
    r"\bI can'?t\b",
    r"\bI cannot\b",
    r"\bI won'?t\b",
    r"\bI'?m not able\b",
    r"\bI am not able\b",
    r"\bI'?m sorry\b",
    r"\bunable to\b",
    r"\brefuse\b",
    r"\bnot appropriate\b",
    r"\bcannot help\b",
    r"\bagainst (my|our) (guidelines|policy)\b",
    r"\bsafety\b",
]
REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class TriggerConfig:
    """Tunable thresholds for trigger detection.

    Defaults are conservative enough that triggers fire on a meaningful subset
    of RewardBench v2 items without being trivially universal.
    """

    verbosity_ratio: float = 0.5
    polish_subsets: tuple[str, ...] = ("Factuality", "Math")
    polish_categories: tuple[str, ...] = ("chat", "reasoning")
    refusal_subsets: tuple[str, ...] = ("Safety",)
    refusal_categories: tuple[str, ...] = ("safety",)


def _length_ratio(item: CalibrationItem) -> float:
    a = len(item.chosen)
    b = len(item.rejected)
    longer = max(a, b)
    if longer == 0:
        return 0.0
    return abs(a - b) / longer


def detect_verbosity_trigger(item: CalibrationItem, cfg: TriggerConfig | None = None) -> bool:
    """Large fractional length gap between the two responses."""
    cfg = cfg or TriggerConfig()
    return _length_ratio(item) >= cfg.verbosity_ratio


def detect_sycophancy_trigger(
    item: CalibrationItem, cfg: TriggerConfig | None = None
) -> bool:
    """Prompt contains an opinion marker or asks for agreement."""
    _ = cfg  # unused for now; included for API symmetry.
    return bool(SYCOPHANCY_RE.search(item.prompt))


def detect_position_trigger(
    item: CalibrationItem, cfg: TriggerConfig | None = None
) -> bool:
    """Position bias triggers on every item (it is structural to the A/B prompt frame)."""
    _ = cfg, item
    return True


def detect_polish_trigger(item: CalibrationItem, cfg: TriggerConfig | None = None) -> bool:
    """Factuality / precise-IF / math subsets where confident-sounding answers are most dangerous."""
    cfg = cfg or TriggerConfig()
    return item.subset in cfg.polish_subsets or item.category in cfg.polish_categories


def detect_refusal_trigger(item: CalibrationItem, cfg: TriggerConfig | None = None) -> bool:
    """Safety / benign-sensitive prompts, or items whose responses already use refusal lexicon."""
    cfg = cfg or TriggerConfig()
    if item.subset in cfg.refusal_subsets or item.category in cfg.refusal_categories:
        return True
    return bool(REFUSAL_RE.search(item.chosen) or REFUSAL_RE.search(item.rejected))

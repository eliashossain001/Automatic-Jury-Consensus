"""Deterministic replay/simulation of biased judge votes.

Replay mode is the default execution path for the CFI experiment: instead of
re-running expensive GPU inference with a biased prompt, we deterministically
derive the biased judge's vote from the calibration item itself, applying a
mechanism-specific decision rule on triggered items and falling back to the
cached clean H1 vote on non-triggered items.

This makes the CFI bank tractable to debug end-to-end on CPU in seconds, and
the deterministic rule produces correlated co-failures by construction:
biased judges share the same rule, so on triggered items they agree (often
incorrectly), inflating naive consensus exactly as the proposal predicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from corrfilter.cfi.banks import CFIBankVariant
from corrfilter.cfi.biases import BiasMechanism
from corrfilter.cfi.manifest import CFITriggers
from corrfilter.data import CalibrationItem

# Vocabularies driving the polish / refusal / sycophancy biased predictions.
_CONFIDENT_RE = re.compile(
    r"\b(definitely|certainly|clearly|obviously|surely|undoubtedly|"
    r"absolutely|in fact|indeed|without (?:a )?doubt|of course)\b",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(I can'?t|I cannot|I won'?t|I'?m sorry|unable to|refuse|"
    r"not appropriate|cannot help|against (?:my|our) (?:guidelines|policy))\b",
    re.IGNORECASE,
)
_AGREE_RE = re.compile(
    r"\b(yes|agree|absolutely|certainly|correct|that'?s right|"
    r"good question|you'?re right|exactly)\b",
    re.IGNORECASE,
)


def _count(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text or ""))


def variant_position_swap(variant_name: str, item_id: str, seed: int = 20260601) -> bool:
    """Per-item position swap that is shared across all biased judges in a variant.

    Position bias only produces correlated errors when the biased judges all see
    the same (A, B) ordering. We pick the swap deterministically from
    ``(seed, variant_name, item_id)`` so the choice is reproducible and stable
    across reruns.
    """
    # Stable hash that does not depend on PYTHONHASHSEED.
    raw = f"{seed}|{variant_name}|{item_id}".encode()
    return bool(sum(raw) & 1)


def biased_prediction(
    mechanism: BiasMechanism,
    item: CalibrationItem,
    position_swap: bool,
) -> int | None:
    """Return the deterministic biased vote on a *triggered* item.

    Returns ``1`` (agrees with chosen ≻ rejected), ``0`` (disagrees), or
    ``None`` (the bias is indifferent on this item; caller should preserve the
    judge's original vote rather than flipping randomly).
    """
    if mechanism is BiasMechanism.VERBOSITY:
        a, b = len(item.chosen), len(item.rejected)
        if a == b:
            return None
        return 1 if a > b else 0

    if mechanism is BiasMechanism.POSITION:
        # Bias prefers A. Without swap, A = chosen ⇒ vote 1; with swap, A = rejected ⇒ vote 0.
        return 0 if position_swap else 1

    if mechanism is BiasMechanism.POLISH:
        c = _count(_CONFIDENT_RE, item.chosen)
        r = _count(_CONFIDENT_RE, item.rejected)
        if c == r:
            return None
        return 1 if c > r else 0

    if mechanism is BiasMechanism.REFUSAL:
        c = _count(_REFUSAL_RE, item.chosen)
        r = _count(_REFUSAL_RE, item.rejected)
        if c == r:
            return None
        return 1 if c > r else 0

    if mechanism is BiasMechanism.SYCOPHANCY:
        c = _count(_AGREE_RE, item.chosen)
        r = _count(_AGREE_RE, item.rejected)
        if c == r:
            return None
        return 1 if c > r else 0

    return None


@dataclass(frozen=True)
class ReplayResult:
    """Per-variant simulated vote matrices ready for analysis."""

    variant: CFIBankVariant
    item_ids: tuple[str, ...]
    logical_ids: tuple[str, ...]
    V: np.ndarray  # (N items × n judges) vote ∈ {0, 1}
    M: np.ndarray  # (N items × n judges) availability ∈ {0, 1}
    biased_flag: np.ndarray  # (n judges,) 1 if judge is biased in this variant
    triggered_flag: np.ndarray  # (N items,) 1 if the variant's mechanism triggers on the item


def simulate_bank_votes(
    variant: CFIBankVariant,
    items: list[CalibrationItem],
    triggers: dict[str, CFITriggers],
    clean_votes: dict[str, dict[str, tuple[int, bool]]],
    seed: int = 20260601,
) -> ReplayResult:
    """Simulate one CFI bank variant by rewriting clean H1 votes on triggered items.

    Parameters
    ----------
    variant
        The bank variant to simulate.
    items
        Calibration items in canonical order; columns of V/M are aligned to
        ``variant.all_logical_ids``.
    triggers
        Mapping ``item_id → CFITriggers`` from the CFI manifest.
    clean_votes
        Mapping ``logical_id → {item_id: (vote, position_swap)}`` from the H1
        vote cache. ``vote`` is 1/0/-1 with -1 meaning the original judge
        abstained.
    seed
        Seed for the per-variant position swap used by the position-bias rule.

    Returns
    -------
    ReplayResult
        Vote matrix, availability mask, and bookkeeping flags suitable for
        feeding to ``corrfilter_score`` and the downstream metrics.
    """
    n_items = len(items)
    logical_ids = list(variant.all_logical_ids)
    n_judges = len(logical_ids)

    V = np.zeros((n_items, n_judges), dtype=np.int8)
    M = np.zeros((n_items, n_judges), dtype=np.int8)
    biased_flag = np.zeros(n_judges, dtype=np.int8)
    triggered_flag = np.zeros(n_items, dtype=np.int8)
    biased_set = set(variant.biased_logical_ids)

    for j, item in enumerate(items):
        trig = triggers.get(item.item_id)
        fires = bool(trig is not None and trig.fires(variant.mechanism))
        triggered_flag[j] = int(fires)
        swap = variant_position_swap(variant.name, item.item_id, seed=seed)
        prediction = biased_prediction(variant.mechanism, item, swap) if fires else None

        for i, lid in enumerate(logical_ids):
            is_biased = lid in biased_set
            biased_flag[i] = int(is_biased)
            cached = clean_votes.get(lid, {}).get(item.item_id)
            if cached is None:
                # No cached clean vote ⇒ this judge abstained on this item in H1.
                continue
            base_vote, _base_swap = cached
            if base_vote == -1:
                # The clean judge abstained — keep abstention regardless of bias.
                continue
            if is_biased and fires and prediction is not None:
                vote = int(prediction)
            else:
                vote = int(base_vote)
            V[j, i] = vote
            M[j, i] = 1

    return ReplayResult(
        variant=variant,
        item_ids=tuple(it.item_id for it in items),
        logical_ids=tuple(logical_ids),
        V=V,
        M=M,
        biased_flag=biased_flag,
        triggered_flag=triggered_flag,
    )


def load_clean_votes_from_cache(
    cache, logical_ids: list[str]
) -> dict[str, dict[str, tuple[int, bool]]]:
    """Materialise ``{logical_id: {item_id: (vote, position_swap)}}`` from the H1 cache.

    Accepts any object exposing ``load(logical_id) -> DataFrame | None`` with
    columns ``item_id``, ``vote``, ``position_swapped`` (i.e. the H1 ``VoteCache``).
    """
    out: dict[str, dict[str, tuple[int, bool]]] = {}
    for lid in logical_ids:
        df = cache.load(lid)
        if df is None:
            out[lid] = {}
            continue
        view: dict[str, tuple[int, bool]] = {}
        for row in df.itertuples(index=False):
            view[str(row.item_id)] = (
                int(row.vote),
                bool(row.position_swapped),
            )
        out[lid] = view
    return out


def replay_to_dataframe(result: ReplayResult) -> pd.DataFrame:
    """Long-form (variant, item, judge, vote, biased, triggered) DataFrame."""
    n_items, n_judges = result.V.shape
    rows = []
    for j in range(n_items):
        for i in range(n_judges):
            if not result.M[j, i]:
                continue
            rows.append({
                "variant": result.variant.name,
                "mechanism": result.variant.mechanism.value,
                "biased_ratio": result.variant.biased_ratio,
                "item_id": result.item_ids[j],
                "logical_id": result.logical_ids[i],
                "biased": bool(result.biased_flag[i]),
                "triggered": bool(result.triggered_flag[j]),
                "vote": int(result.V[j, i]),
            })
    return pd.DataFrame(rows)


__all__ = [
    "ReplayResult",
    "biased_prediction",
    "variant_position_swap",
    "simulate_bank_votes",
    "load_clean_votes_from_cache",
    "replay_to_dataframe",
]

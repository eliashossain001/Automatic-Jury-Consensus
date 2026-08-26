"""Assemble CFI bank variants from cached real votes (Bucket 3 analysis side).

The real-inference runner (scripts/cfi/run_bank_gpu.py) caches biased votes per
``(mechanism, logical_judge)`` and the clean control reuses the H1 cache. This
module assembles, for any ``(mechanism, biased_ratio)`` pair, the vote matrix a
bank with that fraction of biased judges would produce — selecting biased vs
clean votes per judge without any further inference.

Which judges are biased at a given ratio is chosen with the same deterministic,
nested rule used by :func:`corrfilter.cfi.banks.select_biased_judges` (lower
ratios ⊂ higher ratios), but keyed by the mechanism *name* so it works for the
config-driven 8-mechanism bank as well as the enum-based one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corrfilter.cfi.bias_bank import BiasCondition
from corrfilter.data import CalibrationItem
from corrfilter.voting import VoteCache

ABSTAIN = -1


def select_biased_judges_by_name(
    base_logical_ids: list[str], ratio: float, mechanism_name: str, seed: int = 20260601
) -> list[str]:
    """Deterministic biased-judge subset for ``ratio`` (nested across ratios).

    Mirrors :func:`corrfilter.cfi.banks.select_biased_judges` but takes the
    mechanism name directly so config-driven mechanisms (not in the enum) work.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1]; got {ratio}")
    n = len(base_logical_ids)
    k = int(round(ratio * n))
    if k == 0:
        return []
    if k == n:
        return list(base_logical_ids)
    offset = sum(ord(c) for c in mechanism_name)
    rng = np.random.default_rng(seed + offset)
    perm = rng.permutation(n)
    selected = sorted(perm[:k].tolist())
    return [base_logical_ids[i] for i in selected]


def load_vote_views(
    votes_dir, logical_ids: list[str]
) -> dict[str, dict[str, tuple[int, bool]]]:
    """``{logical_id: {item_id: (vote, position_swapped)}}`` from a VoteCache dir."""
    cache = VoteCache(votes_dir)
    out: dict[str, dict[str, tuple[int, bool]]] = {}
    for lid in logical_ids:
        df = cache.load(lid)
        view: dict[str, tuple[int, bool]] = {}
        if df is not None:
            for row in df.itertuples(index=False):
                view[str(row.item_id)] = (
                    int(row.vote),
                    bool(row.position_swapped),
                )
        out[lid] = view
    return out


@dataclass(frozen=True)
class AssembledVariant:
    """Vote matrices + bookkeeping for one (mechanism, ratio) bank."""

    mechanism: str
    biased_ratio: float
    item_ids: tuple[str, ...]
    logical_ids: tuple[str, ...]
    V: np.ndarray            # (N × n) vote ∈ {0, 1}
    M: np.ndarray            # (N × n) availability ∈ {0, 1}
    biased_flag: np.ndarray  # (n,) 1 if judge biased in this variant
    triggered_flag: np.ndarray  # (N,) 1 if the mechanism triggers on the item


def assemble_variant(
    items: list[CalibrationItem],
    logical_ids: list[str],
    condition: BiasCondition,
    ratio: float,
    biased_views: dict[str, dict[str, tuple[int, bool]]],
    clean_views: dict[str, dict[str, tuple[int, bool]]],
    seed: int = 20260601,
) -> AssembledVariant:
    """Build the (N × n) vote matrix for a bank with ``ratio`` biased judges.

    For each judge: if it is biased in this variant, use its cached biased vote
    for the mechanism (falling back to the clean vote when the biased run was
    restricted to triggered items); otherwise use its clean H1 vote.
    """
    n_items, n_judges = len(items), len(logical_ids)
    V = np.zeros((n_items, n_judges), dtype=np.int8)
    M = np.zeros((n_items, n_judges), dtype=np.int8)
    biased_flag = np.zeros(n_judges, dtype=np.int8)
    triggered_flag = np.array([int(condition.fires(it)) for it in items], dtype=np.int8)

    biased_set = set(select_biased_judges_by_name(logical_ids, ratio, condition.name, seed))

    for i, lid in enumerate(logical_ids):
        is_biased = lid in biased_set
        biased_flag[i] = int(is_biased)
        clean_view = clean_views.get(lid, {})
        biased_view = biased_views.get(lid, {})
        for j, it in enumerate(items):
            iid = it.item_id
            if is_biased and iid in biased_view:
                vote = biased_view[iid][0]
            elif iid in clean_view:
                vote = clean_view[iid][0]
            else:
                continue  # judge has no vote for this item → abstain
            if vote == ABSTAIN:
                continue
            V[j, i] = int(vote)
            M[j, i] = 1

    return AssembledVariant(
        mechanism=condition.name,
        biased_ratio=float(ratio),
        item_ids=tuple(it.item_id for it in items),
        logical_ids=tuple(logical_ids),
        V=V,
        M=M,
        biased_flag=biased_flag,
        triggered_flag=triggered_flag,
    )


__all__ = [
    "select_biased_judges_by_name",
    "load_vote_views",
    "AssembledVariant",
    "assemble_variant",
]

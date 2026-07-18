"""On-disk vote cache (parquet per logical judge).

Each logical judge writes one parquet file with rows ``(item_id, vote,
position_swapped, raw_response)``. The runner consults the cache before
loading a judge; if every item is already cached, the judge is skipped
entirely. This decouples downstream training from judge drift
(proposal §7.3: vote caching).

The cache also exposes ``load_vote_matrix`` which materialises an
``(N items × n logical judges)`` integer matrix from disk plus a parallel
error matrix once gold labels are joined.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from corrfilter.judges.base import JudgeVote

ABSTAIN = -1


@dataclass(frozen=True)
class VoteCache:
    """Per-logical-judge parquet cache rooted at ``base_dir``."""

    base_dir: Path

    def __post_init__(self):
        object.__setattr__(self, "base_dir", Path(self.base_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, logical_id: str) -> Path:
        safe = logical_id.replace("::", "__").replace("/", "_")
        return self.base_dir / f"{safe}.parquet"

    def exists(self, logical_id: str) -> bool:
        return self.path_for(logical_id).exists()

    def load(self, logical_id: str) -> pd.DataFrame | None:
        path = self.path_for(logical_id)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def cached_item_ids(self, logical_id: str) -> set[str]:
        df = self.load(logical_id)
        return set() if df is None else set(df["item_id"].astype(str).tolist())

    def write(self, logical_id: str, votes: Iterable[JudgeVote]) -> Path:
        rows = [
            {
                "item_id": str(v.item_id),
                "vote": int(v.vote),
                "position_swapped": bool(v.position_swapped),
                "raw_response": v.raw_response,
            }
            for v in votes
        ]
        path = self.path_for(logical_id)
        existing = self.load(logical_id)
        df_new = pd.DataFrame(rows)
        if existing is not None and len(existing) > 0:
            df = (
                pd.concat([existing, df_new], ignore_index=True)
                .drop_duplicates(subset=["item_id"], keep="last")
                .reset_index(drop=True)
            )
        else:
            df = df_new
        df.to_parquet(path, index=False)
        return path


def load_vote_matrix(
    cache: VoteCache,
    logical_ids: list[str],
    item_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise the ``(N × n)`` vote matrix V plus a missing-mask M.

    ``V[j, i]`` is judge ``i``'s vote on item ``j`` (1 = agree, 0 = disagree).
    ``M[j, i]`` is 1 where the judge cast a parseable vote, 0 where it
    abstained or the entry is absent. Callers should drop or impute on M.
    """
    n_items = len(item_ids)
    n_judges = len(logical_ids)
    V = np.zeros((n_items, n_judges), dtype=np.int8)
    M = np.zeros((n_items, n_judges), dtype=np.int8)
    item_index = {iid: k for k, iid in enumerate(item_ids)}

    for i, lid in enumerate(logical_ids):
        df = cache.load(lid)
        if df is None:
            continue
        for _, row in df.iterrows():
            iid = str(row["item_id"])
            if iid not in item_index:
                continue
            j = item_index[iid]
            vote = int(row["vote"])
            if vote == ABSTAIN:
                M[j, i] = 0
                continue
            V[j, i] = vote
            M[j, i] = 1

    return V, M

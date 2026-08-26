"""Calibration runner: load-score-unload one judge at a time.

Each logical judge in the bank is instantiated, asked to vote on every
calibration item not already cached, and unloaded before the next judge is
loaded. This is the only pattern that fits five 7-9B judges through a
single TITAN RTX, and it matches the version-pinning / vote-caching
posture in proposal §7.3.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from tqdm.auto import tqdm

from corrfilter.data import CalibrationItem
from corrfilter.judges.base import Judge, JudgeSpec
from corrfilter.voting.cache import VoteCache

logger = logging.getLogger(__name__)


def run_calibration(
    items: list[CalibrationItem],
    bank: Iterable[tuple[JudgeSpec, Callable[[], Judge]]],
    cache_dir: Path | str,
    skip_cached: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Vote every (judge, item) pair, caching results to disk.

    Returns a per-judge summary ``{logical_id: {n_voted, n_abstain, seconds}}``.
    """
    cache = VoteCache(Path(cache_dir))
    summary: dict[str, dict[str, float | int]] = {}

    for spec, factory in bank:
        lid = spec.logical_id
        already = cache.cached_item_ids(lid)
        pending = [it for it in items if it.item_id not in already] if skip_cached else list(items)

        if not pending:
            logger.info("judge %s fully cached; skipping load", lid)
            summary[lid] = {"n_voted": len(already), "n_abstain": 0, "seconds": 0.0, "skipped": 1}
            continue

        logger.info("judge %s: voting on %d items (cached %d)", lid, len(pending), len(already))
        judge = factory()
        t0 = time.perf_counter()
        try:
            judge.load()
            votes = []
            for batch_start in tqdm(
                range(0, len(pending), judge.batch_size if hasattr(judge, "batch_size") else 4),
                desc=lid,
                leave=False,
            ):
                bs = judge.batch_size if hasattr(judge, "batch_size") else 4
                batch = pending[batch_start : batch_start + bs]
                votes.extend(judge.vote_batch(batch))
            cache.write(lid, votes)
        finally:
            judge.unload()

        elapsed = time.perf_counter() - t0
        n_abstain = sum(1 for v in votes if v.vote == -1)
        summary[lid] = {
            "n_voted": len(votes),
            "n_abstain": n_abstain,
            "seconds": round(elapsed, 2),
            "skipped": 0,
        }
        logger.info(
            "judge %s done: %d voted (%d abstain) in %.1fs",
            lid,
            len(votes),
            n_abstain,
            elapsed,
        )

    return summary

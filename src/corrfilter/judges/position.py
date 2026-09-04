"""Deterministic candidate-slot assignment.

Replaces the use of Python's built-in ``hash()`` for position randomisation. ``hash()`` is
salted per process for ``str`` and ``bytes`` (PEP 456), so the slot a judge saw was not
reproducible across runs even with ``position_seed`` fixed. Measured consequence in the
published caches: for the same underlying judge, the base and GRPO banks agreed on the slot
for 51.4% of items, i.e. chance. Since position bias is this bank's strongest measured
shared vulnerability (content-independent slot gaps up to 70 points), any comparison that
assumes matched candidate positions was confounded.

``PYTHONHASHSEED`` is not an acceptable fix: it is an environment variable that any caller
can forget, it silently changes behaviour rather than failing loudly, and it does not make
already-written caches interpretable. This module uses BLAKE2b instead, which is stable
across processes, machines, Python versions, and ``PYTHONHASHSEED`` values.

Realized assignments should additionally be persisted alongside experiment outputs; the
vote caches already carry a ``position_swapped`` column for that purpose, and
:func:`realized_assignments` reads them back.
"""

from __future__ import annotations

from hashlib import blake2b
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_POSITION_SEED = 20260601


def position_swap(item_id: str, logical_id: str, position_seed: int = DEFAULT_POSITION_SEED) -> bool:
    """Return whether ``chosen`` is presented in slot B for this (judge, item).

    Deterministic and stable everywhere: BLAKE2b over a length-delimited encoding of the
    three inputs. Length delimiting prevents collisions between, say,
    ``("ab", "c")`` and ``("a", "bc")``.
    """
    payload = b"|".join(
        [str(position_seed).encode(), str(logical_id).encode(), str(item_id).encode()]
    )
    digest = blake2b(payload, digest_size=8).digest()
    return bool(digest[0] & 1)


def assignment_table(item_ids, logical_id: str,
                     position_seed: int = DEFAULT_POSITION_SEED) -> pd.DataFrame:
    """Full (item_id, position_swapped) table for one logical judge."""
    import pandas as pd

    return pd.DataFrame({
        "item_id": [str(i) for i in item_ids],
        "position_swapped": [position_swap(i, logical_id, position_seed) for i in item_ids],
    })


def realized_assignments(vote_parquet: str | Path) -> dict[str, bool]:
    """Read back the slot assignment actually used, from a written vote cache."""
    import pandas as pd

    df = pd.read_parquet(vote_parquet)
    df = df.drop_duplicates(subset="item_id", keep="first")
    return {str(r.item_id): bool(r.position_swapped) for r in df.itertuples()}

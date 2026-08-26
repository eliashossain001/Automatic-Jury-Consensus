"""Regression tests for deterministic candidate-slot assignment.

The old implementation used Python's built-in hash(), salted per process (PEP 456), so slot
assignment varied run to run. These tests fail against it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from corrfilter.judges.position import DEFAULT_POSITION_SEED, assignment_table, position_swap

ROOT = Path(__file__).resolve().parents[1]


ITEMS = ["452", "245", "356", "263", "406", "item-with-dashes", "0", ""]
JUDGE = "qwen-2.5-7b::pairwise"


def test_deterministic_within_process():
    a = [position_swap(i, JUDGE) for i in ITEMS]
    b = [position_swap(i, JUDGE) for i in ITEMS]
    assert a == b


def _child_assignments(hashseed: str) -> list[bool]:
    """Compute assignments in a fresh interpreter with a given PYTHONHASHSEED."""
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=str(ROOT / "src"))
    code = (
        "import json;"
        "from corrfilter.judges.position import position_swap;"
        f"print(json.dumps([position_swap(i, {JUDGE!r}) for i in {ITEMS!r}]))"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True)
    import json
    return json.loads(out.stdout)


def test_stable_across_processes_and_hashseeds():
    """The defect: identical inputs must give identical slots in any interpreter."""
    reference = [position_swap(i, JUDGE) for i in ITEMS]
    for seed in ("0", "1", "12345", "random"):
        assert _child_assignments(seed) == reference, f"unstable under PYTHONHASHSEED={seed}"


def test_builtin_hash_is_actually_unstable():
    """Guard the guard: confirm the mechanism we replaced really was unstable here.

    If this ever fails, the environment pins PYTHONHASHSEED and the original defect would
    have been invisible, which is worth knowing.
    """
    code = "print(hash(('x', 'y')) & 0xFFFFFFFF)"
    seen = set()
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        seen.add(subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, check=True).stdout.strip())
    assert len(seen) > 1, "builtin hash appears stable; PYTHONHASHSEED may be pinned"


def test_depends_on_all_three_inputs():
    base = position_swap("452", JUDGE, DEFAULT_POSITION_SEED)
    assert isinstance(base, bool)
    # Different judge, item, or seed must be able to change the assignment. Each flip is
    # probabilistic per item, so assert over a set rather than a single item.
    def rate(judge=JUDGE, seed=DEFAULT_POSITION_SEED, salt=""):
        return sum(position_swap(f"{i}{salt}", judge, seed) for i in range(500))

    assert rate() != rate(judge="other::likert")
    assert rate() != rate(seed=DEFAULT_POSITION_SEED + 1)
    assert rate() != rate(salt="x")


def test_assignment_is_balanced():
    """Slot assignment should be close to 50/50 so position bias is not induced."""
    swaps = [position_swap(str(i), JUDGE) for i in range(5000)]
    rate = sum(swaps) / len(swaps)
    assert 0.45 < rate < 0.55, f"slot assignment imbalanced: {rate}"


def test_length_delimited_encoding_avoids_collisions():
    """("ab","c") and ("a","bc") must not collide."""
    n_same = sum(position_swap("b" + str(i), "a") == position_swap(str(i), "ab")
                 for i in range(400))
    assert n_same != 400, "concatenation collision: inputs are not length-delimited"


def test_assignment_table_matches_scalar_function():
    df = assignment_table(ITEMS, JUDGE)
    assert list(df.item_id) == ITEMS
    assert list(df.position_swapped) == [position_swap(i, JUDGE) for i in ITEMS]


def test_realized_assignments_roundtrip(tmp_path):
    from corrfilter.judges.position import realized_assignments
    df = assignment_table(ITEMS, JUDGE)
    df["vote"] = 1
    p = tmp_path / "votes.parquet"
    df.to_parquet(p)
    got = realized_assignments(p)
    assert got == {i: position_swap(i, JUDGE) for i in ITEMS}

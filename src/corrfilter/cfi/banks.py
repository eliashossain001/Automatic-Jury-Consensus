"""Bank-variant construction for the CFI experiment.

For each bias mechanism, the experiment compares five banks:

* ``pct000``: clean (no biased judges)
* ``pct025`` / ``pct050`` / ``pct075``: mixed banks
* ``pct100``: all-biased

The same base judge identities are used across variants so that the
clean/biased flip is the only manipulated variable. Which judges are biased
at a given ratio is chosen deterministically from a per-mechanism seed: lower
ratios are always strict subsets of higher ratios, so the 25% biased subset
is contained in the 50% subset, which is contained in the 75% subset. This
makes the curves comparable across ratios.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from corrfilter.cfi.biases import BiasMechanism
from corrfilter.judges.base import JudgeSpec


@dataclass(frozen=True)
class CFIBankVariant:
    """One bank variant: a choice of which judges are biased and which are clean."""

    name: str
    mechanism: BiasMechanism
    biased_ratio: float
    biased_logical_ids: tuple[str, ...]
    clean_logical_ids: tuple[str, ...]

    @property
    def all_logical_ids(self) -> tuple[str, ...]:
        # Preserve clean-then-biased order so consumers can rely on a stable layout.
        return tuple(self.clean_logical_ids) + tuple(self.biased_logical_ids)

    def is_biased(self, logical_id: str) -> bool:
        return logical_id in self.biased_logical_ids


def _mechanism_permutation(
    base_logical_ids: Sequence[str], mechanism: BiasMechanism, seed: int
) -> np.ndarray:
    """Deterministic per-mechanism permutation used to assign biased judges.

    The same seed + mechanism yields the same ordering across ratios so that
    25% biased ⊂ 50% biased ⊂ 75% biased ⊂ 100% biased.
    """
    # Stable hash from the mechanism string so the permutation does not depend
    # on the Python process's PYTHONHASHSEED.
    mech_offset = sum(ord(c) for c in mechanism.value)
    rng = np.random.default_rng(seed + mech_offset)
    return rng.permutation(len(base_logical_ids))


def select_biased_judges(
    base_logical_ids: Sequence[str],
    ratio: float,
    mechanism: BiasMechanism,
    seed: int = 20260601,
) -> list[str]:
    """Pick ``round(ratio * n)`` judges to bias under the deterministic permutation."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1]; got {ratio}")
    n = len(base_logical_ids)
    k = int(round(ratio * n))
    if k == 0:
        return []
    if k == n:
        return list(base_logical_ids)
    perm = _mechanism_permutation(base_logical_ids, mechanism, seed)
    selected = sorted(perm[:k].tolist())
    return [base_logical_ids[i] for i in selected]


def build_bank_variants(
    base_specs: Sequence[JudgeSpec],
    mechanisms: Sequence[BiasMechanism],
    ratios: Sequence[float],
    seed: int = 20260601,
) -> list[CFIBankVariant]:
    """Materialise all (mechanism, ratio) variants for the given base bank."""
    base_logical_ids = [s.logical_id for s in base_specs]
    variants: list[CFIBankVariant] = []
    for mech in mechanisms:
        for r in ratios:
            biased = select_biased_judges(base_logical_ids, r, mech, seed=seed)
            biased_set = set(biased)
            clean = [lid for lid in base_logical_ids if lid not in biased_set]
            variants.append(
                CFIBankVariant(
                    name=f"{mech.value}_pct{int(round(r * 100)):03d}",
                    mechanism=mech,
                    biased_ratio=float(r),
                    biased_logical_ids=tuple(biased),
                    clean_logical_ids=tuple(clean),
                )
            )
    return variants


__all__ = ["CFIBankVariant", "select_biased_judges", "build_bank_variants"]

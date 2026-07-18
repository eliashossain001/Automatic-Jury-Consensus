"""Tests for CFI bank-variant construction."""

from __future__ import annotations

from corrfilter.cfi.banks import build_bank_variants, select_biased_judges
from corrfilter.cfi.biases import BiasMechanism
from corrfilter.judges.base import JudgeSpec


def _specs(n: int) -> list[JudgeSpec]:
    return [
        JudgeSpec(
            base_id=f"j{i}",
            family="fam",
            scale="7B",
            prompt_style="pairwise",
            hf_model=f"org/j{i}",
            quantization="none",
        )
        for i in range(n)
    ]


def test_select_biased_judges_count_matches_ratio():
    ids = [s.logical_id for s in _specs(10)]
    biased = select_biased_judges(ids, ratio=0.5, mechanism=BiasMechanism.VERBOSITY)
    assert len(biased) == 5


def test_select_biased_is_monotone_across_ratios():
    """The 25% biased subset must be contained in the 50% subset, etc."""
    ids = [s.logical_id for s in _specs(10)]
    a = set(select_biased_judges(ids, 0.25, BiasMechanism.POSITION))
    b = set(select_biased_judges(ids, 0.5, BiasMechanism.POSITION))
    c = set(select_biased_judges(ids, 0.75, BiasMechanism.POSITION))
    d = set(select_biased_judges(ids, 1.0, BiasMechanism.POSITION))
    assert a.issubset(b)
    assert b.issubset(c)
    assert c.issubset(d)
    assert d == set(ids)


def test_select_biased_is_deterministic_per_mechanism_seed():
    ids = [s.logical_id for s in _specs(10)]
    a = select_biased_judges(ids, 0.5, BiasMechanism.POSITION, seed=42)
    b = select_biased_judges(ids, 0.5, BiasMechanism.POSITION, seed=42)
    assert a == b


def test_select_biased_mechanisms_can_differ():
    ids = [s.logical_id for s in _specs(10)]
    a = set(select_biased_judges(ids, 0.5, BiasMechanism.POSITION, seed=42))
    b = set(select_biased_judges(ids, 0.5, BiasMechanism.VERBOSITY, seed=42))
    # Different mechanisms should generally pick a different subset.
    assert a != b


def test_build_bank_variants_produces_full_grid():
    specs = _specs(10)
    mechanisms = [BiasMechanism.VERBOSITY, BiasMechanism.POSITION]
    ratios = [0.0, 0.5, 1.0]
    variants = build_bank_variants(specs, mechanisms, ratios)
    assert len(variants) == 6
    for v in variants:
        assert len(v.all_logical_ids) == 10
        assert set(v.biased_logical_ids).isdisjoint(v.clean_logical_ids)


def test_build_bank_variants_clean_variant_is_empty():
    specs = _specs(10)
    variants = build_bank_variants(specs, [BiasMechanism.SYCOPHANCY], [0.0])
    assert variants[0].biased_logical_ids == ()
    assert len(variants[0].clean_logical_ids) == 10

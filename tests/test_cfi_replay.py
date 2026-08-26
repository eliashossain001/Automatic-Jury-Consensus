"""Tests for the deterministic CFI replay engine."""

from __future__ import annotations

from corrfilter.cfi.banks import CFIBankVariant
from corrfilter.cfi.biases import BiasMechanism
from corrfilter.cfi.manifest import CFITriggers
from corrfilter.cfi.prompts import BiasedPrompt
from corrfilter.cfi.replay import (
    biased_prediction,
    simulate_bank_votes,
    variant_position_swap,
)
from corrfilter.data import CalibrationItem
from corrfilter.judges.prompts import PairwisePrompt


def _item(item_id: str, chosen: str = "x", rejected: str = "y", subset: str = "Focus", category: str = "chat", prompt: str = "Q?") -> CalibrationItem:
    return CalibrationItem(
        item_id=item_id, prompt=prompt, chosen=chosen, rejected=rejected, subset=subset, category=category,
    )


def _triggers(item_id: str, mech: BiasMechanism) -> CFITriggers:
    flags = {m.value: False for m in BiasMechanism}
    flags[mech.value] = True
    return CFITriggers(item_id=item_id, **flags)


def _variant(mech: BiasMechanism, biased_ids: tuple[str, ...], clean_ids: tuple[str, ...], biased_ratio: float | None = None) -> CFIBankVariant:
    if biased_ratio is None:
        biased_ratio = len(biased_ids) / max(1, len(biased_ids) + len(clean_ids))
    return CFIBankVariant(
        name=f"{mech.value}_test",
        mechanism=mech,
        biased_ratio=biased_ratio,
        biased_logical_ids=biased_ids,
        clean_logical_ids=clean_ids,
    )


def test_biased_prediction_verbosity_picks_longer_side():
    it = _item("v", chosen="long" * 50, rejected="short")
    assert biased_prediction(BiasMechanism.VERBOSITY, it, position_swap=False) == 1
    it2 = _item("v2", chosen="x", rejected="extended" * 30)
    assert biased_prediction(BiasMechanism.VERBOSITY, it2, position_swap=False) == 0


def test_biased_prediction_position_picks_A():
    it = _item("p")
    # No swap → A = chosen → vote 1.
    assert biased_prediction(BiasMechanism.POSITION, it, position_swap=False) == 1
    # Swap → A = rejected → vote 0.
    assert biased_prediction(BiasMechanism.POSITION, it, position_swap=True) == 0


def test_biased_prediction_polish_prefers_confident_vocab():
    it = _item("pol", chosen="Definitely. Clearly. Indeed.", rejected="Maybe.")
    assert biased_prediction(BiasMechanism.POLISH, it, position_swap=False) == 1
    it2 = _item("pol2", chosen="not sure", rejected="Definitely. Clearly. Without doubt.")
    assert biased_prediction(BiasMechanism.POLISH, it2, position_swap=False) == 0


def test_biased_prediction_refusal_prefers_refusal_lexicon():
    it = _item("r", chosen="I cannot help with that.", rejected="Sure, here you go.")
    assert biased_prediction(BiasMechanism.REFUSAL, it, position_swap=False) == 1


def test_biased_prediction_returns_none_on_tie():
    it = _item("t", chosen="hello", rejected="world")  # same length
    assert biased_prediction(BiasMechanism.VERBOSITY, it, position_swap=False) is None


def test_variant_position_swap_is_deterministic():
    a = variant_position_swap("v", "x", seed=1)
    b = variant_position_swap("v", "x", seed=1)
    assert a == b


def test_simulate_bank_votes_biased_judges_agree_on_triggered():
    items = [_item("i0", chosen="x" * 100, rejected="y")]
    triggers = {"i0": _triggers("i0", BiasMechanism.VERBOSITY)}
    clean_votes = {
        "j0": {"i0": (1, False)},
        "j1": {"i0": (0, False)},
        "j2": {"i0": (1, False)},
        "j3": {"i0": (0, False)},
    }
    variant = _variant(BiasMechanism.VERBOSITY, biased_ids=("j1", "j3"), clean_ids=("j0", "j2"))
    res = simulate_bank_votes(variant, items, triggers, clean_votes)
    # All biased judges should produce the same biased vote (chosen is longer → 1).
    biased_cols = [res.logical_ids.index(lid) for lid in variant.biased_logical_ids]
    assert all(res.V[0, i] == 1 for i in biased_cols)
    # Clean judges retain their original cached vote (one disagrees).
    clean_cols = [res.logical_ids.index(lid) for lid in variant.clean_logical_ids]
    cleaned = sorted([int(res.V[0, i]) for i in clean_cols])
    assert cleaned == [1, 1]  # Both clean judges had vote 1


def test_simulate_bank_votes_falls_back_on_non_triggered_items():
    items = [_item("i0", chosen="x", rejected="y")]  # no length gap → trigger disabled
    flags = {m.value: False for m in BiasMechanism}
    triggers = {"i0": CFITriggers(item_id="i0", **flags)}
    clean_votes = {"j0": {"i0": (1, False)}, "j1": {"i0": (0, False)}}
    variant = _variant(BiasMechanism.VERBOSITY, biased_ids=("j1",), clean_ids=("j0",))
    res = simulate_bank_votes(variant, items, triggers, clean_votes)
    j0 = res.logical_ids.index("j0")
    j1 = res.logical_ids.index("j1")
    # Both judges keep their original cached vote.
    assert int(res.V[0, j0]) == 1
    assert int(res.V[0, j1]) == 0


def test_simulate_bank_votes_preserves_abstention_in_clean_cache():
    items = [_item("i0", chosen="x" * 100, rejected="y")]
    triggers = {"i0": _triggers("i0", BiasMechanism.VERBOSITY)}
    clean_votes = {"j0": {"i0": (-1, False)}, "j1": {"i0": (0, False)}}
    variant = _variant(BiasMechanism.VERBOSITY, biased_ids=("j1",), clean_ids=("j0",))
    res = simulate_bank_votes(variant, items, triggers, clean_votes)
    j0 = res.logical_ids.index("j0")
    assert res.M[0, j0] == 0  # judge 0 abstained in clean cache, stays abstained


def test_biased_prompt_injects_into_system_section():
    base = PairwisePrompt()
    from corrfilter.cfi.biases import BIAS_REGISTRY

    spec = BIAS_REGISTRY[BiasMechanism.POSITION]
    bp = BiasedPrompt(base, spec)
    it = _item("p")
    rendered = bp.render(it, swap=False)
    assert "[Judge guidance]" in rendered.text
    assert "Response A" in rendered.text  # base render still present


def test_biased_prompt_decode_inherits_base_semantics():
    base = PairwisePrompt()
    from corrfilter.cfi.biases import BIAS_REGISTRY

    bp = BiasedPrompt(base, BIAS_REGISTRY[BiasMechanism.POSITION])
    assert bp.decode("A", swapped=False) == 1
    assert bp.decode("B", swapped=False) == 0
    assert bp.decode("A", swapped=True) == 0
    assert bp.decode("B", swapped=True) == 1


def test_simulate_bank_votes_position_uses_shared_per_item_swap():
    """All biased judges in a variant should see the same per-item swap so position bias correlates."""
    item = _item("p")
    triggers = {"p": _triggers("p", BiasMechanism.POSITION)}
    clean_votes = {
        "j0": {"p": (1, False)},
        "j1": {"p": (1, True)},
        "j2": {"p": (0, False)},
    }
    variant = _variant(BiasMechanism.POSITION, biased_ids=("j0", "j1"), clean_ids=("j2",))
    res = simulate_bank_votes(variant, [item], triggers, clean_votes)
    # Shared swap: both biased judges should produce the same vote (0 or 1).
    j0 = res.logical_ids.index("j0")
    j1 = res.logical_ids.index("j1")
    assert int(res.V[0, j0]) == int(res.V[0, j1])

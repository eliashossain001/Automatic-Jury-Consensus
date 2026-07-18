"""Tests for CFI heuristic trigger detection."""

from __future__ import annotations

import pandas as pd

from corrfilter.cfi.biases import BiasMechanism
from corrfilter.cfi.manifest import (
    build_cfi_manifest,
    detect_triggers,
    triggers_from_row,
)
from corrfilter.cfi.triggers import (
    TriggerConfig,
    detect_polish_trigger,
    detect_position_trigger,
    detect_refusal_trigger,
    detect_sycophancy_trigger,
    detect_verbosity_trigger,
)
from corrfilter.data import CalibrationItem


def _item(
    item_id: str = "t",
    prompt: str = "Pick the better answer.",
    chosen: str = "Yes.",
    rejected: str = "No.",
    subset: str = "Focus",
    category: str = "chat",
) -> CalibrationItem:
    return CalibrationItem(
        item_id=item_id,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        subset=subset,
        category=category,
    )


def test_verbosity_trigger_fires_on_large_length_gap():
    it = _item(chosen="a" * 200, rejected="short")
    assert detect_verbosity_trigger(it)


def test_verbosity_trigger_does_not_fire_on_small_gap():
    it = _item(chosen="abc", rejected="ab")
    assert not detect_verbosity_trigger(it)


def test_verbosity_threshold_is_configurable():
    it = _item(chosen="abcd", rejected="ab")  # ratio 0.5
    assert detect_verbosity_trigger(it, TriggerConfig(verbosity_ratio=0.4))
    assert not detect_verbosity_trigger(it, TriggerConfig(verbosity_ratio=0.9))


def test_sycophancy_trigger_detects_opinion_marker():
    it = _item(prompt="I think Python is the best language, right?")
    assert detect_sycophancy_trigger(it)


def test_sycophancy_trigger_ignores_neutral_prompt():
    it = _item(prompt="What is the capital of France?")
    assert not detect_sycophancy_trigger(it)


def test_position_trigger_always_fires():
    assert detect_position_trigger(_item())


def test_polish_trigger_fires_on_factuality_subset():
    it = _item(subset="Factuality")
    assert detect_polish_trigger(it)


def test_polish_trigger_respects_subset_override():
    it = _item(subset="Ties", category="hard")
    assert not detect_polish_trigger(it)


def test_refusal_trigger_fires_on_safety_subset():
    it = _item(subset="Safety", category="safety")
    assert detect_refusal_trigger(it)


def test_refusal_trigger_fires_on_refusal_lexicon_in_responses():
    it = _item(subset="Focus", category="chat", chosen="I cannot help with that request.", rejected="Sure, here goes.")
    assert detect_refusal_trigger(it)


def test_detect_triggers_returns_all_fields():
    it = _item(
        prompt="I believe the moon is made of cheese, isn't it?",
        chosen="x" * 100, rejected="y",
        subset="Factuality",
    )
    trig = detect_triggers(it)
    assert trig.item_id == "t"
    assert trig.verbosity is True
    assert trig.sycophancy is True
    assert trig.position is True
    assert trig.polish is True
    assert isinstance(trig.refusal, bool)


def test_build_cfi_manifest_round_trips_through_row():
    items = [
        _item("a", prompt="What is 2+2?", chosen="four", rejected="five"),
        _item("b", prompt="I think GPT is great, right?", chosen="agree", rejected="disagree", subset="Safety", category="safety"),
    ]
    df = build_cfi_manifest(items)
    assert set(df.columns) >= {
        "item_id", "verbosity", "sycophancy", "position", "polish", "refusal",
    }
    row = df.iloc[1]
    trig = triggers_from_row(row)
    assert trig.item_id == "b"
    assert trig.fires(BiasMechanism.SYCOPHANCY)
    assert trig.fires(BiasMechanism.REFUSAL)
    assert isinstance(df, pd.DataFrame)

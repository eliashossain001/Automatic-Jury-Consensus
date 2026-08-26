"""Regression tests for the APIJudge cost meter (Stage-P pilot bug).

The pilot's per-judge cost read $0.00 because the six pinned frontier model
IDs were missing from PRICES, so the $2.50 budget ceiling was not actively
enforced. These tests pin the fix: prices present, arithmetic correct,
reconstruction matches the exactly-billed pilot spend, and unknown-price
live judges are flagged."""

from pathlib import Path

import pandas as pd
import pytest

from corrfilter.judges.api_judge import PRICES, Usage

ROOT = Path(__file__).resolve().parents[1]


PINNED = ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.5-flash-lite",
          "openai/gpt-5.6-sol", "anthropic/claude-opus-5", "x-ai/grok-4.5"]


def test_pinned_models_have_prices():
    for m in PINNED:
        pin, pout = PRICES.get(m, (None, None))
        assert pin and pout, f"missing PRICES entry for {m}"


def test_usage_cost_arithmetic():
    u = Usage(calls=1, input_tokens=1_000_000, output_tokens=500_000)
    assert u.cost(2.0, 12.0) == pytest.approx(2.0 + 6.0)


@pytest.mark.skipif(
    not (ROOT / "outputs/screening_bank/frontier_code_pilot_runlog.csv").exists(),
    reason="pilot artifacts absent")
def test_pilot_cost_reconstruction_matches_billing():
    """Token counts x fixed prices must reproduce the exactly-billed
    OpenRouter delta of $2.217 (10.379 -> 8.162) within a few cents."""
    log = pd.read_csv(ROOT / "outputs/screening_bank/frontier_code_pilot_runlog.csv")
    or_ids = {"gpt-5.6-sol": "openai/gpt-5.6-sol",
              "claude-opus-5": "anthropic/claude-opus-5",
              "grok-4.5": "x-ai/grok-4.5"}
    tot = 0.0
    for _, r in log[log.judge.isin(or_ids)].iterrows():
        pin, pout = PRICES[or_ids[r.judge]]
        tot += r.in_tok / 1e6 * pin + r.out_tok / 1e6 * pout
    assert tot == pytest.approx(2.217, abs=0.05)


def test_unknown_price_live_judge_is_flagged():
    from corrfilter.judges.api_judge import APIJudge
    from corrfilter.judges.base import JudgeSpec

    spec = JudgeSpec(base_id="x", family="f", scale="s", prompt_style="pairwise",
                     hf_model="", quantization="none", dtype="bfloat16")
    j_unknown = APIJudge(spec, provider="openrouter", model="not/a-real-model",
                         dry_run=True)
    assert j_unknown.cost_known is False
    j_known = APIJudge(spec, provider="openrouter", model="x-ai/grok-4.5",
                       dry_run=True)
    assert j_known.cost_known is True and j_known.price_in > 0

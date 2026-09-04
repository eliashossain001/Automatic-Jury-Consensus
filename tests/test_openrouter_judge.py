"""Tests for the OpenRouter judge adapter: payload shape, parsing, failure posture."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrfilter.data import CalibrationItem
from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.openrouter_judge import OpenRouterJudge


def _judge(**kw):
    kw.setdefault("dry_run", False)
    kw.setdefault("backoff_base_s", 0.0)
    kw.setdefault("backoff_max_s", 0.0)
    spec = JudgeSpec(base_id="or-test", family="openai-gpt", scale="flagship",
                     prompt_style="pairwise", hf_model="openai/test")
    j = OpenRouterJudge(spec, model="openai/test", **kw)
    j._api_key = "fake"
    return j


def _item():
    return CalibrationItem(item_id="i1", prompt="2+2?", chosen="4", rejected="5",
                           subset="Math", category="reasoning")


def _resp(content, finish="stop", reasoning_tokens=0):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 3,
                      "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}}


def test_payload_reasoning_and_temperature_flags():
    j = _judge(reasoning_effort="minimal", supports_temperature=False)
    p = j._payload("s", "u")
    assert p["reasoning"] == {"effort": "minimal"}
    assert "temperature" not in p
    j2 = _judge(reasoning_exclude=True, supports_temperature=True)
    p2 = j2._payload("s", "u")
    assert p2["reasoning"] == {"exclude": True}
    assert p2["temperature"] == 0.0


def test_content_vote_parses_and_reasoning_tracked():
    j = _judge()
    with mock.patch.object(OpenRouterJudge, "_request_once", return_value=_resp("B", reasoning_tokens=77)):
        votes = j.vote_batch([_item()])
    assert votes[0].vote in (0, 1)
    assert j.reasoning_tokens == 77


def test_empty_content_on_length_is_truncation_abstain():
    j = _judge()
    with mock.patch.object(OpenRouterJudge, "_request_once", return_value=_resp(None, finish="length")):
        votes = j.vote_batch([_item()])
    assert votes[0].vote == -1
    assert votes[0].raw_response.startswith("<TRUNCATED")
    assert j.failures[0]["kind"] == "truncated"


def test_402_is_hard_stop_not_retried():
    j = _judge(max_retries=5)
    calls = {"n": 0}

    class FakeResp:
        status_code = 402
        text = "Insufficient credits"

    def post(*a, **k):
        calls["n"] += 1
        return FakeResp()

    with mock.patch("requests.post", side_effect=post):
        votes = j.vote_batch([_item()])
    assert calls["n"] == 1                       # never retried
    assert votes[0].vote == -1
    assert "402" in j.failures[0]["detail"]

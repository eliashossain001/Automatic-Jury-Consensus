"""Tests for the Gemini REST judge: parsing, retries, failure posture, protocol parity."""

from __future__ import annotations

from unittest import mock

from corrfilter.data import CalibrationItem
from corrfilter.judges.api_judge import APIJudge
from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.gemini_judge import GeminiJudge, TransientAPIError


def _spec() -> JudgeSpec:
    return JudgeSpec(
        base_id="gemini-test", family="gemini", scale="flash",
        prompt_style="pairwise", hf_model="gemini-test-model",
    )


def _item(iid: str = "it1") -> CalibrationItem:
    return CalibrationItem(
        item_id=iid, prompt="What is 2+2?", chosen="4", rejected="5",
        subset="Math", category="reasoning",
    )


def _response(text: str, finish: str = "STOP", thoughts: int = 0) -> dict:
    return {
        "candidates": [{
            "content": {"parts": [{"text": text}]},
            "finishReason": finish,
        }],
        "usageMetadata": {
            "promptTokenCount": 50, "candidatesTokenCount": 1,
            "thoughtsTokenCount": thoughts,
        },
    }


def _judge(**kw) -> GeminiJudge:
    kw.setdefault("dry_run", False)
    kw.setdefault("backoff_base_s", 0.0)
    kw.setdefault("backoff_max_s", 0.0)
    j = GeminiJudge(_spec(), model="gemini-test-model", **kw)
    j._api_key = "fake-key"
    return j


def test_dry_run_makes_no_http_calls():
    judge = GeminiJudge(_spec(), model="gemini-test-model", dry_run=True)
    judge.load()
    with mock.patch.object(GeminiJudge, "_request_once", side_effect=AssertionError("HTTP in dry-run")):
        votes = judge.vote_batch([_item()])
    assert len(votes) == 1
    assert votes[0].vote in (0, 1)
    assert judge.usage.calls == 1


def test_parse_vote_and_position_decode():
    judge = _judge()
    swap = judge._position_swap("it1")
    # Model answers "A": agrees iff chosen was presented as A (not swapped).
    with mock.patch.object(GeminiJudge, "_request_once", return_value=_response("A")):
        votes = judge.vote_batch([_item()])
    assert votes[0].vote == (0 if swap else 1)
    assert votes[0].position_swapped == swap


def test_position_swap_matches_open_bank_protocol():
    """Same stable-hash swap as APIJudge/HFJudge: seed+logical_id+item_id keyed."""
    gem = _judge()
    api = APIJudge(_spec(), provider="openai", model="x", dry_run=True)
    for iid in ["a", "b", "c", "42"]:
        assert gem._position_swap(iid) == api._position_swap(iid)


def test_retry_then_success():
    judge = _judge(max_retries=3)
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientAPIError("HTTP 429")
        return _response("B")

    with mock.patch.object(GeminiJudge, "_request_once", side_effect=flaky):
        votes = judge.vote_batch([_item()])
    assert calls["n"] == 3
    assert votes[0].vote in (0, 1)
    assert judge.failures == []


def test_exhausted_retries_records_abstain_and_failure():
    judge = _judge(max_retries=2)
    with mock.patch.object(GeminiJudge, "_request_once", side_effect=TransientAPIError("HTTP 503")):
        votes = judge.vote_batch([_item()])
    assert votes[0].vote == -1
    assert votes[0].raw_response.startswith("<API_ERROR")
    assert judge.failures and judge.failures[0]["kind"] == "api_error"
    assert judge.usage.abstentions == 1


def test_refusal_records_abstain():
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 50}}
    judge = _judge()
    with mock.patch.object(GeminiJudge, "_request_once", return_value=blocked):
        votes = judge.vote_batch([_item()])
    assert votes[0].vote == -1
    assert votes[0].raw_response == "<REFUSAL:SAFETY>"
    assert judge.failures[0]["kind"] == "refusal"


def test_error_sentinel_never_decodes_as_vote():
    """An error message containing a standalone 'A' must not become a vote."""
    judge = _judge(max_retries=0)
    with mock.patch.object(
        GeminiJudge, "_request_once",
        side_effect=TransientAPIError("quota A exceeded"),
    ):
        votes = judge.vote_batch([_item()])
    assert votes[0].vote == -1


def test_thought_tokens_accumulated_separately():
    judge = _judge()
    with mock.patch.object(GeminiJudge, "_request_once", return_value=_response("A", thoughts=146)):
        judge.vote_batch([_item()])
    assert judge.thought_tokens == 146
    assert judge.usage.output_tokens == 1


def test_chatty_response_still_parses():
    judge = _judge()
    with mock.patch.object(GeminiJudge, "_request_once", return_value=_response("The answer is B.")):
        votes = judge.vote_batch([_item()])
    assert votes[0].vote in (0, 1)


def test_workers_preserve_item_order():
    judge = _judge(workers=4)
    with mock.patch.object(GeminiJudge, "_request_once", return_value=_response("A")):
        items = [_item(f"it{k}") for k in range(10)]
        votes = judge.vote_batch(items)
    assert [v.item_id for v in votes] == [f"it{k}" for k in range(10)]


def test_payload_thinking_config():
    j_min = _judge(thinking_level="minimal")
    assert j_min._payload("s", "u")["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    j_none = _judge()
    assert "thinkingConfig" not in j_none._payload("s", "u")["generationConfig"]
    assert j_none._payload("s", "u")["generationConfig"]["temperature"] == 0.0

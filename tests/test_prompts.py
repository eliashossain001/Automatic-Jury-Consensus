"""Tests for prompt rendering + position-aware decoding.

Position-bias mitigation is load-bearing: if decode disagrees with render about
which side is "chosen", every judge vote is inverted on swapped items, and the
resulting correlation matrix is meaningless. These tests pin the contract.
"""

from __future__ import annotations

from corrfilter.data import CalibrationItem
from corrfilter.judges.prompts import LikertPrompt, PairwisePrompt


def _item() -> CalibrationItem:
    return CalibrationItem(
        item_id="t0",
        prompt="What is the capital of France?",
        chosen="Paris.",
        rejected="London.",
        subset="Factuality",
        category="chat",
    )


def test_pairwise_no_swap_a_means_agree():
    p = PairwisePrompt()
    rp = p.render(_item(), swap=False)
    assert "Response A:\nParis." in rp.text
    assert "Response B:\nLondon." in rp.text
    assert rp.swapped is False
    # Judge picks A → agrees that chosen ≻ rejected.
    assert p.decode("A", swapped=False) == 1
    assert p.decode("B", swapped=False) == 0


def test_pairwise_swap_b_means_agree():
    p = PairwisePrompt()
    rp = p.render(_item(), swap=True)
    assert "Response A:\nLondon." in rp.text
    assert "Response B:\nParis." in rp.text
    assert rp.swapped is True
    # With swap, chosen is B → judge picking B agrees with the gold preference.
    assert p.decode("A", swapped=True) == 0
    assert p.decode("B", swapped=True) == 1


def test_pairwise_decode_handles_chatty_judge():
    p = PairwisePrompt()
    assert p.decode("The answer is A.", swapped=False) == 1
    assert p.decode("Clearly B because ...", swapped=True) == 1


def test_pairwise_decode_unparseable_abstains():
    p = PairwisePrompt()
    assert p.decode("I refuse.", swapped=False) == -1
    assert p.decode("", swapped=False) == -1


def test_likert_decode_scores_winner_correctly():
    p = LikertPrompt()
    # A=5, B=2 with no swap: A wins ⇒ agree with chosen ≻ rejected.
    assert p.decode("A=5,B=2", swapped=False) == 1
    # A=2, B=5 with no swap: B wins ⇒ disagrees.
    assert p.decode("A=2,B=5", swapped=False) == 0
    # Swap inverts.
    assert p.decode("A=5,B=2", swapped=True) == 0
    assert p.decode("A=2,B=5", swapped=True) == 1


def test_likert_tie_abstains():
    p = LikertPrompt()
    assert p.decode("A=3,B=3", swapped=False) == -1


def test_likert_decode_malformed_abstains():
    p = LikertPrompt()
    assert p.decode("not a likert response", swapped=False) == -1
    assert p.decode("", swapped=False) == -1

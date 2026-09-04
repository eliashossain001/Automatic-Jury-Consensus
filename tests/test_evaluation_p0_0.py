"""Regression tests for the P0-0 remediation.

Covers the two defects documented in outputs/dpo_judges/D5_RESOLUTION.md:
even-bank ties, constant vs nonconstant gold vectors, abstentions, and
evaluable-pool retention. These tests fail against the pre-P0-0 code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from corrfilter.cfi.consensus import ABSTAIN, majority_consensus
from corrfilter.evaluation import (evaluable_and_correct, frr, matched_k,
                                   top_k_keep)


# --------------------------------------------------------------------------
# Even-bank ties
# --------------------------------------------------------------------------

def test_even_bank_tie_abstains_under_abstain_policy():
    """A 3-3 split on a six-judge bank carries no majority."""
    V = np.array([[1, 1, 1, 0, 0, 0]], dtype=np.int8)
    M = np.ones_like(V, dtype=np.int8)
    assert majority_consensus(V, M, tie_policy="abstain")[0] == ABSTAIN
    assert majority_consensus(V, M, tie_policy="fixed", tie_break=1)[0] == 1
    assert majority_consensus(V, M, tie_policy="fixed", tie_break=0)[0] == 0


def test_odd_bank_never_ties():
    V = np.array([[1, 1, 0, 0, 0]], dtype=np.int8)
    M = np.ones_like(V, dtype=np.int8)
    assert majority_consensus(V, M, tie_policy="abstain")[0] == 0
    assert majority_consensus(V, M, tie_policy="fixed")[0] == 0


def test_tie_after_abstention_on_odd_bank():
    """A five-judge bank with one abstention can still tie 2-2."""
    V = np.array([[1, 1, 0, 0, 1]], dtype=np.int8)
    M = np.array([[1, 1, 1, 1, 0]], dtype=np.int8)
    assert majority_consensus(V, M, tie_policy="abstain")[0] == ABSTAIN


def test_all_abstain_is_abstain_under_both_policies():
    V = np.zeros((1, 4), dtype=np.int8)
    M = np.zeros((1, 4), dtype=np.int8)
    assert majority_consensus(V, M, tie_policy="abstain")[0] == ABSTAIN
    assert majority_consensus(V, M, tie_policy="fixed")[0] == ABSTAIN


def test_invalid_tie_policy_rejected():
    V = np.array([[1, 0]], dtype=np.int8)
    M = np.ones_like(V, dtype=np.int8)
    with pytest.raises(ValueError, match="tie_policy"):
        majority_consensus(V, M, tie_policy="coin_flip")


# --------------------------------------------------------------------------
# Constant vs nonconstant gold
# --------------------------------------------------------------------------

def test_constant_gold_with_fixed_ties_is_rejected():
    """The exact circularity that inflated the base-bank gain by 2.7x."""
    V = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=np.int8)
    M = np.ones_like(V, dtype=np.int8)
    gold = np.ones(2, dtype=np.int8)
    with pytest.raises(ValueError, match="correct by construction"):
        evaluable_and_correct(V, M, gold, tie_policy="fixed")


def test_nonconstant_gold_with_fixed_ties_is_allowed():
    """The poisoning experiments are legitimate: gold varies, both arms share the label."""
    V = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=np.int8)
    M = np.ones_like(V, dtype=np.int8)
    gold = np.array([0, 1], dtype=np.int8)
    label, evaluable, correct = evaluable_and_correct(V, M, gold, tie_policy="fixed")
    assert evaluable.all()
    assert not correct[0] and correct[1]


def test_ties_are_never_free_credit_under_abstain():
    """Under abstain, a tied item cannot be scored correct no matter what gold says."""
    V = np.array([[1, 1, 1, 0, 0, 0]], dtype=np.int8)
    M = np.ones_like(V, dtype=np.int8)
    for g in (0, 1):
        gold = np.full(1, g, dtype=np.int8)
        _, evaluable, correct = evaluable_and_correct(V, M, gold, tie_policy="abstain")
        assert not evaluable[0]
        assert not correct[0]


def test_constant_gold_defaults_are_safe():
    """The default path must be usable with constant gold (that is the common case)."""
    V = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=np.int8)
    M = np.ones_like(V, dtype=np.int8)
    gold = np.ones(2, dtype=np.int8)
    _, evaluable, correct = evaluable_and_correct(V, M, gold)
    assert list(evaluable) == [False, True]
    assert list(correct) == [False, True]


# --------------------------------------------------------------------------
# Evaluable-pool retention
# --------------------------------------------------------------------------

def test_matched_k_uses_evaluable_pool_not_item_count():
    evaluable = np.array([True] * 50 + [False] * 50)
    assert matched_k(0.60, evaluable) == 30
    assert matched_k(0.60, evaluable, retention_mode="all_items") == 60


def test_matched_k_never_exceeds_pool():
    evaluable = np.array([True] * 40 + [False] * 60)
    assert matched_k(1.0, evaluable) == 40


def test_legacy_retention_can_exceed_pool_and_degenerates():
    """Documents the legacy failure mode: k > pool makes every method identical."""
    evaluable = np.array([True] * 40 + [False] * 60)
    k_legacy = matched_k(0.70, evaluable, retention_mode="all_items")
    assert k_legacy == 70 > int(evaluable.sum())
    a = top_k_keep(np.arange(100, dtype=float), k_legacy, evaluable)
    b = top_k_keep(-np.arange(100, dtype=float), k_legacy, evaluable)
    assert (a & evaluable).sum() == (b & evaluable).sum() == 40  # no discrimination left


def test_top_k_keep_excludes_non_evaluable_items():
    score = np.array([9.0, 9.0, 9.0, 1.0])
    evaluable = np.array([False, False, True, True])
    keep = top_k_keep(score, 2, evaluable)
    assert list(keep) == [False, False, True, True]


def test_invalid_retention_mode_rejected():
    with pytest.raises(ValueError, match="retention_mode"):
        matched_k(0.5, np.array([True, True]), retention_mode="per_item")


# --------------------------------------------------------------------------
# Abstentions in the FRR computation
# --------------------------------------------------------------------------

def test_frr_computed_within_evaluable_pool_only():
    keep = np.array([True, True, True, True])
    correct = np.array([True, False, False, False])
    evaluable = np.array([True, True, False, False])
    assert frr(keep, correct, evaluable) == pytest.approx(0.5)


def test_frr_is_nan_when_nothing_kept():
    keep = np.zeros(3, dtype=bool)
    assert np.isnan(frr(keep, np.ones(3, bool), np.ones(3, bool)))


# --------------------------------------------------------------------------
# End-to-end: the defect signature must be gone
# --------------------------------------------------------------------------

def _synthetic_bank(seed=0, n=400):
    """Six judges, constant gold, a substantial block of exact ties."""
    rng = np.random.default_rng(seed)
    V = (rng.random((n, 6)) > 0.35).astype(np.int8)
    V[: n // 4] = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)  # forced tie block
    return V, np.ones((n, 6), dtype=np.int8), np.ones(n, dtype=np.int8)


def test_ties_do_not_manufacture_a_gain():
    """The defect signature, isolated.

    Construct a bank whose items are either exact ties or genuine verdicts, and two
    rankings that differ ONLY in how they order the tie items. Under the legacy
    protocol (ties resolved to 1, gold constant 1) the ranking that retains ties is
    credited for free and a large spurious gain appears. Under the corrected protocol
    ties leave the evaluable pool, both rankings are restricted to the same genuine
    items, and the gain is exactly zero.
    """
    n_tie, n_real = 100, 100
    tie_rows = np.tile(np.array([1, 1, 1, 0, 0, 0], dtype=np.int8), (n_tie, 1))
    right = np.tile(np.array([1, 1, 1, 1, 0, 0], dtype=np.int8), (n_real // 2, 1))
    wrong = np.tile(np.array([0, 0, 0, 0, 1, 1], dtype=np.int8), (n_real // 2, 1))
    V = np.vstack([tie_rows, right, wrong]).astype(np.int8)
    M = np.ones_like(V, dtype=np.int8)
    gold = np.ones(V.shape[0], dtype=np.int8)
    is_tie = np.arange(V.shape[0]) < n_tie

    ties_last = np.where(is_tie, 0.0, 1.0)
    ties_first = np.where(is_tie, 1.0, 0.0)

    # --- legacy protocol, reconstructed by hand (the library now refuses it) ---
    legacy_label = majority_consensus(V, M, tie_break=1, tie_policy="fixed")
    legacy_eval = legacy_label != ABSTAIN
    legacy_correct = legacy_eval & (legacy_label == gold)
    k_legacy = matched_k(0.60, legacy_eval, retention_mode="all_items")
    legacy_gain = (frr(top_k_keep(ties_last, k_legacy, legacy_eval), legacy_correct, legacy_eval)
                   - frr(top_k_keep(ties_first, k_legacy, legacy_eval), legacy_correct, legacy_eval))
    assert legacy_gain > 0.25, "expected the legacy protocol to manufacture a large gain"

    # --- corrected protocol ---
    _, evaluable, correct = evaluable_and_correct(V, M, gold)
    assert evaluable.sum() == n_real and not evaluable[is_tie].any()
    k = matched_k(0.60, evaluable)
    gain = (frr(top_k_keep(ties_last, k, evaluable), correct, evaluable)
            - frr(top_k_keep(ties_first, k, evaluable), correct, evaluable))
    assert gain == pytest.approx(0.0)


def test_paired_frr_gain_rejects_unsafe_combination():
    from bootstrap_ci import paired_frr_gain
    V, M, gold = _synthetic_bank()
    R = np.eye(6)
    with pytest.raises(ValueError, match="correct by construction"):
        paired_frr_gain(V, M, R, gold, retention=0.6, B=10, tie_policy="fixed")


def test_paired_frr_gain_reports_protocol_provenance():
    from bootstrap_ci import paired_frr_gain
    V, M, gold = _synthetic_bank()
    out = paired_frr_gain(V, M, np.eye(6), gold, retention=0.6, B=50)
    assert out["tie_policy"] == "abstain"
    assert out["retention_mode"] == "evaluable"
    assert out["n_keep"] <= out["n_evaluable"]


# --------------------------------------------------------------------------
# P0-1: forced-choice verdict-token resolution
# --------------------------------------------------------------------------

class _FakeSPTokenizer:
    """Mimics the SentencePiece behaviour that broke the naive resolver.

    Faithful detail that matters: the "start of string" marker is emitted only when the
    text begins with a word piece, so ``encode("1")`` yields ``[WS, digit]`` (two tokens,
    which defeats a len==1 check and sends the naive resolver to its fallback, returning
    the SHARED WS id for every digit) while ``encode("ctxA=1")`` yields the bare digit in
    context. Mistral and Phi behave this way; Qwen's BPE does not, which is why the real
    bug was model-specific.
    """
    WS = 900

    def __init__(self):
        self.vocab = {"A": 10, "B": 11, "1": 21, "2": 22, "3": 23, "4": 24, "5": 25,
                      "=": 30, "ctx": 40}

    def encode(self, text, add_special_tokens=False):
        out = []
        i = 0
        while i < len(text):
            if text.startswith("ctx", i):
                out.append(self.vocab["ctx"]); i += 3
            elif text[i] == " ":
                i += 1
            elif text[i] in self.vocab:
                if not out:                       # marker only at start of string
                    out.append(self.WS)
                out.append(self.vocab[text[i]]); i += 1
            else:
                i += 1
        return out


def test_fake_sp_tokenizer_reproduces_the_isolated_encoding_trap():
    """Guard the guard: the fake must exhibit the behaviour the real bug came from."""
    tok = _FakeSPTokenizer()
    assert tok.encode("1") == [_FakeSPTokenizer.WS, 21]      # two tokens, defeats len==1
    assert tok.encode("ctxA=1")[-1] == 21                    # bare digit in context
    naive = {d: tok.encode(d)[0] for d in "12345"}           # the old fallback
    assert len(set(naive.values())) == 1                     # collapses onto WS


def test_next_token_ids_resolves_distinct_digits_on_sentencepiece():
    from corrfilter.judges.forced_choice import LIKERT_DIGITS, _next_token_ids
    tok = _FakeSPTokenizer()
    ids = _next_token_ids(tok, "ctxA=", LIKERT_DIGITS)
    flat = [i for group in ids for i in group]
    assert len(set(flat)) == 5, f"digits must resolve to distinct ids, got {ids}"
    assert _FakeSPTokenizer.WS not in flat, "shared whitespace id must be dropped"


def test_next_token_ids_drops_ids_claimed_by_multiple_surfaces():
    from corrfilter.judges.forced_choice import _next_token_ids
    tok = _FakeSPTokenizer()
    a, b = _next_token_ids(tok, "ctx", ("A", "B"))
    assert set(a).isdisjoint(set(b))
    assert a and b


def test_next_token_ids_raises_when_no_discriminating_id_exists():
    from corrfilter.judges.forced_choice import _next_token_ids

    class AllSame:
        def encode(self, text, add_special_tokens=False):
            return [7] * max(len(text), 1)

    with pytest.raises(RuntimeError, match="no discriminating token id"):
        _next_token_ids(AllSame(), "ctx", ("A", "B"))

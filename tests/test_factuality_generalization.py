"""Stage-1 tests for the pointwise-factuality generalization (LLM-AggreFact).

Covers: generic-batch validation, label orientation and mapping, duplicate
detection, pointwise prompt rendering/parsing incl. malformed and NOT-SUPPORTED
outputs, abstention handling, absence of fake position metadata, exclusion of
position features, swap-free cluster determinism/freezing, error-matrix
compatibility, and zero-shot leakage guards. No GPU or network required.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from corrfilter.data.generic_task import (
    LABEL_MAPPING,
    FactualityItem,
    JudgeTaskBatch,
    _item_hash,
    stratified_sample,
)
from corrfilter.judges.prompts_pointwise import (
    PointwiseAnalysisTemplate,
    PointwiseDirectTemplate,
    _parse_verdict,
)
from corrfilter.routing import correlation_cluster

ROOT = Path(__file__).resolve().parents[1]


def _mini_batch(votes, gold, avail=None):
    votes = np.asarray(votes, np.int8)
    avail = np.ones_like(votes) if avail is None else np.asarray(avail, np.int8)
    return JudgeTaskBatch(item_ids=[f"i{k}" for k in range(votes.shape[0])],
                          gold_labels=np.asarray(gold, np.int8),
                          judge_votes=votes, availability=avail,
                          task_name="t")


def test_batch_validation_accepts_valid():
    _mini_batch([[1, 0], [0, 1]], [1, 0]).validate()


@pytest.mark.parametrize("votes,gold,avail", [
    ([[2, 0], [0, 1]], [1, 0], None),          # non-binary vote
    ([[1, 0], [0, 1]], [1, 2], None),          # non-binary gold
    ([[1, 0]], [1, 0], None),                  # shape mismatch
])
def test_batch_validation_rejects(votes, gold, avail):
    with pytest.raises(ValueError):
        _mini_batch(votes, gold, avail).validate()


def test_batch_masked_votes_ignored():
    # a masked-out cell may hold any placeholder; only available votes checked
    b = _mini_batch([[1, 0], [0, 1]], [1, 0], avail=[[1, 0], [1, 1]])
    b.judge_votes[0, 1] = 0
    b.validate()


def test_label_mapping_is_identity_binary():
    assert LABEL_MAPPING == {1: 1, 0: 0}  # documented inherited binarization


def test_item_hash_dedup_stability():
    assert _item_hash("doc", "claim") == _item_hash("doc ", "claim")  # strip
    assert _item_hash("doc", "claim") != _item_hash("doc", "claim2")


def test_stratified_sample_deterministic_and_capped():
    rng = np.random.default_rng(0)
    man = pd.DataFrame({
        "item_id": [f"fact_{k}" for k in range(400)],
        "document": ["d" * int(l) for l in rng.integers(100, 4000, 400)],
        "claim": ["c"] * 400,
        "gold_supported": rng.integers(0, 2, 400),
        "source_dataset": rng.choice(["A", "B"], 400),
    })
    man["doc_chars"] = man.document.str.len()
    man["claim_chars"] = 1
    s1 = stratified_sample(man, 100, seed=7, max_doc_chars=2000)
    s2 = stratified_sample(man, 100, seed=7, max_doc_chars=2000)
    assert list(s1.item_id) == list(s2.item_id)          # deterministic
    assert (s1.doc_chars <= 2000).all()                   # length control
    ex = set(s1.item_id[:10])
    s3 = stratified_sample(man, 50, seed=8, max_doc_chars=2000, exclude_ids=ex)
    assert not set(s3.item_id) & ex                       # no overlap


ITEM = FactualityItem(item_id="x", document="The sky is blue.",
                      claim="The sky is blue.", gold_supported=1,
                      source_dataset="T")


def test_pointwise_render_contains_fields_and_no_swap():
    for tmpl in (PointwiseDirectTemplate(), PointwiseAnalysisTemplate()):
        rp = tmpl.render(ITEM, swapped=False)
        assert ITEM.document in rp.text and ITEM.claim in rp.text
        assert rp.swapped is False
        with pytest.raises(ValueError):
            tmpl.render(ITEM, swapped=True)   # no fake position semantics


@pytest.mark.parametrize("raw,expect", [
    ("SUPPORTED", 1), ("UNSUPPORTED", 0), (" unsupported \n", 0),
    ("The claim is NOT SUPPORTED", 0), ("NOT FULLY SUPPORTED", 0),
    ("", -1), ("maybe?", -1), ("SUPPORTED because... UNSUPPORTED", 0),
])
def test_direct_parser(raw, expect):
    assert PointwiseDirectTemplate.decode(raw) == expect


@pytest.mark.parametrize("raw,expect", [
    ("evidence...\nFINAL: SUPPORTED", 1),
    ("mentions SUPPORTED early\nFINAL: UNSUPPORTED", 0),
    ("FINAL: NOT SUPPORTED", 0),
    ("no verdict at all", -1),
    ("rambling...\nUNSUPPORTED", 0),      # bare trailing label tolerated
    ("SUPPORTED mentioned mid-analysis only\nmore text", -1),
])
def test_analysis_parser_final_only(raw, expect):
    assert PointwiseAnalysisTemplate.decode(raw) == expect


def test_malformed_is_abstention_not_negative():
    assert _parse_verdict("garbage", final_only=False) == -1  # never 0


def test_correlation_cluster_deterministic_and_frozen():
    R = np.full((8, 8), 0.05)
    R[:4, :4] = 0.55
    np.fill_diagonal(R, 1.0)
    c1 = correlation_cluster(R, k=4)
    c2 = correlation_cluster(R.copy(), k=4)
    assert (c1 == c2).all() and c1[:4].all() and not c1[4:].any()
    assert 2 <= c1.sum() <= 4


def test_error_matrix_generic_gold():
    """Two-class gold works through the standard error construction."""
    V = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], np.int8)
    gold = np.array([1, 0, 0, 1], np.int8)
    E = (V != gold[:, None]).astype(int)
    # item2: judge0 affirms an unsupported claim; item3: judge0 rejects a
    # supported one. Both orientations count as errors symmetrically.
    assert E.sum() == 2 and E[2, 0] == 1 and E[3, 0] == 1 and E[2, 1] == 0


def test_no_position_features_for_pointwise():
    """The pre-registered base features contain no position-derived field."""
    from corrfilter.routing import BASE_FEATS
    banned = {"pdis", "part_contrast", "lojo", "possens", "swap"}
    assert not any(any(b in f for b in banned) for f in BASE_FEATS)


def test_zero_shot_fixed_filter_source_only():
    """best_fixed_on consumes only the rows passed to it (source pools)."""
    from corrfilter.routing import best_fixed_on
    md = pd.DataFrame([
        {"instance": "src1", "method": m, "n_kept": 10,
         "n_kept_clean": 9 if m == "naive_majority" else 5}
        for m in ["naive_majority", "supermajority_75",
                  "corrfilter_small_gold_R", "bias_cluster"]])
    assert best_fixed_on(md, ["src1"]) == "naive_majority"
    assert best_fixed_on(md, []) in md.method.unique()  # no hidden global state

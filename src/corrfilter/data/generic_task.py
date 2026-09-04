"""Generic binary judge-task interface and the LLM-AggreFact factuality adapter.

Vote semantics for generic tasks: ``1`` = the judge AFFIRMS the item (for
factuality: the claim is supported by the document); ``0`` = the judge
REJECTS it. Gold labels use the same orientation. The gold label is never
encoded through response ordering; there is no position swap for pointwise
tasks.

The pairwise loaders (:mod:`corrfilter.data.rewardbench`) are untouched; this
module is a parallel adapter layer mapping factuality items onto the same
``(V, M, gold)`` matrix interface the rest of the pipeline consumes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Generic batch


@dataclass
class JudgeTaskBatch:
    """The minimal generic representation of a binary judge task."""

    item_ids: list[str]
    gold_labels: np.ndarray
    judge_votes: np.ndarray
    availability: np.ndarray
    task_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n_items = len(self.item_ids)
        if len(set(self.item_ids)) != n_items:
            raise ValueError("item_ids must be unique")
        if self.gold_labels.shape != (n_items,):
            raise ValueError(
                f"gold_labels must have shape ({n_items},), got {self.gold_labels.shape}")
        if self.judge_votes.ndim != 2:
            raise ValueError(
                f"judge_votes must be two-dimensional, got shape {self.judge_votes.shape}")
        if self.judge_votes.shape[0] != n_items:
            raise ValueError("judge_votes item dimension does not match item_ids")
        if self.availability.shape != self.judge_votes.shape:
            raise ValueError("availability must have the same shape as judge_votes")
        available = self.judge_votes[self.availability.astype(bool)]
        if not np.isin(available, [0, 1]).all():
            raise ValueError("Available judge votes must be binary")
        if not np.isin(self.gold_labels, [0, 1]).all():
            raise ValueError("Gold labels must be binary")


@dataclass(frozen=True)
class FactualityItem:
    """One pointwise factuality item; duck-compatible with the judge runner
    (only ``item_id`` is read by :class:`HFJudge`; the prompt template reads
    ``document`` and ``claim``)."""

    item_id: str
    document: str
    claim: str
    gold_supported: int
    source_dataset: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM-AggreFact label harmonization
#
# The published aggregation (lytang/LLM-AggreFact) ships ONE binary column
# ``label`` already harmonized by the MiniCheck authors across its component
# datasets (1 = claim supported by the document, 0 = unsupported). We audit
# rather than remap: the loader asserts strict binarity and refuses any other
# value, so no multi-class or ordinal label is ever silently collapsed by our
# code. Component datasets whose ORIGINAL schemes were ternary (e.g.
# partially-supported) were binarized upstream by the benchmark authors;
# this inherited mapping is recorded here for the paper's documentation.
LABEL_MAPPING = {
    # value in the released aggregate -> our gold_supported orientation
    1: 1,   # supported / consistent
    0: 0,   # unsupported / inconsistent / hallucinated
    # anything else -> excluded loudly (see load_aggrefact)
}

REQUIRED_COLUMNS = {"dataset", "doc", "claim", "label"}


def _item_hash(doc: str, claim: str) -> str:
    return hashlib.sha1((doc.strip() + "\x1f" + claim.strip()).encode()).hexdigest()[:16]


def load_aggrefact(split: str = "dev", cache_dir: str | None = None) -> pd.DataFrame:
    """Load LLM-AggreFact and return an audited manifest DataFrame.

    Columns: item_id, document, claim, gold_supported, source_dataset,
    doc_chars, claim_chars, item_hash. Ambiguous / non-binary labels are
    EXCLUDED and counted, never coerced. Exact duplicate (doc, claim) pairs
    are dropped keep-first with a stable hash.
    """
    from datasets import load_dataset

    ds = load_dataset("lytang/LLM-AggreFact", split=split, cache_dir=cache_dir)
    df = ds.to_pandas()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"LLM-AggreFact schema changed; missing columns {missing}")
    n0 = len(df)
    ok = df["label"].isin(list(LABEL_MAPPING))
    excluded_labels = int((~ok).sum())
    df = df[ok].copy()
    df["gold_supported"] = df["label"].map(LABEL_MAPPING).astype(np.int8)
    df["item_hash"] = [_item_hash(d, c) for d, c in zip(df["doc"], df["claim"])]
    n_before_dedup = len(df)
    df = df.drop_duplicates("item_hash", keep="first").reset_index(drop=True)
    df["item_id"] = ["fact_" + h for h in df["item_hash"]]
    df["doc_chars"] = df["doc"].str.len()
    df["claim_chars"] = df["claim"].str.len()
    out = df.rename(columns={"doc": "document", "dataset": "source_dataset"})[
        ["item_id", "document", "claim", "gold_supported", "source_dataset",
         "doc_chars", "claim_chars", "item_hash"]]
    out.attrs["audit"] = {
        "split": split, "n_raw": n0, "n_excluded_nonbinary": excluded_labels,
        "n_duplicates_dropped": n_before_dedup - len(out), "n_final": len(out)}
    return out


# VitaminC (Schuster et al., NAACL 2021; tals/vitaminc, ungated): claim
# verification against short Wikipedia evidence. Ternary labels; NEI is
# EXCLUDED (ambiguous for binary support), never coerced.
LABEL_MAPPING_VITAMINC = {
    "SUPPORTS": 1,
    "REFUTES": 0,
    # "NOT ENOUGH INFO": excluded
}


def load_vitaminc(split: str = "test", cache_dir: str | None = None) -> pd.DataFrame:
    """VitaminC manifest in the same schema as :func:`load_aggrefact`.

    ``source_dataset`` records the revision type (real Wikipedia revisions vs
    synthetic contrastive edits), a useful stratum and a candidate
    real-mechanism axis.
    """
    from datasets import load_dataset

    df = load_dataset("tals/vitaminc", split=split, cache_dir=cache_dir).to_pandas()
    n0 = len(df)
    ok = df["label"].isin(list(LABEL_MAPPING_VITAMINC))
    excluded = int((~ok).sum())
    df = df[ok].copy()
    df["gold_supported"] = df["label"].map(LABEL_MAPPING_VITAMINC).astype(np.int8)
    df["item_hash"] = [_item_hash(d, c) for d, c in zip(df["evidence"], df["claim"])]
    n_before = len(df)
    df = df.drop_duplicates("item_hash", keep="first").reset_index(drop=True)
    df["item_id"] = ["vitc_" + h for h in df["item_hash"]]
    out = df.rename(columns={"evidence": "document"})
    out["source_dataset"] = "vitaminc-" + out["revision_type"].astype(str)
    out["doc_chars"] = out["document"].str.len()
    out["claim_chars"] = out["claim"].str.len()
    out = out[["item_id", "document", "claim", "gold_supported", "source_dataset",
               "doc_chars", "claim_chars", "item_hash"]]
    out.attrs["audit"] = {
        "split": split, "n_raw": n0, "n_excluded_nonbinary": excluded,
        "n_duplicates_dropped": n_before - len(out), "n_final": len(out),
        "label_mapping": {k: v for k, v in LABEL_MAPPING_VITAMINC.items()},
        "excluded_labels": ["NOT ENOUGH INFO"]}
    return out


def stratified_sample(
    manifest: pd.DataFrame,
    n: int,
    seed: int,
    max_doc_chars: int = 2000,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Deterministic stratified sample: source_dataset x label x doc-length
    tercile, documents capped at ``max_doc_chars`` (length control), never
    overlapping ``exclude_ids``."""
    pool = manifest[(manifest.doc_chars <= max_doc_chars)
                    & ~manifest.item_id.isin(exclude_ids)].copy()
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} eligible items for n={n}")
    terc = pool.doc_chars.rank(pct=True)
    pool["len_bin"] = np.minimum((terc * 3).astype(int), 2)
    rng = np.random.default_rng(seed)
    strata = pool.groupby(["source_dataset", "gold_supported", "len_bin"])
    quota = {k: max(1, int(round(n * len(g) / len(pool)))) for k, g in strata}
    taken = []
    for k, g in strata:
        take = min(quota[k], len(g))
        taken.append(g.iloc[np.sort(rng.choice(len(g), take, replace=False))])
    out = pd.concat(taken)
    if len(out) > n:  # trim deterministically, preserving label balance
        out = (out.sort_values("item_id")
               .groupby("gold_supported", group_keys=False)
               .apply(lambda g: g.iloc[:int(round(n * len(g) / len(out)))]))
    return out.sort_values("item_id").reset_index(drop=True)


def manifest_to_items(manifest: pd.DataFrame) -> list[FactualityItem]:
    return [FactualityItem(item_id=r.item_id, document=r.document, claim=r.claim,
                           gold_supported=int(r.gold_supported),
                           source_dataset=r.source_dataset)
            for r in manifest.itertuples(index=False)]


def batch_from_cache(manifest: pd.DataFrame, cache, logical_ids: list[str],
                     task_name: str = "factuality_pointwise") -> JudgeTaskBatch:
    """Assemble a validated JudgeTaskBatch from a VoteCache (no swaps read)."""
    ids = manifest.item_id.tolist()
    idx = {i: k for k, i in enumerate(ids)}
    n, m = len(ids), len(logical_ids)
    V = np.zeros((n, m), np.int8)
    M = np.zeros((n, m), np.int8)
    for j, lid in enumerate(logical_ids):
        dfj = cache.load(lid)
        if dfj is None:
            continue
        for r in dfj.itertuples(index=False):
            i = idx.get(str(r.item_id))
            if i is None or int(r.vote) == -1:
                continue
            V[i, j] = int(r.vote)
            M[i, j] = 1
    batch = JudgeTaskBatch(item_ids=ids,
                           gold_labels=manifest.gold_supported.to_numpy(np.int8),
                           judge_votes=V, availability=M, task_name=task_name,
                           metadata={"logical_ids": logical_ids})
    batch.validate()
    return batch

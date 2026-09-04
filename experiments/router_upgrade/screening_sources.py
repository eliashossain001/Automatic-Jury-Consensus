#!/usr/bin/env python
"""Candidate sources for the routing-feasibility screening campaign.

Builders return manifests in the generic schema (item_id, document, claim,
gold_supported, source_dataset, doc_chars, claim_chars) and are deterministic
(fixed seeds, stable hashes). Nothing here touches the verified pipeline.

Sources:
  halueval_qa   HaluEval QA (ungated): (knowledge, question, answer) with a
                paired hallucinated answer -> two pointwise items per row
                (right answer gold=1, hallucinated gold=0). Model-generated
                plausible hallucinations = a realistic shared-blind-spot pool.
  cjb_point     CodeJudgeBench codegen -> pointwise: (problem, one solution),
                gold from unit-test execution (pos=1, neg=0). Claim = code.
  cjb_pair      CodeJudgeBench codegen -> pairwise CalibrationItems
                (prompt, chosen=pos, rejected=neg), for the existing pairwise
                pipeline.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrfilter.data.generic_task import _item_hash  # noqa: E402

SEED = 20270215
MAX_DOC_CHARS_HALU = 2000
MAX_CODE_CHARS = 6000  # problem + one candidate program


def build_halueval_manifest(n: int = 200, seed: int = SEED) -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset("pminervini/HaluEval", "qa_samples", split="data").to_pandas()
    rng = np.random.default_rng(seed)
    rows = []
    for r in df.itertuples(index=False):
        doc = f"{r.knowledge}\n\nQuestion: {r.question}"
        if len(doc) > MAX_DOC_CHARS_HALU:
            continue
        # HaluEval qa_samples rows carry one answer plus a hallucination flag
        halluc = str(r.hallucination).strip().lower() in ("yes", "true", "1")
        rows.append({"document": doc, "claim": f"Answer: {r.answer}",
                     "gold_supported": 0 if halluc else 1})
    man = pd.DataFrame(rows)
    man["item_hash"] = [_item_hash(d, c) for d, c in zip(man.document, man.claim)]
    man = man.drop_duplicates("item_hash").reset_index(drop=True)
    # balance labels, deterministic subsample
    take = []
    for g in (0, 1):
        sub = man[man.gold_supported == g]
        k = min(n // 2, len(sub))
        take.append(sub.iloc[np.sort(rng.choice(len(sub), k, replace=False))])
    man = pd.concat(take).reset_index(drop=True)
    man["item_id"] = "halu_" + man.item_hash
    man["source_dataset"] = "halueval-qa"
    man["doc_chars"] = man.document.str.len()
    man["claim_chars"] = man.claim.str.len()
    return man[["item_id", "document", "claim", "gold_supported", "source_dataset",
                "doc_chars", "claim_chars", "item_hash"]]


def _cjb_pairs(n_pairs: int, seed: int = SEED) -> pd.DataFrame:
    from datasets import load_dataset

    splits = ["claude_3.7_sonnet", "gemini_2.5_pro", "gemini_2.5_flash",
              "gemini_2.5_flash_lite", "qwen3_235b", "claude_4_opus", "claude_4_sonnet"]
    frames = []
    for s in splits:
        d = load_dataset("mattymchen/codejudgebench", "codegen", split=s).to_pandas()
        d["gen_model"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["prob_chars"] = df.question_content.str.len()
    df["tot_chars"] = (df.prob_chars + df.pos_response.str.len()
                       + df.neg_response.str.len())
    df = df[df.prob_chars + df[["pos_response", "neg_response"]].apply(
        lambda r: max(len(r.pos_response), len(r.neg_response)), axis=1) <= MAX_CODE_CHARS]
    df["pair_hash"] = [hashlib.sha1((a + "\x1f" + b).encode()).hexdigest()[:16]
                       for a, b in zip(df.question_content, df.pos_response)]
    df = df.drop_duplicates("pair_hash").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    # stratify by difficulty x generator model
    take = []
    for (dif, gm), g in df.groupby(["difficulty", "gen_model"]):
        k = max(1, int(round(n_pairs * len(g) / len(df))))
        take.append(g.iloc[np.sort(rng.choice(len(g), min(k, len(g)), replace=False))])
    out = pd.concat(take)
    if len(out) > n_pairs:
        out = out.sort_values("pair_hash").iloc[:n_pairs]
    return out.reset_index(drop=True)


def build_cjb_pointwise_manifest(n_pairs: int = 100, seed: int = SEED) -> pd.DataFrame:
    pairs = _cjb_pairs(n_pairs, seed)
    rows = []
    for r in pairs.itertuples(index=False):
        doc = f"Problem ({r.platform}, difficulty {r.difficulty}):\n{r.question_content}"
        for code, gold in ((r.pos_response, 1), (r.neg_response, 0)):
            rows.append({"document": doc, "claim": code, "gold_supported": gold,
                         "source_dataset": f"cjb-{r.difficulty}",
                         "pair_hash": r.pair_hash})
    man = pd.DataFrame(rows)
    man["item_hash"] = [_item_hash(d, c) for d, c in zip(man.document, man.claim)]
    man = man.drop_duplicates("item_hash").reset_index(drop=True)
    man["item_id"] = "cjb_" + man.item_hash
    man["doc_chars"] = man.document.str.len()
    man["claim_chars"] = man.claim.str.len()
    return man[["item_id", "document", "claim", "gold_supported", "source_dataset",
                "doc_chars", "claim_chars", "item_hash", "pair_hash"]]


def build_cjb_pairwise_items(n_pairs: int = 100, seed: int = SEED):
    """CalibrationItems for the existing pairwise pipeline (gold: chosen=pos)."""
    from corrfilter.data.rewardbench import CalibrationItem

    pairs = _cjb_pairs(n_pairs, seed)
    items, meta = [], []
    for r in pairs.itertuples(index=False):
        iid = "cjbp_" + r.pair_hash
        items.append(CalibrationItem(
            item_id=iid,
            prompt=f"Programming problem ({r.platform}):\n{r.question_content}",
            chosen=r.pos_response, rejected=r.neg_response,
            subset="codejudgebench", category=str(r.difficulty), gold_label="A"))
        meta.append({"item_id": iid, "difficulty": r.difficulty,
                     "gen_model": r.gen_model, "pair_hash": r.pair_hash})
    return items, pd.DataFrame(meta)

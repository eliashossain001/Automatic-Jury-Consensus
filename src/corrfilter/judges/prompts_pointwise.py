"""Pointwise factuality prompt templates: direct verdict and analysis-then-verdict.

Both templates ignore position (there is no candidate pair); ``render`` takes
``(item, swapped)`` for interface compatibility with :class:`HFJudge` but
raises if ``swapped`` is ever True, and ``decode`` maps the model output to
``1`` (SUPPORTED), ``0`` (UNSUPPORTED), or ``-1`` (abstain: malformed output,
never treated as a negative vote).

The two styles are genuinely distinct judging procedures (verdict-only vs
evidence-analysis-then-verdict), preserving the bank's prompt-diversity axis
without paraphrase-only variation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RenderedPointwisePrompt:
    text: str
    swapped: bool  # always False; kept for cache/interface parity


class PointwiseDirectTemplate:
    """Direct binary verdict: 'respond with exactly one label'."""

    name = "pw_direct"
    max_new_tokens = 6

    def render(self, item, swapped: bool = False) -> RenderedPointwisePrompt:
        if swapped:
            raise ValueError("pointwise tasks have no position swap")
        text = (
            "<|system|>You are evaluating whether a claim is supported by a "
            "source document.<|user|>Source document:\n"
            f"{item.document}\n\nClaim:\n{item.claim}\n\n"
            "Decide whether the claim is fully supported by the source "
            "document.\n\nRespond with exactly one label:\n\nSUPPORTED\n"
            "UNSUPPORTED<|assistant|>"
        )
        return RenderedPointwisePrompt(text=text, swapped=False)

    @staticmethod
    def decode(raw: str, swapped: bool = False) -> int:
        return _parse_verdict(raw, final_only=False)


class PointwiseAnalysisTemplate:
    """Analysis-then-decision: brief evidence check, then 'FINAL: <label>'."""

    name = "pw_analysis"
    max_new_tokens = 192

    def render(self, item, swapped: bool = False) -> RenderedPointwisePrompt:
        if swapped:
            raise ValueError("pointwise tasks have no position swap")
        text = (
            "<|system|>You are checking factual consistency.<|user|>"
            "Read the source document and claim carefully.\n\n"
            "Determine whether every material part of the claim is supported "
            "by the source document. A claim is UNSUPPORTED if it introduces "
            "incorrect, contradictory, or unverifiable information.\n\n"
            f"Source document:\n{item.document}\n\nClaim:\n{item.claim}\n\n"
            "Briefly identify the relevant evidence, then end with exactly "
            "one line:\n\nFINAL: SUPPORTED\n\nor\n\nFINAL: UNSUPPORTED"
            "<|assistant|>"
        )
        return RenderedPointwisePrompt(text=text, swapped=False)

    @staticmethod
    def decode(raw: str, swapped: bool = False) -> int:
        return _parse_verdict(raw, final_only=True)


_LABEL = r"(NOT\s+(?:FULLY\s+)?SUPPORTED|UNSUPPORTED|SUPPORTED)"
_FINAL_RE = re.compile(r"FINAL\s*[:\-]?\s*" + _LABEL, re.IGNORECASE)
_BARE_RE = re.compile(r"\b" + _LABEL + r"\b", re.IGNORECASE)


def _parse_verdict(raw: str, final_only: bool) -> int:
    """Map output text to a vote. Only the FINAL label counts for the
    analysis style; the LAST occurrence wins in either style so preamble
    mentions of the labels cannot leak in. 'NOT (FULLY) SUPPORTED' maps to
    UNSUPPORTED. Malformed output -> -1 (abstain), never a negative vote."""
    if not raw or not raw.strip():
        return -1
    matches = list((_FINAL_RE if final_only else _BARE_RE).finditer(raw))
    if not matches and final_only:
        # tolerate a bare trailing label if the model dropped the FINAL tag
        tail = raw.strip().splitlines()[-1]
        matches = list(_BARE_RE.finditer(tail))
    if not matches:
        return -1
    label = matches[-1].group(1).upper()
    return 1 if label == "SUPPORTED" else 0


POINTWISE_TEMPLATES = {
    PointwiseDirectTemplate.name: PointwiseDirectTemplate(),
    PointwiseAnalysisTemplate.name: PointwiseAnalysisTemplate(),
}


def register_pointwise_templates() -> None:
    """Additively register the pointwise styles in the shared template
    registry so API judges (Gemini/OpenRouter) resolve them via
    ``get_prompt_template``. Existing pairwise/likert styles are untouched."""
    from corrfilter.judges.prompts import PROMPT_TEMPLATES

    PROMPT_TEMPLATES.setdefault("pw_direct", PointwiseDirectTemplate)
    PROMPT_TEMPLATES.setdefault("pw_analysis", PointwiseAnalysisTemplate)


class PointwiseDecomposeTemplate:
    """Bank-D reasoning mode: decompose the claim into atomic facts, verify
    each against the document, support only if every part is supported."""

    name = "pw_decompose"
    max_new_tokens = 224

    def render(self, item, swapped: bool = False) -> RenderedPointwisePrompt:
        if swapped:
            raise ValueError("pointwise tasks have no position swap")
        text = (
            "<|system|>You verify claims by decomposition.<|user|>"
            "List the distinct factual assertions contained in the claim, one "
            "per line. For each assertion, check it against the source "
            "document. The claim is SUPPORTED only if every assertion is "
            "supported; otherwise it is UNSUPPORTED.\n\n"
            f"Source document:\n{item.document}\n\nClaim:\n{item.claim}\n\n"
            "End with exactly one line:\n\nFINAL: SUPPORTED\n\nor\n\n"
            "FINAL: UNSUPPORTED<|assistant|>"
        )
        return RenderedPointwisePrompt(text=text, swapped=False)

    @staticmethod
    def decode(raw: str, swapped: bool = False) -> int:
        return _parse_verdict(raw, final_only=True)


class PointwiseContradictTemplate:
    """Bank-D reasoning mode: contradiction-first search."""

    name = "pw_contradict"
    max_new_tokens = 160

    def render(self, item, swapped: bool = False) -> RenderedPointwisePrompt:
        if swapped:
            raise ValueError("pointwise tasks have no position swap")
        text = (
            "<|system|>You are an adversarial fact checker.<|user|>"
            "Actively search the claim for any detail that contradicts the "
            "source document or is not verifiable from it. If you find one, "
            "the claim is UNSUPPORTED; only if you fail to find any is it "
            "SUPPORTED.\n\n"
            f"Source document:\n{item.document}\n\nClaim:\n{item.claim}\n\n"
            "Name the problematic detail if any, then end with exactly one "
            "line:\n\nFINAL: SUPPORTED\n\nor\n\nFINAL: UNSUPPORTED<|assistant|>"
        )
        return RenderedPointwisePrompt(text=text, swapped=False)

    @staticmethod
    def decode(raw: str, swapped: bool = False) -> int:
        return _parse_verdict(raw, final_only=True)


_CODE_LABEL = r"(INCORRECT|CORRECT)"
_CODE_FINAL_RE = re.compile(r"FINAL\s*[:\-]?\s*" + _CODE_LABEL, re.IGNORECASE)
_CODE_BARE_RE = re.compile(r"\b" + _CODE_LABEL + r"\b", re.IGNORECASE)


def _parse_code_verdict(raw: str, final_only: bool) -> int:
    if not raw or not raw.strip():
        return -1
    matches = list((_CODE_FINAL_RE if final_only else _CODE_BARE_RE).finditer(raw))
    if not matches and final_only:
        tail = raw.strip().splitlines()[-1]
        matches = list(_CODE_BARE_RE.finditer(tail))
    if not matches:
        return -1
    return 1 if matches[-1].group(1).upper() == "CORRECT" else 0


class CodePointwiseDirectTemplate:
    """Pointwise code correctness: direct verdict."""

    name = "pwc_direct"
    max_new_tokens = 6

    def render(self, item, swapped: bool = False) -> RenderedPointwisePrompt:
        if swapped:
            raise ValueError("pointwise tasks have no position swap")
        text = (
            "<|system|>You judge whether a candidate solution correctly "
            "solves a programming problem.<|user|>"
            f"{item.document}\n\nCandidate solution:\n{item.claim}\n\n"
            "Does this solution correctly satisfy the problem specification "
            "for all valid inputs?\n\nRespond with exactly one label:\n\n"
            "CORRECT\nINCORRECT<|assistant|>"
        )
        return RenderedPointwisePrompt(text=text, swapped=False)

    @staticmethod
    def decode(raw: str, swapped: bool = False) -> int:
        return _parse_code_verdict(raw, final_only=False)


class CodePointwiseTraceTemplate:
    """Pointwise code correctness: reason about edge cases, then verdict."""

    name = "pwc_trace"
    max_new_tokens = 224

    def render(self, item, swapped: bool = False) -> RenderedPointwisePrompt:
        if swapped:
            raise ValueError("pointwise tasks have no position swap")
        text = (
            "<|system|>You judge code by reasoning about its behaviour."
            "<|user|>"
            f"{item.document}\n\nCandidate solution:\n{item.claim}\n\n"
            "Briefly reason about the algorithm and check boundary cases "
            "(empty input, duplicates, off-by-one, overflow, extreme sizes). "
            "Then end with exactly one line:\n\nFINAL: CORRECT\n\nor\n\n"
            "FINAL: INCORRECT<|assistant|>"
        )
        return RenderedPointwisePrompt(text=text, swapped=False)

    @staticmethod
    def decode(raw: str, swapped: bool = False) -> int:
        return _parse_code_verdict(raw, final_only=True)


POINTWISE_TEMPLATES.update({
    PointwiseDecomposeTemplate.name: PointwiseDecomposeTemplate(),
    PointwiseContradictTemplate.name: PointwiseContradictTemplate(),
})
CODE_TEMPLATES = {
    CodePointwiseDirectTemplate.name: CodePointwiseDirectTemplate(),
    CodePointwiseTraceTemplate.name: CodePointwiseTraceTemplate(),
}

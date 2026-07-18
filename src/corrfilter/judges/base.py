"""Abstract Judge contract.

A judge maps a calibration item to a binary vote ``v ∈ {0, 1}`` indicating
agreement (1) or disagreement (0) with the labelled preference. To mitigate
position-bias as a confound (proposal §3.5: position and order bias), each
call randomises which candidate is presented as "A" vs "B" and decodes the
position-aware vote back to the chosen/rejected frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from corrfilter.data import CalibrationItem


@dataclass(frozen=True)
class JudgeSpec:
    """Static configuration that identifies a logical judge.

    A logical judge is one base model paired with one prompt style. Voting
    over the calibration set indexes columns of the error matrix E by
    ``logical_id``.
    """

    base_id: str          # e.g. "qwen-2.5-7b"
    family: str           # e.g. "qwen"
    scale: str            # e.g. "7B"
    prompt_style: str     # "pairwise" or "likert"
    hf_model: str
    hf_revision: str | None = None
    quantization: str = "nf4"
    dtype: str = "bfloat16"
    # Optional PEFT/LoRA adapter directory applied on top of ``hf_model`` after
    # load. Used by the GRPO judge bank: base weights stay shared/cached and only
    # the small trained adapter is checkpointed per judge. None ⇒ plain base model.
    adapter_path: str | None = None
    # Vendored modeling files (e.g., Phi-3.5's bundled modeling_phi3.py) are
    # frequently stale against modern transformers; prefer the maintained
    # in-library implementation by default. Set True only for models that
    # genuinely need their HF-hub code path.
    trust_remote_code: bool = False

    @property
    def logical_id(self) -> str:
        return f"{self.base_id}::{self.prompt_style}"


@dataclass(frozen=True)
class JudgeVote:
    """A single judge's vote on one calibration item."""

    judge_id: str
    item_id: str
    vote: int                 # 1 ⇒ judge agrees chosen ≻ rejected, 0 ⇒ disagrees, -1 ⇒ abstain
    raw_response: str
    position_swapped: bool    # whether chosen was presented as B (true) or A (false)


class Judge(ABC):
    """Abstract base for any judge implementation (HF, API-backed, ...)."""

    spec: JudgeSpec

    def __init__(self, spec: JudgeSpec):
        self.spec = spec

    @property
    def logical_id(self) -> str:
        return self.spec.logical_id

    @abstractmethod
    def load(self) -> None:
        """Bring the judge into a state where ``vote`` can be called.

        Heavy resources (model weights, tokeniser) are acquired here so that
        the runner can load-score-unload one judge at a time.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release heavy resources (free GPU memory)."""

    @abstractmethod
    def vote_batch(self, items: Iterable[CalibrationItem]) -> list[JudgeVote]:
        """Score a batch of items, returning one JudgeVote per item in order."""

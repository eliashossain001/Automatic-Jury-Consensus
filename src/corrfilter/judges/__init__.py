"""Judge interfaces and configuration helpers.

Exports are resolved lazily so lightweight utilities such as deterministic position
assignment do not import model, dataset, or dataframe dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corrfilter.judges.base import Judge, JudgeSpec, JudgeVote
    from corrfilter.judges.hf_judge import HFJudge
    from corrfilter.judges.prompts import LikertPrompt, PairwisePrompt, PromptTemplate
    from corrfilter.judges.registry import build_bank, load_bank_config

_EXPORTS = {
    "Judge": ("corrfilter.judges.base", "Judge"),
    "JudgeSpec": ("corrfilter.judges.base", "JudgeSpec"),
    "JudgeVote": ("corrfilter.judges.base", "JudgeVote"),
    "HFJudge": ("corrfilter.judges.hf_judge", "HFJudge"),
    "PromptTemplate": ("corrfilter.judges.prompts", "PromptTemplate"),
    "PairwisePrompt": ("corrfilter.judges.prompts", "PairwisePrompt"),
    "LikertPrompt": ("corrfilter.judges.prompts", "LikertPrompt"),
    "build_bank": ("corrfilter.judges.registry", "build_bank"),
    "load_bank_config": ("corrfilter.judges.registry", "load_bank_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

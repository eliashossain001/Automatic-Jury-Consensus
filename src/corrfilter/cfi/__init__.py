"""Correlated Failure Injection (CFI) — Experiment 2.

The CFI experiment plants a shared bias into a controllable fraction of the
judge bank and asks: when biased judges share the same blind spot, does naive
majority consensus inflate to "high agreement, still wrong", and does
CorrFilter (subset-aware decorrelated agreement) discount the correlated
agreement enough to reduce false retention at matched retention?

Public API:

* ``BiasMechanism``, ``BiasSpec``, ``BIAS_REGISTRY`` — the five mechanisms.
* ``BiasedPrompt`` — prompt-level bias injection wrapping a base template.
* ``detect_triggers`` / ``build_cfi_manifest`` — per-item heuristic triggers.
* ``CFIBankVariant`` / ``build_bank_variants`` — clean / mixed / all-biased banks.
* ``simulate_bank_votes`` — deterministic replay of biased votes from the H1 cache.
* ``majority_consensus`` / ``corrfilter_score`` / ``retention_match_threshold``.
* ``bootstrap_ci`` — item-level bootstrap CIs.
"""

from corrfilter.cfi.banks import CFIBankVariant, build_bank_variants
from corrfilter.cfi.biases import BIAS_REGISTRY, BiasMechanism, BiasSpec
from corrfilter.cfi.bootstrap import bootstrap_ci
from corrfilter.cfi.consensus import (
    accuracy_by_consensus_level,
    consensus_level,
    majority_consensus,
    supermajority_consensus,
)
from corrfilter.cfi.corrfilter_score import (
    CorrFilterScores,
    RMode,
    corrfilter_filter,
    corrfilter_score,
    resolve_R,
    retention_match_threshold,
)
from corrfilter.cfi.manifest import CFITriggers, build_cfi_manifest, detect_triggers
from corrfilter.cfi.metrics import (
    accuracy_by_consensus_curve,
    false_retention_by_ratio,
    false_retention_rate,
    retention_rate,
)
from corrfilter.cfi.prompts import BiasedPrompt
from corrfilter.cfi.replay import biased_prediction, simulate_bank_votes

__all__ = [
    "BiasMechanism",
    "BiasSpec",
    "BIAS_REGISTRY",
    "BiasedPrompt",
    "CFITriggers",
    "detect_triggers",
    "build_cfi_manifest",
    "CFIBankVariant",
    "build_bank_variants",
    "biased_prediction",
    "simulate_bank_votes",
    "majority_consensus",
    "supermajority_consensus",
    "consensus_level",
    "accuracy_by_consensus_level",
    "corrfilter_score",
    "CorrFilterScores",
    "RMode",
    "resolve_R",
    "corrfilter_filter",
    "retention_match_threshold",
    "retention_rate",
    "false_retention_rate",
    "accuracy_by_consensus_curve",
    "false_retention_by_ratio",
    "bootstrap_ci",
]

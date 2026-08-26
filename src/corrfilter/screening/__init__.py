"""Routing-feasibility screening: generators, deployment features, and gates.

* ``generators`` -- synthetic dependence-regime instance generators and the
  shared matched-retention evaluation protocol.
* ``features``   -- the label-free deployment feature extractor.
* ``feasibility``-- the cheap pre-registered gate that decides whether a
  candidate task/bank can support a routing experiment at all.
* ``sources``    -- candidate task adapters for the screening campaign.
"""

from corrfilter.screening.feasibility import screen_candidate
from corrfilter.screening.features import extract_features
from corrfilter.screening.generators import (
                                             Source,
                                             evaluate,
                                             majority_regime,
                                             make_instance,
                                             make_mixed_instance,
                                             marginal_params,
                                             mixtures,
                                             wrong_probs,
)

__all__ = [
    "Source",
    "evaluate",
    "extract_features",
    "majority_regime",
    "make_instance",
    "make_mixed_instance",
    "marginal_params",
    "mixtures",
    "screen_candidate",
    "wrong_probs",
]

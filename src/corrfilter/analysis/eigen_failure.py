"""Item-level failure-mode decomposition from the eigenstructure of R.

The eigenvectors of the error-correlation matrix ``R`` index directions in
judge-error space along which the bank co-varies (proposal §3.5). This module
turns those judge-space directions into *item-level* evidence:

* it projects each calibration item's error pattern onto a leading eigenvector
  to get a per-item loading score (reusing :func:`top_eigen_directions`);
* it attaches the item-side metadata (prompt, candidates, gold, per-judge
  vote/error, consensus ratio, position-swap) needed for manual review; and
* it computes a deterministic heuristic failure tag plus a set of
  interpretable per-item signals (verbosity, polish, refusal, sycophancy,
  position, common-difficulty, prompt-divergence, cross-family co-failure) so a
  qualitative review can confirm or correct the automatic interpretation.

Everything here is analysis-only and deterministic: no inference, no RNG.
The error matrix ``E``, correlation ``R``, and judge order are reused verbatim
from the H1 ``correlation.npz`` produced by ``scripts/dependence/compute_correlation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corrfilter.analysis.failure_modes import EigenDirection
from corrfilter.cfi.replay import _AGREE_RE, _CONFIDENT_RE, _REFUSAL_RE, _count
from corrfilter.data import CalibrationItem

# Heuristic failure tags. These align to the candidate failure families the
# proposal asks a leading direction to be matched against (§3.5).
TAG_POSITION = "position_bias"
TAG_COMMON_DIFFICULT = "common_difficult"
TAG_PROMPT_DIVERGENCE = "prompt_divergence"
TAG_CROSS_FAMILY = "cross_family_cofailure"
TAG_SAFETY_REFUSAL = "safety_refusal"
TAG_VERBOSITY_STYLE = "verbosity_style"
TAG_HALLUCINATION = "hallucination_factuality"
TAG_MIXED = "mixed_unclear"

ALL_TAGS = (
    TAG_POSITION,
    TAG_COMMON_DIFFICULT,
    TAG_PROMPT_DIVERGENCE,
    TAG_CROSS_FAMILY,
    TAG_SAFETY_REFUSAL,
    TAG_VERBOSITY_STYLE,
    TAG_HALLUCINATION,
    TAG_MIXED,
)

# The alignment families reported per eigenvector in the summary, paired with
# the per-item signal that proxies each one.
ALIGNMENT_FAMILIES = (
    ("position_bias", "position_swap_rate"),
    ("common_difficult", "error_count"),
    ("prompt_divergence", "prompt_divergence_score"),
    ("cross_family_cofailure", "cross_family_score"),
    ("safety_refusal", "safety_refusal_signal"),
    ("verbosity_style", "verbosity_signal"),
    ("hallucination_factuality", "factuality_signal"),
)


def _family_of(logical_id: str) -> str:
    """Base model family from a ``model::method`` logical id (e.g. ``llama``)."""
    return logical_id.split("::", 1)[0].split("-", 1)[0]


def _method_of(logical_id: str) -> str:
    """Prompt method (``pairwise`` / ``likert``) from a logical id."""
    return logical_id.split("::", 1)[1] if "::" in logical_id else ""


def orient_directions(directions: list[EigenDirection]) -> list[EigenDirection]:
    """Fix each eigenvector's arbitrary sign for deterministic interpretation.

    Eigenvectors are defined up to sign. We orient each so its dominant
    component is positive (ties broken by the sum), which makes a positive item
    loading mean "this item sits on the failure side of the direction" — for v1
    (uniform sign) that is simply "many judges erred".
    """
    oriented: list[EigenDirection] = []
    for d in directions:
        v = d.judge_loading
        dominant = v[int(np.argmax(np.abs(v)))]
        sign = 1.0
        if dominant < 0 or (dominant == 0 and v.sum() < 0):
            sign = -1.0
        oriented.append(
            EigenDirection(
                rank=d.rank,
                eigenvalue=d.eigenvalue,
                variance_explained=d.variance_explained,
                judge_loading=v * sign,
                item_loading=d.item_loading * sign,
            )
        )
    return oriented


@dataclass(frozen=True)
class ItemSignals:
    """Interpretable per-item heuristic signals over all calibration items."""

    error_count: np.ndarray            # (N,) number of judges that erred
    consensus_ratio: np.ndarray        # (N,) max(p, 1-p) over voting judges
    position_swap_rate: np.ndarray     # (N,) fraction of judges that saw a swap
    verbosity_signal: np.ndarray       # (N,) |len(chosen) - len(rejected)| / max
    polish_signal: np.ndarray          # (N,) |confident(chosen) - confident(rej)|
    sycophancy_signal: np.ndarray      # (N,) |agree(chosen) - agree(rejected)|
    safety_refusal_signal: np.ndarray  # (N,) Safety subset or refusal-word delta
    factuality_signal: np.ndarray      # (N,) Factuality/Math subset indicator
    prompt_divergence_score: np.ndarray  # (N,) |err(pairwise) - err(likert)|
    cross_family_score: np.ndarray     # (N,) # of distinct families that erred


def compute_item_signals(
    items: list[CalibrationItem],
    E: np.ndarray,
    M: np.ndarray,
    swap: np.ndarray,
    logical_ids: list[str],
) -> ItemSignals:
    """Compute deterministic per-item signals used for tagging and alignment.

    ``E`` and ``M`` are aligned ``(N × n)`` error / availability matrices over
    the same items and judge columns; ``swap`` is the ``(N × n)`` per-judge
    position-swap flag pulled from the vote cache.
    """
    n_items, n_judges = E.shape
    families = np.array([_family_of(lid) for lid in logical_ids])
    methods = np.array([_method_of(lid) for lid in logical_ids])
    pairwise_cols = methods == "pairwise"
    likert_cols = methods == "likert"

    error_count = (E * M).sum(axis=1)
    counts = M.sum(axis=1)
    # consensus ratio uses errors as a proxy for agreement on the *gold* label:
    # gold is constant (chosen ≻ rejected), so the fraction voting correctly is
    # 1 - mean error; consensus is max(p, 1-p).
    with np.errstate(invalid="ignore", divide="ignore"):
        p_correct = np.where(counts > 0, 1.0 - error_count / np.maximum(counts, 1), 0.5)
    consensus_ratio = np.maximum(p_correct, 1.0 - p_correct)

    with np.errstate(invalid="ignore", divide="ignore"):
        position_swap_rate = np.where(
            counts > 0, (swap * M).sum(axis=1) / np.maximum(counts, 1), 0.0
        )

    verbosity = np.zeros(n_items)
    polish = np.zeros(n_items)
    syco = np.zeros(n_items)
    safety = np.zeros(n_items)
    factual = np.zeros(n_items)
    for j, it in enumerate(items):
        lc, lr = len(it.chosen), len(it.rejected)
        verbosity[j] = abs(lc - lr) / max(lc, lr, 1)
        polish[j] = abs(_count(_CONFIDENT_RE, it.chosen) - _count(_CONFIDENT_RE, it.rejected))
        syco[j] = abs(_count(_AGREE_RE, it.chosen) - _count(_AGREE_RE, it.rejected))
        refusal_delta = abs(_count(_REFUSAL_RE, it.chosen) - _count(_REFUSAL_RE, it.rejected))
        is_safety = it.subset == "Safety" or it.category == "safety"
        safety[j] = max(float(is_safety), min(refusal_delta, 1.0))
        factual[j] = float(it.subset in ("Factuality", "Math") or it.category == "reasoning")

    # Prompt divergence: gap between pairwise-judge and likert-judge error rates.
    def _rate(cols: np.ndarray) -> np.ndarray:
        c = (M[:, cols]).sum(axis=1)
        e = (E[:, cols] * M[:, cols]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, e / np.maximum(c, 1), 0.0)

    prompt_div = np.abs(_rate(pairwise_cols) - _rate(likert_cols))

    # Cross-family co-failure: how many distinct base families erred on the item.
    cross_family = np.zeros(n_items)
    erred = (E.astype(bool)) & (M.astype(bool))
    uniq_families = sorted(set(families.tolist()))
    for j in range(n_items):
        fams = {families[i] for i in range(n_judges) if erred[j, i]}
        cross_family[j] = len(fams)
    cross_family = cross_family / max(len(uniq_families), 1)

    return ItemSignals(
        error_count=error_count,
        consensus_ratio=consensus_ratio,
        position_swap_rate=position_swap_rate,
        verbosity_signal=verbosity,
        polish_signal=polish,
        sycophancy_signal=syco,
        safety_refusal_signal=safety,
        factuality_signal=factual,
        prompt_divergence_score=prompt_div,
        cross_family_score=cross_family,
    )


def heuristic_tag(item: CalibrationItem, sig: ItemSignals, idx: int, n_judges: int) -> str:
    """Assign a single deterministic heuristic failure tag to an item.

    Priority order favours the most specific, content-grounded signals first
    (safety, factuality, style asymmetry), then structural signals
    (position, prompt divergence, cross-family), then falls back to the
    bank-wide "common difficult" bucket and finally "mixed_unclear".
    """
    if sig.safety_refusal_signal[idx] >= 1.0:
        return TAG_SAFETY_REFUSAL
    if sig.verbosity_signal[idx] >= 0.6 or sig.polish_signal[idx] >= 3:
        return TAG_VERBOSITY_STYLE
    if sig.factuality_signal[idx] >= 1.0 and sig.error_count[idx] >= max(2, n_judges // 3):
        return TAG_HALLUCINATION
    if sig.position_swap_rate[idx] >= 0.5 and sig.prompt_divergence_score[idx] < 0.25:
        # A swap-heavy item where pairwise/likert agree points at position layout.
        return TAG_POSITION
    if sig.prompt_divergence_score[idx] >= 0.34:
        return TAG_PROMPT_DIVERGENCE
    if sig.error_count[idx] >= max(3, (2 * n_judges) // 3):
        return TAG_COMMON_DIFFICULT
    if sig.cross_family_score[idx] >= 0.6:
        return TAG_CROSS_FAMILY
    return TAG_MIXED


def alignment_scores(item_loading: np.ndarray, sig: ItemSignals) -> dict[str, float]:
    """Pearson correlation between an eigenvector's item loadings and each signal.

    A high positive correlation means items lying far along the failure
    direction also score high on that interpretable signal, i.e. the direction
    is *aligned* with that failure family.
    """

    def _corr(x: np.ndarray) -> float:
        a = item_loading.astype(np.float64)
        b = x.astype(np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    signal_map = {
        "position_swap_rate": sig.position_swap_rate,
        "error_count": sig.error_count,
        "prompt_divergence_score": sig.prompt_divergence_score,
        "cross_family_score": sig.cross_family_score,
        "safety_refusal_signal": sig.safety_refusal_signal,
        "verbosity_signal": sig.verbosity_signal,
        "factuality_signal": sig.factuality_signal,
    }
    out: dict[str, float] = {}
    for family, signal_key in ALIGNMENT_FAMILIES:
        out[family] = _corr(signal_map[signal_key])
    return out


def describe_judge_loading(judge_loading: np.ndarray, logical_ids: list[str]) -> str:
    """One-line interpretation of an eigenvector's judge-space loading pattern.

    Three signatures are recognised, in order: a near-uniform sign (every judge
    loads the same way → a common direction); a within-model pairwise-vs-likert
    sign flip (the same base model loads with opposite signs across its two
    prompt styles → a prompt-divergence axis, even when the global per-method
    means cancel); and otherwise a model/family-specific contrast.
    """
    v = judge_loading
    same_sign = float(np.mean(np.sign(v) == np.sign(v[np.argmax(np.abs(v))])))
    families = np.array([_family_of(lid) for lid in logical_ids])
    methods = np.array([_method_of(lid) for lid in logical_ids])

    if same_sign >= 0.9:
        return (
            "near-uniform sign across the bank: a common direction every judge "
            f"loads on ({same_sign:.0%} share the dominant sign)"
        )

    # Within-model prompt-method divergence: per base model, does its pairwise
    # loading sit opposite its likert loading? Weight each flip by how much
    # loading mass the model carries so noise near zero does not dominate.
    flip_mass = 0.0
    flipped_models: list[str] = []
    for fam in sorted(set(families.tolist())):
        pw_i = [i for i in range(len(v)) if families[i] == fam and methods[i] == "pairwise"]
        lk_i = [i for i in range(len(v)) if families[i] == fam and methods[i] == "likert"]
        if not pw_i or not lk_i:
            continue
        pw_w, lk_w = v[pw_i[0]], v[lk_i[0]]
        if np.sign(pw_w) != np.sign(lk_w) and min(abs(pw_w), abs(lk_w)) >= 0.1:
            flip_mass += abs(pw_w) + abs(lk_w)
            flipped_models.append(fam)
    if flip_mass >= 0.5 and flipped_models:
        return (
            "within-model pairwise-vs-likert sign flips on "
            f"{', '.join(flipped_models)}: a prompt-divergence axis"
        )

    pw = v[methods == "pairwise"]
    lk = v[methods == "likert"]
    method_gap = abs(pw.mean() - lk.mean()) if pw.size and lk.size else 0.0
    if method_gap >= 0.15:
        sign = "pairwise-positive / likert-negative" if pw.mean() > lk.mean() else "likert-positive / pairwise-negative"
        return f"global prompt-method shift ({sign}, |Δmean|={method_gap:.2f}): a prompt-divergence axis"
    return "mixed sign with no single dominant axis: a model/family-specific contrast"

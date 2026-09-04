"""Forced-choice verdict-token scoring (P0-1).

Additive: the generation path in ``hf_judge.HFJudge`` is untouched. This module reads the
model's next-token distribution at the verdict position instead of generating and parsing,
which makes abstention structurally impossible and therefore removes abstention as a
selection confound in the base / GRPO / DPO comparison.

Prompts, chat templating, position assignment, truncation, and quantization are identical
to the generation path; only the readout differs.

- ``pairwise``: one forward pass, compare probability mass on tokenizations of "A" vs "B".
- ``likert``: two forward passes. Teacher-force ``A=`` and read the 1-5 digit distribution
  for an expected score ``E_A``; teacher-force ``A=<argmax>,B=`` and read it again for
  ``E_B``. Verdict is ``A`` iff ``E_A > E_B``. Expected scores rather than argmax digits
  remove the integer-tie abstention that dominates the likert judges' abstention rate.

Exact float ties are resolved toward the *rejected* candidate so the protocol can never
favour gold, and are counted so the count can be reported.

See experiments/abstention_control/PREREG.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import torch

from corrfilter.data import CalibrationItem
from corrfilter.judges.base import JudgeVote
from corrfilter.judges.hf_judge import HFJudge

logger = logging.getLogger(__name__)

LIKERT_DIGITS = ("1", "2", "3", "4", "5")

# Total probability on the verdict alternatives below which the readout is not
# meaningful (usually a truncated prompt, where the model is completing prose).
DEGENERATE_MASS = 1e-3


def _next_token_ids(tokenizer, prefix: str, surfaces: tuple[str, ...]) -> list[list[int]]:
    """Ids the model must emit next, in context, to produce each surface after ``prefix``.

    Resolving ids by encoding the surface in isolation is WRONG for SentencePiece
    tokenizers: ``encode("1")`` returns ``["", "1"]`` (two tokens) on Mistral and Phi, so a
    naive "first token of the encoding" fallback returns the shared whitespace marker for
    every digit. That silently collapsed all five Likert digits onto one id and produced a
    fully degenerate readout (mean verdict mass 2e-05 on 1195/1195 items). BPE tokenizers
    such as Qwen's happen to be immune, which is why the bug was model-specific.

    Strategy: diff ``encode(prefix)`` against ``encode(prefix + surface)`` and take the
    first token the surface contributes. Fall back to the last token of the surface encoded
    alone if appending perturbs the prefix tokenization. Finally drop any id claimed by more
    than one surface, since a shared id cannot discriminate between them.
    """
    base = tokenizer.encode(prefix, add_special_tokens=False)
    cands: list[set[int]] = []
    for surface in surfaces:
        ids: set[int] = set()
        full = tokenizer.encode(prefix + surface, add_special_tokens=False)
        if full[: len(base)] == base and len(full) > len(base):
            ids.add(int(full[len(base)]))
        else:  # appending perturbed the prefix tokenization
            solo = tokenizer.encode(surface, add_special_tokens=False)
            if solo:
                ids.add(int(solo[-1]))
        for variant in (surface, " " + surface):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        cands.append(ids)

    seen: dict[int, int] = {}
    for group in cands:
        for i in group:
            seen[i] = seen.get(i, 0) + 1
    out = [sorted(i for i in group if seen[i] == 1) for group in cands]
    for surface, ids in zip(surfaces, out):
        if not ids:
            raise RuntimeError(
                f"no discriminating token id for surface {surface!r} after prefix "
                f"...{prefix[-40:]!r}"
            )
    return out


@dataclass
class ForcedChoiceStats:
    """Diagnostics accumulated over a run."""

    n_items: int = 0
    n_exact_ties: int = 0
    n_degenerate: int = 0          # verdict-mass below DEGENERATE_MASS (unreliable readout)
    margin_sum: float = 0.0
    mass_sum: float = 0.0
    per_style: dict = field(default_factory=dict)

    def mean_margin(self) -> float:
        return self.margin_sum / max(self.n_items, 1)

    def mean_mass(self) -> float:
        return self.mass_sum / max(self.n_items, 1)


class ForcedChoiceJudge(HFJudge):
    """HFJudge whose verdict is read from logits rather than generated text."""

    def __init__(self, *args, position_override: dict[str, bool] | None = None, **kwargs):
        """``position_override`` pins the candidate slot per item_id.

        The current inherited ``_position_swap`` is deterministic across processes.
        ``position_override`` additionally reproduces the exact slot assignments stored in
        an earlier cache, which is necessary for like-for-like comparisons with historical
        runs created before stable position hashing was introduced.
        """
        super().__init__(*args, **kwargs)
        self.stats = ForcedChoiceStats()
        self.position_override = position_override
        self._ab_ids: dict[str, list[int]] | None = None
        self._digit_ids: list[list[int]] | None = None

    def _position_swap(self, item_id: str) -> bool:
        if self.position_override is not None:
            try:
                return bool(self.position_override[str(item_id)])
            except KeyError as exc:
                raise KeyError(
                    f"no pinned position for item {item_id!r} on {self.spec.logical_id}"
                ) from exc
        return super()._position_swap(item_id)

    def _ensure_vocab_maps(self, chat_example: str) -> None:
        """Resolve verdict-token ids in the actual prompt context (see _next_token_ids)."""
        if self._ab_ids is None:
            a, b = _next_token_ids(self._tokenizer, chat_example, ("A", "B"))
            self._ab_ids = {"A": a, "B": b}
            logger.info("%s verdict ids: A=%s B=%s", self.spec.logical_id, a, b)
        if self._digit_ids is None and self.spec.prompt_style == "likert":
            self._digit_ids = _next_token_ids(
                self._tokenizer, chat_example + "A=", LIKERT_DIGITS)
            logger.info("%s digit ids: %s", self.spec.logical_id, self._digit_ids)

    @torch.inference_mode()
    def _next_token_probs(self, texts: list[str]) -> torch.Tensor:
        """Probability distribution over the vocabulary at the next position, per text.

        ``logits_to_keep=1`` restricts the LM head to the final position. Without it the
        model materialises a ``[batch, seq, vocab]`` tensor (6+ GB at batch 8 x 2.5k tokens
        x 152k vocab), which is pure waste here and OOMs a 24 GB card.
        """
        enc = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length,
        ).to(self._model.device)
        try:
            out = self._model(**enc, logits_to_keep=1)
        except TypeError:  # older transformers spelling
            out = self._model(**enc, num_logits_to_keep=1)
        return torch.softmax(out.logits[:, -1, :].float(), dim=-1)

    def _token_budget_batches(self, texts: list[str], budget: int):
        """Greedy length-sorted packing: yield index lists with max_len * size <= budget."""
        lens = [len(self._tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lens[i], reverse=True)
        batch, longest = [], 0
        for i in order:
            cand_longest = max(longest, lens[i])
            if batch and cand_longest * (len(batch) + 1) > budget:
                yield batch
                batch, longest = [i], lens[i]
            else:
                batch.append(i)
                longest = cand_longest
        if batch:
            yield batch

    def _mass(self, probs: torch.Tensor, ids: list[int]) -> torch.Tensor:
        return probs[:, ids].sum(dim=-1)

    @torch.inference_mode()
    def vote_batch(self, items: Iterable[CalibrationItem]) -> list[JudgeVote]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"judge {self.spec.logical_id} not loaded")
        items = list(items)
        votes: list[JudgeVote] = []
        style = self.spec.prompt_style

        # Token-budget batching: pack length-sorted items so that
        # max_len_in_batch * batch_size stays under a fixed budget. Long items therefore
        # run in small batches and short items in large ones, which avoids the OOM that a
        # fixed batch size hits on the 2.5k-token tail while keeping throughput high.
        # Votes are keyed by item_id and the cache is order-independent, so batching
        # affects throughput only.
        all_swaps = [self._position_swap(it.item_id) for it in items]
        all_chat = [self._render_chat(self.prompt.render(it, sw).text)
                    for it, sw in zip(items, all_swaps)]
        budget = self.batch_size * 1024
        self._ensure_vocab_maps(all_chat[0])

        for idxs in self._token_budget_batches(all_chat, budget):
            batch = [items[i] for i in idxs]
            swaps = [all_swaps[i] for i in idxs]
            chat = [all_chat[i] for i in idxs]

            if style == "pairwise":
                probs = self._next_token_probs(chat)
                pa = self._mass(probs, self._ab_ids["A"])
                pb = self._mass(probs, self._ab_ids["B"])
                scores_a, scores_b = pa, pb
                masses = (pa + pb)
                raws = [f"forced_choice:pairwise p(A)={a:.6g} p(B)={b:.6g} mass={m:.3g}"
                        for a, b, m in zip(pa.tolist(), pb.tolist(), masses.tolist())]
            elif style == "likert":
                # Pass 1: expected score for A.
                probs_a = self._next_token_probs([c + "A=" for c in chat])
                digit_mass_a = torch.stack(
                    [self._mass(probs_a, ids) for ids in self._digit_ids], dim=-1)
                norm_a = digit_mass_a / digit_mass_a.sum(-1, keepdim=True).clamp_min(1e-12)
                weights = torch.arange(1, 6, device=norm_a.device, dtype=norm_a.dtype)
                exp_a = (norm_a * weights).sum(-1)
                top_a = (norm_a.argmax(-1) + 1).tolist()
                # Pass 2: expected score for B, conditioned on A's most likely score.
                probs_b = self._next_token_probs(
                    [f"{c}A={d},B=" for c, d in zip(chat, top_a)])
                digit_mass_b = torch.stack(
                    [self._mass(probs_b, ids) for ids in self._digit_ids], dim=-1)
                norm_b = digit_mass_b / digit_mass_b.sum(-1, keepdim=True).clamp_min(1e-12)
                exp_b = (norm_b * weights).sum(-1)
                scores_a, scores_b = exp_a, exp_b
                masses = torch.minimum(digit_mass_a.sum(-1), digit_mass_b.sum(-1))
                raws = [f"forced_choice:likert E[A]={a:.6g} E[B]={b:.6g} topA={d} mass={m:.3g}"
                        for a, b, d, m in zip(exp_a.tolist(), exp_b.tolist(), top_a,
                                              masses.tolist())]
            else:
                raise ValueError(f"forced choice not defined for prompt style {style!r}")

            for it, swap, sa, sb, mass, raw in zip(
                batch, swaps, scores_a.tolist(), scores_b.tolist(), masses.tolist(), raws
            ):
                self.stats.n_items += 1
                self.stats.margin_sum += abs(sa - sb)
                self.stats.mass_sum += mass
                if mass < DEGENERATE_MASS:
                    self.stats.n_degenerate += 1
                if sa == sb:
                    self.stats.n_exact_ties += 1
                    # Resolve toward the candidate that is NOT gold, so the protocol can
                    # never manufacture accuracy. Gold is always "chosen"; chosen sits in
                    # slot A when not swapped, slot B when swapped.
                    letter = "B" if not swap else "A"
                else:
                    letter = "A" if sa > sb else "B"
                agrees = (letter == "A" and not swap) or (letter == "B" and swap)
                votes.append(JudgeVote(
                    judge_id=self.spec.logical_id,
                    item_id=it.item_id,
                    vote=1 if agrees else 0,
                    raw_response=raw,
                    position_swapped=swap,
                ))

        return votes

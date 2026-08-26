"""Bias-injected prompt wrappers for the CFI experiment.

A ``BiasedPrompt`` wraps a base ``PromptTemplate`` (pairwise or Likert) and
prepends a bias instruction to the system prompt at render time. The decoder
is inherited unchanged so that the position-aware vote semantics in
[src/corrfilter/judges/prompts.py](../judges/prompts.py) keep working: a vote
of 1 still means "agrees with chosen ≻ rejected" regardless of which bias was
injected.

This is the only intentional difference between a clean judge and a biased
judge in real-inference mode. The dataset, gold labels, and decoder are all
untouched.
"""

from __future__ import annotations

from corrfilter.cfi.biases import BiasSpec
from corrfilter.data import CalibrationItem
from corrfilter.judges.prompts import PromptTemplate, RenderedPrompt


class BiasedPrompt(PromptTemplate):
    """Inject a bias instruction into the system prompt of a base template."""

    def __init__(self, base: PromptTemplate, bias: BiasSpec):
        self.base = base
        self.bias = bias
        self.name = f"{base.name}__{bias.mechanism.value}"

    @classmethod
    def from_instruction(
        cls, base: PromptTemplate, name: str, instruction: str
    ) -> BiasedPrompt:
        """Build a biased prompt from a raw instruction string.

        Used by the config-driven 8-mechanism CFI bank
        (:mod:`corrfilter.cfi.bias_bank`), which carries instructions in YAML
        rather than the enum-based :data:`BIAS_REGISTRY`. Bypasses ``__init__``
        so no :class:`BiasSpec` is required.
        """
        obj = cls.__new__(cls)
        obj.base = base
        obj.bias = None
        obj._instruction = instruction
        obj.name = f"{base.name}__{name}"
        return obj

    @property
    def instruction(self) -> str:
        if self.bias is not None:
            return self.bias.instruction
        return getattr(self, "_instruction", "")

    def render(self, item: CalibrationItem, swap: bool) -> RenderedPrompt:
        rp = self.base.render(item, swap)
        instruction = self.instruction
        if not instruction.strip():
            # No-op injection (clean control) — return the base prompt verbatim.
            return rp
        marker = "<|system|>\n"
        if marker in rp.text:
            text = rp.text.replace(
                marker,
                f"{marker}[Judge guidance]: {instruction}\n\n",
                1,
            )
        else:
            text = f"[Judge guidance]: {instruction}\n\n" + rp.text
        return RenderedPrompt(text=text, swapped=swap)

    def decode(self, raw: str, swapped: bool) -> int:
        return self.base.decode(raw, swapped)

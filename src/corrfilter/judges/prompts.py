"""Judge prompt templates: pairwise compare and Likert-then-compare.

Two prompt styles are crossed with each base judge (proposal §3.3) so that
prompt-template-induced correlation can be separated from model-family
correlation in the error-correlation matrix R.

Both templates randomise position: candidate A and candidate B are assigned
from (chosen, rejected) according to a per-item swap bit. The judge's
A/B verdict is then decoded back to the agree/disagree frame so that the
returned vote always answers "did the judge agree with chosen ≻ rejected?".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from corrfilter.data import CalibrationItem


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered prompt plus the position bit needed to decode the vote."""

    text: str
    swapped: bool        # True ⇒ chosen was presented as B; False ⇒ chosen was A


class PromptTemplate(ABC):
    """Abstract template; concrete subclasses implement ``render`` and ``decode``."""

    name: str

    @abstractmethod
    def render(self, item: CalibrationItem, swap: bool) -> RenderedPrompt: ...

    @abstractmethod
    def decode(self, raw: str, swapped: bool) -> int:
        """Decode raw model output to vote ∈ {0, 1, -1}.

        Returns 1 if the judge agrees with the gold preference (chosen ≻
        rejected), 0 if it disagrees, -1 if the output is unparseable
        (abstain). The runner records abstentions as missing values.
        """


def _extract_first_letter(raw: str, allowed: tuple[str, ...] = ("A", "B")) -> str | None:
    """Return the first standalone A/B token in ``raw``, else None.

    Resolution order:
      1. If the stripped output is exactly one of the allowed letters
         (case-insensitive), return that letter. Handles compliant judges
         that obey 'reply with exactly one letter'.
      2. Otherwise look for an uppercase A or B at a word boundary. This
         tolerates chatty judges that say 'The answer is A.' but avoids
         false positives like the 'a' inside 'Clearly' or 'because'.
      3. Otherwise abstain.
    """
    import re

    if not raw:
        return None
    stripped = raw.strip().rstrip(".,!?\"' ")
    if len(stripped) == 1 and stripped.upper() in allowed:
        return stripped.upper()
    pattern = r"\b[" + "".join(allowed) + r"]\b"
    match = re.search(pattern, raw)
    if match:
        return match.group(0)
    return None


class PairwisePrompt(PromptTemplate):
    """Direct pairwise comparison: 'output exactly one letter, A or B'."""

    name = "pairwise"

    SYSTEM = (
        "You are an impartial judge comparing two assistant responses to the same prompt. "
        "Decide which response is more helpful, accurate, and appropriate. "
        "Reply with exactly one letter: A or B. No other text."
    )

    USER_TEMPLATE = (
        "Prompt:\n{prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}\n\n"
        "Which response is better? Answer with exactly one letter, A or B."
    )

    def render(self, item: CalibrationItem, swap: bool) -> RenderedPrompt:
        if swap:
            a, b = item.rejected, item.chosen
        else:
            a, b = item.chosen, item.rejected
        text = (
            f"<|system|>\n{self.SYSTEM}\n<|user|>\n"
            + self.USER_TEMPLATE.format(prompt=item.prompt, response_a=a, response_b=b)
            + "\n<|assistant|>\n"
        )
        return RenderedPrompt(text=text, swapped=swap)

    def decode(self, raw: str, swapped: bool) -> int:
        letter = _extract_first_letter(raw)
        if letter is None:
            return -1
        # Without swap, A == chosen; with swap, B == chosen.
        agrees = (letter == "A" and not swapped) or (letter == "B" and swapped)
        return 1 if agrees else 0


class LikertPrompt(PromptTemplate):
    """Likert-then-compare: judge rates A on 1-5 and B on 1-5, then we compare.

    The proposal contrasts pairwise with Likert-then-compare to expose whether
    different prompt styles produce systematically different error patterns
    (a candidate prompt-template-driven correlation cluster in R).
    """

    name = "likert"

    SYSTEM = (
        "You are an impartial judge scoring assistant responses for quality. "
        "Quality covers helpfulness, factual accuracy, and appropriateness. "
        "You will be shown two responses and asked to score them on a 1 to 5 integer scale, "
        "where 1 is very poor and 5 is excellent. "
        "You MUST give the two responses different scores; ties are not allowed. "
        "If the responses seem close, pick a winner on whichever subtle difference matters most "
        "(precision, completeness, tone, formatting). "
        "Reply with exactly two integers separated by a comma, formatted 'A=<n>,B=<n>'."
    )

    USER_TEMPLATE = (
        "Prompt:\n{prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}\n\n"
        "Score each response from 1 to 5. The two scores must differ. "
        "Respond exactly as 'A=<n>,B=<n>'."
    )

    def render(self, item: CalibrationItem, swap: bool) -> RenderedPrompt:
        if swap:
            a, b = item.rejected, item.chosen
        else:
            a, b = item.chosen, item.rejected
        text = (
            f"<|system|>\n{self.SYSTEM}\n<|user|>\n"
            + self.USER_TEMPLATE.format(prompt=item.prompt, response_a=a, response_b=b)
            + "\n<|assistant|>\n"
        )
        return RenderedPrompt(text=text, swapped=swap)

    def decode(self, raw: str, swapped: bool) -> int:
        import re

        if not raw:
            return -1
        match = re.search(r"A\s*=\s*(\d+).*?B\s*=\s*(\d+)", raw, re.IGNORECASE | re.DOTALL)
        if not match:
            return -1
        try:
            score_a = int(match.group(1))
            score_b = int(match.group(2))
        except ValueError:
            return -1
        if score_a == score_b:
            return -1  # Likert tie ⇒ abstain rather than break the preference frame.
        winner = "A" if score_a > score_b else "B"
        agrees = (winner == "A" and not swapped) or (winner == "B" and swapped)
        return 1 if agrees else 0


PROMPT_TEMPLATES: dict[str, type[PromptTemplate]] = {
    PairwisePrompt.name: PairwisePrompt,
    LikertPrompt.name: LikertPrompt,
}


def get_prompt_template(style: str) -> PromptTemplate:
    if style not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {list(PROMPT_TEMPLATES)}")
    return PROMPT_TEMPLATES[style]()

"""API-backed judge adapter (OpenAI / OpenRouter / Anthropic / Google).

Drop-in alternative to :class:`HFJudge` for adding proprietary judges (e.g.
GPT-4o-mini, Gemini Flash, Claude Haiku) to the bank. It reuses the SAME prompt
templates, the SAME position-swap protocol, and the SAME vote decoder as the
local judges, so an API judge contributes a directly comparable column to the
error matrix R.

Design notes:
* No paid call is made unless ``dry_run=False`` AND a key is present. In
  ``dry_run=True`` the adapter renders every prompt, counts tokens, and returns
  a deterministic placeholder vote, so the full pipeline (and the cost estimate)
  can be exercised for free.
* Token usage and an estimated dollar cost are accumulated in ``self.usage``.
* Position swap is a stable hash of (seed, logical_id, item_id) so runs are
  reproducible regardless of ``PYTHONHASHSEED``.

This module is self-contained and imports its provider SDK lazily; none is
required for ``dry_run`` accounting.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from corrfilter.data import CalibrationItem
from corrfilter.judges.base import Judge, JudgeSpec, JudgeVote
from corrfilter.judges.position import position_swap
from corrfilter.judges.prompts import get_prompt_template

logger = logging.getLogger(__name__)

# Approximate USD per 1M tokens (input, output); update as vendor pricing changes.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # The paper's six pinned frontier judges. Gemini rates come from the
    # project's own billed run logs (outputs/gemini_bank/run_log_*.json);
    # OpenRouter rates from the live /api/v1/models pricing endpoint
    # (2026-08-07), validated against the Stage-P pilot's exact balance
    # delta ($2.224 reconstructed vs $2.217 billed).
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.6-flash": (0.3, 2.5),
    "gemini-3.5-flash-lite": (0.1, 0.4),
    "openai/gpt-5.6-sol": (5.0, 30.0),
    "anthropic/claude-opus-5": (5.0, 25.0),
    "x-ai/grok-4.5": (2.0, 6.0),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    abstentions: int = 0

    def cost(self, price_in: float, price_out: float) -> float:
        return self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out


class APIJudge(Judge):
    """A judge backed by a hosted chat-completion API."""

    def __init__(
        self,
        spec: JudgeSpec,
        provider: str,                       # "openai" | "openrouter" | "anthropic" | "google"
        model: str,                          # vendor model id, e.g. "gpt-4o-mini"
        *,
        api_key_env: str | None = None,      # env var holding the key
        base_url: str | None = None,         # e.g. https://openrouter.ai/api/v1
        max_new_tokens: int = 8,
        temperature: float = 0.0,
        position_seed: int = 20260601,
        shuffle_position: bool = True,
        dry_run: bool = True,
        price_in: float | None = None,
        price_out: float | None = None,
        request_pause_s: float = 0.0,
    ):
        super().__init__(spec)
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env or {
            "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY",
        }.get(provider, "API_KEY")
        self.base_url = base_url
        self.prompt = get_prompt_template(spec.prompt_style)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.position_seed = position_seed
        self.shuffle_position = shuffle_position
        self.dry_run = dry_run
        pin, pout = PRICES.get(model, (None, None))
        self.price_in = price_in if price_in is not None else (pin or 0.0)
        self.price_out = price_out if price_out is not None else (pout or 0.0)
        # A live judge with no known price would silently report $0 and defeat
        # any budget ceiling (observed in the Stage-P pilot). Track and warn.
        self.cost_known = bool((price_in is not None and price_out is not None)
                               or model in PRICES)
        if not dry_run and not self.cost_known:
            logger.warning(
                "APIJudge model %r has no PRICES entry and no explicit price: "
                "reported cost will be $0 and budget ceilings will NOT be "
                "enforced for this judge", model)
        self.request_pause_s = request_pause_s
        self._client = None
        self.usage = Usage()

    # ---- lifecycle ----
    def load(self) -> None:
        if self.dry_run:
            logger.info("APIJudge %s in DRY-RUN: no client, no paid calls", self.spec.logical_id)
            return
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} not set; cannot run {self.provider} judge live")
        if self.provider in ("openai", "openrouter"):
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._client = genai
        else:
            raise ValueError(f"unknown provider {self.provider!r}")

    def unload(self) -> None:
        self._client = None

    # ---- helpers ----
    def _position_swap(self, item_id: str) -> bool:
        """Deterministic slot assignment (see corrfilter.judges.position).

        Previously used Python's built-in hash(), which is salted per process, so slot
        assignment was not reproducible across runs and cross-bank comparisons silently
        saw different layouts. BLAKE2b is stable across processes, machines, and
        PYTHONHASHSEED values.
        """
        if not self.shuffle_position:
            return False
        return position_swap(item_id, self.spec.logical_id, self.position_seed)

    @staticmethod
    def _split_system_user(text: str) -> tuple[str, str]:
        """Parse the rendered <|system|>..<|user|>..<|assistant|> prompt into chat turns."""
        system = text.split("<|user|>", 1)[0].replace("<|system|>", "").strip()
        rest = text.split("<|user|>", 1)[1] if "<|user|>" in text else text
        user = rest.split("<|assistant|>", 1)[0].strip()
        return system, user

    @staticmethod
    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)            # ~4 chars/token heuristic for dry-run accounting

    def _call(self, system: str, user: str) -> tuple[str, int, int]:
        """Return (raw_text, input_tokens, output_tokens). Dry-run returns a placeholder."""
        if self.dry_run:
            return "A", self._est_tokens(system + user), 1
        if self.provider in ("openai", "openrouter"):
            r = self._client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_new_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            txt = r.choices[0].message.content or ""
            u = r.usage
            return txt, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)
        if self.provider == "anthropic":
            r = self._client.messages.create(
                model=self.model, max_tokens=self.max_new_tokens, temperature=self.temperature,
                system=system, messages=[{"role": "user", "content": user}],
            )
            txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return txt, r.usage.input_tokens, r.usage.output_tokens
        if self.provider == "google":
            mdl = self._client.GenerativeModel(self.model, system_instruction=system)
            r = mdl.generate_content(user, generation_config={"temperature": self.temperature,
                                                              "max_output_tokens": self.max_new_tokens})
            txt = r.text or ""
            um = getattr(r, "usage_metadata", None)
            return txt, getattr(um, "prompt_token_count", 0), getattr(um, "candidates_token_count", 0)
        raise ValueError(self.provider)

    # ---- voting ----
    def vote_batch(self, items: Iterable[CalibrationItem]) -> list[JudgeVote]:
        import time
        votes: list[JudgeVote] = []
        for it in items:
            swap = self._position_swap(it.item_id)
            rp = self.prompt.render(it, swap)
            system, user = self._split_system_user(rp.text)
            raw, tin, tout = self._call(system, user)
            self.usage.calls += 1
            self.usage.input_tokens += tin
            self.usage.output_tokens += tout
            vote = self.prompt.decode(raw, swap)
            if vote == -1:
                self.usage.abstentions += 1
            votes.append(JudgeVote(judge_id=self.spec.logical_id, item_id=it.item_id,
                                   vote=int(vote), raw_response=raw, position_swapped=swap))
            if self.request_pause_s and not self.dry_run:
                time.sleep(self.request_pause_s)
        return votes

    def cost_so_far(self) -> float:
        return self.usage.cost(self.price_in, self.price_out)


def estimate_cost(model: str, n_items: int, n_prompt_styles: int = 2, position_orders: int = 2,
                  avg_input_tokens: int = 2000, avg_output_tokens: int = 8) -> dict:
    """Budget estimate for adding one API judge model across the calibration set.

    position_orders=2 doubles calls to also measure each judge's position bias
    (set to 1 for a single presentation order).
    """
    pin, pout = PRICES.get(model, (None, None))
    if pin is None:
        raise ValueError(f"no price on file for {model}; pass price_in/out explicitly")
    calls = n_items * n_prompt_styles * position_orders
    cost = calls * (avg_input_tokens / 1e6 * pin + avg_output_tokens / 1e6 * pout)
    return {"model": model, "calls": calls, "input_tokens": calls * avg_input_tokens,
            "output_tokens": calls * avg_output_tokens, "usd_per_1M_in_out": [pin, pout],
            "estimated_cost_usd": round(cost, 2)}

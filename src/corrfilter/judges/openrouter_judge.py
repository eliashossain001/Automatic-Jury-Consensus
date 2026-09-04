"""OpenRouter judge for the multi-provider frontier replication.

Same contract as :class:`GeminiJudge` (same prompts, position-swap protocol,
decoder, retry/failure posture) but speaking OpenRouter's OpenAI-compatible
chat-completions REST API. Reasoning-capable models are handled via the
``reasoning`` request field (effort level or full exclusion); the decoder reads
only the final ``content`` (never the ``reasoning`` trace), and an empty
content (e.g. reasoning consumed ``max_tokens``) is an abstain, logged as a
truncation failure rather than silently dropped.

No paid call is made unless ``dry_run=False`` and a key is present.
"""

from __future__ import annotations

import json
import logging
import os

from corrfilter.judges.base import JudgeSpec
from corrfilter.judges.gemini_judge import GeminiJudge, TransientAPIError

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

KEY_ENV_VARS = ("OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY")

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class OpenRouterJudge(GeminiJudge):
    """A judge backed by OpenRouter (OpenAI-compatible API, many providers).

    Inherits the retry loop, thread-pooled ``vote_batch``, usage accounting,
    checkpoints-compatible interface, and failure log from ``GeminiJudge``;
    only the transport (payload/request/parse) differs.
    """

    def __init__(
        self,
        spec: JudgeSpec,
        model: str,
        *,
        reasoning_effort: str | None = None,   # "minimal"|"low"|... or None to omit
        reasoning_exclude: bool = False,        # request no reasoning at all where supported
        supports_temperature: bool = True,
        max_output_tokens: int = 2048,
        **kw,
    ):
        super().__init__(spec, model, max_output_tokens=max_output_tokens, **kw)
        self.reasoning_effort = reasoning_effort
        self.reasoning_exclude = reasoning_exclude
        self.supports_temperature = supports_temperature
        self.reasoning_tokens = 0   # billed as output; tracked separately like thought_tokens
        # Fallback detection: count the (provider, model) pairs echoed back by
        # OpenRouter so any silent substitution is visible in the run log.
        self.providers_seen: dict[str, int] = {}

    # ---- lifecycle ----
    def load(self) -> None:
        if self.dry_run:
            logger.info("OpenRouterJudge %s in DRY-RUN: no paid calls", self.logical_id)
            return
        for env in KEY_ENV_VARS:
            key = os.environ.get(env)
            if key:
                self._api_key = key
                return
        raise RuntimeError(f"none of {KEY_ENV_VARS} set; cannot run OpenRouter judge live")

    # ---- transport ----
    def _payload(self, system: str, user: str) -> dict:
        p: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.max_new_tokens,
        }
        if self.supports_temperature:
            p["temperature"] = self.temperature
        if self.reasoning_exclude:
            p["reasoning"] = {"exclude": True}
        elif self.reasoning_effort is not None:
            p["reasoning"] = {"effort": self.reasoning_effort}
        return p

    def _request_once(self, payload: dict) -> dict:
        import requests

        resp = requests.post(
            API_URL,
            json=payload,
            timeout=self.request_timeout_s,
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"},
        )
        if resp.status_code in RETRYABLE_STATUS:
            raise TransientAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code == 402:
            # Insufficient credits: hard stop, never retried.
            raise RuntimeError(f"OpenRouter 402 insufficient credits: {resp.text[:300]}")
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def _parse_response(self, item_id: str, data: dict) -> tuple[str, int, int]:
        if "error" in data:
            err = data["error"]
            code = int(err.get("code", 0) or 0)
            if code in RETRYABLE_STATUS:
                raise TransientAPIError(json.dumps(err)[:200])
            raise RuntimeError(f"OpenRouter error: {json.dumps(err)[:300]}")
        usage = data.get("usage", {}) or {}
        tin = int(usage.get("prompt_tokens", 0))
        tout = int(usage.get("completion_tokens", 0))
        details = usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)
        echo = f"{data.get('provider', '?')}::{data.get('model', '?')}"
        self.providers_seen[echo] = self.providers_seen.get(echo, 0) + 1
        choices = data.get("choices") or []
        if not choices:
            self.failures.append({"item_id": item_id, "kind": "refusal", "detail": "NO_CHOICES"})
            return "<REFUSAL:NO_CHOICES>", tin, tout
        ch = choices[0]
        text = (ch.get("message") or {}).get("content") or ""
        finish = str(ch.get("finish_reason", ""))
        if not text:
            kind = "truncated" if finish == "length" else "refusal"
            self.failures.append({"item_id": item_id, "kind": kind, "detail": finish or "EMPTY"})
            return f"<{'TRUNCATED' if kind == 'truncated' else 'REFUSAL'}:{finish or 'EMPTY'}>", tin, tout
        if finish not in ("stop", "length", ""):
            self.failures.append({"item_id": item_id, "kind": "refusal", "detail": finish})
        return text, tin, tout

    def cost_so_far(self) -> float:
        # OpenRouter bills reasoning tokens inside completion_tokens, so the
        # base Usage accounting already covers them; no extra term needed.
        return self.usage.cost(self.price_in, self.price_out)

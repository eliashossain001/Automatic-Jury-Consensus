"""Gemini REST judge for the frontier-judge pilot.

Subclass of :class:`APIJudge` that talks to the Google Generative Language
REST API (``v1beta``) directly instead of the deprecated ``google.generativeai``
SDK. The REST path is required to set per-model ``thinkingConfig`` (Gemini 3.x
models reject ``thinkingBudget`` and use ``thinkingLevel``), and it lets us
implement bounded-exponential-backoff retries and explicit failure logging.

Comparability contract (identical to the open-weight bank):
* same prompt templates (``corrfilter.judges.prompts``);
* same stable-hash position swap keyed by (seed, logical_id, item_id);
* same vote decoder (A/B -> agree/disagree, unparseable -> abstain -1).

Failure posture: transient HTTP errors (429/5xx/timeouts) are retried with
bounded exponential backoff; after ``max_retries`` the vote is recorded as an
abstain with a ``<API_ERROR:...>`` raw response and the failure is appended to
``self.failures`` so coverage can be reported. Nothing is silently dropped.
No paid call is made unless ``dry_run=False`` and a key is present.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from corrfilter.data import CalibrationItem
from corrfilter.judges.api_judge import APIJudge
from corrfilter.judges.base import JudgeSpec, JudgeVote

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Env vars checked in order for the API key. The first name matches the
# variable present in this project's .env (sic).
KEY_ENV_VARS = ("GEMNI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiJudge(APIJudge):
    """A judge backed by the Gemini REST API with retries and failure logging."""

    def __init__(
        self,
        spec: JudgeSpec,
        model: str,
        *,
        thinking_level: str | None = None,
        max_output_tokens: int = 64,
        temperature: float = 0.0,
        position_seed: int = 20260601,
        shuffle_position: bool = True,
        dry_run: bool = True,
        price_in: float = 0.0,
        price_out: float = 0.0,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        request_timeout_s: float = 120.0,
        workers: int = 1,
    ):
        super().__init__(
            spec,
            provider="google",
            model=model,
            api_key_env=KEY_ENV_VARS[0],
            max_new_tokens=max_output_tokens,
            temperature=temperature,
            position_seed=position_seed,
            shuffle_position=shuffle_position,
            dry_run=dry_run,
            price_in=price_in,
            price_out=price_out,
        )
        self.thinking_level = thinking_level
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.request_timeout_s = request_timeout_s
        self.workers = max(1, int(workers))
        self._api_key: str | None = None
        # Explicit failure log: list of dicts with item-level API problems
        # (exhausted retries, refusals, truncations). Reset per vote_batch call.
        self.failures: list[dict] = []
        self.thought_tokens = 0

    # ---- lifecycle ----
    def load(self) -> None:
        if self.dry_run:
            logger.info("GeminiJudge %s in DRY-RUN: no client, no paid calls", self.logical_id)
            return
        for env in KEY_ENV_VARS:
            key = os.environ.get(env)
            if key:
                self._api_key = key
                return
        raise RuntimeError(f"none of {KEY_ENV_VARS} set; cannot run Gemini judge live")

    def unload(self) -> None:
        self._api_key = None

    # ---- request plumbing ----
    def _payload(self, system: str, user: str) -> dict:
        gen_cfg: dict = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_new_tokens,
        }
        if self.thinking_level is not None:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen_cfg,
        }

    def _request_once(self, payload: dict) -> dict:
        """One HTTP POST; raises on transport errors, returns parsed JSON."""
        import requests

        url = f"{API_ROOT}/{self.model}:generateContent?key={self._api_key}"
        resp = requests.post(
            url,
            json=payload,
            timeout=self.request_timeout_s,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code in RETRYABLE_STATUS:
            raise TransientAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def _call_item(self, item_id: str, system: str, user: str) -> tuple[str, int, int]:
        """Return (raw_text, input_tokens, output_tokens) with retries.

        On exhausted retries returns an ``<API_ERROR:...>`` sentinel text that
        the decoder maps to abstain, and logs the failure.
        """
        if self.dry_run:
            return "A", self._est_tokens(system + user), 1

        payload = self._payload(system, user)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = self._request_once(payload)
                return self._parse_response(item_id, data)
            except TransientAPIError as e:  # bounded exponential backoff + jitter
                last_err = e
                if attempt < self.max_retries:
                    delay = min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))
                    delay *= 0.5 + random.random()
                    logger.warning(
                        "%s item %s attempt %d/%d failed (%s); retrying in %.1fs",
                        self.logical_id, item_id, attempt + 1, self.max_retries, e, delay,
                    )
                    time.sleep(delay)
            except Exception as e:  # non-retryable transport/HTTP error
                last_err = e
                break
        self.failures.append(
            {"item_id": item_id, "kind": "api_error", "detail": str(last_err)[:300]}
        )
        return f"<API_ERROR:{str(last_err)[:120]}>", 0, 0

    def _parse_response(self, item_id: str, data: dict) -> tuple[str, int, int]:
        if "error" in data:
            err = data["error"]
            if int(err.get("code", 0)) in RETRYABLE_STATUS:
                raise TransientAPIError(json.dumps(err)[:200])
            raise RuntimeError(f"Gemini API error: {json.dumps(err)[:300]}")
        usage = data.get("usageMetadata", {})
        tin = int(usage.get("promptTokenCount", 0))
        tout = int(usage.get("candidatesTokenCount", 0))
        self.thought_tokens += int(usage.get("thoughtsTokenCount", 0))
        candidates = data.get("candidates") or []
        if not candidates:
            # Prompt-level block (e.g. safety); a refusal, not a transport error.
            reason = str(data.get("promptFeedback", {}).get("blockReason", "NO_CANDIDATES"))
            self.failures.append({"item_id": item_id, "kind": "refusal", "detail": reason})
            return f"<REFUSAL:{reason}>", tin, tout
        cand = candidates[0]
        finish = str(cand.get("finishReason", ""))
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if finish not in ("STOP", "MAX_TOKENS", ""):
            self.failures.append({"item_id": item_id, "kind": "refusal", "detail": finish})
            if not text:
                return f"<REFUSAL:{finish}>", tin, tout
        if finish == "MAX_TOKENS" and not text:
            self.failures.append({"item_id": item_id, "kind": "truncated", "detail": finish})
            return "<TRUNCATED>", tin, tout
        return text, tin, tout

    # ---- voting ----
    def _vote_one(self, item: CalibrationItem) -> tuple[JudgeVote, int, int]:
        swap = self._position_swap(item.item_id)
        rendered = self.prompt.render(item, swap)
        system, user = self._split_system_user(rendered.text)
        raw, tin, tout = self._call_item(item.item_id, system, user)
        # Sentinel raws (API error / refusal / truncation) are always abstains;
        # never let the letter extractor scrape an A/B out of an error message.
        if raw.startswith("<"):
            vote = -1
        else:
            vote = self.prompt.decode(raw, swap)
        jv = JudgeVote(
            judge_id=self.logical_id,
            item_id=item.item_id,
            vote=int(vote),
            raw_response=raw,
            position_swapped=swap,
        )
        return jv, tin, tout

    def vote_batch(self, items: Iterable[CalibrationItem]) -> list[JudgeVote]:
        import threading

        items = list(items)
        self.failures = []
        results: list[JudgeVote | None] = [None] * len(items)
        usage_lock = threading.Lock()

        def work(k: int) -> None:
            vote, tin, tout = self._vote_one(items[k])
            results[k] = vote
            with usage_lock:
                self.usage.calls += 1
                self.usage.input_tokens += tin
                self.usage.output_tokens += tout
                if vote.vote == -1:
                    self.usage.abstentions += 1

        if self.workers == 1 or self.dry_run:
            for k in range(len(items)):
                work(k)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(work, range(len(items))))
        return [v for v in results if v is not None]


class TransientAPIError(RuntimeError):
    """A retryable API failure (rate limit, transient 5xx, timeout)."""

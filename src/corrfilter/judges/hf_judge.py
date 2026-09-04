"""HuggingFace local-weights judge with optional 4-bit nf4 quantization.

Loads one judge at a time so the runner can iterate the five base models
without exceeding the 24 GB-per-card budget of a TITAN RTX. The judge's
chat template (when provided by the tokenizer) is used to wrap the rendered
prompt; otherwise the prompt is fed as raw text.
"""

from __future__ import annotations

import gc
import logging
from typing import Iterable

import torch

from corrfilter.data import CalibrationItem
from corrfilter.judges.base import Judge, JudgeSpec, JudgeVote
from corrfilter.judges.position import position_swap
from corrfilter.judges.prompts import PromptTemplate, get_prompt_template

logger = logging.getLogger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HFJudge(Judge):
    """A HuggingFace causal-LM judge wrapped behind the Judge contract."""

    def __init__(
        self,
        spec: JudgeSpec,
        prompt_template: PromptTemplate | None = None,
        batch_size: int = 4,
        max_new_tokens: int = 8,
        device: str = "cuda",
        position_seed: int = 20260601,
        shuffle_position: bool = True,
        max_length: int = 4096,
    ):
        super().__init__(spec)
        self.prompt = prompt_template or get_prompt_template(spec.prompt_style)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.device = device
        self.position_seed = position_seed
        self.shuffle_position = shuffle_position
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        trc = bool(getattr(self.spec, "trust_remote_code", False))
        kwargs: dict = {"trust_remote_code": trc}
        if self.spec.hf_revision:
            kwargs["revision"] = self.spec.hf_revision

        dtype = _DTYPE_MAP[self.spec.dtype]
        if self.spec.quantization == "nf4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = quant_config
        elif self.spec.quantization in ("none", None):
            kwargs["torch_dtype"] = dtype
        else:
            raise ValueError(f"unsupported quantization {self.spec.quantization!r}")

        logger.info(
            "loading judge %s (%s, trust_remote_code=%s)",
            self.spec.logical_id,
            self.spec.hf_model,
            trc,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.spec.hf_model, trust_remote_code=trc
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"  # left-pad so generation aligns to prompt tail.

        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_model, device_map="auto", **kwargs
        )

        adapter_path = getattr(self.spec, "adapter_path", None)
        if adapter_path:
            from peft import PeftModel

            logger.info("attaching LoRA adapter %s to %s", adapter_path, self.spec.hf_model)
            self._model = PeftModel.from_pretrained(self._model, adapter_path)

        self._model.eval()

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def _render_chat(self, raw_prompt_text: str) -> str:
        """If the tokenizer ships a chat template, use it; else return the raw text."""
        assert self._tokenizer is not None
        if not getattr(self._tokenizer, "chat_template", None):
            return raw_prompt_text
        # Heuristic split of our rendered prompt format <|system|>..<|user|>..<|assistant|>.
        system_part = raw_prompt_text.split("<|user|>", 1)[0].replace("<|system|>", "").strip()
        rest = raw_prompt_text.split("<|user|>", 1)[1]
        user_part = rest.split("<|assistant|>", 1)[0].strip()
        messages = []
        if system_part:
            messages.append({"role": "system", "content": system_part})
        messages.append({"role": "user", "content": user_part})
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat template failed for %s: %s; falling back to raw", self.spec.logical_id, exc)
            return raw_prompt_text

    @torch.inference_mode()
    def vote_batch(self, items: Iterable[CalibrationItem]) -> list[JudgeVote]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"judge {self.spec.logical_id} not loaded")

        items = list(items)
        votes: list[JudgeVote] = []

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            swaps = [self._position_swap(it.item_id) for it in batch]
            rendered = [self.prompt.render(it, swap) for it, swap in zip(batch, swaps)]
            chat_texts = [self._render_chat(rp.text) for rp in rendered]

            enc = self._tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._model.device)

            out = self._model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            gen_only = out[:, enc["input_ids"].shape[1] :]
            decoded = self._tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for it, swap, raw in zip(batch, swaps, decoded):
                vote = self.prompt.decode(raw, swap)
                votes.append(
                    JudgeVote(
                        judge_id=self.spec.logical_id,
                        item_id=it.item_id,
                        vote=int(vote),
                        raw_response=raw,
                        position_swapped=swap,
                    )
                )

        return votes

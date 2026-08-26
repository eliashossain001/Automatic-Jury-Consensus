"""GRPO-train ONE pairwise-preference judge into a LoRA adapter (real RL, no simulation).

Dr. Lim asks for *original* GRPO-trained LLM judges. This script takes one base
model (already cached under /shared/models/huggingface) and fine-tunes it with
GRPO on a preference-label reward:

    reward = 1.0  if the judge's A/B verdict selects the gold (chosen) response
    reward = 0.0  otherwise

The judging prompt is the SAME pairwise format the paper's base bank uses
(``corrfilter.judges.prompts.PairwisePrompt``) so the GRPO judge and the base
judge are directly comparable. Position (which response is shown as A vs B) is
randomised per example so the reward rewards preference, not position.

Only the LoRA adapter is written to disk (~a few hundred MB): the 7-9B base
weights stay shared/cached. That is what keeps this runnable on the box's
23 GB of free disk.

Usage (one model, one seed, pinned to one GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_grpo_judge.py \
        --model Qwen/Qwen2.5-7B-Instruct --base-id qwen-2.5-7b --family qwen \
        --scale 7B --seed 0 --max-steps 300 \
        --output results/grpo_judges/adapters/qwen-2.5-7b_seed0
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Base models are pre-cached; never hit the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import numpy as np
import torch
import yaml
from datasets import Dataset

from corrfilter.data import load_calibration_set
from corrfilter.judges.prompts import PairwisePrompt, _extract_first_letter

ROOT = Path(__file__).resolve().parents[2]


PAIRWISE = PairwisePrompt()


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + " …[truncated]"


def build_dataset(manifest_path: str, seed: int, limit: int | None,
                  prompt_chars: int, resp_chars: int) -> Dataset:
    """Conversational GRPO dataset: {prompt: [messages], gold_letter: 'A'|'B'}."""
    items = load_calibration_set(manifest_path)
    if limit is not None:
        items = items[:limit]
    rng = random.Random(seed)
    rows = []
    for it in items:
        swap = rng.random() < 0.5  # randomise position per example
        chosen = _truncate(it.chosen, resp_chars)
        rejected = _truncate(it.rejected, resp_chars)
        a, b = (rejected, chosen) if swap else (chosen, rejected)
        gold_letter = "B" if swap else "A"  # where the gold (chosen) response landed
        user = PAIRWISE.USER_TEMPLATE.format(
            prompt=_truncate(it.prompt, prompt_chars), response_a=a, response_b=b
        )
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": PAIRWISE.SYSTEM},
                    {"role": "user", "content": user},
                ],
                "gold_letter": gold_letter,
                "item_id": it.item_id,
            }
        )
    return Dataset.from_list(rows)


def _completion_text(completion) -> str:
    """GRPO passes conversational completions as [{'role':...,'content':...}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)
    return str(completion)


def make_reward_fn():
    """reward = 1.0 iff the parsed A/B verdict equals the gold letter."""

    def preference_reward(completions, gold_letter, **kwargs):
        rewards = []
        for comp, gold in zip(completions, gold_letter):
            letter = _extract_first_letter(_completion_text(comp))
            rewards.append(1.0 if letter == gold else 0.0)
        return rewards

    preference_reward.__name__ = "preference_reward"
    return preference_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (cached)")
    ap.add_argument("--base-id", required=True, help="short id, e.g. qwen-2.5-7b")
    ap.add_argument("--family", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help="adapter output dir")
    ap.add_argument("--calibration", default="configs/calibration.yaml")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-length", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap #training prompts")
    ap.add_argument("--prompt-chars", type=int, default=2000)
    ap.add_argument("--resp-chars", type=int, default=1200)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    cal_cfg = yaml.safe_load(Path(args.calibration).read_text())
    manifest_path = cal_cfg["output"]["manifest_path"]
    ds = build_dataset(manifest_path, args.seed, args.limit, args.prompt_chars, args.resp_chars)
    print(f"[28] training prompts: {len(ds)}  (model={args.model}, seed={args.seed})")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GRPOConfig(
        output_dir=str(out_dir / "_trainer"),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        scale_rewards=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(),
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )

    trainer.train()

    # Save ONLY the adapter (+ tokenizer) — keeps disk footprint small.
    trainer.model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # Record the mean training reward trajectory for the writeup.
    try:
        hist = [h for h in trainer.state.log_history if "reward" in h]
        meta = {
            "model": args.model, "base_id": args.base_id, "family": args.family,
            "scale": args.scale, "seed": args.seed, "max_steps": args.max_steps,
            "num_generations": args.num_generations, "lr": args.lr, "beta": args.beta,
            "n_train_prompts": len(ds),
            "reward_first": hist[0].get("reward") if hist else None,
            "reward_last": hist[-1].get("reward") if hist else None,
        }
        (out_dir / "grpo_train_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        print(f"[28] reward {meta['reward_first']} -> {meta['reward_last']}")
    except Exception as e:  # noqa: BLE001
        print("[28] meta save skipped:", e)

    print(f"[28] adapter saved -> {out_dir}")


if __name__ == "__main__":
    main()

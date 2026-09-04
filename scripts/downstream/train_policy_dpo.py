"""Bucket 3 — DPO-train a small policy model on a filtered preference file (SLURM/GPU).

Reads a {prompt,chosen,rejected} jsonl produced by downstream/build_filtered_dpo_data.py and
DPO-trains a small policy with LoRA. bf16 by default (H100/A100); pass --fp16 for
Turing (TITAN). Saves only the LoRA adapter + a train_meta.json. Deterministic seed.

Usage (the cluster):
  python scripts/downstream/train_policy_dpo.py --data <dpo_majority.jsonl> \
      --base-model Qwen/Qwen2.5-1.5B-Instruct --output <out/models/majority> \
      --epochs 2 --beta 0.1 --lr 5e-6 --seed 20260707 --bf16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("json", data_files=args.data, split="train")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    cfg = DPOConfig(
        output_dir=str(out / "_trainer"), num_train_epochs=args.epochs, beta=args.beta,
        learning_rate=args.lr, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, max_length=args.max_length,
        warmup_ratio=0.1, lr_scheduler_type="cosine",
        bf16=args.bf16, fp16=args.fp16, logging_steps=10, save_strategy="no",
        report_to=[], seed=args.seed, gradient_checkpointing=True)

    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    hist = [h for h in trainer.state.log_history if "loss" in h]
    meta = {"data": args.data, "base_model": args.base_model, "n_train": len(ds),
            "epochs": args.epochs, "beta": args.beta, "lr": args.lr, "seed": args.seed,
            "final_loss": hist[-1].get("loss") if hist else None}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train_policy_dpo] saved adapter -> {out} (n_train={len(ds)}, final_loss={meta['final_loss']})")


if __name__ == "__main__":
    main()

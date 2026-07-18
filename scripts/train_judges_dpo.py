"""Bucket 2 — DPO-train ONE pairwise-preference judge into a LoRA adapter (Newton/GPU).

Preference-optimization alternative to GRPO (scripts/28). For each calibration item we
build a DPO pair on the SAME pairwise judging prompt used elsewhere:
  chosen   = the correct verdict letter  (A or B, per the position swap)
  rejected = the incorrect verdict letter
so DPO trains the judge to prefer emitting the correct preference. Position is randomised
per item. Saves only the LoRA adapter. bf16 by default (H100/A100); --fp16 for Turing.

Target claim: "preference optimization (not only GRPO) can increase inter-judge dependence."

Usage (Newton):
  python scripts/train_judges_dpo.py --model Qwen/Qwen2.5-7B-Instruct \
      --base-id qwen-2.5-7b --family qwen --scale 7B --seed 0 \
      --output results/strengthening_phase/dpo_judges/adapters/qwen-2.5-7b_seed0 --bf16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges.prompts import PairwisePrompt  # noqa: E402

PAIRWISE = PairwisePrompt()


def _trunc(t, n):
    t = t or ""
    return t if len(t) <= n else t[:n] + " …[truncated]"


def build_pairs(manifest, seed, limit, prompt_chars, resp_chars):
    from datasets import Dataset
    items = load_calibration_set(manifest)
    if limit:
        items = items[:limit]
    rng = random.Random(seed)
    rows = []
    for it in items:
        swap = rng.random() < 0.5
        a, b = (_trunc(it.rejected, resp_chars), _trunc(it.chosen, resp_chars)) if swap \
            else (_trunc(it.chosen, resp_chars), _trunc(it.rejected, resp_chars))
        correct = "B" if swap else "A"
        wrong = "A" if swap else "B"
        user = PAIRWISE.USER_TEMPLATE.format(prompt=_trunc(it.prompt, prompt_chars), response_a=a, response_b=b)
        rows.append({
            "prompt": [{"role": "system", "content": PAIRWISE.SYSTEM}, {"role": "user", "content": user}],
            "chosen": [{"role": "assistant", "content": correct}],
            "rejected": [{"role": "assistant", "content": wrong}]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-id", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--calibration", default="configs/calibration.yaml")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-chars", type=int, default=800)
    ap.add_argument("--resp-chars", type=int, default=600)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load-4bit", action="store_true",
                    help="QLoRA: load base in 4-bit nf4 (fits a 7B on a 24GB TITAN)")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import torch
    import yaml
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = yaml.safe_load(Path(args.calibration).read_text())["output"]["manifest_path"]
    ds = build_pairs(manifest, args.seed, args.limit, args.prompt_chars, args.resp_chars)
    print(f"[train_judges_dpo] {args.model} seed={args.seed} pairs={len(ds)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if args.load_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=args.trust_remote_code)
        model.config.use_cache = False
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg = DPOConfig(output_dir=str(out / "_trainer"), num_train_epochs=args.epochs, beta=args.beta,
                    learning_rate=args.lr, per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.grad_accum, max_length=args.max_length,
                    warmup_ratio=0.1, lr_scheduler_type="cosine", bf16=args.bf16, fp16=args.fp16,
                    logging_steps=10, save_strategy="no", report_to=[], seed=args.seed,
                    gradient_checkpointing=True)
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    hist = [h for h in trainer.state.log_history if "loss" in h]
    (out / "train_meta.json").write_text(json.dumps({
        "method": "dpo", "model": args.model, "base_id": args.base_id, "family": args.family,
        "scale": args.scale, "seed": args.seed, "n_pairs": len(ds), "beta": args.beta,
        "epochs": args.epochs, "final_loss": hist[-1].get("loss") if hist else None}, indent=2))
    print(f"[train_judges_dpo] saved adapter -> {out}")


if __name__ == "__main__":
    main()

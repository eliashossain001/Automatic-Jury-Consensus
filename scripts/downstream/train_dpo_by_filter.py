"""Phase 1, step 2 — train one DPO policy per filter method, identical settings.

For every ``data/<method>.jsonl`` produced by ``prepare_dpo_datasets.py`` we fine-tune
the SAME base model (Qwen2.5-0.5B-Instruct) with the SAME LoRA config, learning rate,
epochs, batch size, beta, and seed. The only thing that differs between runs is the
filtered preference set. Adapters are written to ``models/<method>/``.

Datasets with identical content (e.g. the router collapsing onto supermajority) are
trained once and reused, logged in ``results/train_log.json`` — no silent skips.

Usage:
    python scripts/downstream/train_dpo_by_filter.py            # all methods
    python scripts/downstream/train_dpo_by_filter.py --only corrfilter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time

from corrfilter import downstream as C


def dataset_fingerprint(records: list[dict]) -> str:
    """Content hash over (item_id, chosen, rejected) so duplicate sets reuse a model."""
    key = sorted(f"{r['item_id']}|{r['stated_label']}" for r in records)
    return hashlib.sha1("\n".join(key).encode()).hexdigest()[:12]


def format_for_dpo(records: list[dict], tokenizer):
    """Apply the chat template: prompt -> user turn w/ generation prompt; responses raw."""
    rows = {"prompt": [], "chosen": [], "rejected": []}
    for r in records:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        rows["prompt"].append(prompt)
        rows["chosen"].append(r["chosen"])
        rows["rejected"].append(r["rejected"])
    return rows


def train_one(method: str, records: list[dict], cfg: dict, model_dir):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    t = cfg["train"]
    C.seed_everything(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(t["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = Dataset.from_dict(format_for_dpo(records, tok))
    dtype = torch.bfloat16 if t["bf16"] else (torch.float16 if t.get("fp16") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(t["base_model"], dtype=dtype)
    lora = LoraConfig(
        r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    ) if t["use_lora"] else None

    args = DPOConfig(
        output_dir=str(model_dir / "_trainer"),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        num_train_epochs=t["num_train_epochs"],
        beta=t["beta"],
        max_length=t["max_length"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        bf16=t["bf16"],
        fp16=bool(t.get("fp16")),
        logging_steps=t["logging_steps"],
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    t0 = time.time()
    trainer.train()
    dur = time.time() - t0
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    shutil.rmtree(model_dir / "_trainer", ignore_errors=True)
    final_loss = trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None
    del trainer, model
    torch.cuda.empty_cache()
    return {"train_seconds": round(dur, 1), "final_train_loss": final_loss}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None, help="train a single method")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    C.set_run_dir(cfg.get("run_dir", "."))
    C.ensure_dirs(C.MODELS_DIR, C.RESULTS_DIR)

    methods = [args.only] if args.only else list(cfg["filters"])
    log: dict[str, dict] = {}
    seen: dict[str, str] = {}   # fingerprint -> method already trained

    for method in methods:
        path = C.DATA_DIR / f"{method}.jsonl"
        if not path.exists():
            print(f"[skip] {method}: no dataset at {path}")
            continue
        records = C.read_jsonl(path)
        fp = dataset_fingerprint(records)
        model_dir = C.MODELS_DIR / method

        if fp in seen:
            src = seen[fp]
            print(f"[reuse] {method}: identical dataset to '{src}' (fp={fp}); copying adapter")
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.copytree(C.MODELS_DIR / src, model_dir)
            log[method] = {**log[src], "reused_from": src, "fingerprint": fp}
            continue

        print(f"\n=== training {method} (n={len(records)}, fp={fp}) ===")
        info = train_one(method, records, cfg, model_dir)
        info.update({"fingerprint": fp, "train_size": len(records)})
        log[method] = info
        seen[fp] = method
        print(f"  done in {info['train_seconds']}s, final_loss={info['final_train_loss']}")

    (C.RESULTS_DIR / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {C.RESULTS_DIR / 'train_log.json'}")


if __name__ == "__main__":
    main()

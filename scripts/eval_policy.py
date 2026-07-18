"""Bucket 3 — evaluate a DPO-trained policy on held-out preference pairs (Newton/GPU).

Reward accuracy = fraction of held-out (prompt, chosen, rejected) pairs where the
policy assigns higher length-normalized log-probability to `chosen`. Also reports the
mean reward margin. Works on RewardBench (default) or any jsonl/csv/parquet manifest
with prompt/chosen/rejected (or response_a/response_b/true_label). Emits a per-item
CSV so downstream can compute paired bootstrap CIs across the two policies.

Usage (Newton):
  python scripts/eval_policy.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter <out/models/corrfilter> --manifest <rewardbench_eval.parquet> \
      --n 400 --out <out/results/eval_corrfilter.csv> --tag corrfilter --bf16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _load_pairs(manifest, n, seed):
    import numpy as np
    import pandas as pd
    p = Path(manifest)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".jsonl", ".json"):
        df = pd.read_json(p, lines=(p.suffix == ".jsonl"))
    else:
        df = pd.read_csv(p)
    if not {"chosen", "rejected"} <= set(df.columns) and {"response_a", "response_b", "true_label"} <= set(df.columns):
        a = df["true_label"].astype(str).str.upper() == "A"
        df["chosen"] = np.where(a, df["response_a"], df["response_b"])
        df["rejected"] = np.where(a, df["response_b"], df["response_a"])
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-length", type=int, default=640)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = _load_pairs(args.manifest, args.n, args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=args.trust_remote_code)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    @torch.inference_mode()
    def logp(prompt, completion):
        # length-normalized log-prob of `completion` given `prompt`
        p_ids = tok(prompt, truncation=True, max_length=args.max_length, return_tensors="pt").input_ids
        full = tok(prompt + "\n" + completion, truncation=True, max_length=args.max_length, return_tensors="pt").input_ids.to(model.device)
        out = model(full)
        logits = out.logits[:, :-1].log_softmax(-1)
        tgt = full[:, 1:]
        tok_lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        start = max(p_ids.shape[1] - 1, 0)
        comp_lp = tok_lp[start:]
        return float(comp_lp.mean().item()) if comp_lp.numel() else float("nan")

    rows = []
    for _, r in df.iterrows():
        lc, lr = logp(str(r["prompt"]), str(r["chosen"])), logp(str(r["prompt"]), str(r["rejected"]))
        rows.append({"item_id": str(r.get("item_id", _)), "logp_chosen": lc, "logp_rejected": lr,
                     "correct": int(lc > lr), "margin": lc - lr})
    res = pd.DataFrame(rows)
    acc = float(res["correct"].mean())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    summ = {"tag": args.tag, "n": len(res), "reward_accuracy": round(acc, 4),
            "mean_margin": round(float(res["margin"].mean()), 4)}
    pd.DataFrame([summ]).to_csv(str(out).replace(".csv", "_summary.csv"), index=False)
    print(f"[eval_policy] {args.tag}: reward_accuracy={acc:.4f} (n={len(res)}) -> {out}")


if __name__ == "__main__":
    main()

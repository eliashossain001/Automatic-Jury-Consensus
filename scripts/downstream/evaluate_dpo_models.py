"""Phase 1, step 3 — evaluate each trained DPO policy.

Two held-out metrics, both computed via the DPO *implicit reward*
    r(y|x) = log p_policy(y|x) - log p_ref(y|x)      (summed over response tokens)
so a pair is scored "correct" when r(chosen) > r(rejected):

  1. clean_pref_acc  — the held-out CLEAN UltraFeedback preferences (true labels),
     disjoint from any training set, identical across methods.
  2. rewardbench_acc — a fixed random subset of RewardBench v2 (cached locally),
     a generic out-of-distribution preference benchmark.

The reference log-probs depend only on the base model, so they are computed ONCE
and reused for every policy. Results -> results/eval_results.csv.

Usage: python scripts/downstream/evaluate_dpo_models.py
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from corrfilter import downstream as C


def load_rewardbench(cfg) -> list[dict]:
    from datasets import Dataset

    e = cfg["eval"]
    files = sorted(glob.glob(str(C.resolve(e["rewardbench_arrow_glob"])), recursive=True))
    if not files:
        print("[warn] no RewardBench arrow found; skipping that metric")
        return []
    ds = Dataset.from_file(files[0])
    rng = np.random.default_rng(e["rewardbench_seed"])
    idx = rng.choice(len(ds), size=min(e["rewardbench_n"], len(ds)), replace=False)
    out = []
    for i in idx:
        r = ds[int(i)]
        chosen = r["chosen"][0] if isinstance(r["chosen"], list) else r["chosen"]
        rejected = r["rejected"][0] if isinstance(r["rejected"], list) else r["rejected"]
        out.append({"prompt": r["prompt"], "chosen": chosen, "rejected": rejected,
                    "subset": r.get("subset", "")})
    return out


def score_pairs(model, tok, pairs, max_tokens, device):
    """Return (lp_chosen, lp_rejected): summed response-token log-probs under model."""
    import torch

    def seq_logprob(prompt, response):
        p_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        p_ids = tok(p_text, return_tensors="pt", add_special_tokens=False).input_ids
        r_ids = tok(response, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_tokens]
        input_ids = torch.cat([p_ids, r_ids], dim=1).to(device)
        with torch.no_grad():
            logits = model(input_ids).logits.float()
        # predict token t from logits at t-1; response tokens start at len(p_ids).
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = input_ids[:, 1:]
        tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        start = p_ids.shape[1] - 1
        return float(tok_lp[start:].sum().item())

    lp_c, lp_r = [], []
    for pr in pairs:
        lp_c.append(seq_logprob(pr["prompt"], pr["chosen"]))
        lp_r.append(seq_logprob(pr["prompt"], pr["rejected"]))
    return np.array(lp_c), np.array(lp_r)


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    C.set_run_dir(cfg.get("run_dir", "."))
    C.seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = cfg["train"]["base_model"]
    max_tokens = cfg["eval"]["max_eval_tokens"]
    t = cfg["train"]
    dtype = torch.bfloat16 if t["bf16"] else (torch.float16 if t.get("fp16") else torch.float32)

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eval_clean = C.read_jsonl(C.DATA_DIR / "eval_clean.jsonl")
    rb = load_rewardbench(cfg)
    print(f"eval sets: clean={len(eval_clean)} rewardbench={len(rb)}")

    # Reference log-probs (base model) — computed once, reused for every policy.
    print("scoring reference (base) model ...")
    ref = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device).eval()
    ref_clean = score_pairs(ref, tok, eval_clean, max_tokens, device)
    ref_rb = score_pairs(ref, tok, rb, max_tokens, device) if rb else (None, None)
    del ref
    torch.cuda.empty_cache()

    methods = [m for m in cfg["filters"] if (C.MODELS_DIR / m).exists()]
    if not methods:
        print("[error] no trained models found; run train_dpo_by_filter.py first")
        return

    def acc(lp_c_pol, lp_r_pol, ref_pair):
        ref_c, ref_r = ref_pair
        margin = (lp_c_pol - ref_c) - (lp_r_pol - ref_r)
        return float((margin > 0).mean()), float(margin.mean())

    rows = []
    for method in methods:
        print(f"scoring policy: {method}")
        policy = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        policy = PeftModel.from_pretrained(policy, str(C.MODELS_DIR / method)).to(device).eval()
        pc, pr_ = score_pairs(policy, tok, eval_clean, max_tokens, device)
        clean_acc, clean_margin = acc(pc, pr_, ref_clean)
        row = {"filter_method": method, "clean_pref_acc": round(clean_acc, 4),
               "clean_reward_margin": round(clean_margin, 4)}
        if rb:
            rc, rr = score_pairs(policy, tok, rb, max_tokens, device)
            rb_acc, rb_margin = acc(rc, rr, ref_rb)
            row.update({"rewardbench_acc": round(rb_acc, 4),
                        "rewardbench_reward_margin": round(rb_margin, 4)})
        rows.append(row)
        print(f"  {method}: clean_acc={row['clean_pref_acc']}"
              + (f"  rb_acc={row.get('rewardbench_acc')}" if rb else ""))
        del policy
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "eval_results.csv", index=False)
    print(f"\nwrote {C.RESULTS_DIR / 'eval_results.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

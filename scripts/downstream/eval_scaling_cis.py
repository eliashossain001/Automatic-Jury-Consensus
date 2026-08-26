#!/usr/bin/env python
"""Per-item implicit-reward margins + paired bootstrap CIs for the scaling study.

Re-scores every trained contamination-scaling policy on the held-out clean set and
the RewardBench subset, saving PER-ITEM margins so that paired (same-item)
bootstrap CIs can be computed for the key contrasts: each contamination level (and
filtered arm) versus the clean-trained (contam_00) policy, on margin and accuracy.

Reuses evaluate_dpo_models.score_pairs / load_rewardbench verbatim. GPU, ~minutes.
Usage:
  python scripts/downstream/eval_scaling_cis.py \
      [--config experiments/downstream_dpo_validation/config_scaling.yaml]
Outputs -> contamination_scaling/results/{per_item_margins.parquet, scaling_cis.csv}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_dpo_models import load_rewardbench, score_pairs

from corrfilter import downstream as C

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models


N_BOOT = 2000
BOOT_SEED = 20260706


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(EXP_DIR / "config_scaling.yaml"))
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    C.set_run_dir(cfg.get("run_dir", "contamination_scaling"))
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

    ref = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device).eval()
    refc = score_pairs(ref, tok, eval_clean, max_tokens, device)
    refr = score_pairs(ref, tok, rb, max_tokens, device) if rb else None
    del ref
    torch.cuda.empty_cache()

    frames = []
    for m in [x for x in cfg["filters"] if (C.MODELS_DIR / x).exists()]:
        print("scoring", m)
        pol = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        pol = PeftModel.from_pretrained(pol, str(C.MODELS_DIR / m)).to(device).eval()
        for split, pairs, refp in (("clean", eval_clean, refc), ("rewardbench", rb, refr)):
            if not pairs:
                continue
            lc, lr = score_pairs(pol, tok, pairs, max_tokens, device)
            margin = (lc - refp[0]) - (lr - refp[1])
            frames.append(pd.DataFrame({"method": m, "split": split,
                                        "item": np.arange(len(margin)), "margin": margin}))
        del pol
        torch.cuda.empty_cache()
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(C.RESULTS_DIR / "per_item_margins.parquet", index=False)

    # paired bootstrap vs contam_00, per split
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for split in df.split.unique():
        d = df[df.split == split].pivot(index="item", columns="method", values="margin")
        base_m = d["contam_00"].to_numpy()
        for m in [c for c in d.columns if c != "contam_00"]:
            x = d[m].to_numpy()
            dm = x - base_m
            da = (x > 0).astype(float) - (base_m > 0).astype(float)
            n = len(dm)
            bm, ba = [], []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, n)
                bm.append(dm[idx].mean())
                ba.append(da[idx].mean())
            rows.append({
                "split": split, "method": m,
                "margin": round(float(x.mean()), 4),
                "margin_minus_clean": round(float(dm.mean()), 4),
                "margin_ci": (round(float(np.quantile(bm, .025)), 4), round(float(np.quantile(bm, .975)), 4)),
                "acc": round(float((x > 0).mean()), 4),
                "acc_minus_clean_pts": round(100 * float(da.mean()), 2),
                "acc_ci_pts": (round(100 * float(np.quantile(ba, .025)), 2), round(100 * float(np.quantile(ba, .975)), 2)),
                "margin_sig": bool(np.quantile(bm, .975) < 0 or np.quantile(bm, .025) > 0),
            })
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "scaling_cis.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

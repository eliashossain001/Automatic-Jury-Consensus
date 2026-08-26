"""Option 1 — targeted position-bias evaluation probe (no retraining).

Question: does the naive-vs-bias_cluster difference appear specifically on the axis
the position attack targets, rather than on random clean preferences?

We score every held-out CLEAN eval item by the SAME position-bias mass that drives the
poison selection (scripts/attacks/position_aligned_poisoning.py):  s_i = sum_j w_j * (1 - V[i,j]) * M[i,j], where w_j is
the H1 position-bias gap and V=1 iff judge j picked the truly-better response. High s_i
= the position-biased subgroup co-fails on this item = the attacked/vulnerable axis. We
also flag leave-cluster-out (LCO) items where dropping the H1 position cluster flips the
clean majority. Then we evaluate the existing adapters (naive, bias_cluster, corrfilter,
oracle) on the targeted slice vs the easy slice vs full eval, using the DPO implicit
reward margin, and run a paired (bias_cluster - naive) bootstrap on the targeted slice.

Outputs: position_aligned_poisoning/targeted_eval/{targeted_eval.json,
TARGETED_REPORT.md}.

Usage: python position_aligned_poisoning/targeted_eval_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corrfilter import downstream as C
from corrfilter.cfi.consensus import ABSTAIN
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "position_aligned_poisoning"                                    # this experiment's outputs


BETA = 0.1
H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}


def build_VM(cache, logical_ids, item_ids):
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), np.int8); M = np.zeros((n, m), np.int8)
    idx = {it: i for i, it in enumerate(item_ids)}
    for j, lid in enumerate(logical_ids):
        dfj = cache.load(lid)
        if dfj is None:
            continue
        for r in dfj.itertuples(index=False):
            i = idx.get(str(r.item_id))
            if i is None:
                continue
            v = int(r.vote)
            if v == ABSTAIN:
                continue
            V[i, j] = v; M[i, j] = 1
    return V, M


def seq_logprobs(model, tok, pairs, max_tokens, device):
    import torch

    def lp(prompt, response):
        p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True)
        pid = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
        rid = tok(response, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_tokens]
        ids = torch.cat([pid, rid], 1).to(device)
        with torch.no_grad():
            lg = model(ids).logits.float()
        logp = torch.log_softmax(lg[:, :-1], -1)
        tl = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        return float(tl[pid.shape[1] - 1:].sum().item())

    c = np.array([lp(p["prompt"], p["chosen"]) for p in pairs])
    r = np.array([lp(p["prompt"], p["rejected"]) for p in pairs])
    return c, r


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    mu = x[idx].mean(1)
    return float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))


def slice_metrics(margin, mask):
    mg = margin[mask]
    if len(mg) == 0:
        return {}
    return {"n": int(mask.sum()),
            "win_rate": round(float((mg > 0).mean()), 4),
            "mean_margin": round(float(mg.mean()), 4),
            "median_margin": round(float(np.median(mg)), 4)}


def main() -> None:
    import argparse

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="position_aligned_poisoning/pos10",
                    help="run_dir holding data/ and models/ to probe")
    ap.add_argument("--out", default=None, help="output dir (default <run_dir>/targeted_eval)")
    args = ap.parse_args()
    cfg = C.load_config()
    C.set_run_dir(args.run_dir)
    OUT = Path(args.out) if args.out else (EXP_DIR / args.run_dir / "targeted_eval")
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = cfg["train"]["base_model"]
    dtype = torch.float16
    maxtok = cfg["eval"]["max_eval_tokens"]

    eval_clean = C.read_jsonl(C.DATA_DIR / "eval_clean.jsonl")
    item_ids = [r["item_id"] for r in eval_clean]
    bank = load_bank_config(str(C.resolve(cfg["data"]["bank"])))
    logical_ids = [s.logical_id for s in bank.specs]
    cache = VoteCache(C.resolve(cfg["data"]["votes_dir"]))
    V, M = build_VM(cache, logical_ids, item_ids)

    # position-bias mass (attacked axis) + leave-cluster-out flip on clean votes
    w = np.array([H1_POSITION_GAP.get(l, 0.0) / 100.0 for l in logical_ids])
    score = ((1 - V) * M * w[None, :]).sum(1)
    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids])
    cb = np.zeros(len(logical_ids), bool); cb[np.argsort(-gaps)[:5]] = True
    Mb = M.astype(bool)
    # clean majority (V=1=correct) over all vs over non-cluster judges
    def frac_correct(sel):
        num = (V * Mb * sel[None, :]).sum(1); den = (Mb * sel[None, :]).sum(1)
        return np.divide(num, den, out=np.full(len(V), 0.5), where=den > 0)
    all_corr = frac_correct(np.ones(len(logical_ids), bool))
    noncl_corr = frac_correct(~cb)
    lco_flip = (all_corr < 0.5) != (noncl_corr < 0.5)     # cluster presence flips majority

    # targeted slice = top tertile by position-bias mass; easy = bottom tertile
    t_hi, t_lo = np.quantile(score, [2 / 3, 1 / 3])
    targeted = score >= t_hi
    easy = score <= t_lo
    full = np.ones(len(score), bool)
    print(f"slices: full={full.sum()} targeted(top-third)={targeted.sum()} "
          f"easy(bottom-third)={easy.sum()} lco_flip={int(lco_flip.sum())} "
          f"score range [{score.min():.2f},{score.max():.2f}] cut@{t_hi:.2f}")

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ref = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device).eval()
    rc, rr = seq_logprobs(ref, tok, eval_clean, maxtok, device)
    ref_dlp = rc - rr
    del ref; torch.cuda.empty_cache()

    methods = [m for m in ["naive_consensus", "bias_cluster", "corrfilter", "oracle"]
               if (C.MODELS_DIR / m).exists()]
    margins = {}
    report = {"slice_sizes": {"full": int(full.sum()), "targeted_top_third": int(targeted.sum()),
                              "easy_bottom_third": int(easy.sum()), "lco_flip": int(lco_flip.sum())},
              "by_method": {}}
    for m in methods:
        pol = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        pol = PeftModel.from_pretrained(pol, str(C.MODELS_DIR / m)).to(device).eval()
        pc, pr = seq_logprobs(pol, tok, eval_clean, maxtok, device)
        mg = BETA * ((pc - pr) - ref_dlp)
        margins[m] = mg
        report["by_method"][m] = {
            "full": slice_metrics(mg, full),
            "targeted_top_third": slice_metrics(mg, targeted),
            "easy_bottom_third": slice_metrics(mg, easy),
            "lco_flip": slice_metrics(mg, lco_flip),
        }
        del pol; torch.cuda.empty_cache()

    # paired bias_cluster - naive on each slice
    report["paired_bc_minus_naive"] = {}
    if "bias_cluster" in margins and "naive_consensus" in margins:
        d = margins["bias_cluster"] - margins["naive_consensus"]
        for name, mask in [("full", full), ("targeted_top_third", targeted),
                           ("easy_bottom_third", easy), ("lco_flip", lco_flip)]:
            if mask.sum() < 5:
                continue
            dd = d[mask]; lo, hi = boot_ci(dd)
            report["paired_bc_minus_naive"][name] = {
                "n": int(mask.sum()), "mean_margin_diff": round(float(dd.mean()), 5),
                "boot95_ci": [round(lo, 5), round(hi, 5)],
                "win_rate_diff_pts": round(100 * float((margins["bias_cluster"][mask] > 0).mean()
                                                        - (margins["naive_consensus"][mask] > 0).mean()), 2),
                "detectable_at_95pct": bool(lo > 0 or hi < 0)}

    (OUT / "targeted_eval.json").write_text(json.dumps(report, indent=2))
    write_report(report, OUT)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'targeted_eval.json'} and TARGETED_REPORT.md")


def write_report(r, OUT):
    bm = r["by_method"]; pj = r["paired_bc_minus_naive"]
    L = ["# Targeted Position-Bias Evaluation Probe (pos10, no retraining)\n",
         "Held-out clean eval split by position-bias mass (the attacked axis). "
         "Targeted = top tertile (biased subgroup co-fails); easy = bottom tertile; "
         "lco_flip = items where dropping the H1 position cluster flips the clean majority.\n",
         f"Slice sizes: {r['slice_sizes']}\n",
         "## Win rate by method and slice\n",
         "| method | full | targeted | easy | lco_flip |", "|---|---|---|---|---|"]
    for m, d in bm.items():
        def wr(s): return d[s].get("win_rate", "-")
        L.append(f"| {m} | {wr('full')} | {wr('targeted_top_third')} | {wr('easy_bottom_third')} | {wr('lco_flip')} |")
    L += ["\n## Mean reward margin by method and slice\n",
          "| method | full | targeted | easy | lco_flip |", "|---|---|---|---|---|"]
    for m, d in bm.items():
        def mm(s): return d[s].get("mean_margin", "-")
        L.append(f"| {m} | {mm('full')} | {mm('targeted_top_third')} | {mm('easy_bottom_third')} | {mm('lco_flip')} |")
    L += ["\n## Paired (bias_cluster − naive), per slice\n",
          "| slice | n | mean margin diff | 95% CI | win-rate diff (pts) | detectable |",
          "|---|---|---|---|---|---|"]
    for s, v in pj.items():
        L.append(f"| {s} | {v['n']} | {v['mean_margin_diff']} | {v['boot95_ci']} | "
                 f"{v['win_rate_diff_pts']} | {'YES' if v['detectable_at_95pct'] else 'no'} |")
    tgt = pj.get("targeted_top_third", {})
    detect = tgt.get("detectable_at_95pct", False)
    L += ["\n## Answer\n",
          "**Does the bias_cluster advantage appear on the attacked position-bias axis?** "
          + ("**Yes** — on the targeted top-tertile slice the paired bias_cluster − naive margin "
             f"separates from zero (mean {tgt.get('mean_margin_diff')}, CI {tgt.get('boot95_ci')}). "
             "The earlier null was an evaluation mismatch: the random clean eval diluted the effect."
             if detect else
             f"**No** — even on the targeted slice (n={tgt.get('n')}) the paired margin difference "
             f"stays within the bootstrap CI ({tgt.get('boot95_ci')}; win-rate diff "
             f"{tgt.get('win_rate_diff_pts')} pts). The filtering label-quality gain does not translate "
             "into measurable model behavior even on the relevant failure axis."),
          "\n## Interpretation\n",
          ("Signal exists but the eval was mistargeted -> a position-axis metric is the right lever, "
           "and a capacity sweep may now be justified."
           if detect else
           "Combined with the diagnostic (datasets 82% identical, 37-pair net correction, paired margin "
           "CI around zero), the conclusion is that bias_cluster's filtering improvement does not produce "
           "a measurable downstream model difference at this scale even on the attacked axis. A larger "
           "model cannot exploit a difference that leaves no trace on the very items it should affect; "
           "amplifying the regime (no matched-retention dilution, or higher contamination) is the more "
           "informative next step than scaling."),
          ]
    (OUT / "TARGETED_REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()

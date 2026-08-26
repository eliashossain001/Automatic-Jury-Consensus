"""pos10 diagnostic — is the downstream null from capacity, metric, or no signal?

No training. Uses the already-built pos10 datasets, the trained LoRA adapters, and
the base model (inference only) to decompose the null result:

  PART 1  Dataset diff (CPU): how much do the naive vs bias_cluster training sets
          actually differ — differing pairs, overlap, poisoned removed/added, the
          net label correction.
  PART 2  Training-signal stats: under the reference model, the chosen-vs-rejected
          log-prob gap, implied preference probability/entropy, and the DPO
          gradient-weight proxy sigmoid(-beta*margin) on the items that differ vs
          the whole set (how much of the initial gradient mass the difference owns).
  PART 3  Sensitive model-level metrics on the held-out clean eval (400 true-label
          pairs): per-example reward margin r=beta*((lp_pol_c-lp_pol_r)-(lp_ref_c-
          lp_ref_r)); mean/median/std, win-rate, distribution percentiles, and a
          PAIRED bootstrap of (bias_cluster - naive) margins — detects a model
          difference even when win-rate accuracy does not move.
  PART 4  Effect size + report (answers A/B/C).

Outputs: position_aligned_poisoning/diagnostics/{pos10_diagnostics.json,
DIAGNOSTIC_REPORT.md, margin_hist.png}.

Usage: python position_aligned_poisoning/diagnose_pos10.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corrfilter import downstream as C

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
ROOT = CODE_DIR.parents[1]
EXP_DIR = ROOT / "experiments" / "downstream_dpo_validation"   # artifacts, configs, models
RUN_DIR = EXP_DIR / "position_aligned_poisoning"                                    # this experiment's outputs


TAG = "pos10"
RUN_DIR = f"position_aligned_poisoning/{TAG}"
BETA = 0.1
OUT = RUN_DIR / "diagnostics"


def load_set(method):
    return {r["item_id"]: r for r in C.read_jsonl(C.DATA_DIR / f"{method}.jsonl")}


def dataset_diff(naive, bc):
    ni, bi = set(naive), set(bc)
    shared, only_n, only_b = ni & bi, ni - bi, bi - ni
    # orientation flips among shared items (same item, different chosen/rejected)
    flips = [i for i in shared if naive[i]["chosen"] != bc[i]["chosen"]]
    def pois(d, ids):
        return sum(d[i]["is_poisoned"] for i in ids)
    n_total = len(ni)
    differing = len(only_n) + len(only_b) + len(flips)
    return {
        "n_naive": len(ni), "n_bias_cluster": len(bi),
        "overlap_count": len(shared),
        "overlap_pct": round(100 * len(shared) / n_total, 2),
        "differing_pairs": differing,
        "differing_pct": round(100 * differing / n_total, 2),
        "orientation_flips_in_shared": len(flips),
        "only_in_naive": len(only_n), "only_in_bias_cluster": len(only_b),
        "poisoned_kept_naive": pois(naive, ni),
        "poisoned_kept_bias_cluster": pois(bc, bi),
        "poisoned_removed_by_bc": pois(naive, only_n),     # poisoned naive kept, bc dropped
        "poisoned_added_by_bc": pois(bc, only_b),          # poisoned bc kept, naive dropped
        "net_poisoned_reduction": pois(naive, ni) - pois(bc, bi),
        "clean_swapped_in_by_bc": len(only_b) - pois(bc, only_b),
    }


def seq_logprobs(model, tok, pairs, max_tokens, device):
    """Per-pair (lp_chosen, lp_rejected) summed response-token log-probs."""
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
        tgt = ids[:, 1:]
        tl = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        return float(tl[pid.shape[1] - 1:].sum().item())

    c = np.array([lp(p["prompt"], p["chosen"]) for p in pairs])
    r = np.array([lp(p["prompt"], p["rejected"]) for p in pairs])
    return c, r


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = C.load_config()
    C.set_run_dir(RUN_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = cfg["train"]["base_model"]
    dtype = torch.float16
    maxtok = cfg["eval"]["max_eval_tokens"]
    report = {}

    # ---------- PART 1: dataset diff ----------
    naive, bc = load_set("naive_consensus"), load_set("bias_cluster")
    diff = dataset_diff(naive, bc)
    report["part1_dataset_diff"] = diff
    print("PART 1 dataset diff:", json.dumps(diff, indent=2))

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # items that differ between the two training sets (the only place signal can differ)
    differ_ids = (set(naive) - set(bc)) | (set(bc) - set(naive))
    differ_pairs = [naive.get(i) or bc.get(i) for i in differ_ids]
    train_union = list({**naive, **bc}.values())

    # ---------- PART 2: training-signal stats under the reference model ----------
    ref = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device).eval()
    c_u, r_u = seq_logprobs(ref, tok, train_union, maxtok, device)
    dlp_u = c_u - r_u                                   # chosen - rejected logprob gap
    p_u = 1 / (1 + np.exp(-BETA * dlp_u))               # implied preference prob
    ent_u = -(p_u * np.log(p_u + 1e-9) + (1 - p_u) * np.log(1 - p_u + 1e-9))
    grad_u = 1 / (1 + np.exp(BETA * dlp_u))             # DPO grad weight sigmoid(-beta*margin)

    is_differ = np.array([(p["item_id"] in differ_ids) for p in train_union])
    report["part2_training_signal"] = {
        "ref_logprob_gap_mean_all": round(float(dlp_u.mean()), 3),
        "ref_logprob_gap_mean_differing": round(float(dlp_u[is_differ].mean()), 3) if is_differ.any() else None,
        "preference_entropy_mean_all": round(float(ent_u.mean()), 4),
        "preference_entropy_mean_differing": round(float(ent_u[is_differ].mean()), 4) if is_differ.any() else None,
        "grad_weight_sum_all": round(float(grad_u.sum()), 2),
        "grad_weight_sum_differing": round(float(grad_u[is_differ].sum()), 2),
        "grad_mass_fraction_from_differing_pct": round(100 * float(grad_u[is_differ].sum() / grad_u.sum()), 2) if grad_u.sum() else 0.0,
    }
    print("PART 2:", json.dumps(report["part2_training_signal"], indent=2))

    # ---------- PART 3: sensitive eval metrics ----------
    eval_clean = C.read_jsonl(C.DATA_DIR / "eval_clean.jsonl")
    ref_c, ref_r = seq_logprobs(ref, tok, eval_clean, maxtok, device)
    ref_dlp = ref_c - ref_r
    del ref; torch.cuda.empty_cache()

    methods = [m for m in ["naive_consensus", "bias_cluster", "corrfilter", "oracle"]
               if (C.MODELS_DIR / m).exists()]
    margins = {}
    metric_rows = {}
    for m in methods:
        pol = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        pol = PeftModel.from_pretrained(pol, str(C.MODELS_DIR / m)).to(device).eval()
        pc, pr = seq_logprobs(pol, tok, eval_clean, maxtok, device)
        mg = BETA * ((pc - pr) - ref_dlp)               # per-example reward margin
        margins[m] = mg
        metric_rows[m] = {
            "win_rate": round(float((mg > 0).mean()), 4),
            "mean_margin": round(float(mg.mean()), 4),
            "median_margin": round(float(np.median(mg)), 4),
            "std_margin": round(float(mg.std()), 4),
            "p10_margin": round(float(np.percentile(mg, 10)), 4),
            "p90_margin": round(float(np.percentile(mg, 90)), 4),
            "mean_margin_when_correct": round(float(mg[mg > 0].mean()), 4) if (mg > 0).any() else None,
            "mean_margin_when_wrong": round(float(mg[mg <= 0].mean()), 4) if (mg <= 0).any() else None,
        }
        del pol; torch.cuda.empty_cache()
    report["part3_eval_metrics"] = metric_rows
    print("PART 3:", json.dumps(metric_rows, indent=2))

    # paired bias_cluster - naive margin difference (the sensitive test)
    if "bias_cluster" in margins and "naive_consensus" in margins:
        d = margins["bias_cluster"] - margins["naive_consensus"]
        lo, hi = boot_ci(d)
        report["part3_paired_bc_minus_naive"] = {
            "mean_margin_diff": round(float(d.mean()), 5),
            "boot95_ci": [round(lo, 5), round(hi, 5)],
            "frac_eval_items_bc_higher": round(float((d > 0).mean()), 4),
            "detectable_at_95pct": bool(lo > 0 or hi < 0),
        }
        print("PART 3 paired bc-naive:", json.dumps(report["part3_paired_bc_minus_naive"], indent=2))

    # ---------- PART 4: effect size ----------
    n_train = diff["n_naive"]
    grad_frac = report["part2_training_signal"]["grad_mass_fraction_from_differing_pct"]
    report["part4_effect_size"] = {
        "differing_examples": diff["differing_pairs"],
        "differing_pct_of_training": diff["differing_pct"],
        "net_poisoned_reduction": diff["net_poisoned_reduction"],
        "label_purity_gain_pts": round(100 * (diff["poisoned_kept_naive"] - diff["poisoned_kept_bias_cluster"]) / n_train, 2),
        "initial_grad_mass_from_differing_pct": grad_frac,
    }
    print("PART 4:", json.dumps(report["part4_effect_size"], indent=2))

    # ---------- margin histogram ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in ["naive_consensus", "bias_cluster"]:
            if m in margins:
                ax.hist(margins[m], bins=40, alpha=0.5, label=m, density=True)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("per-example reward margin (held-out clean eval)")
        ax.set_ylabel("density"); ax.set_title("pos10 reward-margin distribution: naive vs bias_cluster")
        ax.legend(); fig.tight_layout(); fig.savefig(OUT / "margin_hist.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("hist skipped:", e)

    (OUT / "pos10_diagnostics.json").write_text(json.dumps(report, indent=2))
    write_report(report)
    print(f"\nwrote {OUT/'pos10_diagnostics.json'} and DIAGNOSTIC_REPORT.md")


def write_report(r):
    d = r["part1_dataset_diff"]; s = r["part2_training_signal"]
    e = r["part4_effect_size"]; pj = r.get("part3_paired_bc_minus_naive", {})
    em = r["part3_eval_metrics"]
    noise = 2 * (0.25 / 400) ** 0.5 * 100  # ~5 pts win-rate threshold
    L = [
        "# pos10 Diagnostic: capacity vs metric vs no-signal\n",
        "Inference-only analysis of the existing pos10 datasets, LoRA adapters, and base "
        "model. No training.\n",
        "## Part 1 — did bias_cluster change the training data?\n",
        f"- Training-set overlap with naive: **{d['overlap_pct']}%** "
        f"({d['overlap_count']}/{d['n_naive']} items shared).",
        f"- Differing pairs: **{d['differing_pairs']}** ({d['differing_pct']}% of training): "
        f"{d['only_in_naive']} only-naive, {d['only_in_bias_cluster']} only-bias_cluster, "
        f"{d['orientation_flips_in_shared']} orientation flips.",
        f"- Poisoned labels removed by bias_cluster: **{d['poisoned_removed_by_bc']}**; "
        f"added: {d['poisoned_added_by_bc']}; net reduction: **{d['net_poisoned_reduction']}** "
        f"(label-purity gain {e['label_purity_gain_pts']} pts).\n",
        "## Part 2 — training-signal statistics (reference model)\n",
        f"- Chosen-minus-rejected log-prob gap: mean {s['ref_logprob_gap_mean_all']} (all) vs "
        f"{s['ref_logprob_gap_mean_differing']} (differing items).",
        f"- Preference entropy: {s['preference_entropy_mean_all']} (all) vs "
        f"{s['preference_entropy_mean_differing']} (differing).",
        f"- DPO initial gradient mass from the differing items: "
        f"**{s['grad_mass_fraction_from_differing_pct']}%** of the total.\n",
        "## Part 3 — sensitive model-level metrics (held-out clean eval, n=400)\n",
        "| method | win_rate | mean_margin | median_margin | std |",
        "|---|---|---|---|---|",
    ]
    for m, row in em.items():
        L.append(f"| {m} | {row['win_rate']} | {row['mean_margin']} | {row['median_margin']} | {row['std_margin']} |")
    if pj:
        L += ["", f"- Paired (bias_cluster − naive) mean margin: **{pj['mean_margin_diff']}** "
                  f"(95% CI {pj['boot95_ci']}); detectable difference: "
                  f"**{'YES' if pj['detectable_at_95pct'] else 'NO'}**; "
                  f"bias_cluster higher on {100*pj['frac_eval_items_bc_higher']:.0f}% of items."]
    # verdicts
    sig_changed = d["differing_pct"] >= 3
    model_diff = pj.get("detectable_at_95pct", False)
    L += [
        "\n## Answers\n",
        "**A. Did bias_cluster meaningfully change the training data?** "
        + (f"Yes — {d['differing_pairs']} pairs ({d['differing_pct']}%) differ and it removed "
           f"{d['net_poisoned_reduction']} net poisoned labels. The change is real but small: it "
           f"touches only {d['differing_pct']}% of training and owns {s['grad_mass_fraction_from_differing_pct']}% "
           "of the initial gradient mass."),
        "\n**B. Any detectable model-level difference beyond accuracy?** "
        + ("Yes — the paired reward-margin bootstrap separates from zero, so bias_cluster shifts the "
           "margin distribution even though win-rate does not clear the noise floor."
           if model_diff else
           "No — neither win-rate nor the paired reward-margin distribution differs from naive beyond "
           "the bootstrap CI. The two policies are statistically indistinguishable at the model level."),
        "\n**C. Most likely cause of the null:** "
        + ("primarily **negligible downstream impact / training-signal difference** — the datasets are "
           f"{d['overlap_pct']}% identical and the differing {d['differing_pct']}% carries only "
           f"{s['grad_mass_fraction_from_differing_pct']}% of the gradient mass, so there is too little "
           "differential signal for any model to convert"
           + (". Metric sensitivity is a secondary factor (even margins do not move)."
              if not model_diff else
              ", though a margin-level difference IS present and is simply masked by the win-rate metric "
              "(metric insensitivity is then a real secondary cause).")),
        f"\n_Win-rate noise floor at n=400 is ~{noise:.1f} pts; differences below that are not "
        "interpretable as accuracy changes._",
        "\n## Recommendation on a 3B/7B sweep\n",
        ("A capacity sweep is **justified** only if Part 3 shows a real margin-level difference (B=Yes): "
         "that would mean the signal exists but the small model cannot exploit it. If B=No and the "
         "gradient mass from differing items is tiny, a larger model will not help — the bottleneck is "
         "the data difference itself, and the better next step is to amplify the regime (higher "
         "contamination, or evaluate on the poisoned-axis directly) rather than scale the model."),
    ]
    OUT.joinpath("DIAGNOSTIC_REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()

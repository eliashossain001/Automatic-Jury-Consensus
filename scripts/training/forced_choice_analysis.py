#!/usr/bin/env python
"""P0-1 analysis: forced-choice base / GRPO / DPO dependence comparison.

CPU-only, deterministic, no inference. Consumes the forced-choice vote caches written by
scripts/training/forced_choice_bank.py plus the published generated caches (read-only) for the protocol-validity check.

Three outputs:
  1. truncation_strata.csv  - exploratory: cached votes stratified by prompt length (the
                              analysis that motivated PREREG Deviation 1)
  2. validation.csv         - forced-choice vs cached base votes: agreement, accuracy and
                              rho_bar deltas, split by prompt-length stratum and by
                              position-swap bit
  3. forced_choice.csv      - the primary comparison, with paired item-bootstrap CIs on
                              Delta rho_bar and Delta n_eff for every bank pair

All retention-matched machinery goes through src/corrfilter/evaluation.py (post-P0-0).
This script performs no filtering comparison; H1 is a dependence claim.

Usage: python scripts/training/forced_choice_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.analysis.dependence import cofailure, estimate_R, load_votes
from corrfilter.correlation.effective_size import (
    effective_eig_rank,
    effective_size,
    mean_off_diagonal,
)
from corrfilter.data import load_calibration_set
from corrfilter.evaluation import evaluable_and_correct
from corrfilter.judges.prompts import get_prompt_template
from corrfilter.voting import VoteCache, load_vote_matrix

ROOT = Path(__file__).resolve().parents[2]


MANIFEST = ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet"
FC = ROOT / "outputs/forced_choice/votes"
OUT = ROOT / "outputs/forced_choice"
SEED = 20260706
B = 2000
TRUNC_LIMIT = 1024          # the max_length the trained banks were originally run at

MODELS = ("qwen-2.5-7b", "mistral-7b-v0.3", "phi-3.5-mini")
STYLES = ("pairwise", "likert")
GEN_BANKS = {
    "base": ("configs/judge_bank.yaml", [f"{m}::{s}" for m in MODELS for s in STYLES]),
    "grpo": ("configs/grpo_judge_bank.yaml", None),
    "dpo": ("configs/dpo_judge.yaml", None),
}


# ---------------------------------------------------------------- prompt lengths
def prompt_token_lengths(items) -> dict[str, np.ndarray]:
    """Token length of the chat-templated prompt per style, using the Qwen tokenizer.

    All three base models are within a few percent of each other on these prompts; Qwen is
    used as the reference so the stratum boundary is one fixed set of items.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    out = {}
    for style in STYLES:
        tmpl = get_prompt_template(style)
        lens = []
        for it in items:
            raw = tmpl.render(it, False).text
            sysp = raw.split("<|user|>", 1)[0].replace("<|system|>", "").strip()
            usr = raw.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0].strip()
            s = tok.apply_chat_template(
                [{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
                tokenize=False, add_generation_prompt=True)
            lens.append(len(tok.encode(s, add_special_tokens=False)))
        out[style] = np.array(lens)
    return out


# ---------------------------------------------------------------- metric block
def bank_block(V, M, gold, label) -> dict:
    R, estimator, _, retention = estimate_R(V, M, gold)
    _, evaluable, correct = evaluable_and_correct(V, M, gold, tie_policy="abstain")
    Mb = M.astype(bool)
    per_judge_acc = [
        float((V[Mb[:, j], j] == gold[Mb[:, j]]).mean()) if Mb[:, j].any() else np.nan
        for j in range(V.shape[1])
    ]
    return {
        "bank": label,
        "n_judges": V.shape[1],
        "n_items": V.shape[0],
        "estimator": estimator,
        "listwise_retention": round(retention, 4),
        "abstention_rate": round(float(1 - Mb.mean()), 4),
        "accuracy_mean_per_judge": round(float(np.nanmean(per_judge_acc)), 4),
        "rho_bar": round(float(mean_off_diagonal(R)), 4),
        "n_eff": round(float(effective_size(R)), 3),
        "eigen_rank": round(float(effective_eig_rank(R)), 3),
        "majority_accuracy": round(float(correct.sum() / max(int(evaluable.sum()), 1)), 4),
        "majority_evaluable": int(evaluable.sum()),
        "majority_tie_rate": round(float(1 - evaluable.mean() - (1 - Mb.any(1).mean())), 4),
        **{k: v for k, v in cofailure(V, M, gold).items()
           if k in ("conditional_cofailure", "conditional_lift", "majority_cofailure")},
    }


def _rho_neff(V, M, gold):
    R, _, _, _ = estimate_R(V, M, gold)
    return float(mean_off_diagonal(R)), float(effective_size(R))


def paired_delta(Va, Ma, Vb, Mb_, gold, seed=SEED, B=B) -> dict:
    """Paired item bootstrap of (bank_b minus bank_a) in rho_bar and n_eff."""
    n = Va.shape[0]
    rng = np.random.default_rng(seed)
    d_rho, d_neff = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            ra, na = _rho_neff(Va[idx], Ma[idx], gold[idx])
            rb, nb = _rho_neff(Vb[idx], Mb_[idx], gold[idx])
        except Exception:
            continue
        d_rho.append(rb - ra)
        d_neff.append(nb - na)
    d_rho, d_neff = np.array(d_rho), np.array(d_neff)
    ra, na = _rho_neff(Va, Ma, gold)
    rb, nb = _rho_neff(Vb, Mb_, gold)
    lo_r, hi_r = np.quantile(d_rho, [0.025, 0.975])
    lo_n, hi_n = np.quantile(d_neff, [0.025, 0.975])
    return {
        "delta_rho_bar": round(rb - ra, 4),
        "rho_ci_low": round(float(lo_r), 4), "rho_ci_high": round(float(hi_r), 4),
        "rho_p_gt_0": round(float((d_rho > 0).mean()), 4),
        "delta_n_eff": round(nb - na, 3),
        "neff_ci_low": round(float(lo_n), 3), "neff_ci_high": round(float(hi_n), 3),
        "n_boot": int(len(d_rho)), "seed": seed,
    }


# ---------------------------------------------------------------- main
def main():
    items = load_calibration_set(MANIFEST)
    ids = [str(it.item_id) for it in items]
    gold = np.ones(len(ids), dtype=np.int8)
    lens = prompt_token_lengths(items)

    # ---- 1. exploratory truncation strata on the PUBLISHED generated caches ----
    rows = []
    for bank, (cfg, sel) in GEN_BANKS.items():
        V, M, lids = load_votes(ROOT / cfg, ids, sel)
        styles = [x.split("::")[1] for x in lids]
        over = np.zeros(len(ids), bool)
        for st in styles:
            over |= lens[st] > TRUNC_LIMIT
        for lab, mask in ((f"<= {TRUNC_LIMIT} tok", ~over), (f"> {TRUNC_LIMIT} tok", over)):
            blk = bank_block(V[mask], M[mask], gold[mask], f"{bank}|generated|{lab}")
            blk["stratum"] = lab
            blk["protocol"] = "generated (published caches)"
            rows.append(blk)
    trunc = pd.DataFrame(rows)
    (OUT).mkdir(parents=True, exist_ok=True)
    trunc.to_csv(OUT / "truncation_strata.csv", index=False)
    print("\n=== 1. cached votes stratified by prompt length (exploratory) ===")
    print(trunc[["bank", "n_items", "abstention_rate", "accuracy_mean_per_judge",
                 "rho_bar", "n_eff"]].to_string(index=False))

    # ---- 2. protocol-validity check: forced choice vs cached base ----
    fc_base = VoteCache(FC / "base")
    base_lids = GEN_BANKS["base"][1]
    Vg, Mg, _ = load_votes(ROOT / GEN_BANKS["base"][0], ids, base_lids)
    Vf, Mf = load_vote_matrix(fc_base, base_lids, ids)
    vrows = []
    for j, lid in enumerate(base_lids):
        style = lid.split("::")[1]
        over = lens[style] > TRUNC_LIMIT
        committed = Mg[:, j].astype(bool)          # cached run committed
        for lab, mask in (("all", committed),
                          (f"<= {TRUNC_LIMIT} tok", committed & ~over),
                          (f"> {TRUNC_LIMIT} tok", committed & over)):
            if mask.sum() == 0:
                continue
            agree = float((Vg[mask, j] == Vf[mask, j]).mean())
            vrows.append({
                "logical_id": lid, "style": style, "stratum": lab, "n": int(mask.sum()),
                "agreement": round(agree, 4),
                "acc_generated": round(float((Vg[mask, j] == 1).mean()), 4),
                "acc_forced_choice": round(float((Vf[mask, j] == 1).mean()), 4),
            })
    val = pd.DataFrame(vrows)
    # position-swap breakdown (systematic disagreement check)
    swap_rows = []
    for j, lid in enumerate(base_lids):
        df = (fc_base.load(lid).drop_duplicates(subset="item_id", keep="first")
              .set_index("item_id").reindex(ids))
        sw = df["position_swapped"].to_numpy().astype(bool)
        committed = Mg[:, j].astype(bool)
        for lab, m in (("chosen in slot A", committed & ~sw), ("chosen in slot B", committed & sw)):
            swap_rows.append({"logical_id": lid, "slot": lab, "n": int(m.sum()),
                              "agreement": round(float((Vg[m, j] == Vf[m, j]).mean()), 4)})
    swaps = pd.DataFrame(swap_rows)
    val.to_csv(OUT / "validation.csv", index=False)
    swaps.to_csv(OUT / "validation_by_slot.csv", index=False)
    print("\n=== 2. protocol validity: forced choice vs cached base ===")
    print(val[val.stratum == "all"].to_string(index=False))
    print("\noverall agreement (all committed):",
          round(float(np.average(val[val.stratum == 'all'].agreement,
                                 weights=val[val.stratum == 'all'].n)), 4))

    # ---- 3. primary: forced-choice bank comparison ----
    FCB = {}
    for bank in ("base", "grpo", "dpo"):
        lids = [f"{m}::{s}" for m in MODELS for s in STYLES] if bank == "base" else \
               [f"{m}-{bank}::{s}" for m in MODELS for s in STYLES]
        V, M = load_vote_matrix(VoteCache(FC / bank), lids, ids)
        FCB[bank] = (V, M, lids)
    blocks = [bank_block(*FCB[b][:2], gold, f"{b}|forced_choice") for b in ("base", "grpo", "dpo")]
    fc = pd.DataFrame(blocks)
    fc.to_csv(OUT / "forced_choice.csv", index=False)
    print("\n=== 3. forced-choice banks (identical items, zero abstention) ===")
    print(fc[["bank", "n_items", "abstention_rate", "accuracy_mean_per_judge", "rho_bar",
              "n_eff", "conditional_cofailure", "conditional_lift",
              "majority_accuracy"]].to_string(index=False))

    print("\n=== 4. paired item-bootstrap deltas (primary endpoint) ===")
    drows = []
    for a, b in (("base", "grpo"), ("base", "dpo"), ("grpo", "dpo")):
        d = paired_delta(FCB[a][0], FCB[a][1], FCB[b][0], FCB[b][1], gold)
        d = {"comparison": f"{b} - {a}", **d}
        drows.append(d)
        print(f"  {d['comparison']:12} d_rho={d['delta_rho_bar']:+.4f} "
              f"[{d['rho_ci_low']:+.4f},{d['rho_ci_high']:+.4f}]   "
              f"d_neff={d['delta_n_eff']:+.3f} [{d['neff_ci_low']:+.3f},{d['neff_ci_high']:+.3f}]")
    deltas = pd.DataFrame(drows)
    deltas.to_csv(OUT / "paired_deltas.csv", index=False)

    summary = {
        "n_items": len(ids), "bootstrap_B": B, "seed": SEED,
        "trunc_limit": TRUNC_LIMIT,
        "H1a_grpo_gt_base": bool(deltas.iloc[0].rho_ci_low > 0),
        "H1b_dpo_gt_base": bool(deltas.iloc[1].rho_ci_low > 0),
        "ordering_base_lt_grpo_lt_dpo": bool(
            fc.set_index("bank").loc["base|forced_choice"].rho_bar
            < fc.set_index("bank").loc["grpo|forced_choice"].rho_bar
            < fc.set_index("bank").loc["dpo|forced_choice"].rho_bar),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n=== verdict ===")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

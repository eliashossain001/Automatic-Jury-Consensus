"""Bucket 5 — qualitative examples where consensus is wrong but CorrFilter removes the item.

Finds items that naive majority/supermajority RETAINS with the WRONG label, yet CorrFilter's
alpha_subset ranks low enough to be REMOVED at the target retention. For each, dumps the
prompt, the two responses, gold, per-judge votes, the majority decision, the CorrFilter score,
its retain/remove status, and a short automatic failure-type note. Deterministic; CPU-only.

Outputs failure_examples.jsonl and a compact failure_examples.tex table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402
from compute_dependence import estimate_R, load_votes  # noqa: E402

GRPO_IDS = [f"{m}::{s}" for m in ("qwen-2.5-7b-grpo", "mistral-7b-grpo", "phi-3.5-mini-grpo")
            for s in ("pairwise", "likert")]


def _clip(s, n=240):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " …"


def failure_note(subset, n_agree_wrong, n_committed, alpha, set_size):
    reason = []
    if str(subset).lower() in ("math", "precise if", "reasoning"):
        reason.append("shared reasoning blind spot")
    frac = n_agree_wrong / max(n_committed, 1)
    if frac >= 0.75:
        reason.append("near-unanimous correlated co-failure")
    elif frac > 0.5:
        reason.append("correlated majority co-failure")
    reason.append(f"low decorrelated agreement (alpha_subset={alpha:.2f} over |S|={set_size})")
    return "; ".join(reason)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/judge_bank.yaml",
                    help="default = base 10-judge bank (low abstention -> cleanest co-failure cases)")
    ap.add_argument("--manifest", default="experiments/h1_measurement/results/calibration_manifest.parquet")
    ap.add_argument("--retention", type=float, default=0.60)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/qualitative")
    ap.add_argument("--judge-id", action="append", default=None,
                    help="restrict to these logical ids; default uses all judges in --bank")
    args = ap.parse_args()

    p = Path(args.manifest)
    man = (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
    man["item_id"] = man["item_id"].astype(str)
    item_ids = man["item_id"].tolist()
    man = man.set_index("item_id")

    V, M, lids = load_votes(args.bank, item_ids, args.judge_id)
    gold = np.ones(len(item_ids), dtype=np.int8)
    R, _, _, _ = estimate_R(V, M, gold)
    n = len(item_ids)

    maj = majority_consensus(V, M)
    level = consensus_level(V, M)
    labellable = maj != ABSTAIN
    cf = corrfilter_score(V, M, R, maj)
    # rank & keep-sets at target retention
    k = int(round(args.retention * n))
    order_cf = np.argsort(-np.nan_to_num(cf.score, nan=-np.inf), kind="stable")
    keep_cf = np.zeros(n, bool); keep_cf[order_cf[:k]] = True

    # candidate: consensus retains (labellable) & WRONG & CorrFilter removes, with a
    # SUBSTANTIAL agreeing set so the example shows genuine correlated co-failure
    # (not a high-abstention artifact where |S|=1).
    n_judges = len(lids)
    min_commit = max(4, int(np.ceil(0.6 * n_judges)))
    min_set = max(3, int(np.ceil(0.5 * n_judges)))
    committed = M.sum(1)
    cand = np.where(labellable & (maj != gold) & (~keep_cf)
                    & (committed >= min_commit) & (cf.set_size >= min_set))[0]
    # prioritise the most confidently-wrong-yet-correlated ones (high consensus, low alpha)
    cand = sorted(cand, key=lambda i: (-level[i], cf.score[i] if not np.isnan(cf.score[i]) else 1e9))
    if len(cand) < args.n_examples:  # relax if too few clean cases at this abstention level
        extra = np.where(labellable & (maj != gold) & (~keep_cf) & (cf.set_size >= 2))[0]
        cand = list(dict.fromkeys(list(cand) + sorted(extra, key=lambda i: -level[i])))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in cand[: args.n_examples]:
        iid = item_ids[i]
        row = man.loc[iid]
        M_b = M[i].astype(bool)
        votes = {lids[j]: int(V[i, j]) if M_b[j] else -1 for j in range(len(lids))}
        n_committed = int(M_b.sum())
        n_agree_wrong = int(((V[i] == maj[i]) & M_b & (maj[i] != gold[i])).sum())
        subset = row.get("subset", row.get("category", "?"))
        rec = {
            "item_id": iid, "subset": str(subset),
            "prompt": _clip(row.get("prompt", ""), 400),
            "response_chosen_gold": _clip(row.get("chosen", "")),
            "response_rejected_gold": _clip(row.get("rejected", "")),
            "gold_label": "chosen>rejected",
            "judge_votes": votes,
            "consensus_level": round(float(level[i]), 3),
            "majority_decision": "chosen" if maj[i] == 1 else "rejected",
            "majority_correct": bool(maj[i] == gold[i]),
            "corrfilter_alpha_subset": round(float(cf.score[i]), 3) if not np.isnan(cf.score[i]) else None,
            "corrfilter_status": "REMOVED" if not keep_cf[i] else "kept",
            "set_size_S": int(cf.set_size[i]),
            "failure_type": failure_note(subset, n_agree_wrong, n_committed,
                                         float(cf.score[i]) if not np.isnan(cf.score[i]) else 0.0,
                                         int(cf.set_size[i])),
        }
        recs.append(rec)

    (out / "failure_examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    # compact LaTeX table
    lines = [r"\begin{tabular}{p{0.30\linewidth} c c c c p{0.28\linewidth}}", r"\toprule",
             r"Prompt (truncated) & subset & consensus & $\alpha_{\mathrm{subset}}$ & CorrFilter & Failure type \\",
             r"\midrule"]
    for r in recs:
        prompt = r["prompt"][:60].replace("&", r"\&")
        lines.append(
            f"{prompt}\\ldots & {r['subset']} & "
            f"{r['consensus_level']:.2f} (wrong) & {r['corrfilter_alpha_subset']} & "
            f"{r['corrfilter_status']} & {r['failure_type'][:60]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "failure_examples.tex").write_text("\n".join(lines) + "\n")

    print(f"[extract_failure_examples] wrote {len(recs)} examples -> {out}/failure_examples.{{jsonl,tex}}")
    for r in recs[:3]:
        print(f"  {r['item_id']} [{r['subset']}] consensus={r['consensus_level']} wrong, "
              f"alpha={r['corrfilter_alpha_subset']} -> {r['corrfilter_status']} :: {r['failure_type']}")


if __name__ == "__main__":
    main()

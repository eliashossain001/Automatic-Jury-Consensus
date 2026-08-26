#!/usr/bin/env python
"""Analyze the screening campaign: all candidates, bank compositions, unified
routing-feasibility table (Sec 9 of the instruction).

Candidates screened (given cached votes from scripts/crosstask/run_screening_votes.py):
  factuality sources: vitc_all / vitc_real / vitc_synth (bank A, free),
                      halueval (bank A styles)
  factuality banks on the vitc bank-screen sample:
                      bankA (2 styles x 5 models), bankD_only (2 new
                      reasoning modes), bankAD (4 styles x 5 models = 20),
                      bankB_mix (bank A + qwen-14b + qwen-1.5b = 14),
                      bankABD (24)
  code:               cjb_point (pointwise, 10 judges),
                      cjb_pair (pairwise+likert, 10 judges; chance 0.5,
                      flip-mode regimes)

Usage: python scripts/crosstask/analyze_screening.py
Outputs -> outputs/router_upgrade/screening/screen_summary_all.csv (+ per-
candidate JSON), printed unified table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from corrfilter.screening.feasibility import screen_candidate
from corrfilter.voting import VoteCache

ROOT = Path(__file__).resolve().parents[2]


OUT = ROOT / "outputs/router_upgrade/screening"
SB = ROOT / "outputs/screening_bank"
FBV = ROOT / "outputs/factuality_bank/votes"
MODELS = ["gemma-2-9b", "llama-3.1-8b", "mistral-7b-v0.3", "phi-3.5-mini", "qwen-2.5-7b"]


class MultiCache:
    """First-hit-wins union of several VoteCaches."""

    def __init__(self, *dirs):
        self.caches = [VoteCache(d) for d in dirs]

    def load(self, lid):
        for c in self.caches:
            df = c.load(lid)
            if df is not None:
                return df
        return None


def lids_for(models, styles):
    return [f"{m}::{s}" for m in models for s in styles]


def main() -> None:
    rows = []
    free = OUT / "screen_summary_free.csv"
    if free.exists():
        rows += pd.read_csv(free).to_dict("records")

    # factuality source: halueval (bank A styles)
    man_h = pd.read_parquet(SB / "manifest_halueval.parquet")
    rows.append(screen_candidate(
        "halueval", man_h, SB / "votes/halueval",
        lids_for(MODELS, ("pw_direct", "pw_analysis")), OUT,
        task_name="factuality"))

    # bank compositions on the vitc bank-screen sample
    man_v = pd.read_parquet(SB / "manifest_vitc_bankscreen.parquet")
    mc = MultiCache(FBV, SB / "votes/vitc_extra")
    banks = {
        "bankA": lids_for(MODELS, ("pw_direct", "pw_analysis")),
        "bankD_only": lids_for(MODELS, ("pw_decompose", "pw_contradict")),
        "bankAD": lids_for(MODELS, ("pw_direct", "pw_analysis",
                                    "pw_decompose", "pw_contradict")),
        "bankB_mix": lids_for(MODELS, ("pw_direct", "pw_analysis"))
        + lids_for(["qwen-2.5-14b", "qwen-2.5-1.5b"], ("pw_direct", "pw_analysis")),
        "bankABD": lids_for(MODELS, ("pw_direct", "pw_analysis",
                                     "pw_decompose", "pw_contradict"))
        + lids_for(["qwen-2.5-14b", "qwen-2.5-1.5b"], ("pw_direct", "pw_analysis")),
    }
    for name, lids in banks.items():
        k = 5 if len(lids) >= 10 else 3
        rows.append(screen_candidate(
            f"vitc_{name}", man_v, None, lids, OUT, cluster_k=k,
            task_name="factuality-bank", cache=mc,
            extra={"nominal_judges": len(lids)}))

    # code pointwise: the pwc_trace judges abstained 52-100% in the pilot
    # (224-token budget exhausted before the FINAL line), so the effective
    # bank is the five pwc_direct judges; reported as a pilot finding.
    man_c = pd.read_parquet(SB / "manifest_cjb_point.parquet")
    rows.append(screen_candidate(
        "cjb_point", man_c, SB / "votes/cjb_point",
        lids_for(MODELS, ("pwc_direct",)), OUT, cluster_k=3,
        task_name="code-pointwise", extra={"note": "trace judges parse-dead"}))

    # code pairwise (oriented gold=1; chance 0.5; flip-mode regimes)
    meta = pd.read_parquet(SB / "manifest_cjb_pair.parquet")
    man_p = pd.DataFrame({"item_id": meta.item_id, "gold_supported": 1,
                          "doc_chars": 0, "claim_chars": 0})
    rows.append(screen_candidate(
        "cjb_pair", man_p, SB / "votes/cjb_pair",
        lids_for(MODELS, ("pairwise", "likert")), OUT,
        task_name="code-pairwise", chance_override=0.5, inject_mode="flip"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "screen_summary_all.csv", index=False)
    cols = ["candidate", "task", "n_items", "majority_acc", "chance",
            "majority_wrong", "rho_err", "n_eff", "headroom_mean_pts",
            "nonzero_headroom_rate", "corrfilter_delta_global_pts",
            "biascluster_delta_subgroup_pts", "decision"]
    print("== Unified routing-feasibility table ==")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

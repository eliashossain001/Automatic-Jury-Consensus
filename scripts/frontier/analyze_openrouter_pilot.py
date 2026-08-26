#!/usr/bin/env python
"""Multi-provider pilot analysis: OpenRouter frontier bank vs Gemini vs open bank.

Analysis-only. On the 200-item stratified RewardBench pilot subset (identical
items for every judge), compares:

  openrouter_3   gpt-5.6-sol + claude-opus-5 + grok-4.5 (three providers)
  gemini_3       the Gemini pilot bank (one provider, three tiers)
  open_pairwise_5 / open_all_10   the open-weight bank
  frontier_6     gemini_3 + openrouter_3 (four providers)
  mixed_11       open_pairwise_5 + gemini_3 + openrouter_3

Primary question: does a cross-provider frontier bank show lower error
correlation than the single-provider (Gemini) frontier bank of the same size?
Reported with paired item-bootstrap CIs on the rho_bar differences.

Usage: python scripts/frontier/analyze_openrouter_pilot.py [--n-boot 2000]
Outputs -> outputs/openrouter_bank/analysis_pilot200/
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from corrfilter.analysis.dependence import dependence_metrics
from corrfilter.correlation import compute_error_matrix, correlation_shrunk
from corrfilter.correlation.effective_size import mean_off_diagonal
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache, load_vote_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load_script(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s34 = _load_script("gba", "analyze_gemini_bank.py")

BOOT_SEED = 20260706

OR_JUDGES = ["gpt-5.6-sol::pairwise", "claude-opus-5::pairwise", "grok-4.5::pairwise"]
GEM_JUDGES = ["gemini-3.1-pro::pairwise", "gemini-3.6-flash::pairwise", "gemini-3.5-flash-lite::pairwise"]

PROVIDER = {}
for j in OR_JUDGES:
    PROVIDER[j] = {"gpt-5.6-sol::pairwise": "openai", "claude-opus-5::pairwise": "anthropic",
                   "grok-4.5::pairwise": "xai"}[j]
for j in GEM_JUDGES:
    PROVIDER[j] = "google"


def cond_cosuccess(V, M, gold) -> float:
    Mb = M.astype(bool)
    C = (V == gold[:, None]) & Mb
    vals = []
    for i in range(V.shape[1]):
        for j in range(V.shape[1]):
            if i == j:
                continue
            both = Mb[:, i] & Mb[:, j]
            ci = C[both, i]
            if ci.sum() > 0:
                vals.append(float((C[both, i] & C[both, j]).sum() / ci.sum()))
    return float(np.mean(vals))


def disagreement_matrix(V, M, lids) -> pd.DataFrame:
    Mb = M.astype(bool)
    n = len(lids)
    D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = 0.0
                continue
            both = Mb[:, i] & Mb[:, j]
            if both.sum():
                D[i, j] = float((V[both, i] != V[both, j]).mean())
    return pd.DataFrame(D, index=lids, columns=lids)


def provider_blocks(R: np.ndarray, lids: list[str]) -> pd.DataFrame:
    provs = [PROVIDER.get(l, "open") for l in lids]
    uniq = sorted(set(provs))
    rows = []
    for a in uniq:
        for b in uniq:
            vals = [R[i, j] for i in range(len(lids)) for j in range(len(lids))
                    if i != j and provs[i] == a and provs[j] == b]
            if vals:
                rows.append({"provider_a": a, "provider_b": b,
                             "mean_corr": round(float(np.mean(vals)), 4), "n_pairs": len(vals)})
    return pd.DataFrame(rows)


def rho_of(E_draw) -> float:
    R, _ = correlation_shrunk(E_draw)
    return mean_off_diagonal(R)


def paired_rho_diff_ci(Va, Ma, Vb, Mb_, gold, n_boot, seed=BOOT_SEED):
    """Paired item bootstrap of rho_bar(bank A) - rho_bar(bank B) on shared items."""
    Ea, ca = compute_error_matrix(Va, Ma, gold)
    Eb, cb = compute_error_matrix(Vb, Mb_, gold)
    both = ca & cb                                  # items listwise-complete in BOTH banks
    Ea = (Va[both] != gold[both, None]).astype(float)
    Eb = (Vb[both] != gold[both, None]).astype(float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(Ea), len(Ea))
        diffs.append(rho_of(Ea[idx]) - rho_of(Eb[idx]))
    point = rho_of(Ea) - rho_of(Eb)
    return (round(point, 4), round(float(np.quantile(diffs, 0.025)), 4),
            round(float(np.quantile(diffs, 0.975)), 4), int(both.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.project_root)
    out_dir = root / "outputs" / "openrouter_bank" / "analysis_pilot200"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_ids = [l.strip() for l in (root / "outputs/openrouter_bank/items_rewardbench_pilot200.txt").read_text().splitlines() if l.strip()]
    n_items = len(item_ids)
    gold = np.ones(n_items, dtype=np.int8)

    or_cache = VoteCache(root / "outputs/openrouter_bank/votes/frontier/rewardbench")
    gem_cache = VoteCache(root / "outputs/gemini_bank/votes/rewardbench")
    open_bank = load_bank_config(root / "configs/judge_bank.yaml")
    open_cache = VoteCache(open_bank.votes_dir if Path(open_bank.votes_dir).is_absolute()
                           else root / open_bank.votes_dir)
    open_all = [s.logical_id for s in open_bank.specs]
    open_pw = [l for l in open_all if l.endswith("::pairwise")]

    # ---- per-judge metrics ----
    rows = []
    for cache, lids, prov in ((or_cache, OR_JUDGES, "openrouter"), (gem_cache, GEM_JUDGES, "gemini"),
                              (open_cache, open_all, "open")):
        for lid in lids:
            df = cache.load(lid)
            if df is None:
                print(f"[39] WARNING: no votes for {lid}")
                continue
            df = df[df["item_id"].astype(str).isin(set(item_ids))].drop_duplicates("item_id", keep="last")
            rows.append({**_s34.judge_row(lid, df, n_items), "source": prov})
    judge_df = pd.DataFrame(rows)
    judge_df.to_csv(out_dir / "judge_metrics.csv", index=False)

    # ---- banks on identical items ----
    Vor, Mor = load_vote_matrix(or_cache, OR_JUDGES, item_ids)
    Vg, Mg = load_vote_matrix(gem_cache, GEM_JUDGES, item_ids)
    Vp, Mp = load_vote_matrix(open_cache, open_pw, item_ids)
    Va, Ma = load_vote_matrix(open_cache, open_all, item_ids)

    banks = {
        "openrouter_3": (Vor, Mor, OR_JUDGES),
        "gemini_3": (Vg, Mg, GEM_JUDGES),
        "open_pairwise_5": (Vp, Mp, open_pw),
        "open_all_10": (Va, Ma, open_all),
        "frontier_6": (np.hstack([Vg, Vor]), np.hstack([Mg, Mor]), GEM_JUDGES + OR_JUDGES),
        "mixed_11": (np.hstack([Vp, Vg, Vor]), np.hstack([Mp, Mg, Mor]), open_pw + GEM_JUDGES + OR_JUDGES),
    }
    bank_rows = {}
    for name, (V, M, lids) in banks.items():
        row, R = dependence_metrics(V, M, gold, lids, label=name)
        row.update(_s34.majority_accuracy(V, M, gold))
        row["cond_co_success"] = round(cond_cosuccess(V, M, gold), 4)
        row.update({f"boot_{k}": v for k, v in _s34.bootstrap_bank(V, M, gold, args.n_boot).items()})
        bank_rows[name] = row
        pd.DataFrame(R, index=lids, columns=lids).to_csv(out_dir / f"R_{name}.csv")
        disagreement_matrix(V, M, lids).to_csv(out_dir / f"disagreement_{name}.csv")
        if name in ("frontier_6", "mixed_11"):
            provider_blocks(R, lids).to_csv(out_dir / f"provider_blocks_{name}.csv", index=False)
    bank_df = pd.DataFrame(list(bank_rows.values()))
    bank_df.to_csv(out_dir / "bank_metrics.csv", index=False)

    # ---- primary comparisons: paired rho_bar differences ----
    comps = []
    for a, b, note in (
        ("gemini_3", "openrouter_3", "same-size frontier: single-provider minus cross-provider"),
        ("open_pairwise_5", "openrouter_3", "open 5-judge minus cross-provider frontier 3"),
        ("gemini_3", "frontier_6", "adding 3 providers to Gemini: gemini-only minus combined"),
        ("open_all_10", "mixed_11", "open 10 minus full mixed 11"),
    ):
        Va_, Ma_, _ = banks[a]
        Vb_, Mb_, _ = banks[b]
        point, lo, hi, n_shared = paired_rho_diff_ci(Va_, Ma_, Vb_, Mb_, gold, args.n_boot)
        comps.append({"comparison": f"rho({a}) - rho({b})", "note": note,
                      "diff": point, "ci95_low": lo, "ci95_high": hi,
                      "n_shared_items": n_shared,
                      "significant": bool(lo > 0 or hi < 0)})
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(out_dir / "comparisons_ci.csv", index=False)

    # ---- summary ----
    cols = ["label", "n_judges", "rho_bar", "n_eff", "conditional_cofailure",
            "cond_co_success", "majority_accuracy", "majority_jointly_wrong"]
    with open(out_dir / "summary.md", "w") as f:
        f.write("# Multi-provider pilot (200 stratified RewardBench items)\n\n")
        f.write("Identical items for all judges; identical prompt/position/decoding protocol.\n\n")
        f.write("## Per-judge\n\n")
        f.write(judge_df.to_markdown(index=False))
        f.write("\n\n## Banks\n\n")
        f.write(bank_df[cols].to_markdown(index=False))
        f.write("\n\n## Paired rho_bar comparisons (item bootstrap, 95% CI)\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n")
    print(f"[39] wrote {out_dir}")
    print(bank_df[cols].to_string(index=False))
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()

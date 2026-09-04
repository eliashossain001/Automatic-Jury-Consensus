#!/usr/bin/env python
"""Analyze the Gemini frontier-judge pilot and compare with the open-weight bank.

Analysis-only (no API calls). For each dataset (rewardbench, pku) computes:
  per judge : coverage, abstention, accuracy (+95% Wilson CI), slot-A/slot-B
              accuracy, position-bias gap;
  per bank  : error-correlation matrix R, rho_bar, n_eff, eigen rank,
              conditional co-failure P(B wrong | A wrong), majority-vote
              accuracy, majority-jointly-wrong rate, bootstrap CIs on
              rho_bar / n_eff (item resampling, seed 20260706);
  cross     : within-Gemini vs within-open vs cross-provider mean error
              correlation on the combined pairwise bank.

Banks compared on IDENTICAL items:
  gemini_3          the 3 Gemini judges (pairwise)
  open_pairwise_5   open-weight bank, pairwise prompt only (prompt parity)
  open_all_10       full open-weight bank (both prompt styles)
  combined_8        open_pairwise_5 + gemini_3

Usage:
  python scripts/frontier/analyze_gemini_bank.py [--dataset rewardbench pku] [--n-boot 2000]

Outputs -> outputs/gemini_bank/analysis_<dataset>/
  judge_metrics.csv, bank_metrics.csv, R_<bank>.csv, cross_provider.csv, summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from compute_dependence import cofailure, dependence_metrics, estimate_R  # noqa: E402

from corrfilter.correlation import compute_error_matrix, correlation_shrunk  # noqa: E402
from corrfilter.correlation.effective_size import effective_size, mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

BOOT_SEED = 20260706

OPEN_BANK_YAML = {
    "rewardbench": "configs/judge_bank.yaml",
    "pku": "configs/judge_bank_pku.yaml",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 4), round(centre + half, 4))


def judge_row(name: str, df: pd.DataFrame, n_items: int) -> dict:
    """Per-judge metrics from a raw vote parquet (vote, position_swapped)."""
    committed = df[df["vote"] != -1]
    acc_k, acc_n = int((committed["vote"] == 1).sum()), len(committed)
    # chosen shown in slot A <=> not swapped
    slot_a = committed[~committed["position_swapped"]]
    slot_b = committed[committed["position_swapped"]]
    acc_a = float((slot_a["vote"] == 1).mean()) if len(slot_a) else float("nan")
    acc_b = float((slot_b["vote"] == 1).mean()) if len(slot_b) else float("nan")
    lo, hi = wilson_ci(acc_k, acc_n)
    return {
        "judge": name,
        "n_items": n_items,
        "n_responses": len(df),
        "n_committed": acc_n,
        "coverage": round(len(df) / n_items, 4),
        "abstention_rate": round(1 - acc_n / max(len(df), 1), 4),
        "accuracy": round(acc_k / acc_n, 4) if acc_n else float("nan"),
        "acc_ci_low": lo,
        "acc_ci_high": hi,
        "slotA_accuracy": round(acc_a, 4),
        "slotB_accuracy": round(acc_b, 4),
        "position_gap": round(acc_a - acc_b, 4),
        "n_slotA": len(slot_a),
        "n_slotB": len(slot_b),
    }


def bootstrap_bank(V, M, gold, n_boot: int, seed: int = BOOT_SEED) -> dict:
    """Item-resampled CIs for rho_bar and n_eff on the listwise-complete error matrix."""
    E, complete = compute_error_matrix(V, M, gold)
    if E.shape[0] < 10:
        return {}
    rng = np.random.default_rng(seed)
    rhos, neffs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, E.shape[0], size=E.shape[0])
        R_b, _ = correlation_shrunk(E[idx])
        rhos.append(mean_off_diagonal(R_b))
        neffs.append(effective_size(R_b))
    q = lambda a: (round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4))
    return {
        "rho_bar_ci": q(rhos),
        "n_eff_ci": q(neffs),
        "n_boot": n_boot,
        "n_listwise_items": int(E.shape[0]),
    }


def majority_accuracy(V, M, gold) -> dict:
    """Majority vote over committed votes; ties are counted as not-correct."""
    M_b = M.astype(bool)
    correct = ((V == gold[:, None]) & M_b).sum(1)
    n_c = M_b.sum(1)
    has = n_c > 0
    frac = correct[has] / n_c[has]
    return {
        "n_items_with_vote": int(has.sum()),
        "majority_accuracy": round(float((frac > 0.5).mean()), 4),
        "majority_jointly_wrong": round(float((frac < 0.5).mean()), 4),
        "majority_tie_rate": round(float((frac == 0.5).mean()), 4),
    }


def cross_provider(R: np.ndarray, lids: list[str]) -> dict:
    gem = np.array([l.startswith("gemini") for l in lids])
    blocks = {"within_gemini": (gem, gem), "within_open": (~gem, ~gem), "cross": (gem, ~gem)}
    out = {}
    for name, (a, b) in blocks.items():
        vals = []
        for i in np.where(a)[0]:
            for j in np.where(b)[0]:
                if i != j:
                    vals.append(R[i, j])
        out[name] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return out


def cross_provider_boot(V, M, gold, lids, n_boot: int, seed: int = BOOT_SEED) -> dict:
    """Item-bootstrap CIs for the block means and the within-minus-cross gaps."""
    E, _ = compute_error_matrix(V, M, gold)
    if E.shape[0] < 10:
        return {}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("within_gemini", "within_open", "cross",
                             "gemini_minus_cross", "open_minus_cross")}
    for _ in range(n_boot):
        R_b, _ = correlation_shrunk(E[rng.integers(0, E.shape[0], E.shape[0])])
        b = cross_provider(R_b, lids)
        draws["within_gemini"].append(b["within_gemini"])
        draws["within_open"].append(b["within_open"])
        draws["cross"].append(b["cross"])
        draws["gemini_minus_cross"].append(b["within_gemini"] - b["cross"])
        draws["open_minus_cross"].append(b["within_open"] - b["cross"])
    q = lambda a: (round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4))
    return {f"{k}_ci": q(v) for k, v in draws.items()}


def analyze_dataset(dataset: str, root: Path, n_boot: int) -> None:
    out_dir = root / "outputs" / "gemini_bank" / f"analysis_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gcfg = yaml.safe_load((root / "configs" / "judge_bank_gemini.yaml").read_text())
    item_ids = [l.strip() for l in (root / "outputs" / "gemini_bank" / f"items_{dataset}.txt").read_text().splitlines() if l.strip()]
    n_items = len(item_ids)
    gold = np.ones(n_items, dtype=np.int8)

    gem_cache = VoteCache(root / "outputs" / "gemini_bank" / "votes" / dataset)
    gem_lids = [f"{e['id']}::pairwise" for e in gcfg["bank"]]

    open_bank = load_bank_config(root / OPEN_BANK_YAML[dataset])
    open_cache = VoteCache(open_bank.votes_dir if Path(open_bank.votes_dir).is_absolute()
                           else root / open_bank.votes_dir)
    open_all = [s.logical_id for s in open_bank.specs]
    open_pw = [l for l in open_all if l.endswith("::pairwise")]

    # ---- per-judge metrics (raw parquets keep position_swapped) ----
    rows = []
    for lid in gem_lids:
        df = gem_cache.load(lid)
        if df is None:
            print(f"[34] WARNING: no votes for {lid} on {dataset}; skipping")
            continue
        df = df[df["item_id"].astype(str).isin(set(item_ids))]
        df = df.drop_duplicates(subset="item_id", keep="last")
        rows.append({**judge_row(lid, df, n_items), "provider": "gemini"})
    for lid in open_all:
        df = open_cache.load(lid)
        if df is None:
            continue
        df = df[df["item_id"].astype(str).isin(set(item_ids))]
        df = df.drop_duplicates(subset="item_id", keep="last")
        rows.append({**judge_row(lid, df, n_items), "provider": "open"})
    judge_df = pd.DataFrame(rows)
    judge_df.to_csv(out_dir / "judge_metrics.csv", index=False)

    # ---- bank-level metrics on identical items ----
    banks = {
        "gemini_3": (gem_cache, gem_lids),
        "open_pairwise_5": (open_cache, open_pw),
        "open_all_10": (open_cache, open_all),
    }
    bank_rows, R_store = [], {}
    for name, (cache, lids) in banks.items():
        V, M = load_vote_matrix(cache, lids, item_ids)
        if M.sum() == 0:
            continue
        row, R = dependence_metrics(V, M, gold, lids, label=name)
        row.update(majority_accuracy(V, M, gold))
        row.update({f"boot_{k}": v for k, v in bootstrap_bank(V, M, gold, n_boot).items()})
        bank_rows.append(row)
        R_store[name] = (R, lids)
        pd.DataFrame(R, index=lids, columns=lids).to_csv(out_dir / f"R_{name}.csv")

    # combined pairwise bank (open 5 + gemini 3)
    Vg, Mg = load_vote_matrix(gem_cache, gem_lids, item_ids)
    Vo, Mo = load_vote_matrix(open_cache, open_pw, item_ids)
    V = np.concatenate([Vo, Vg], axis=1)
    M = np.concatenate([Mo, Mg], axis=1)
    lids = open_pw + gem_lids
    row, R = dependence_metrics(V, M, gold, lids, label="combined_8")
    row.update(majority_accuracy(V, M, gold))
    row.update({f"boot_{k}": v for k, v in bootstrap_bank(V, M, gold, n_boot).items()})
    bank_rows.append(row)
    R_store["combined_8"] = (R, lids)
    pd.DataFrame(R, index=lids, columns=lids).to_csv(out_dir / "R_combined_8.csv")
    xp = cross_provider(R, lids)
    xp.update(cross_provider_boot(V, M, gold, lids, n_boot))
    pd.DataFrame([xp]).to_csv(out_dir / "cross_provider.csv", index=False)

    bank_df = pd.DataFrame(bank_rows)
    bank_df.to_csv(out_dir / "bank_metrics.csv", index=False)

    # ---- heatmaps: R and pairwise conditional co-failure, per bank ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for name, (R, lids) in R_store.items():
        V, M = (np.concatenate([Vo, Vg], axis=1), np.concatenate([Mo, Mg], axis=1)) \
            if name == "combined_8" else (None, None)
        if V is None:
            cache, ls = banks[name]
            V, M = load_vote_matrix(cache, ls, item_ids)
        Mb = M.astype(bool)
        W = (V != gold[:, None]) & Mb
        n = len(lids)
        CF = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                both = Mb[:, i] & Mb[:, j]
                wi = W[both, i]
                if wi.sum() > 0:
                    CF[i, j] = (W[both, i] & W[both, j]).sum() / wi.sum()
        short = [l.replace("::pairwise", "").replace("::likert", "-lk") for l in lids]
        for mat, title, fname, vmax in ((R, "error correlation $R$", f"R_heatmap_{name}.png", 1.0),
                                        (CF, "P(col wrong | row wrong)", f"cofail_heatmap_{name}.png", 1.0)):
            fig, ax = plt.subplots(figsize=(1.2 + 0.6 * n, 1.0 + 0.6 * n))
            im = ax.imshow(mat, vmin=0, vmax=vmax, cmap="viridis")
            ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7)
            for i in range(n):
                for j in range(n):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if mat[i, j] < 0.6 * vmax else "black")
            ax.set_title(f"{name} — {title}", fontsize=9)
            fig.colorbar(im, shrink=0.8)
            fig.tight_layout()
            fig.savefig(fig_dir / fname, dpi=150)
            plt.close(fig)

    # ---- summary ----
    with open(out_dir / "summary.md", "w") as f:
        f.write(f"# Gemini frontier-judge pilot — {dataset}\n\n")
        f.write(f"Items: {n_items} (identical across all judges/banks). ")
        f.write("Single-provider pilot (Gemini only); see run_log for model IDs, settings, cost.\n\n")
        f.write("## Per-judge metrics\n\n")
        f.write(judge_df.to_markdown(index=False))
        f.write("\n\n## Bank-level dependence (identical items)\n\n")
        f.write(bank_df.to_markdown(index=False))
        f.write("\n\n## Cross-provider mean error correlation (combined_8)\n\n")
        f.write(pd.DataFrame([xp]).to_markdown(index=False))
        f.write("\n")
    print(f"[34] {dataset}: wrote {out_dir}")
    print(bank_df[["label", "rho_bar", "n_eff", "conditional_cofailure",
                   "majority_accuracy", "majority_jointly_wrong"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["rewardbench", "pku"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    for ds in args.dataset:
        analyze_dataset(ds, Path(args.project_root), args.n_boot)


if __name__ == "__main__":
    main()

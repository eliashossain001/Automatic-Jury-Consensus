#!/usr/bin/env python
"""Correlated NON-error analysis: are correct predictions over-counted too?

For judge j define the correctness indicator C_ij = 1[judge j correct on item i]
(on committed votes). By definition C = 1 - E (unconditionally; no gold-direction
assumption is needed), and Pearson correlation is invariant under jointly
complementing both binary variables (Cov(1-X,1-Y) = Cov(X,Y), variances unchanged;
Ledoit-Wolf depends on the data only through the covariance, so the shrunk estimate
is invariant too). The correctness-correlation matrix is therefore IDENTICAL to the
error-correlation matrix when both use the same committed-vote mask and item set.
This script quantifies the quantities that are NOT identical:

  * conditional co-success P(B correct | A correct) and its lift over the
    marginal accuracy (vs conditional co-failure P(B wrong | A wrong) and its
    larger lift over the marginal error rate);
  * effective number of independent CORRECT votes per item,
    n_eff+(i) = |S+|^2 / (1' R_{S+} 1) = alpha_subset(S+)^2, vs the nominal
    count |S+| (and the same for error votes);
  * difficulty-stratified residual correlation: within strata of leave-pair-out
    item difficulty (fraction of the OTHER judges correct), how much pairwise
    correlation remains -- separating shared item easiness/hardness from shared
    judge bias.

Analysis-only; replays cached votes. Usage:
  python scripts/dependence/analyze_nonerror_correlation.py                     # open bank, RewardBench 1195
  python scripts/dependence/analyze_nonerror_correlation.py --bank gemini_rewardbench ...

Outputs -> outputs/nonerror_correlation/<label>/
  nonerror_summary.csv     one-row bank summary (paper table source)
  consensus_evidence.csv   nominal correct-consensus size vs effective evidence
  difficulty_adjusted.csv  raw vs difficulty-stratified mean pair correlation
  summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from compute_dependence import estimate_R  # noqa: E402

from corrfilter.cfi.corrfilter_score import corrfilter_score  # noqa: E402
from corrfilter.correlation import compute_error_matrix, correlation_shrunk  # noqa: E402
from corrfilter.correlation.effective_size import effective_size, mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

BOOT_SEED = 20260706

BANKS = {
    # label: (votes_dir, logical_ids source, items source)
    "open_rewardbench": {
        "bank_yaml": "configs/judge_bank.yaml",
        "manifest": "experiments/h1_measurement/results/calibration_manifest.parquet",
    },
    "open_pku": {
        "bank_yaml": "configs/judge_bank_pku.yaml",
        "manifest": "outputs/pku_saferlhf/pku_manifest.parquet",
    },
    "gemini_rewardbench": {
        "votes_dir": "outputs/gemini_bank/votes/rewardbench",
        "gemini": True,
        "items_file": "outputs/gemini_bank/items_rewardbench.txt",
    },
    "gemini_pku": {
        "votes_dir": "outputs/gemini_bank/votes/pku",
        "gemini": True,
        "items_file": "outputs/gemini_bank/items_pku.txt",
    },
}


def conditional_rates(C: np.ndarray, Mb: np.ndarray) -> dict:
    """Mean over ordered pairs of P(B correct|A correct) and P(B wrong|A wrong)."""
    n = C.shape[1]
    co_succ, co_fail = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            both = Mb[:, i] & Mb[:, j]
            ci, cj = C[both, i], C[both, j]
            if ci.sum() > 0:
                co_succ.append(float((ci * cj).sum() / ci.sum()))
            wi, wj = 1 - ci, 1 - cj
            if wi.sum() > 0:
                co_fail.append(float((wi * wj).sum() / wi.sum()))
    acc = float(np.concatenate([C[Mb[:, j], j] for j in range(n)]).mean())
    return {
        "marginal_accuracy": round(acc, 4),
        "marginal_error": round(1 - acc, 4),
        "cond_co_success": round(float(np.mean(co_succ)), 4),
        "cond_co_failure": round(float(np.mean(co_fail)), 4),
        "co_success_lift": round(float(np.mean(co_succ)) / acc, 3),
        "co_failure_lift": round(float(np.mean(co_fail)) / (1 - acc), 3),
    }


def effective_evidence(V, M, gold, R) -> tuple[dict, pd.DataFrame]:
    """Nominal vs effective (alpha^2) independent-vote counts, correct and wrong sets."""
    n_items = V.shape[0]
    plus = corrfilter_score(V, M, R, np.asarray(gold))            # agreeing-with-gold = correct set
    minus = corrfilter_score(V, M, R, np.asarray(1 - gold))       # disagreeing = wrong set
    rows = []
    for name, cs in (("correct", plus), ("wrong", minus)):
        sizes, quad = cs.set_size, cs.quad_form
        ok = sizes > 0
        eff = np.where(quad[ok] > 0, sizes[ok] ** 2 / quad[ok], np.nan)
        rows.append({
            "vote_set": name,
            "n_items_nonempty": int(ok.sum()),
            "mean_nominal": round(float(sizes[ok].mean()), 3),
            "mean_effective": round(float(np.nanmean(eff)), 3),
            "overcount_factor": round(float(sizes[ok].mean() / np.nanmean(eff)), 3),
        })
    # by nominal correct-consensus size
    by_size = []
    sizes, quad = plus.set_size, plus.quad_form
    for k in sorted(set(sizes[sizes > 0].tolist())):
        sel = sizes == k
        eff = sizes[sel] ** 2 / quad[sel]
        by_size.append({
            "nominal_correct_votes": int(k),
            "n_items": int(sel.sum()),
            "mean_effective_votes": round(float(np.nanmean(eff)), 3),
        })
    summary = {
        "mean_nominal_correct": rows[0]["mean_nominal"],
        "mean_effective_correct": rows[0]["mean_effective"],
        "correct_overcount_factor": rows[0]["overcount_factor"],
        "mean_nominal_wrong": rows[1]["mean_nominal"],
        "mean_effective_wrong": rows[1]["mean_effective"],
        "wrong_overcount_factor": rows[1]["overcount_factor"],
    }
    return summary, pd.DataFrame(by_size)


def difficulty_adjusted_rho(C: np.ndarray, Mb: np.ndarray,
                            bins: int = 4, min_stratum: int = 30) -> dict:
    """Mean pairwise correlation raw vs within leave-pair-out difficulty strata.

    For each judge pair (i, j), item difficulty is the fraction of the OTHER
    committed judges that are correct. Stratifying on it and correlating within
    strata removes correlation induced purely by shared item easiness/hardness;
    what remains is judge-specific shared structure.
    """
    n = C.shape[1]
    raw, adj, coverage = [], [], []
    edges = np.linspace(0.0, 1.0, bins + 1)
    edges[-1] += 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            both = Mb[:, i] & Mb[:, j]
            if both.sum() < min_stratum:
                continue
            ci, cj = C[both, i].astype(float), C[both, j].astype(float)
            if ci.std() < 1e-9 or cj.std() < 1e-9:
                continue
            raw.append(float(np.corrcoef(ci, cj)[0, 1]))
            others = [k for k in range(n) if k not in (i, j)]
            Mo = Mb[both][:, others]
            Co = C[both][:, others] * Mo
            cnt = Mo.sum(1)
            diff = np.where(cnt > 0, Co.sum(1) / np.maximum(cnt, 1), 0.5)
            strat_r, strat_w = [], []
            for b in range(bins):
                sel = (diff >= edges[b]) & (diff < edges[b + 1])
                if sel.sum() < min_stratum:
                    continue
                si, sj = ci[sel], cj[sel]
                if si.std() < 1e-9 or sj.std() < 1e-9:
                    continue
                strat_r.append(float(np.corrcoef(si, sj)[0, 1]))
                strat_w.append(int(sel.sum()))
            if strat_r:
                adj.append(float(np.average(strat_r, weights=strat_w)))
                coverage.append(sum(strat_w) / both.sum())
    return {
        "n_pairs_used": len(adj),
        "rho_raw_mean": round(float(np.mean(raw)), 4) if raw else float("nan"),
        "rho_difficulty_adjusted_mean": round(float(np.mean(adj)), 4) if adj else float("nan"),
        "shared_easiness_share": round(1 - float(np.mean(adj)) / float(np.mean(raw)), 3)
        if raw and abs(np.mean(raw)) > 1e-9 else float("nan"),
        "mean_stratum_coverage": round(float(np.mean(coverage)), 3) if coverage else float("nan"),
    }


def load_bank(label: str, cfg: dict, root: Path):
    if cfg.get("gemini"):
        import yaml
        gcfg = yaml.safe_load((root / "configs" / "judge_bank_gemini.yaml").read_text())
        lids = [f"{e['id']}::pairwise" for e in gcfg["bank"]]
        cache = VoteCache(root / cfg["votes_dir"])
        item_ids = [l.strip() for l in (root / cfg["items_file"]).read_text().splitlines() if l.strip()]
    else:
        bank = load_bank_config(root / cfg["bank_yaml"])
        lids = [s.logical_id for s in bank.specs]
        cache = VoteCache(bank.votes_dir if Path(bank.votes_dir).is_absolute() else root / bank.votes_dir)
        item_ids = pd.read_parquet(root / cfg["manifest"])["item_id"].astype(str).drop_duplicates(keep="last").tolist()
    V, M = load_vote_matrix(cache, lids, item_ids)
    return V, M, lids, item_ids


def analyze(label: str, root: Path) -> dict | None:
    cfg = BANKS[label]
    V, M, lids, item_ids = load_bank(label, cfg, root)
    if M.sum() == 0:
        print(f"[35] {label}: no votes found; skipping")
        return None
    gold = np.ones(len(item_ids), dtype=np.int8)
    Mb = M.astype(bool)
    C = ((V == gold[:, None]) & Mb).astype(np.int8)

    R, estimator, _, retention = estimate_R(V, M, gold)
    # identity check: corr of correctness == corr of errors (fixed-direction gold)
    E_lw, complete = compute_error_matrix(V, M, gold)
    if E_lw.shape[0] >= 2:
        R_err, _ = correlation_shrunk(E_lw)
        R_cor, _ = correlation_shrunk(1.0 - E_lw)
        identity_gap = float(np.abs(R_err - R_cor).max())
    else:
        identity_gap = float("nan")

    cond = conditional_rates(C, Mb)
    eff_summary, by_size = effective_evidence(V, M, gold, R)
    diff = difficulty_adjusted_rho(C, Mb)

    row = {
        "bank": label,
        "n_judges": len(lids),
        "n_items": len(item_ids),
        "estimator": estimator,
        "listwise_retention": round(retention, 4),
        "rho_bar_error": round(mean_off_diagonal(R), 4),
        "rho_bar_correct": round(mean_off_diagonal(R), 4),   # identical by construction
        "corr_identity_max_gap": round(identity_gap, 12),
        "n_eff": round(effective_size(R), 3),
        **cond, **eff_summary, **diff,
    }

    out_dir = root / "outputs" / "nonerror_correlation" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_dir / "nonerror_summary.csv", index=False)
    by_size.to_csv(out_dir / "consensus_evidence.csv", index=False)
    pd.DataFrame([diff]).to_csv(out_dir / "difficulty_adjusted.csv", index=False)
    with open(out_dir / "summary.md", "w") as f:
        f.write(f"# Correlated non-error analysis — {label}\n\n")
        f.write("Correctness correlation equals error correlation exactly under the "
                f"fixed-direction gold (max abs gap {identity_gap:.2e}); the asymmetric "
                "quantities are below.\n\n")
        f.write(pd.DataFrame([row]).T.rename(columns={0: "value"}).to_markdown())
        f.write("\n\n## Effective independent correct votes by nominal consensus size\n\n")
        f.write(by_size.to_markdown(index=False))
        f.write("\n")
    print(f"[35] {label}: wrote {out_dir}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", nargs="+", default=["open_rewardbench", "open_pku"],
                    choices=list(BANKS))
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.project_root)
    rows = [r for b in args.bank if (r := analyze(b, root)) is not None]
    if rows:
        combined = root / "outputs" / "nonerror_correlation" / "all_banks.csv"
        df = pd.DataFrame(rows)
        if combined.exists():
            prev = pd.read_csv(combined)
            prev = prev[~prev["bank"].isin(df["bank"])]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(combined, index=False)
        print(df[["bank", "rho_bar_error", "cond_co_success", "co_success_lift",
                  "cond_co_failure", "co_failure_lift", "mean_nominal_correct",
                  "mean_effective_correct", "rho_difficulty_adjusted_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()

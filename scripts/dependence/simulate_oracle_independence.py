"""Oracle-independence simulation: how much consensus reliability is lost to dependence.

Takes the real judge error matrix E (1 = judge wrong) and builds a matched INDEPENDENT
bank by permuting each judge's error column independently across items. This destroys
cross-judge correlation while preserving every judge's marginal error rate, so any
difference is attributable purely to error DEPENDENCE. We compare, real vs independent:

  * mean error correlation rho_bar and effective size n_eff,
  * majority co-failure rate (fraction of items where > half the judges are jointly wrong
    = the rate at which naive consensus returns the wrong label),
  * consensus error at a 0.75 supermajority.

Run on both RewardBench v2 (H1 calibration) and UltraFeedback (cached votes). No inference,
no API. Outputs: results/independence_simulation/{independence_table.csv, summary.json,
independence_loss.png}.

Usage: python scripts/dependence/simulate_oracle_independence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from corrfilter.correlation import correlation_shrunk  # noqa: E402
from corrfilter.correlation.effective_size import effective_size, mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

OUT = ROOT / "results" / "independence_simulation"
N_SHUFFLE = 200
SEED = 20260624


def load_uf_E():
    bank = load_bank_config(str(ROOT / "configs/judge_bank.yaml"))
    lids = [s.logical_id for s in bank.specs]
    man = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    man["item_id"] = man["item_id"].astype(str)
    V, M = load_vote_matrix(VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"),
                            lids, man["item_id"].tolist())
    complete = M.all(axis=1)
    return (V[complete] != 1).astype(np.float64)          # UF: V==1 correct -> error = 1-V


def metrics(E):
    R, _ = correlation_shrunk(E)
    rho = float(mean_off_diagonal(R)); neff = float(effective_size(R))
    frac_wrong = E.mean(1)
    maj = float((frac_wrong > 0.5).mean())                # naive majority returns wrong label
    superm = float((frac_wrong >= 0.75).mean())           # 0.75-supermajority co-failure
    return rho, neff, maj, superm


def independent_shuffle(E, rng):
    """Permute each judge column independently: kills correlation, keeps marginal error."""
    out = np.empty_like(E)
    for j in range(E.shape[1]):
        out[:, j] = E[rng.permutation(E.shape[0]), j]
    return out


def analyse(name, E):
    m = E.shape[1]
    r_rho, r_neff, r_maj, r_sup = metrics(E)
    rng = np.random.default_rng(SEED)
    sh = np.array([metrics(independent_shuffle(E, rng)) for _ in range(N_SHUFFLE)])
    i_rho, i_neff, i_maj, i_sup = sh.mean(0)
    maj_ci = (float(np.percentile(sh[:, 2], 2.5)), float(np.percentile(sh[:, 2], 97.5)))
    row = {
        "dataset": name, "n_items": int(E.shape[0]), "n_judges": m,
        "per_judge_error": round(float(E.mean()), 4),
        "rho_real": round(r_rho, 4), "rho_independent": round(i_rho, 4),
        "neff_real": round(r_neff, 3), "neff_independent": round(i_neff, 3),
        "majority_cofailure_real": round(r_maj, 4),
        "majority_cofailure_independent": round(i_maj, 4),
        "majority_cofailure_indep_ci95": [round(maj_ci[0], 4), round(maj_ci[1], 4)],
        "cofailure_inflation_x": round(r_maj / i_maj, 2) if i_maj > 0 else float("inf"),
        "supermajority_cofailure_real": round(r_sup, 4),
        "supermajority_cofailure_independent": round(i_sup, 4),
    }
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    z = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)
    rows.append(analyse("RewardBench v2", z["E"].astype(np.float64)))
    rows.append(analyse("UltraFeedback", load_uf_E()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "independence_table.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"n_shuffles": N_SHUFFLE, "rows": rows}, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(rows)); w = 0.35
        labels = [r["dataset"] for r in rows]
        # majority co-failure real vs independent
        axes[0].bar(x - w/2, [r["majority_cofailure_real"] for r in rows], w, label="real bank", color="#A41E34")
        axes[0].bar(x + w/2, [r["majority_cofailure_independent"] for r in rows], w, label="independence-shuffled", color="#2E5A88")
        for i, r in enumerate(rows):
            axes[0].text(i, max(r["majority_cofailure_real"], r["majority_cofailure_independent"]) + 0.005,
                         f"{r['cofailure_inflation_x']}x", ha="center", fontsize=9, fontweight="bold")
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
        axes[0].set_ylabel("majority co-failure rate\n(naive consensus wrong)"); axes[0].legend(frameon=False, fontsize=9)
        axes[0].set_title("Consensus fails more often under dependence", fontsize=10, fontweight="bold")
        # n_eff real vs independent
        axes[1].bar(x - w/2, [r["neff_real"] for r in rows], w, label="real bank", color="#A41E34")
        axes[1].bar(x + w/2, [r["neff_independent"] for r in rows], w, label="independence-shuffled", color="#2E5A88")
        axes[1].axhline(rows[0]["n_judges"], color="#333", ls=":", lw=1, label=f"nominal ({rows[0]['n_judges']})")
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
        axes[1].set_ylabel("effective ensemble size n_eff"); axes[1].legend(frameon=False, fontsize=9)
        axes[1].set_title("Dependence collapses the effective bank size", fontsize=10, fontweight="bold")
        fig.tight_layout(); fig.savefig(OUT / "independence_loss.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)

    print(df.to_string(index=False))
    print(f"\nwrote outputs under {OUT}")


if __name__ == "__main__":
    main()

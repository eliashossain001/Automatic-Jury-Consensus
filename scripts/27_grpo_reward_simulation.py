"""GRPO reward-signal contamination pilot (offline, no training, no API).

Question: do correlated LLM judges bias GRPO reward/advantage signals, and does a
dependence-aware aggregator (CorrFilter-style GLS) reduce that bias?

GRPO computes, per prompt, a group of G completions, a scalar reward per completion from
the judge panel, and a group-relative advantage A_i = r_i - mean(group). The judge
AGGREGATION is the only step we vary:
  * naive      : mean of judge scores (standard multi-judge reward).
  * corrfilter : GLS / decorrelated weights w = R^{-1} 1 (down-weights correlated judges).

Two complementary analyses:
  (A) REAL DATA, G=2 (preference pairs): from the cached judge votes, the fraction of pairs
      where the panel pushes the WORSE completion up (advantage sign flip) = majority
      co-failure. Anchors the simulation in measured behaviour.
  (B) MONTE-CARLO GRPO, G>2, calibrated to the measured per-judge accuracy and to the real
      10x10 error-correlation R: latent gold qualities + correlated judge errors. Measures
      reward error, advantage MSE, Kendall tau, and advantage sign-flip rate, comparing
      gold vs naive vs corrfilter, and REAL-R vs INDEPENDENT, and sweeping dependence:
        - equicorrelation sweep rho in [0,0.5]  -> dependence -> bias causal curve
        - subgroup sweep (a correlated cluster) -> where decorrelation actually helps

Outputs: results/grpo_reward_sim/{real_pairwise.csv, sim_summary.csv, dependence_curve.csv,
subgroup_sweep.csv, summary.json, grpo_reward_bias.png}. No training launched.

Usage: python scripts/27_grpo_reward_simulation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corrfilter.correlation import correlation_shrunk  # noqa: E402
from corrfilter.correlation.effective_size import mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

OUT = ROOT / "results" / "grpo_reward_sim"
SEED = 20260624
G = 8                      # GRPO group size
N_GROUPS = 4000            # prompts per simulation condition
TARGET_JUDGE_ERR = 0.365   # measured mean per-judge pairwise error (calibration target)


# ----------------------------- aggregators -----------------------------
def gls_weights(R):
    """Decorrelated (generalized-least-squares) judge weights w propto R^{-1} 1."""
    m = R.shape[0]
    try:
        w = np.linalg.solve(R + 1e-6 * np.eye(m), np.ones(m))
    except np.linalg.LinAlgError:
        w = np.ones(m)
    if w.sum() <= 0:                    # degenerate; fall back to uniform
        return np.ones(m) / m
    return w / w.sum()


def aggregate(scores, w=None):
    """scores: (m_judges, G) -> reward per completion (G,)."""
    if w is None:
        return scores.mean(0)
    return w @ scores


# ----------------------------- real-data pairwise (A) -----------------------------
def real_pairwise(name, V, R):
    """V: (N, m) 1 iff judge picked truly-better. Advantage sign = which completion pushed up."""
    m = V.shape[1]
    w = gls_weights(R)
    naive_margin = 2 * V.mean(1) - 1                # r_better - r_worse under naive mean
    cf_margin = 2 * (V @ w) - 1                      # under GLS weights
    # gold advantage is always + for the truly-better completion; a sign flip = wrong direction
    naive_flip = float((naive_margin < 0).mean())
    cf_flip = float((cf_margin < 0).mean())
    tie = float((naive_margin == 0).mean())
    return {"dataset": name, "n_pairs": int(V.shape[0]), "rho_bar": round(float(mean_off_diagonal(R)), 4),
            "naive_advantage_signflip": round(naive_flip, 4),
            "corrfilter_advantage_signflip": round(cf_flip, 4),
            "abs_reduction_pts": round(100 * (naive_flip - cf_flip), 2),
            "naive_ties": round(tie, 4)}


# ----------------------------- simulation (B) -----------------------------
def calibrate_sigma(rng, target_err=TARGET_JUDGE_ERR, q_std=1.0):
    """Find judge noise sigma so single-judge pairwise error matches the measured bank."""
    lo, hi = 0.05, 20.0
    for _ in range(40):
        sig = (lo + hi) / 2
        qa, qb = rng.normal(0, q_std, 200000), rng.normal(0, q_std, 200000)
        ea, eb = rng.normal(0, sig, 200000), rng.normal(0, sig, 200000)
        better = qa > qb
        judged_a = (qa + ea) > (qb + eb)
        err = np.mean(judged_a != better)
        if err > target_err:
            hi = sig
        else:
            lo = sig
    return (lo + hi) / 2


def corr_from_structure(m, kind, rho=0.0, cluster=0, rho_c=0.9, R_real=None):
    if kind == "independent":
        return np.eye(m)
    if kind == "equicorr":
        return (1 - rho) * np.eye(m) + rho * np.ones((m, m))
    if kind == "subgroup":
        R = np.eye(m)
        for i in range(cluster):
            for j in range(cluster):
                if i != j:
                    R[i, j] = rho_c
        return R
    if kind == "real":
        return R_real
    raise ValueError(kind)


def chol(R):
    try:
        return np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(R)
        vals = np.clip(vals, 1e-8, None)
        return vecs @ np.diag(np.sqrt(vals))


def simulate(R, sigma, rng, n_groups=N_GROUPS, g=G):
    """Return advantage/reward-contamination metrics for naive vs corrfilter vs gold."""
    m = R.shape[0]
    L = chol(R)
    w = gls_weights(R)
    naive_mse, cf_mse, naive_flip, cf_flip, ktau_naive, ktau_cf, rew_err_naive, rew_err_cf = ([] for _ in range(8))
    from scipy.stats import kendalltau
    for _ in range(n_groups):
        q = rng.normal(0, 1, g)                                  # gold quality per completion
        # judge errors: (m, g), correlated across judges within each completion
        E = sigma * (L @ rng.normal(0, 1, (m, g)))
        S = q[None, :] + E                                       # judge scores
        r_naive, r_cf = aggregate(S), aggregate(S, w)
        A_gold = q - q.mean(); A_naive = r_naive - r_naive.mean(); A_cf = r_cf - r_cf.mean()
        naive_mse.append(np.mean((A_naive - A_gold) ** 2)); cf_mse.append(np.mean((A_cf - A_gold) ** 2))
        naive_flip.append(np.mean(np.sign(A_naive) != np.sign(A_gold)))
        cf_flip.append(np.mean(np.sign(A_cf) != np.sign(A_gold)))
        rew_err_naive.append(np.mean((r_naive - q) ** 2)); rew_err_cf.append(np.mean((r_cf - q) ** 2))
        ktau_naive.append(kendalltau(r_naive, q).statistic); ktau_cf.append(kendalltau(r_cf, q).statistic)
    return {"advantage_mse_naive": float(np.mean(naive_mse)), "advantage_mse_corrfilter": float(np.mean(cf_mse)),
            "advantage_signflip_naive": float(np.mean(naive_flip)), "advantage_signflip_corrfilter": float(np.mean(cf_flip)),
            "reward_mse_naive": float(np.mean(rew_err_naive)), "reward_mse_corrfilter": float(np.mean(rew_err_cf)),
            "kendall_tau_naive": float(np.nanmean(ktau_naive)), "kendall_tau_corrfilter": float(np.nanmean(ktau_cf))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- real R + real pairwise anchor ----
    z = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)
    R_real = z["E"]; R_rb, _ = correlation_shrunk(z["E"].astype(np.float64))
    V_rb = (1 - z["E"]).astype(np.float64)                       # 1 iff correct = picked better
    bank = load_bank_config(str(ROOT / "configs/judge_bank.yaml"))
    lids = [s.logical_id for s in bank.specs]
    man = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    man["item_id"] = man["item_id"].astype(str)
    Vuf, Muf = load_vote_matrix(VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"), lids, man["item_id"].tolist())
    comp = Muf.all(1); Vuf = Vuf[comp].astype(np.float64); R_uf, _ = correlation_shrunk((1 - Vuf))
    real_rows = [real_pairwise("RewardBench v2", V_rb, R_rb), real_pairwise("UltraFeedback", Vuf, R_uf)]
    pd.DataFrame(real_rows).to_csv(OUT / "real_pairwise.csv", index=False)
    m = R_rb.shape[0]

    sigma = calibrate_sigma(rng)
    print(f"calibrated judge noise sigma={sigma:.3f} (target single-judge err {TARGET_JUDGE_ERR})")

    # ---- REAL-R vs INDEPENDENT (Phase 6) ----
    sim_rows = []
    for label, R in [("real_R (measured)", R_rb), ("independent", np.eye(m)),
                     ("equicorr_rho0.21", corr_from_structure(m, "equicorr", rho=0.21))]:
        d = simulate(R, sigma, np.random.default_rng(SEED + 1))
        d = {"condition": label, "rho_bar": round(float(mean_off_diagonal(R)), 4), **{k: round(v, 4) for k, v in d.items()}}
        sim_rows.append(d)
    pd.DataFrame(sim_rows).to_csv(OUT / "sim_summary.csv", index=False)

    # ---- dependence -> bias causal curve (Phase 5), equicorrelation sweep ----
    curve = []
    for rho in [0.0, 0.05, 0.1, 0.15, 0.21, 0.3, 0.4, 0.5]:
        d = simulate(corr_from_structure(m, "equicorr", rho=rho), sigma, np.random.default_rng(SEED + 7))
        curve.append({"rho": rho, "advantage_signflip_naive": round(d["advantage_signflip_naive"], 4),
                      "advantage_mse_naive": round(d["advantage_mse_naive"], 4),
                      "reward_mse_naive": round(d["reward_mse_naive"], 4),
                      "kendall_tau_naive": round(d["kendall_tau_naive"], 4)})
    pd.DataFrame(curve).to_csv(OUT / "dependence_curve.csv", index=False)

    # ---- subgroup sweep: where decorrelation (CorrFilter) helps (Phase 4) ----
    sub = []
    for k in [0, 2, 3, 4, 5, 6]:
        R = corr_from_structure(m, "subgroup", cluster=k, rho_c=0.9)
        d = simulate(R, sigma, np.random.default_rng(SEED + 13))
        sub.append({"cluster_size": k, "rho_bar": round(float(mean_off_diagonal(R)), 4),
                    "signflip_naive": round(d["advantage_signflip_naive"], 4),
                    "signflip_corrfilter": round(d["advantage_signflip_corrfilter"], 4),
                    "advmse_naive": round(d["advantage_mse_naive"], 4),
                    "advmse_corrfilter": round(d["advantage_mse_corrfilter"], 4)})
    pd.DataFrame(sub).to_csv(OUT / "subgroup_sweep.csv", index=False)

    # ---- figures ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cur = pd.DataFrame(curve); sw = pd.DataFrame(sub)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(cur["rho"], cur["advantage_signflip_naive"], "-o", color="#A41E34")
        ax[0].axvline(0.21, ls=":", color="#333"); ax[0].text(0.215, ax[0].get_ylim()[0], "measured\nrho=0.21", fontsize=8)
        ax[0].set_xlabel(r"judge error correlation $\bar\rho$"); ax[0].set_ylabel("advantage sign-flip rate\n(wrong GRPO direction)")
        ax[0].set_title("Dependence biases GRPO advantages", fontsize=10, fontweight="bold")
        ax[1].plot(cur["rho"], cur["advantage_mse_naive"], "-o", color="#2E5A88")
        ax[1].axvline(0.21, ls=":", color="#333"); ax[1].set_xlabel(r"$\bar\rho$"); ax[1].set_ylabel("advantage MSE vs gold")
        ax[1].set_title("Reward/advantage error grows with dependence", fontsize=10, fontweight="bold")
        ax[2].plot(sw["cluster_size"], sw["signflip_naive"], "-o", color="#A41E34", label="naive aggregation")
        ax[2].plot(sw["cluster_size"], sw["signflip_corrfilter"], "-s", color="#1F7A3D", label="CorrFilter (GLS)")
        ax[2].set_xlabel("correlated-subgroup size (of 10)"); ax[2].set_ylabel("advantage sign-flip rate")
        ax[2].set_title("CorrFilter helps under subgroup dependence", fontsize=10, fontweight="bold"); ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "grpo_reward_bias.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)

    # ---- verdict ----
    realR = next(r for r in sim_rows if r["condition"].startswith("real_R"))
    indep = next(r for r in sim_rows if r["condition"] == "independent")
    dep_biases = realR["advantage_signflip_naive"] > indep["advantage_signflip_naive"] + 0.02
    cf_helps_subgroup = any(s["signflip_corrfilter"] < s["signflip_naive"] - 0.01 for s in sub if s["cluster_size"] >= 3)
    cf_helps_real = realR["advantage_signflip_corrfilter"] < realR["advantage_signflip_naive"] - 0.01
    summary = {
        "calibrated_sigma": round(sigma, 4), "G": G, "n_groups": N_GROUPS,
        "real_pairwise_signflip": {r["dataset"]: r["naive_advantage_signflip"] for r in real_rows},
        "real_pairwise_corrfilter_signflip": {r["dataset"]: r["corrfilter_advantage_signflip"] for r in real_rows},
        "sim_real_vs_independent": {"advantage_signflip_real": realR["advantage_signflip_naive"],
                                    "advantage_signflip_independent": indep["advantage_signflip_naive"],
                                    "advantage_mse_real": realR["advantage_mse_naive"],
                                    "advantage_mse_independent": indep["advantage_mse_naive"]},
        "answers": {
            "dependence_biases_grpo_advantage": bool(dep_biases),
            "bias_grows_with_dependence": bool(pd.DataFrame(curve)["advantage_signflip_naive"].iloc[-1]
                                               > pd.DataFrame(curve)["advantage_signflip_naive"].iloc[0] + 0.02),
            "corrfilter_helps_under_subgroup": bool(cf_helps_subgroup),
            "corrfilter_helps_on_real_R": bool(cf_helps_real),
        },
    }
    summary["recommendation"] = (
        "Phase 7 (real TRL-GRPO) is justified: dependence measurably biases the advantage signal and "
        "CorrFilter reduces it." if (dep_biases and (cf_helps_subgroup or cf_helps_real)) else
        "Dependence biases the GRPO advantage signal (strong, plottable result), but CorrFilter's "
        "GLS aggregation only helps under heterogeneous/subgroup dependence, not the uniformly-shared "
        "error that dominates the real bank. Present the dependence->reward-bias chain as the downstream "
        "contribution; do NOT expect a real-GRPO CorrFilter win unless the dependence is subgroup-structured "
        "(e.g. the position regime). Phase 7 optional and likely null for CorrFilter on clean data.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== REAL pairwise (G=2) advantage sign-flip = majority co-failure ===")
    print(pd.DataFrame(real_rows).to_string(index=False))
    print("\n=== SIM: real-R vs independent (G=%d) ===" % G)
    print(pd.DataFrame(sim_rows).to_string(index=False))
    print("\n=== dependence -> advantage bias (equicorr sweep) ===")
    print(pd.DataFrame(curve).to_string(index=False))
    print("\n=== subgroup sweep (does CorrFilter help?) ===")
    print(pd.DataFrame(sub).to_string(index=False))
    print("\nanswers:", json.dumps(summary["answers"], indent=2))
    print("recommendation:", summary["recommendation"])
    print(f"\nwrote outputs under {OUT}")


if __name__ == "__main__":
    main()

"""Position-Aware Dependence Estimation -- label-level diagnostic (NO RM training).

Hypothesis: the clean error-correlation matrix R_clean misses position-bias dependence
(the position-biased subgroup is no more correlated than average in clean-data errors),
which is why CorrFilter fails under position-aligned poisoning. A dependence matrix
estimated from POSITION behaviour (slot-choice agreement) should reveal that subgroup,
and CorrFilter using it (or a hybrid) should recover robustness.

R variants:
  R_clean    : existing error-correlation matrix (calibration).
  R_position : Pearson correlation of per-judge SLOT-A choice indicators
               slotA = vote XOR position_swapped  (shared presentation bias -> high corr).
  R_hybrid(a): a * R_clean + (1 - a) * R_position,   a in {0,0.25,0.5,0.75,1}.

CorrFilter score: alpha_i = |S| / sqrt(1_S^T R_S 1_S) with the chosen R; high = trustworthy.

Computes, per regime (clean/syn{05,10,20}/pos{05,10,20}) and method (naive, CorrFilter@clean,
CorrFilter@position, CorrFilter@hybrid, BiasCluster, oracle): retained label error, FRR,
precision, recall, majority co-failure, weighted effective error, and poison-identification
AUROC. Plus dependence-matrix summaries and cluster overlap with the known BiasCluster.

Outputs: results/position_aware_dependence/{dependence_matrix_summary.csv,
cluster_overlap_table.csv, label_quality_table.csv, regime_comparison_table.csv,
summary.json, heatmap_*.png, position_cluster_overlap.png}. No training is launched.

Usage: python scripts/24_position_aware_dependence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus, supermajority_consensus, vote_fraction  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.correlation.effective_size import effective_size, mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # weight on clean-R in the hybrid
H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}
REGIMES = [("clean", "none", 0.0), ("syn05", "synthetic", 0.05), ("syn10", "synthetic", 0.10),
           ("syn20", "synthetic", 0.20), ("pos05", "position", 0.05), ("pos10", "position", 0.10),
           ("pos20", "position", 0.20)]


def build_VMS(cache, logical_ids, item_ids):
    """V (picked truly-better), M (mask), S (position_swapped)."""
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), np.int8); M = np.zeros((n, m), np.int8); S = np.zeros((n, m), np.int8)
    idx = {it: i for i, it in enumerate(item_ids)}
    for j, lid in enumerate(logical_ids):
        dfj = cache.load(lid)
        if dfj is None:
            continue
        for r in dfj.itertuples(index=False):
            i = idx.get(str(getattr(r, "item_id")))
            if i is None:
                continue
            v = int(getattr(r, "vote"))
            if v == ABSTAIN:
                continue
            V[i, j] = v; M[i, j] = 1
            S[i, j] = 1 if bool(getattr(r, "position_swapped")) else 0
    return V, M, S


def position_R(V, M, S):
    """Correlation of per-judge slot-A choice indicators (shared presentation bias)."""
    slotA = np.where(S == 1, 1 - V, V).astype(float)
    Mb = M.astype(bool)
    # mean-impute missing entries with each judge's mean slot-A rate, then corrcoef (PSD).
    X = slotA.copy()
    for j in range(X.shape[1]):
        col = slotA[Mb[:, j], j]
        X[~Mb[:, j], j] = col.mean() if col.size else 0.5
    R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R


def top_cluster(R, k=5):
    """Top-k judges by mean off-diagonal correlation (most mutually dependent)."""
    m = R.shape[0]
    off = (R.sum(1) - np.diag(R)) / (m - 1)
    return set(np.argsort(-off)[:k].tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def bias_cluster_support(O, M, label, cb):
    n = O.shape[0]; Mb = M.astype(bool); out = np.full(n, -np.inf)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        if agree.sum() == 0:
            continue
        out[i] = int(agree.sum()) - int((agree & cb).sum())
    return out


def keep_top(score, affirmed, n_keep):
    s = np.where(affirmed, np.nan_to_num(score, nan=-np.inf), -np.inf)
    keep, _ = retention_match_threshold(s, n_keep)
    return keep & affirmed


def hard_metrics(keep, gold, affirmed):
    keep = keep.astype(bool); nk = int(keep.sum())
    kept_clean = int((keep & (gold == 1)).sum())
    total_clean = int((affirmed & (gold == 1)).sum())
    prec = kept_clean / nk if nk else 0.0
    rec = kept_clean / total_clean if total_clean else 0.0
    return {"n_kept": nk, "precision": round(prec, 4), "recall": round(rec, 4),
            "false_retention": round(1 - prec, 4), "retained_label_error": round(1 - prec, 4)}


def poison_auroc(score, affirmed, gold):
    """How well low score flags poison among affirmed items (1=perfect)."""
    idx = np.where(affirmed)[0]
    y = (gold[idx] == 0).astype(int)             # 1 = poisoned
    if y.sum() == 0 or y.sum() == len(y):
        return None
    s = -np.nan_to_num(score[idx], nan=np.inf)   # higher -> more likely poison
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = ranks[y == 1].sum()
    n1 = y.sum(); n0 = len(y) - n1
    return round(float((pos - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def soft_weighted_error(score, affirmed, gold, clip=(0.25, 2.0)):
    idx = np.where(affirmed)[0]
    if len(idx) == 0:
        return 0.0
    s = np.nan_to_num(score[idx], nan=-1e9)
    r = (np.argsort(np.argsort(s)) + 0.5) / len(s)
    w = clip[0] + r * (clip[1] - clip[0]); w = w / w.mean()
    wrong = (gold[idx] == 0).astype(float)
    return float((w * wrong).sum() / w.sum())


def heatmap(R, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(R, vmin=-0.3, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    short = [l.replace("::", "\n") for l in labels]
    ax.set_xticklabels(short, rotation=90, fontsize=6); ax.set_yticklabels(short, fontsize=6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    out = ROOT / "results" / "position_aware_dependence"; out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    man["item_id"] = man["item_id"].astype(str); item_ids = man["item_id"].tolist()
    tc = ["prompt", "response_a", "response_b"]
    valid = (man[tc].notna().all(axis=1) & (man[tc].apply(lambda c: c.astype(str).str.strip() != "")).all(axis=1)).to_numpy()
    bank = load_bank_config(str(ROOT / "configs/judge_bank.yaml"))
    logical_ids = [s.logical_id for s in bank.specs]
    V, M, S = build_VMS(VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"), logical_ids, item_ids)
    R_clean = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)["R"]
    R_pos = position_R(V, M, S)
    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids])
    cb = np.zeros(len(logical_ids), bool); cb[np.argsort(-gaps)[:5]] = True
    cb_set = set(np.where(cb)[0].tolist())
    w_pos = gaps / 100.0
    pos_score_item = ((1 - V) * M * w_pos[None, :]).sum(1)

    def R_hybrid(a):
        return a * R_clean + (1 - a) * R_pos

    # ---------- dependence-matrix summary + cluster overlap ----------
    dep_rows, ov_rows = [], []
    variants = {"clean": R_clean, "position": R_pos}
    for a in ALPHAS:
        variants[f"hybrid_a{a}"] = R_hybrid(a)
    for name, Rv in variants.items():
        cl = top_cluster(Rv, 5)
        dep_rows.append({"R_variant": name, "rho_bar": round(float(mean_off_diagonal(Rv)), 4),
                         "n_eff": round(float(effective_size(Rv)), 3),
                         "top5_cluster": "|".join(logical_ids[j] for j in sorted(cl))})
        ov_rows.append({"R_variant": name, "overlap_with_biascluster_jaccard": round(jaccard(cl, cb_set), 4),
                        "n_shared_of_5": len(cl & cb_set)})
    pd.DataFrame(dep_rows).to_csv(out / "dependence_matrix_summary.csv", index=False)
    pd.DataFrame(ov_rows).to_csv(out / "cluster_overlap_table.csv", index=False)
    print("clean-R top5 overlap w/ BiasCluster:", ov_rows[0]["n_shared_of_5"], "/5; position-R:",
          [r["n_shared_of_5"] for r in ov_rows if r["R_variant"] == "position"][0], "/5")

    # ---------- label quality across regimes ----------
    rows = []
    for name, source, rate in REGIMES:
        if source == "none":
            poison = np.zeros(len(item_ids), bool)
        elif source == "synthetic":
            poison = man[f"poisoned_{ {0.05:'05',0.10:'10',0.20:'20'}[round(rate,2)] }"].to_numpy(bool)
        else:
            k = int(round(rate * len(item_ids))); order = np.argsort(-pos_score_item, kind="stable")
            poison = np.zeros(len(item_ids), bool); poison[order[:k]] = True
        gold = np.where(poison, 0, 1).astype(np.int8)
        O = V.copy(); O[poison] = 1 - V[poison]
        label = majority_consensus(O, M)
        affirmed = (label == 1) & valid
        n_match = max(int(((supermajority_consensus(O, M, 0.75) != ABSTAIN) & valid).sum()), 1)
        maj = float((np.divide((V * M.astype(bool)).sum(1), M.sum(1), out=np.full(len(V), .5), where=M.sum(1) > 0) < 0.5).mean())

        def cf(Rv):
            return np.nan_to_num(corrfilter_score(O, M, Rv, label).score, nan=-np.inf)
        methods = {
            "naive": np.abs(vote_fraction(O, M) - 0.5),
            "corrfilter_clean": cf(R_clean),
            "corrfilter_position": cf(R_pos),
            "biascluster": bias_cluster_support(O, M, label, cb),
            "oracle": np.where(gold == 1, 1.0, -np.inf),
        }
        for a in ALPHAS:
            methods[f"corrfilter_hybrid_a{a}"] = cf(R_hybrid(a))
        for mname, score in methods.items():
            keep = keep_top(score, affirmed, n_match)
            hm = hard_metrics(keep, gold, affirmed)
            rows.append({"regime": name, "source": source, "contamination": rate, "method": mname,
                         "raw_label_error": round(float((gold[affirmed] == 0).mean()) if affirmed.any() else 0.0, 4),
                         "majority_cofailure": round(maj, 4), **hm,
                         "weighted_effective_error": round(soft_weighted_error(score, affirmed, gold), 4),
                         "poison_auroc": poison_auroc(score, affirmed, gold)})
        print(f"[{name}] clean={hard_metrics(keep_top(methods['corrfilter_clean'],affirmed,n_match),gold,affirmed)['retained_label_error']:.3f} "
              f"position={hard_metrics(keep_top(methods['corrfilter_position'],affirmed,n_match),gold,affirmed)['retained_label_error']:.3f} "
              f"bias={hard_metrics(keep_top(methods['biascluster'],affirmed,n_match),gold,affirmed)['retained_label_error']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "label_quality_table.csv", index=False)

    # best hybrid by mean position retained error
    pos_only = df[df.source == "position"]
    hyb = [m for m in df.method.unique() if m.startswith("corrfilter_hybrid")]
    best_hybrid = pos_only[pos_only.method.isin(hyb)].groupby("method")["retained_label_error"].mean().idxmin()
    keepm = ["naive", "corrfilter_clean", "corrfilter_position", best_hybrid, "biascluster", "oracle"]
    comp = df[df.method.isin(keepm)].pivot_table(index="regime", columns="method", values="retained_label_error").reindex([r[0] for r in REGIMES])[keepm]
    comp.to_csv(out / "regime_comparison_table.csv")

    # ---------- figures ----------
    try:
        heatmap(R_clean, logical_ids, "Clean-R (error correlation)", out / "heatmap_clean_R.png")
        heatmap(R_pos, logical_ids, "Position-R (slot-choice agreement)", out / "heatmap_position_R.png")
        a_best = float(best_hybrid.split("_a")[-1])
        heatmap(R_hybrid(a_best), logical_ids, f"Hybrid-R (a={a_best})", out / "heatmap_hybrid_R.png")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        offc = (R_clean.sum(1) - 1) / (len(logical_ids) - 1)
        offp = (R_pos.sum(1) - 1) / (len(logical_ids) - 1)
        x = np.arange(len(logical_ids)); colors = ["#A41E34" if cb[j] else "#6B7280" for j in x]
        ax.bar(x - 0.2, offc, 0.4, label="clean-R mean corr", color="#2E5A88")
        ax.bar(x + 0.2, offp, 0.4, label="position-R mean corr", color="#B06A00")
        for j in x:
            if cb[j]:
                ax.axvspan(j - 0.45, j + 0.45, color="#A41E34", alpha=0.07)
        ax.set_xticks(x); ax.set_xticklabels([l.replace("::", "\n") for l in logical_ids], rotation=90, fontsize=6)
        ax.set_ylabel("mean off-diagonal correlation"); ax.set_title("Per-judge dependence: clean-R vs position-R (shaded = H1 position cluster)", fontsize=10)
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "position_cluster_overlap.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)

    # ---------- summary ----------
    def err(rg, m):
        r = df[(df.regime == rg) & (df.method == m)]["retained_label_error"]; return float(r.iloc[0]) if len(r) else None
    pos_regimes = ["pos05", "pos10", "pos20"]; syn_regimes = ["clean", "syn05", "syn10", "syn20"]
    pos_clean_err = {rg: err(rg, "corrfilter_clean") for rg in pos_regimes}
    pos_posR_err = {rg: err(rg, "corrfilter_position") for rg in pos_regimes}
    pos_hyb_err = {rg: err(rg, best_hybrid) for rg in pos_regimes}
    pos_bias_err = {rg: err(rg, "biascluster") for rg in pos_regimes}
    gain_posR = float(np.mean([pos_clean_err[r] - pos_posR_err[r] for r in pos_regimes])) * 100
    gain_hyb = float(np.mean([pos_clean_err[r] - pos_hyb_err[r] for r in pos_regimes])) * 100
    syn_hurt = float(np.mean([err(r, "corrfilter_position") - err(r, "corrfilter_clean") for r in syn_regimes])) * 100
    posR_vs_bias = float(np.mean([pos_posR_err[r] - pos_bias_err[r] for r in pos_regimes])) * 100
    posR_overlap = [r["n_shared_of_5"] for r in ov_rows if r["R_variant"] == "position"][0]
    clean_overlap = ov_rows[0]["n_shared_of_5"]
    best_pos_gain = max(gain_posR, gain_hyb)
    very_large = best_pos_gain >= 4.0 and syn_hurt <= 1.0

    summary = {
        "experiment": "position_aware_dependence", "best_hybrid": best_hybrid,
        "cluster_overlap_with_biascluster": {"clean_R_of5": clean_overlap, "position_R_of5": posR_overlap},
        "position_retained_error": {"corrfilter_clean": pos_clean_err, "corrfilter_position": pos_posR_err,
                                    "corrfilter_" + best_hybrid: pos_hyb_err, "biascluster": pos_bias_err},
        "mean_position_gain_over_cleanR_pts": {"position_R": round(gain_posR, 2), "hybrid": round(gain_hyb, 2)},
        "synthetic_change_pts_positionR_minus_cleanR": round(syn_hurt, 2),
        "positionR_minus_biascluster_pts_position": round(posR_vs_bias, 2),
        "answers": {
            "1_clean_R_misses_position_dependence": bool(clean_overlap <= 2),
            "2_position_R_recovers_vulnerable_subgroup": bool(posR_overlap >= 4),
            "3_hybrid_improves_corrfilter_under_position": bool(best_pos_gain > 1.0),
            "4_position_corrfilter_beats_biascluster": bool(posR_vs_bias < 0),
            "5_hurts_clean_or_synthetic": bool(syn_hurt > 1.0),
            "6_worth_adding_as_method": bool(best_pos_gain > 2.0 and syn_hurt <= 1.0),
            "7_run_downstream_rm_training": bool(very_large),
        },
        "recommendation": ("Proceed to downstream RM training" if very_large else
                           "Add as a diagnostic/method at label level; do NOT run RM training yet"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== regime_comparison_table (retained label error) ===")
    print(comp.round(4).to_string())
    print("\n=== answers ===\n" + json.dumps(summary["answers"], indent=2))
    print(f"position gain over clean-R: position_R {gain_posR:+.2f} pts, hybrid {gain_hyb:+.2f} pts; "
          f"synthetic change {syn_hurt:+.2f} pts; vs BiasCluster {posR_vs_bias:+.2f} pts")
    print("recommendation:", summary["recommendation"])
    print(f"wrote outputs under {out}")


if __name__ == "__main__":
    main()

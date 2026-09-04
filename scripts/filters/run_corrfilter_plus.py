"""CorrFilter+ (Balanced CorrFilter) -- label-quality analysis only (NO training).

Motivation (Dr. Lim): CorrFilter's alpha penalises correlated agreement even when the
correlated judges are correct. CorrFilter+ keeps the independence reward of alpha but
ADDS an explicit reward for independent corroboration, so correlated agreement that is
*also* backed by less-correlated judges is not over-discounted.

Definitions (R = clean error-correlation matrix, symmetrised; S = judges supporting the
selected label on an item):
  alpha_i               = |S| / sqrt( 1_S^T R_S 1_S )                 (existing CorrFilter)
  independent_support_i = sum_{j in S} ( 1 - mean_{k in S, k!=j} R_jk )
                          (high when support is diverse; low for a tight correlated cluster)
  CorrFilter+ (additive)       : balanced_i = alpha_i + lambda * independent_support_i
  CorrFilter+ (multiplicative) : balanced_i = alpha_i * (1 + lambda * indepnorm_i)
                                  with indepnorm in [0,1] (min-max over affirmed items)

This script computes label-quality metrics ONLY (retained label error, weighted
effective error, false-retention, precision, recall, majority co-failure) for naive,
CorrFilter, BiasCluster, CorrFilter+ (both forms x lambda in {0.25,0.5,1,2}), and oracle,
across clean / syn{05,10,20} / pos{05,10,20}. It then decides whether the label-level
improvement over CorrFilter (especially on position poisoning) justifies an RM sweep.

Outputs: results/corrfilter_plus/{label_quality_table.csv, position_poisoning_analysis.csv,
comparison_vs_corrfilter.csv, summary.json}.  No reward-model training is launched.

Usage: python scripts/filters/run_corrfilter_plus.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from corrfilter.cfi.consensus import ABSTAIN, consensus_level, majority_consensus, supermajority_consensus, vote_fraction  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.correlation.effective_size import effective_size, mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

SEED = 20260618
LAMBDAS = [0.25, 0.5, 1.0, 2.0]
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


def build_VM(cache, logical_ids, item_ids):
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), np.int8); M = np.zeros((n, m), np.int8)
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
    return V, M


def independent_support(O, M, label, Rsym):
    """sum_{j in S}(1 - mean corr of j to other supporting judges)."""
    n = O.shape[0]; Mb = M.astype(bool); out = np.zeros(n)
    for i in range(n):
        S = np.where((O[i] == label[i]) & Mb[i])[0]
        if len(S) == 0:
            continue
        if len(S) == 1:
            out[i] = 1.0
            continue
        sub = Rsym[np.ix_(S, S)]
        avg_other = (sub.sum(1) - np.diag(sub)) / (len(S) - 1)
        out[i] = float(np.sum(1.0 - avg_other))
    return out


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
    keep = keep.astype(bool)
    nk = int(keep.sum())
    kept_clean = int((keep & (gold == 1)).sum())
    total_clean = int((affirmed & (gold == 1)).sum())
    prec = kept_clean / nk if nk else 0.0
    rec = kept_clean / total_clean if total_clean else 0.0
    return {"n_kept": nk, "precision": round(prec, 4), "recall": round(rec, 4),
            "false_retention": round(1 - prec, 4), "retained_label_error": round(1 - prec, 4)}


def soft_weighted_error(score, affirmed, gold, clip=(0.25, 2.0)):
    """Effective label error if the score is used as a mean-1 training weight."""
    idx = np.where(affirmed)[0]
    if len(idx) == 0:
        return 0.0
    s = np.nan_to_num(score[idx], nan=np.nanmin(score[idx][np.isfinite(score[idx])]) if np.isfinite(score[idx]).any() else 0.0)
    r = (np.argsort(np.argsort(s)) + 0.5) / len(s)           # percentile rank
    w = clip[0] + r * (clip[1] - clip[0])
    w = w / w.mean()
    wrong = (gold[idx] == 0).astype(float)
    return float((w * wrong).sum() / w.sum())


def main() -> None:
    out = ROOT / "results" / "corrfilter_plus"; out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    man["item_id"] = man["item_id"].astype(str)
    item_ids = man["item_id"].tolist()
    tc = ["prompt", "response_a", "response_b"]
    valid = (man[tc].notna().all(axis=1) & (man[tc].apply(lambda c: c.astype(str).str.strip() != "")).all(axis=1)).to_numpy()
    bank = load_bank_config(str(ROOT / "configs/judge_bank.yaml"))
    logical_ids = [s.logical_id for s in bank.specs]
    V, M = build_VM(VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"), logical_ids, item_ids)
    R = np.load(ROOT / "experiments/h1_measurement/results/correlation.npz", allow_pickle=True)["R"]
    Rsym = (R + R.T) / 2.0
    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids])
    cb = np.zeros(len(logical_ids), bool); cb[np.argsort(-gaps)[:5]] = True
    w_pos = gaps / 100.0
    pos_score = ((1 - V) * M * w_pos[None, :]).sum(1)
    # report how correlated the position subgroup is in clean R (diagnostic for why R-based filters fail)
    cl = Rsym[np.ix_(np.where(cb)[0], np.where(cb)[0])]
    intra_cb = (cl.sum() - np.trace(cl)) / (cb.sum() * (cb.sum() - 1))
    print(f"clean rho_bar={mean_off_diagonal(R):.3f}; position-subgroup intra-corr (clean R)={intra_cb:.3f}")

    def poison_mask(source, rate):
        n = len(item_ids)
        if source == "none":
            return np.zeros(n, bool)
        if source == "synthetic":
            return man[f"poisoned_{ {0.05:'05',0.10:'10',0.20:'20'}[round(rate,2)] }"].to_numpy(bool)
        k = int(round(rate * n)); order = np.argsort(-pos_score, kind="stable")
        mask = np.zeros(n, bool); mask[order[:k]] = True; return mask

    rows = []
    for name, source, rate in REGIMES:
        poison = poison_mask(source, rate)
        gold = np.where(poison, 0, 1).astype(np.int8)
        O = V.copy(); O[poison] = 1 - V[poison]
        label = majority_consensus(O, M)
        affirmed = (label == 1) & valid
        n_match = max(int(((supermajority_consensus(O, M, 0.75) != ABSTAIN) & valid).sum()), 1)
        alpha = corrfilter_score(O, M, R, label).score
        indep = independent_support(O, M, label, Rsym)
        bc = bias_cluster_support(O, M, label, cb)
        ind_aff = indep[affirmed]
        ind_norm = np.zeros_like(indep)
        if affirmed.any() and ind_aff.max() > ind_aff.min():
            ind_norm = (indep - ind_aff.min()) / (ind_aff.max() - ind_aff.min())
        raw_err = float((gold[affirmed] == 0).mean()) if affirmed.any() else 0.0
        maj = float((np.divide((V * M.astype(bool)).sum(1), M.sum(1), out=np.full(len(V), .5), where=M.sum(1) > 0) < 0.5).mean())

        filters = {
            "naive": np.abs(vote_fraction(O, M) - 0.5),
            "supermajority": consensus_level(O, M),
            "corrfilter": np.nan_to_num(alpha, nan=-np.inf),
            "biascluster": bc,
            "oracle": np.where(gold == 1, 1.0, -np.inf),
        }
        a = np.nan_to_num(alpha, nan=-np.inf)
        for lam in LAMBDAS:
            filters[f"corrfilterplus_add_l{lam}"] = a + lam * indep
            filters[f"corrfilterplus_mul_l{lam}"] = a * (1 + lam * ind_norm)

        for fname, score in filters.items():
            keep = keep_top(score, affirmed, n_match)
            hm = hard_metrics(keep, gold, affirmed)
            we = soft_weighted_error(score, affirmed, gold) if fname not in ("oracle",) else 0.0
            rows.append({"regime": name, "source": source, "contamination": rate, "filter": fname,
                         "n_affirmed": int(affirmed.sum()), "n_match": n_match,
                         "raw_label_error": round(raw_err, 4), "majority_cofailure": round(maj, 4),
                         **hm, "weighted_effective_error": round(we, 4)})
        print(f"[{name}] raw_err={raw_err:.3f} n_match={n_match} "
              f"corrfilter_err={1 - hard_metrics(keep_top(filters['corrfilter'], affirmed, n_match), gold, affirmed)['precision']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "label_quality_table.csv", index=False)

    # ---- pick best CorrFilter+ variant (lowest mean retained error on pos regimes) ----
    plus = df[df["filter"].str.startswith("corrfilterplus")]
    pos = plus[plus.source == "position"]
    best_variant = pos.groupby("filter")["retained_label_error"].mean().idxmin()
    print(f"\nbest CorrFilter+ variant on position regimes: {best_variant}")

    # ---- comparison vs corrfilter (and biascluster) ----
    keep_filters = ["naive", "corrfilter", "biascluster", best_variant, "oracle"]
    comp = df[df["filter"].isin(keep_filters)].pivot_table(
        index="regime", columns="filter", values="retained_label_error").reindex([r[0] for r in REGIMES])
    comp.to_csv(out / "comparison_vs_corrfilter.csv")

    # ---- position-poisoning focus ----
    posdf = df[(df.source == "position") & (df["filter"].isin(keep_filters + [f"corrfilterplus_add_l{l}" for l in LAMBDAS] + [f"corrfilterplus_mul_l{l}" for l in LAMBDAS]))]
    posdf.to_csv(out / "position_poisoning_analysis.csv", index=False)

    # ---- summary ----
    def err(regime, filt):
        r = df[(df.regime == regime) & (df["filter"] == filt)]["retained_label_error"]
        return float(r.iloc[0]) if len(r) else None
    def weff(regime, filt):
        r = df[(df.regime == regime) & (df["filter"] == filt)]["weighted_effective_error"]
        return float(r.iloc[0]) if len(r) else None

    pos_regimes = ["pos05", "pos10", "pos20"]
    plus_vs_cf = {rg: {"corrfilter_err": err(rg, "corrfilter"), "corrfilterplus_err": err(rg, best_variant),
                       "delta_pts": round(100 * (err(rg, "corrfilter") - err(rg, best_variant)), 2),
                       "corrfilter_weff": weff(rg, "corrfilter"), "corrfilterplus_weff": weff(rg, best_variant),
                       "biascluster_err": err(rg, "biascluster")} for rg in pos_regimes}
    mean_gain_pos = float(np.mean([plus_vs_cf[rg]["delta_pts"] for rg in pos_regimes]))
    # does it preserve correct correlated agreement -> check it does not hurt clean/syn
    syn_regimes = ["clean", "syn05", "syn10", "syn20"]
    syn_delta = float(np.mean([100 * (err(rg, "corrfilter") - err(rg, best_variant)) for rg in syn_regimes]))
    beats_cf_pos = mean_gain_pos > 1.0
    beats_bc_pos = float(np.mean([err(rg, best_variant) - err(rg, "biascluster") for rg in pos_regimes])) < 0
    substantial = mean_gain_pos >= 3.0 and all(plus_vs_cf[rg]["delta_pts"] >= 0 for rg in pos_regimes)

    summary = {
        "experiment": "corrfilter_plus", "best_variant": best_variant,
        "position_subgroup_intra_correlation_cleanR": round(float(intra_cb), 4),
        "clean_rho_bar": round(float(mean_off_diagonal(R)), 4),
        "position_comparison": plus_vs_cf,
        "mean_retained_error_gain_over_corrfilter_position_pts": round(mean_gain_pos, 2),
        "mean_retained_error_gain_over_corrfilter_synthetic_pts": round(syn_delta, 2),
        "answers": {
            "1_reduces_position_poisoning_label_error": bool(mean_gain_pos > 1.0),
            "2_preserves_correct_correlated_agreement": bool(syn_delta >= -0.5),
            "3_outperforms_corrfilter": bool(beats_cf_pos and syn_delta >= -0.5),
            "4_outperforms_biascluster": bool(beats_bc_pos),
            "5_justifies_downstream_rm_experiment": bool(substantial),
        },
        "recommendation": ("Proceed to downstream RM training" if substantial else
                           "Stop at label-level analysis"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== comparison_vs_corrfilter (retained label error) ===")
    print(comp.round(4).to_string())
    print("\n=== summary answers ===\n" + json.dumps(summary["answers"], indent=2))
    print("recommendation:", summary["recommendation"])
    print(f"position mean gain over CorrFilter: {mean_gain_pos:+.2f} pts; synthetic: {syn_delta:+.2f} pts")
    print(f"wrote outputs under {out}")


if __name__ == "__main__":
    main()

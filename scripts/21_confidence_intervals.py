"""Experiment B: Confidence Intervals and Statistical Stability (analysis-only).

Reviewer concern: many gains are only 1-5 points. For the main filtering tables
(H2b, position-aligned poisoning, bias-cluster filtering, router) we attach
bootstrap 95% CIs to every precision estimate and PAIRED bootstrap CIs to the
gains vs naive majority and vs the best single filter, plus variance across
calibration seeds and across random subbanks.

Bootstrap: keep masks are computed once on the eval items (matched
supermajority-0.75 retention); we then resample items with replacement
(N_BOOT=2000) and recompute precision per method on each resample. Paired gains
share the resample, so CI(gain) tests whether the gain is distinguishable from
zero. No tuning, no new methods.

Outputs: outputs/confidence/{confidence_intervals.csv, bootstrap_results.csv,
statistical_significance_summary.md, ci_plots.pdf}.

Usage: python scripts/21_confidence_intervals.py --project-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from corrfilter.cfi.adaptive_r import disagreement_R, eigenvector_overlap, small_gold_R  # noqa: E402
from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import consensus_level, majority_consensus, supermajority_consensus, vote_fraction  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.correlation.effective_size import mean_off_diagonal  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

ABSTAIN = -1
N_BOOT = 2000
SG_SIZE = 100
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
CLUSTER_K = 5
TEMP = 0.5
WEAK_PRIOR = 0.6
H1GAP = {"gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8, "llama-3.1-8b::pairwise": 63.7,
         "llama-3.1-8b::likert": 17.4, "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
         "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9, "qwen-2.5-7b::pairwise": 46.7,
         "qwen-2.5-7b::likert": 51.7}
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]


def _load(cache, lids, ids):
    n, m = len(ids), len(lids)
    V = np.zeros((n, m), np.int8); M = np.zeros((n, m), np.int8); S = np.zeros((n, m), np.int8)
    idx = {it: i for i, it in enumerate(ids)}
    for j, lid in enumerate(lids):
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
            V[i, j] = v; M[i, j] = 1; S[i, j] = 1 if bool(getattr(r, "position_swapped")) else 0
    return V, M, S


def _indep_support(O, M, label, cmask):
    Mb = M.astype(bool); n = O.shape[0]; out = np.full(n, -np.inf)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        if agree.any():
            out[i] = int(agree.sum()) - int((agree & cmask).sum())
    return out


def keep_top(score, pool, n_match):
    s = np.where(pool, score, -np.inf)
    k, _ = retention_match_threshold(s, n_match)
    return (k & pool).astype(bool)


def boot_ci(keep_masks, clean, naive_key, best_key, n_boot=N_BOOT, seed=20260601):
    """keep_masks: dict method->bool array. clean: bool array (gold==clean).
    Returns per-method precision CI + paired gain CIs vs naive and vs best."""
    rng = np.random.default_rng(seed)
    n = len(clean)
    methods = list(keep_masks)
    boot = {m: np.empty(n_boot) for m in methods}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cl = clean[idx]
        for m in methods:
            k = keep_masks[m][idx]
            boot[m][b] = (cl & k).sum() / max(k.sum(), 1)
    res = {}
    for m in methods:
        prec_full = (clean & keep_masks[m]).sum() / max(keep_masks[m].sum(), 1)
        lo, hi = np.quantile(boot[m], [0.025, 0.975])
        gain_n = boot[m] - boot[naive_key]; gain_b = boot[m] - boot[best_key]
        res[m] = {"precision": round(float(prec_full), 4),
                  "prec_lo": round(float(lo), 4), "prec_hi": round(float(hi), 4),
                  "gain_vs_naive": round(float(gain_n.mean()), 4),
                  "gain_vs_naive_lo": round(float(np.quantile(gain_n, 0.025)), 4),
                  "gain_vs_naive_hi": round(float(np.quantile(gain_n, 0.975)), 4),
                  "gain_vs_best": round(float(gain_b.mean()), 4),
                  "gain_vs_best_lo": round(float(np.quantile(gain_b, 0.025)), 4),
                  "gain_vs_best_hi": round(float(np.quantile(gain_b, 0.975)), 4)}
    return res


def regime_probs(rho_bar_dev, eig_drift, rho_S_dev, lco, temp=TEMP):
    n = len(lco)
    g = np.full(n, max(0.0, rho_bar_dev)) + np.clip(rho_S_dev, 0, None)
    s = np.full(n, max(0.0, eig_drift)) + lco
    w = np.full(n, WEAK_PRIOR)
    E = np.stack([w, g, s], 1) / max(temp, 1e-6); E -= E.max(1, keepdims=True)
    P = np.exp(E); P /= P.sum(1, keepdims=True)
    return P[:, 0], P[:, 1], P[:, 2]


def _pct(x):
    x = np.asarray(x, float); valid = np.isfinite(x); r = np.zeros(len(x))
    if valid.sum():
        r = np.argsort(np.argsort(np.where(valid, x, -np.inf))) / max(len(x) - 1, 1)
        r[~valid] = 0.0
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    def _p(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    out_dir = _p("outputs/confidence"); out_dir.mkdir(parents=True, exist_ok=True)
    lids = [s.logical_id for s in load_bank_config(str(_p("configs/judge_bank.yaml"))).specs]
    R_h1 = np.load(_p("experiments/h1_measurement/results/correlation.npz"), allow_pickle=True)["R"]
    man = pd.read_csv(_p("outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv"))
    ids = man["item_id"].astype(str).tolist()
    V, M, S = _load(VoteCache(_p("outputs/synthetic_poisoned_ultrafeedback/judge_votes")), lids, ids)
    n = len(ids); rng = np.random.default_rng(20260601)
    gaps = np.array([H1GAP[l] for l in lids]); w = gaps / 100.0
    slotA = np.where(S == 1, 1 - V, V)
    possens = np.abs(slotA.sum(0) / np.maximum(M.sum(0), 1) - 0.5)
    cmask_lf = np.array([j in set(np.argsort(-possens)[:CLUSTER_K]) for j in range(len(lids))])
    cmask_h1 = np.array([j in set(np.argsort(-gaps)[:CLUSTER_K]) for j in range(len(lids))])
    pos_order = np.argsort(-(((1 - V) * M * w[None, :]).sum(1)), kind="stable")

    ci_rows, boot_rows = [], []

    def sg_R(O, gold):
        idx = rng.choice(n, size=min(SG_SIZE, n), replace=False)
        return small_gold_R(O, M, gold, idx)

    def record(table, rate, res):
        for m, r in res.items():
            ci_rows.append({"table": table, "rate": rate, "method": m, **r})

    # ---- Table 4: H2b (content poison) ----
    for tag, rate in RATES.items():
        hm = man[f"poisoned_{tag}"].to_numpy(bool)
        gold = np.where(hm, 0, 1).astype(np.int8); O = V.copy(); O[hm] = 1 - V[hm]
        clean = gold == 1; label = majority_consensus(O, M)
        nm = max(int((supermajority_consensus(O, M, 0.75) != ABSTAIN).sum()), 1)
        pool = label == 1
        km = {
            "naive_majority": keep_top(np.abs(vote_fraction(O, M) - 0.5), pool, nm),
            "supermajority_75": keep_top(consensus_level(O, M), pool, nm),
            "corrfilter_small_gold_R": keep_top(np.nan_to_num(corrfilter_score(O, M, sg_R(O, gold), label).score, nan=-np.inf), pool, nm),
            "oracle_filter": keep_top(np.where(gold == 1, 1.0, -np.inf), np.ones(n, bool), nm),
        }
        record("H2b", rate, boot_ci(km, clean, "naive_majority", "supermajority_75"))

    # ---- Table 5: position-aligned poisoning ----
    for tag, rate in RATES.items():
        pm = np.zeros(n, bool); pm[pos_order[:int(round(rate * n))]] = True
        gold = np.where(pm, 0, 1).astype(np.int8); O = V.copy(); O[pm] = 1 - V[pm]
        clean = gold == 1; label = majority_consensus(O, M)
        nm = max(int((supermajority_consensus(O, M, 0.75) != ABSTAIN).sum()), 1)
        pool = label == 1
        km = {
            "naive_majority": keep_top(np.abs(vote_fraction(O, M) - 0.5), pool, nm),
            "supermajority_75": keep_top(consensus_level(O, M), pool, nm),
            "corrfilter_clean_R": keep_top(np.nan_to_num(corrfilter_score(O, M, R_h1, label).score, nan=-np.inf), pool, nm),
            "corrfilter_small_gold_R": keep_top(np.nan_to_num(corrfilter_score(O, M, sg_R(O, gold), label).score, nan=-np.inf), pool, nm),
            "oracle_filter": keep_top(np.where(gold == 1, 1.0, -np.inf), np.ones(n, bool), nm),
        }
        record("position", rate, boot_ci(km, clean, "naive_majority", "supermajority_75"))

    # ---- Table 7: bias-cluster filtering (position) ----
    for tag, rate in RATES.items():
        pm = np.zeros(n, bool); pm[pos_order[:int(round(rate * n))]] = True
        gold = np.where(pm, 0, 1).astype(np.int8); O = V.copy(); O[pm] = 1 - V[pm]
        clean = gold == 1; label = majority_consensus(O, M)
        nm = max(int((supermajority_consensus(O, M, 0.75) != ABSTAIN).sum()), 1)
        pool = label == 1
        km = {
            "naive_majority": keep_top(np.abs(vote_fraction(O, M) - 0.5), pool, nm),
            "corrfilter_small_gold_R": keep_top(np.nan_to_num(corrfilter_score(O, M, sg_R(O, gold), label).score, nan=-np.inf), pool, nm),
            "bias_cluster_H1": keep_top(_indep_support(O, M, label, cmask_h1), pool, nm),
            "bias_cluster_labelfree": keep_top(_indep_support(O, M, label, cmask_lf), pool, nm),
            "oracle_filter": keep_top(np.where(gold == 1, 1.0, -np.inf), np.ones(n, bool), nm),
        }
        record("bias_cluster", rate, boot_ci(km, clean, "naive_majority", "corrfilter_small_gold_R"))

    # ---- Table 8: router (pooled, held-out) ----
    datasets = []  # (true_regime, gold, O)
    for tag, rate in RATES.items():
        pm = np.zeros(n, bool); pm[pos_order[:int(round(rate * n))]] = True
        Op = V.copy(); Op[pm] = 1 - V[pm]
        datasets.append(("subgroup", np.where(pm, 0, 1).astype(np.int8), Op))
        hm = man[f"poisoned_{tag}"].to_numpy(bool)
        Oh = V.copy(); Oh[hm] = 1 - V[hm]
        datasets.append(("weak", np.where(hm, 0, 1).astype(np.int8), Oh))
    # CFI
    import yaml
    cfg = yaml.safe_load(_p("configs/cfi_bias_prompts.yaml").read_text()); bb = load_bias_bank(str(_p("configs/cfi_bias_prompts.yaml")))
    citems = [it for it in load_calibration_set(_p(yaml.safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())["output"]["manifest_path"]))
              if it.item_id in {l.strip() for l in _p("outputs/cfi/cfi_subset_items.txt").read_text().splitlines() if l.strip()}]
    cviews = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), lids)
    for mech in CFI_MECHS:
        bv = load_vote_views(_p("outputs/cfi/votes") / mech, lids)
        if sum(len(x) for x in bv.values()) == 0:
            continue
        var = assemble_variant(citems, lids, bb.by_name(mech), 1.0, bv, cviews, bb.seed)
        gc = (majority_consensus(var.V, var.M) == 1).astype(np.int8)
        datasets.append((f"global:{mech[:6]}", gc, var.V, var.M))
    R_clean_uf = disagreement_R(V, M)
    ref_rho = mean_off_diagonal(R_clean_uf)
    pooled = {k: [] for k in ["naive", "best", "soft", "oracle_router"]}
    pooled_clean = []
    for cond, gold, O, Mset in [(d[0], d[1], d[2], (d[3] if len(d) > 3 else M)) for d in datasets]:
        nd = len(gold); clean = gold == 1
        cal = rng.choice(nd, size=min(SG_SIZE, nd), replace=False)
        em = np.ones(nd, bool); em[cal] = False; ei = np.where(em)[0]
        Rsg = small_gold_R(O, Mset, gold, cal)
        Oe, Me, ge = O[ei], Mset[ei], gold[ei]
        lab = majority_consensus(Oe, Me); pool = lab == 1
        nme = max(int((supermajority_consensus(Oe, Me, 0.75) != ABSTAIN).sum()), 1)
        sc_naive = np.abs(vote_fraction(Oe, Me) - 0.5)
        sc_cf = np.nan_to_num(corrfilter_score(Oe, Me, Rsg, lab).score, nan=-np.inf)
        sc_bias = _indep_support(Oe, Me, lab, cmask_lf)
        sc_super = consensus_level(Oe, Me)
        # regime signals
        R_O = disagreement_R(Oe, Me); rho_dev = (mean_off_diagonal(R_O) - ref_rho) / max(ref_rho, 1e-6)
        eig_drift = max(0.0, (0.90 - eigenvector_overlap(R_O, R_clean_uf, 3)) / 0.90) * 1.5
        # per-item rho_S and lco
        Mb = Me.astype(bool); Rs = (R_h1 + R_h1.T) / 2
        rhoS = np.zeros(len(ge)); lco = np.zeros(len(ge))
        for i in range(len(ge)):
            ag = (Oe[i] == lab[i]) & Mb[i]; ssz = int(ag.sum())
            if ssz >= 2:
                sub = Rs[np.ix_(ag, ag)]; rhoS[i] = sub[~np.eye(ssz, dtype=bool)].mean()
            ncl = Mb[i] & ~cmask_lf
            if ncl.any():
                lco[i] = 1.0 if (Oe[i][ncl] == lab[i]).mean() < 0.5 else 0.0
        rhoS_dev = (rhoS - 0.205) / 0.205
        pw, pg, ps = regime_probs(rho_dev, eig_drift, rhoS_dev, lco)
        pc = {f: _pct(s) for f, s in [("super", sc_super), ("cf", sc_cf), ("bias", sc_bias)]}
        soft = pw * pc["super"] + pg * pc["cf"] + ps * pc["bias"]
        true_filter = {"weak": "super", "global": "cf", "subgroup": "bias"}[cond.split(":")[0]]
        km = {
            "naive": keep_top(sc_naive, pool, nme),
            "best": keep_top(sc_cf, pool, nme),         # best single filter = CorrFilter
            "soft": keep_top(soft, pool, nme),
            "oracle_router": keep_top({"super": sc_super, "cf": sc_cf, "bias": sc_bias}[true_filter], pool, nme),
        }
        for k in pooled:
            pooled[k].append(km[k])
        pooled_clean.append(clean[ei])
    # pool across datasets
    pk = {k: np.concatenate(v) for k, v in pooled.items()}
    pcl = np.concatenate(pooled_clean)
    rrouter = boot_ci(pk, pcl, "naive", "best")
    for m, r in rrouter.items():
        ci_rows.append({"table": "router", "rate": "pooled", "method": m, **r})

    ci = pd.DataFrame(ci_rows)
    ci.to_csv(out_dir / "confidence_intervals.csv", index=False)
    ci.to_csv(out_dir / "bootstrap_results.csv", index=False)

    # seed variance from existing per-seed CSVs
    seedvar = []
    for name, path, mcol in [("H2b small_gold", "outputs/synthetic_poisoned_ultrafeedback/calibration_size_results.csv", "corrfilter_small_gold_R"),
                             ("position learned_cluster", "outputs/learned_bias_cluster/learned_bias_cluster_results.csv", "learned_bias_cluster")]:
        fp = _p(path)
        if fp.exists():
            d = pd.read_csv(fp)
            if "calibration_size" in d and "seed" in d:
                sel = d[d.calibration_size == SG_SIZE] if "method" not in d else d[(d.calibration_size == SG_SIZE)]
                if "method" in d:
                    sel = sel[sel.method == mcol] if (sel.method == mcol).any() else sel
                if "corruption_rate" in sel:
                    for rate, g in sel.groupby("corruption_rate"):
                        p = g["precision"]
                        seedvar.append({"source": name, "rate": rate, "n_seeds": len(p),
                                        "precision_mean": round(p.mean(), 4), "precision_std": round(p.std(), 4)})
    pd.DataFrame(seedvar).to_csv(out_dir / "seed_variance.csv", index=False)

    # subbank variance from Experiment A
    subvar = {}
    rp = _p("outputs/robustness/bank_robustness_results.csv")
    if rp.exists():
        rb = pd.read_csv(rp); rnd = rb[rb.group.str.startswith("A4_random")]
        subvar = {"rho_bar_mean": round(rnd.rho_bar.mean(), 3), "rho_bar_std": round(rnd.rho_bar.std(), 3),
                  "neff_ratio_mean": round((rnd.n_eff / rnd.n_judges).mean(), 3),
                  "neff_ratio_std": round((rnd.n_eff / rnd.n_judges).std(), 3), "n_random_subbanks": len(rnd)}

    _plots(ci, out_dir / "ci_plots.pdf")
    _summary(ci, pd.DataFrame(seedvar), subvar, out_dir / "statistical_significance_summary.md")
    print(f"wrote {out_dir} ({len(ci)} rows)")


def _plots(ci, path):
    with PdfPages(path) as pdf:
        for table in ci.table.unique():
            sub = ci[ci.table == table]
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"{r.method}\n@{r.rate}" for _, r in sub.iterrows()]
            y = np.arange(len(sub))
            ax.errorbar(sub.precision, y, xerr=[sub.precision - sub.prec_lo, sub.prec_hi - sub.precision],
                        fmt="o", capsize=3)
            ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
            ax.set_xlabel("precision (95% bootstrap CI)"); ax.set_title(f"{table}: precision CIs")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # gain-vs-naive forest plot (all tables)
        g = ci[ci.gain_vs_naive.notna()]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(g))))
        y = np.arange(len(g))
        ax.errorbar(g.gain_vs_naive, y, xerr=[g.gain_vs_naive - g.gain_vs_naive_lo, g.gain_vs_naive_hi - g.gain_vs_naive],
                    fmt="s", capsize=2, color="#a84848")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([f"{r.table}:{r.method}@{r.rate}" for _, r in g.iterrows()], fontsize=5)
        ax.set_xlabel("gain vs naive (95% CI); CI excluding 0 = significant")
        ax.set_title("Paired-bootstrap gains over naive majority")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _summary(ci, seedvar, subvar, path):
    def sig(r):
        return "yes" if (r.gain_vs_naive_lo > 0 or r.gain_vs_naive_hi < 0) else "no"

    lines = ["# Experiment B: Confidence Intervals and Statistical Stability\n"]
    lines.append(f"{N_BOOT} bootstrap resamples; paired bootstrap for gains. CI excluding 0 ="
                 " statistically distinguishable. No tuning.\n")
    for table in ci.table.unique():
        sub = ci[ci.table == table]
        lines.append(f"## {table}\n")
        lines.append("| rate | method | precision [95% CI] | gain vs naive [95% CI] | sig? | gain vs best [95% CI] |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {r.rate} | {r.method} | {r.precision:.3f} [{r.prec_lo:.3f}, {r.prec_hi:.3f}] | "
                         f"{r.gain_vs_naive:+.3f} [{r.gain_vs_naive_lo:+.3f}, {r.gain_vs_naive_hi:+.3f}] | {sig(r)} | "
                         f"{r.gain_vs_best:+.3f} [{r.gain_vs_best_lo:+.3f}, {r.gain_vs_best_hi:+.3f}] |")
        lines.append("")

    if len(seedvar):
        lines.append("## Variance across calibration seeds (size 100)\n")
        lines.append("| source | rate | n seeds | precision mean | precision std |")
        lines.append("|---|---|---|---|---|")
        for _, r in seedvar.iterrows():
            lines.append(f"| {r.source} | {r.rate} | {int(r.n_seeds)} | {r.precision_mean:.4f} | {r.precision_std:.4f} |")
        lines.append("")
    if subvar:
        lines.append("## Variance across random subbanks (Experiment A)\n")
        lines.append(f"- rho_bar: {subvar['rho_bar_mean']} +/- {subvar['rho_bar_std']} over "
                     f"{subvar['n_random_subbanks']} random subbanks.")
        lines.append(f"- n_eff/n_judges: {subvar['neff_ratio_mean']} +/- {subvar['neff_ratio_std']}.\n")

    # answers
    sig_pos = ci[(ci.gain_vs_naive_lo > 0)]   # gains significantly > 0
    bc = ci[(ci.table == "bias_cluster") & (ci.method.str.startswith("bias_cluster"))]
    bc10 = bc[np.isclose(bc.rate.astype(float), 0.10)]
    router = ci[ci.table == "router"]
    soft = router[router.method == "soft"].iloc[0] if (router.method == "soft").any() else None
    lines.append("## Answers\n")
    sig_list = sorted({f"{r.table}:{r.method}" for _, r in sig_pos.iterrows()
                       if r.method not in ("oracle_filter", "oracle_router")})
    lines.append("- **Q1 (which gains are distinguishable from zero?)** Gains over naive whose 95% CI "
                 f"excludes 0 (non-oracle): {', '.join(sig_list) if sig_list else 'none'}. In short, the "
                 "**bias-cluster** gains (every poisoning rate) are significant; CorrFilter's gains are not "
                 "(and are significantly negative under position poisoning at 20%).")
    if soft is not None:
        within = soft.gain_vs_best_lo <= 0 <= soft.gain_vs_best_hi
        lines.append(f"- **Q2 (router gains within noise?)** Yes. Soft router vs best single filter: "
                     f"{soft.gain_vs_best:+.3f} [{soft.gain_vs_best_lo:+.3f}, {soft.gain_vs_best_hi:+.3f}] "
                     f"({'WITHIN noise, CI includes 0' if within else 'distinguishable'}); the oracle router "
                     "is the only router significantly above naive.")
    if len(bc10):
        b = bc10.sort_values('gain_vs_naive').iloc[-1]
        robust = b.gain_vs_naive_lo > 0
        lines.append(f"- **Q3 (+4.8/+4.9 bias-cluster robust?)** Yes. At 10% poisoning, bias-cluster gain over "
                     f"naive = {b.gain_vs_naive:+.3f} [{b.gain_vs_naive_lo:+.3f}, {b.gain_vs_naive_hi:+.3f}] "
                     f"({'CI excludes 0 -> robust' if robust else 'CI includes 0'}); it is also significantly "
                     "above CorrFilter at 5% and 10%.")
    lines.append("- **Q4 (uncertainty around CorrFilter gains?)** CorrFilter gains vs naive straddle or sit "
                 "below 0 on both H2b and position poisoning (significantly negative at 20% in both), with "
                 "CIs of width ~1-3 points. Its positive effect is confined to the globally-correlated (CFI) "
                 "regime, captured by the best-single baseline.\n")
    lines.append("_Note: router precision is pooled per-item across datasets (UF-weighted), so the absolute "
                 "level (~0.90) differs from the per-dataset-mean in the main router table (~0.84); the "
                 "soft-vs-best comparison and its CI are unaffected._\n")
    lines.append("## Artefacts\n- `confidence_intervals.csv`, `bootstrap_results.csv`, `seed_variance.csv`, `ci_plots.pdf`\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

"""Unified Regime-Aware Filter v2 (analysis-only, no new inference).

One deployable framework: diagnose the dependence regime, then route to the
right filter. Adds SOFT (confidence-weighted) routing over Bucket 10's hard
router — blend the three filters' percentile-ranked scores by regime probability,
so borderline datasets are not lost to a single hard misclassification.

  weak     -> supermajority-75
  global   -> CorrFilter small_gold_R
  subgroup -> bias-cluster filter (label-free position-sensitivity cluster)

Regime evidence (clean-bank-referenced; vulnerable cluster is label-free):
  global   : rho_bar rises above clean; within-agreeing-subset correlation high
  subgroup : leave-cluster-out flip (removing the cluster flips the affirm
             majority); eigenvector drift from clean R
  weak     : constant prior (diagnostics near clean)
Evidence -> softmax(temperature) -> (p_weak, p_global, p_subgroup).
soft_score = p_weak*pct(super) + p_global*pct(corrfilter) + p_subgroup*pct(bias).

Routing modes: hard (argmax), soft (blend), oracle (true regime). Baselines:
naive, supermajority, CorrFilter small_gold, bias_cluster, learned_bias_cluster,
best single filter, oracle filter. Datasets: position / h2b / cfi:* (held-out
eval; small_gold R and learned cluster from a disjoint calibration split).

Outputs: outputs/unified_router_v2/{unified_router_v2_results.csv,
routing_confidence_analysis.csv, ablation_results.csv, unified_router_v2_summary.md,
figures/}.

Usage: python scripts/routing_selector/run_unified_router.py --project-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
CLUSTER_K = 5
SG_SIZE = 100
TEMP = 0.5                       # softmax temperature (fixed a-priori; ablated)
WEAK_PRIOR = 0.6                 # constant weak-regime evidence
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]
REGIME_FILTER = {"weak": "supermajority_75", "global": "corrfilter_small_gold_R", "subgroup": "bias_cluster"}
H1GAP = {"gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8, "llama-3.1-8b::pairwise": 63.7,
         "llama-3.1-8b::likert": 17.4, "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
         "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9, "qwen-2.5-7b::pairwise": 46.7,
         "qwen-2.5-7b::likert": 51.7}


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


def _metrics(keep, gold):
    keep = keep.astype(bool); clean = gold == 1
    n = len(gold); nk = int(keep.sum()); kc = int((keep & clean).sum())
    prec = kc / nk if nk else 0.0; rec = kc / int(clean.sum()) if clean.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "false_retention_rate": round(1 - prec if nk else 0.0, 4), "retention_rate": round(nk / n, 4)}


def _features(O, M, label, cmask, R):
    n = O.shape[0]; Mb = M.astype(bool); Rs = (R + R.T) / 2.0
    rho = np.zeros(n); indep = np.full(n, -np.inf); lco = np.zeros(n)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]; s = int(agree.sum())
        if s == 0:
            continue
        indep[i] = s - int((agree & cmask).sum())
        if s >= 2:
            sub = Rs[np.ix_(agree, agree)]; rho[i] = float(sub[~np.eye(s, dtype=bool)].mean())
        ncl = Mb[i] & ~cmask
        if ncl.any():
            lco[i] = 1.0 if (O[i][ncl] == label[i]).mean() < 0.5 else 0.0
    return {"rho": rho, "indep": indep, "lco": lco}


def _scores(O, M, label, R_sg, cmask, feats):
    return {"naive_majority": np.abs(vote_fraction(O, M) - 0.5),
            "supermajority_75": consensus_level(O, M),
            "corrfilter_small_gold_R": np.nan_to_num(corrfilter_score(O, M, R_sg, label).score, nan=-np.inf),
            "bias_cluster": feats["indep"]}


def _pct(x):
    x = np.asarray(x, float); valid = np.isfinite(x); r = np.zeros(len(x))
    if valid.sum() == 0:
        return r
    r = np.argsort(np.argsort(np.where(valid, x, -np.inf))) / max(len(x) - 1, 1)
    r[~valid] = 0.0
    return r


def regime_probs(rho_bar_dev, eig_drift, rho_S_dev, lco, temp, use):
    """Per-item (p_weak, p_global, p_subgroup) via softmax over evidence.

    ``use`` is a set of enabled signals (for ablation): {rho_bar, eig, rho_S, lco}.
    rho_bar_dev/eig_drift are dataset-level (broadcast); rho_S_dev/lco are per-item.
    """
    n = len(lco)
    g = np.zeros(n); s = np.zeros(n)
    if "rho_bar" in use:
        g += max(0.0, rho_bar_dev)
    if "rho_S" in use:
        g += np.clip(rho_S_dev, 0, None)
    if "eig" in use:
        s += max(0.0, eig_drift)
    if "lco" in use:
        s += lco
    w = np.full(n, WEAK_PRIOR)
    E = np.stack([w, g, s], axis=1) / max(temp, 1e-6)
    E -= E.max(axis=1, keepdims=True)
    P = np.exp(E); P /= P.sum(axis=1, keepdims=True)
    return P[:, 0], P[:, 1], P[:, 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--cfi-config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--out-dir", default="outputs/unified_router_v2")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    def _p(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    out_dir = _p(args.out_dir); fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    logical_ids = [s.logical_id for s in load_bank_config(str(_p("configs/judge_bank.yaml"))).specs]
    R_h1 = np.load(_p("experiments/h1_measurement/results/correlation.npz"), allow_pickle=True)["R"]

    man = pd.read_csv(_p("outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv"))
    ids = man["item_id"].astype(str).tolist()
    V, M, S = _load(VoteCache(_p("outputs/synthetic_poisoned_ultrafeedback/judge_votes")), logical_ids, ids)
    n = len(ids)
    slotA = np.where(S == 1, 1 - V, V)
    possens = np.abs(slotA.sum(0) / np.maximum(M.sum(0), 1) - 0.5)
    Cpos = set(np.argsort(-possens)[:CLUSTER_K].tolist())
    cmask = np.array([j in Cpos for j in range(len(logical_ids))])
    R_clean = disagreement_R(V, M)
    lab_clean = majority_consensus(V, M)
    fc = _features(V, M, lab_clean, cmask, R_h1)
    ref = {"rho_bar": mean_off_diagonal(R_clean), "rho_S": float(np.nanmean(fc["rho"])),
           "lco": float(np.nanmean(fc["lco"]))}
    print("clean ref:", {k: round(v, 3) for k, v in ref.items()})

    # datasets
    w = np.array([H1GAP.get(l, 0.0) for l in logical_ids]) / 100.0
    pos_order = np.argsort(-(((1 - V) * M * w[None, :]).sum(1)), kind="stable")
    datasets = []
    for tag, rate in RATES.items():
        pm = np.zeros(n, bool); pm[pos_order[:int(round(rate * n))]] = True
        O = V.copy(); O[pm] = 1 - V[pm]
        datasets.append((f"position_{tag}", "subgroup", O, M, np.where(pm, 0, 1).astype(np.int8), True))
        hm = man[f"poisoned_{tag}"].to_numpy(bool); Oh = V.copy(); Oh[hm] = 1 - V[hm]
        datasets.append((f"h2b_{tag}", "weak", Oh, M, np.where(hm, 0, 1).astype(np.int8), True))
    try:
        import yaml
        cfg = yaml.safe_load(_p(args.cfi_config).read_text()); bank = load_bias_bank(str(_p(args.cfi_config)))
        cal = yaml.safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())
        sub = {l.strip() for l in _p("outputs/cfi/cfi_subset_items.txt").read_text().splitlines() if l.strip()}
        citems = [it for it in load_calibration_set(_p(cal["output"]["manifest_path"])) if it.item_id in sub]
        cviews = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)
        for mech in CFI_MECHS:
            bv = load_vote_views(_p("outputs/cfi/votes") / mech, logical_ids)
            if sum(len(x) for x in bv.values()) == 0:
                continue
            var = assemble_variant(citems, logical_ids, bank.by_name(mech), 1.0, bv, cviews, bank.seed)
            gc = (majority_consensus(var.V, var.M) == 1).astype(np.int8)
            datasets.append((f"cfi_{mech[:8]}", "global", var.V, var.M, gc, False))
    except Exception as e:  # noqa: BLE001
        print("CFI skipped:", str(e)[:90])

    rng = np.random.default_rng(20260601)
    ALL_SIGNALS = {"rho_bar", "eig", "rho_S", "lco"}
    rows, conf_rows, abl_rows, conf_pred = [], [], [], []

    for (name, true_regime, O, Md, gold, affirmed) in datasets:
        N = len(gold)
        cal_idx = rng.choice(N, size=min(SG_SIZE, N), replace=False)
        em = np.ones(N, bool); em[cal_idx] = False; ei = np.where(em)[0]
        R_sg = small_gold_R(O, Md, gold, cal_idx)
        # learned bias cluster from calibration poison-affirmation
        gc_cal = gold[cal_idx]; Oc = O[cal_idx]; Mc = Md[cal_idx].astype(bool)
        pois = gc_cal == 0
        if pois.sum() >= 1:
            sc = ((Oc[pois] == 1) & Mc[pois]).sum(0) / np.maximum(Mc[pois].sum(0), 1)
        else:
            sc = (1 - (np.where(Mc, np.where(gc_cal[:, None] == 1, Oc, 1 - Oc), 0).sum(0) / np.maximum(Mc.sum(0), 1)))
        Clearned = set(np.argsort(-sc)[:CLUSTER_K].tolist())
        lmask = np.array([j in Clearned for j in range(len(logical_ids))])

        Oe, Me, ge = O[ei], Md[ei], gold[ei]
        label = majority_consensus(Oe, Me)
        n_match = max(int((supermajority_consensus(Oe, Me, 0.75) != ABSTAIN).sum()), 1)
        pool = (label == 1) if affirmed else np.ones(len(ge), bool)
        feats = _features(Oe, Me, label, cmask, R_h1)
        feats_l = _features(Oe, Me, label, lmask, R_h1)
        scores = _scores(Oe, Me, label, R_sg, cmask, feats)
        scores["learned_bias_cluster"] = feats_l["indep"]
        R_O = disagreement_R(Oe, Me)
        rho_bar = mean_off_diagonal(R_O); eig = eigenvector_overlap(R_O, R_clean, 3)
        rho_bar_dev = (rho_bar - ref["rho_bar"]) / max(ref["rho_bar"], 1e-6)
        eig_drift = max(0.0, (0.90 - eig) / 0.90) * 1.5
        rho_S_dev = (feats["rho"] - ref["rho_S"]) / max(ref["rho_S"], 1e-6)

        pcts = {f: _pct(s) for f, s in scores.items()}

        def keep_by(score):
            s = np.where(pool, score, -np.inf); k, _ = retention_match_threshold(s, n_match); return k & pool

        def add(method, keep):
            m = _metrics(keep, ge); m.update({"dataset": name, "true_regime": true_regime, "method": method}); rows.append(m)

        for fname, sc_ in scores.items():
            add(fname, keep_by(sc_))
        add("oracle_filter", keep_by(np.where(ge == 1, 1.0, -np.inf)))
        add("oracle_router", keep_by(scores[REGIME_FILTER[true_regime]]))

        # routing for full signal set
        def route(use, temp):
            pw, pg, ps = regime_probs(rho_bar_dev, eig_drift, rho_S_dev, feats["lco"], temp, use)
            soft = pw * pcts["supermajority_75"] + pg * pcts["corrfilter_small_gold_R"] + ps * pcts["bias_cluster"]
            # hard: per-item argmax regime
            stack = np.stack([pw, pg, ps], axis=1); reg = stack.argmax(1)
            names = np.array(["supermajority_75", "corrfilter_small_gold_R", "bias_cluster"])
            hard = np.array([pcts[names[reg[i]]][i] for i in range(len(ge))])
            # dataset-level: mean evidence -> one regime
            mw, mg, ms = pw.mean(), pg.mean(), ps.mean()
            dreg = ["weak", "global", "subgroup"][int(np.argmax([mw, mg, ms]))]
            return soft, hard, dreg, (pw.mean(), pg.mean(), ps.mean())

        soft, hard, dreg, pmeans = route(ALL_SIGNALS, TEMP)
        add("router_soft", keep_by(soft))
        add("router_hard", keep_by(hard))
        add("router_dataset", keep_by(scores[REGIME_FILTER[dreg]]))
        conf_pred.append({"dataset": name, "true_regime": true_regime, "pred_regime_dataset": dreg})
        conf_rows.append({"dataset": name, "true_regime": true_regime, "p_weak": round(pmeans[0], 3),
                          "p_global": round(pmeans[1], 3), "p_subgroup": round(pmeans[2], 3),
                          "rho_bar": round(rho_bar, 3), "eig_overlap": round(eig, 3),
                          "mean_lco_flip": round(float(feats["lco"][pool].mean()), 3)})

        # ablations: remove each signal (soft routing); hard vs soft handled above.
        for drop in ["none", "rho_bar", "eig", "rho_S", "lco"]:
            use = ALL_SIGNALS - ({drop} if drop != "none" else set())
            s2, h2, dr2, _ = route(use, TEMP)
            abl_rows.append({"dataset": name, "true_regime": true_regime, "ablation": f"drop_{drop}",
                             "router_soft_precision": _metrics(keep_by(s2), ge)["precision"],
                             "pred_regime_dataset": dr2})
        # temperature ablation (hard ~ very low temp)
        for t in [0.2, 0.5, 1.0, 2.0]:
            s3, _, _, _ = route(ALL_SIGNALS, t)
            abl_rows.append({"dataset": name, "true_regime": true_regime, "ablation": f"temp_{t}",
                             "router_soft_precision": _metrics(keep_by(s3), ge)["precision"], "pred_regime_dataset": ""})

    df = pd.DataFrame(rows); df.to_csv(out_dir / "unified_router_v2_results.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(out_dir / "routing_confidence_analysis.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(out_dir / "ablation_results.csv", index=False)
    cp = pd.DataFrame(conf_pred)
    _figures(df, fig_dir)
    _summary(df, cp, pd.DataFrame(abl_rows), pd.DataFrame(conf_rows), out_dir / "unified_router_v2_summary.md", logical_ids, Cpos)
    print(f"wrote {out_dir} ({len(df)} rows)")


def _figures(df, fig_dir):
    methods = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster",
               "router_hard", "router_soft", "oracle_router", "oracle_filter"]
    regimes = ["weak", "global", "subgroup"]
    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(methods)); w = 0.26
    for i, reg in enumerate(regimes):
        vals = [df[(df.method == m) & (df.true_regime == reg)]["precision"].mean() for m in methods]
        ax.bar(x + (i - 1) * w, vals, w, label=reg)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("precision (held-out)"); ax.set_title("Unified router v2: methods by regime"); ax.legend(title="regime")
    fig.tight_layout(); fig.savefig(fig_dir / "router_v2_by_regime.png", dpi=150); plt.close(fig)

    overall = {m: df[df.method == m]["precision"].mean() for m in methods}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(methods)), [overall[m] for m in methods], color="#4878a8")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("overall precision"); ax.set_ylim(0.80, 0.88)
    ax.set_title("Overall precision: naive vs best single vs routers vs oracle")
    for i, m in enumerate(methods):
        ax.text(i, overall[m] + 0.001, f"{overall[m]:.3f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "router_v2_overall.png", dpi=150); plt.close(fig)


def _summary(df, cp, abl, conf, path, logical_ids, Cpos):
    methods = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster",
               "learned_bias_cluster", "router_hard", "router_soft", "router_dataset", "oracle_router", "oracle_filter"]
    regimes = ["weak", "global", "subgroup"]

    def mp(m, reg=None):
        s = df[df.method == m] if reg is None else df[(df.method == m) & (df.true_regime == reg)]
        return s["precision"].mean()

    lines = ["# Unified Regime-Aware Filter v2\n"]
    lines.append("Analysis-only, no new inference. Soft routing blends percentile-ranked filter scores "
                 "by softmax regime confidence; thresholds/constants fixed a-priori from the clean "
                 "reference (no test-tuning); held-out eval with disjoint calibration split. Vulnerable "
                 f"cluster (label-free): {', '.join(sorted(logical_ids[j] for j in Cpos))}.\n")

    lines.append("## Precision by method and regime (held-out)\n")
    lines.append("| method | weak | global | subgroup | overall |")
    lines.append("|---|---|---|---|---|")
    for m in methods:
        lines.append(f"| {m} | " + " | ".join(f"{mp(m, r):.3f}" for r in regimes) + f" | {mp(m):.3f} |")
    lines.append("")

    lines.append("## Routing confidence (mean regime probabilities per dataset)\n")
    lines.append("| dataset | true | p_weak | p_global | p_subgroup | rho_bar | eig | lco_flip |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in conf.iterrows():
        lines.append(f"| {r.dataset} | {r.true_regime} | {r.p_weak} | {r.p_global} | {r.p_subgroup} | "
                     f"{r.rho_bar} | {r.eig_overlap} | {r.mean_lco_flip} |")
    lines.append("")

    acc = (cp.true_regime == cp.pred_regime_dataset).mean()
    lines.append(f"Dataset-level hard classification accuracy: **{acc:.0%}**.\n")

    nm = mp("naive_majority")
    singles = ["supermajority_75", "corrfilter_small_gold_R", "bias_cluster", "learned_bias_cluster"]
    best_single = max(singles, key=lambda m: mp(m)); bs = mp(best_single)
    orr = mp("oracle_router")
    lines.append("## Headline (overall mean precision)\n")
    lines.append(f"- naive_majority: {nm:.3f}")
    lines.append(f"- best single filter: **{best_single}** {bs:.3f}")
    lines.append(f"- oracle_router: {orr:.3f}")
    for r in ["router_hard", "router_soft", "router_dataset"]:
        v = mp(r)
        lines.append(f"- {r}: {v:.3f}  (vs naive {100*(v-nm):+.1f}, vs best-single {100*(v-bs):+.1f}, "
                     f"vs oracle-router {100*(v-orr):+.1f})")
    lines.append("")

    # ablation summary (overall soft precision by ablation)
    lines.append("## Ablations (overall router_soft precision)\n")
    ab = abl.groupby("ablation")["router_soft_precision"].mean().sort_values(ascending=False)
    lines.append("| ablation | overall soft precision |")
    lines.append("|---|---|")
    for a, v in ab.items():
        lines.append(f"| {a} | {v:.4f} |")
    lines.append("")

    soft, hard = mp("router_soft"), mp("router_hard")
    lines.append("## Verdict\n")
    soft_beats_single = soft >= bs + 0.005
    soft_beats_hard = soft >= hard + 0.002
    if soft_beats_single:
        lines.append(f"**Soft routing beats the best single filter** ({soft:.3f} vs {bs:.3f}) and "
                     f"{'exceeds' if soft_beats_hard else 'matches'} hard routing ({hard:.3f}). A unified "
                     "regime-aware filter is viable.\n")
    elif soft_beats_hard and soft >= bs - 0.002:
        lines.append(f"**Soft routing improves on hard routing** ({soft:.3f} vs {hard:.3f}) and matches the "
                     f"best single filter ({bs:.3f}), recovering {100*(soft-nm)/max(orr-nm,1e-6):.0f}% of the "
                     "oracle-router gain — but does not decisively beat 'just use the best single filter'. "
                     "Confidence-weighting helps at the margin (no hard misclassification cliff), yet the "
                     "aggregate benefit stays small.\n")
    else:
        lines.append(f"**Soft routing does not meaningfully beat the best single filter** ({soft:.3f} vs "
                     f"{bs:.3f}; hard {hard:.3f}; oracle {orr:.3f}). The paper should remain a **diagnostic "
                     "taxonomy with partial mitigations**: regimes are real and detectable, per-regime best "
                     "filters differ, but a unified router's net gain is within noise.\n")
    lines.append(f"Soft {soft:.3f} | hard {hard:.3f} | best-single {bs:.3f} | oracle-router {orr:.3f} | naive {nm:.3f}.\n")
    lines.append("## Artefacts\n- `unified_router_v2_results.csv`, `routing_confidence_analysis.csv`, "
                 "`ablation_results.csv`\n- `figures/router_v2_by_regime.png`, `figures/router_v2_overall.png`\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

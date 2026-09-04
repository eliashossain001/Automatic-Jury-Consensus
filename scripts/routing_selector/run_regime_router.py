"""Regime Router for judge-bank filtering (analysis-only, no new inference).

Prior buckets established three dependence regimes, each with a different best
filter:
  weak dependence            -> supermajority-75
  globally-correlated cofail -> CorrFilter small_gold_R   (CFI)
  vulnerable subgroup        -> bias-cluster filter        (position-aligned)
No single filter wins everywhere. This builds a router that picks the filter from
diagnostics, without being told the regime, and tests whether it matches the
per-regime best.

Router signals (all from votes / clean-calibration R; vulnerable cluster is the
LABEL-FREE position-sensitivity cluster from votes+swap, per Bucket 9):
  rho_bar, n_eff, eigenvector drift vs clean R, within-agreeing-subset
  correlation, agreement concentration in the vulnerable cluster,
  cluster-vs-non-cluster disagreement.

Thresholds are derived from the CLEAN bank reference (no test-label tuning); an
ablation sweeps them. Three routers: dataset-level, item-level (percentile-
combined), hybrid (dataset prior + item override). Baselines: naive majority,
supermajority-75, CorrFilter small_gold_R, bias_cluster, learned/ label-free
bias-cluster, oracle filter, and an ORACLE ROUTER (best filter per true regime).

Datasets: position-aligned (subgroup), h2b (weak), cfi:* (global).
Outputs: outputs/regime_router/{regime_router_results.csv, regime_confusion_matrix.csv,
router_ablation_results.csv, regime_router_summary.md, figures/}.

Usage: python scripts/routing_selector/run_regime_router.py --project-root .
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
from corrfilter.correlation import effective_size  # noqa: E402
from corrfilter.correlation.effective_size import mean_off_diagonal  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

ABSTAIN = -1
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
CLUSTER_K = 5
SG_SIZE = 100
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]
REGIME_FILTER = {"weak": "supermajority_75", "global": "corrfilter_small_gold_R", "subgroup": "bias_cluster"}


def _load_votes_swaps(cache, logical_ids, item_ids):
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
            V[i, j] = v; M[i, j] = 1; S[i, j] = 1 if bool(getattr(r, "position_swapped")) else 0
    return V, M, S


def _metrics(keep, gold):
    keep = keep.astype(bool); clean = gold == 1
    n = len(gold); nk = int(keep.sum()); kc = int((keep & clean).sum())
    prec = kc / nk if nk else 0.0
    rec = kc / int(clean.sum()) if clean.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"retention_rate": round(nk / n, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "false_retention_rate": round(1 - prec if nk else 0.0, 4)}


def _agree_features(O, M, label, cpos_mask, R_clean):
    """Per-item: cluster concentration, within-subset rho, cluster-noncluster disagreement, indep support."""
    n, m = O.shape; Mb = M.astype(bool); Rs = (R_clean + R_clean.T) / 2.0
    conc = np.zeros(n); rho = np.zeros(n); disagree = np.zeros(n); indep = np.full(n, -np.inf)
    lco_flip = np.zeros(n)   # does removing the vulnerable cluster flip the affirm-majority?
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        s = int(agree.sum())
        if s == 0:
            continue
        inC = int((agree & cpos_mask).sum())
        conc[i] = inC / s
        indep[i] = s - inC
        if s >= 2:
            sub = Rs[np.ix_(agree, agree)]
            rho[i] = float(sub[~np.eye(s, dtype=bool)].mean())
        cl = Mb[i] & cpos_mask; ncl = Mb[i] & ~cpos_mask
        ar_c = (O[i][cl] == label[i]).mean() if cl.any() else 0.0
        ar_n = (O[i][ncl] == label[i]).mean() if ncl.any() else 0.0
        disagree[i] = abs(ar_c - ar_n)
        # leave-cluster-out: item "passed" only because of the cluster if the
        # non-cluster judges' majority disagrees with the retained label.
        if ncl.any():
            lco_flip[i] = 1.0 if ar_n < 0.5 else 0.0
    return {"conc": conc, "rho": rho, "disagree": disagree, "indep": indep, "lco_flip": lco_flip}


def _filter_scores(O, M, label, R_h1, R_sg, cpos_mask):
    """Per-item scores (higher = keep) for each base filter."""
    return {
        "naive_majority": np.abs(vote_fraction(O, M) - 0.5),
        "supermajority_75": consensus_level(O, M),
        "corrfilter_small_gold_R": np.nan_to_num(corrfilter_score(O, M, R_sg, label).score, nan=-np.inf),
        "bias_cluster": _agree_features(O, M, label, cpos_mask, R_h1)["indep"],
    }


def _pct(x):
    """Percentile-rank a score array to [0,1] (NaN/-inf -> 0)."""
    x = np.asarray(x, float); valid = np.isfinite(x)
    r = np.zeros(len(x))
    if valid.sum() == 0:
        return r
    order = np.argsort(np.argsort(np.where(valid, x, -np.inf)))
    r = order / max(len(x) - 1, 1)
    r[~valid] = 0.0
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--uf-manifest", default="outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    ap.add_argument("--uf-votes", default="outputs/synthetic_poisoned_ultrafeedback/judge_votes")
    ap.add_argument("--npz", default="experiments/h1_measurement/results/correlation.npz")
    ap.add_argument("--cfi-config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--out-dir", default="outputs/regime_router")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    out_dir = _p(args.out_dir); fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    bank_cfg = load_bank_config(str(_p(args.bank))); logical_ids = [s.logical_id for s in bank_cfg.specs]
    R_h1 = np.load(_p(args.npz), allow_pickle=True)["R"]

    # UF votes + label-free position-sensitivity cluster
    man = pd.read_csv(_p(args.uf_manifest)); item_ids = man["item_id"].astype(str).tolist()
    V, M, Sw = _load_votes_swaps(VoteCache(_p(args.uf_votes)), logical_ids, item_ids)
    n = len(item_ids)
    slotA = np.where(Sw == 1, 1 - V, V)
    possens = np.abs(slotA.sum(0) / np.maximum(M.sum(0), 1) - 0.5)
    Cpos = set(np.argsort(-possens)[:CLUSTER_K].tolist())
    cpos_mask = np.array([j in Cpos for j in range(len(logical_ids))])
    R_clean_uf = disagreement_R(V, M)

    # Clean-bank reference diagnostics (calibration; no test labels).
    lab_clean = majority_consensus(V, M)
    feat_clean = _agree_features(V, M, lab_clean, cpos_mask, R_h1)
    ref = {"rho_bar": mean_off_diagonal(R_clean_uf), "conc": float(np.nanmean(feat_clean["conc"])),
           "rho_S": float(np.nanmean(feat_clean["rho"])), "disagree": float(np.nanmean(feat_clean["disagree"])),
           "lco_flip": float(np.nanmean(feat_clean["lco_flip"]))}
    print("clean reference:", {k: round(v, 3) for k, v in ref.items()})

    # Assemble datasets: (name, true_regime, O, M, gold, affirmed_only)
    datasets = []
    pos_score = ((1 - V) * M * (np.array([_h1gap(l) for l in logical_ids]) / 100.0)[None, :]).sum(1)
    pos_order = np.argsort(-pos_score, kind="stable")
    for tag, rate in RATES.items():
        pm = np.zeros(n, bool); pm[pos_order[:int(round(rate * n))]] = True
        O = V.copy(); O[pm] = 1 - V[pm]
        datasets.append((f"position_{tag}", "subgroup", O, M, np.where(pm, 0, 1).astype(np.int8), True))
        hm = man[f"poisoned_{tag}"].to_numpy(bool)
        Oh = V.copy(); Oh[hm] = 1 - V[hm]
        datasets.append((f"h2b_{tag}", "weak", Oh, M, np.where(hm, 0, 1).astype(np.int8), True))
    # CFI
    try:
        import yaml
        cfg = yaml.safe_load(_p(args.cfi_config).read_text()); bank = load_bias_bank(str(_p(args.cfi_config)))
        cal = yaml.safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())
        citems = [it for it in load_calibration_set(_p(cal["output"]["manifest_path"]))
                  if it.item_id in {l.strip() for l in _p("outputs/cfi/cfi_subset_items.txt").read_text().splitlines() if l.strip()}]
        clean_views = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)
        for mech in CFI_MECHS:
            bv = load_vote_views(_p("outputs/cfi/votes") / mech, logical_ids)
            if sum(len(x) for x in bv.values()) == 0:
                continue
            var = assemble_variant(citems, logical_ids, bank.by_name(mech), 1.0, bv, clean_views, bank.seed)
            gc = (majority_consensus(var.V, var.M) == 1).astype(np.int8)
            datasets.append((f"cfi_{mech[:8]}", "global", var.V, var.M, gc, False))
    except Exception as e:  # noqa: BLE001
        print("CFI skipped:", str(e)[:100])

    rng = np.random.default_rng(20260601)
    rows, diag_rows, conf_rows, ablation_rows = [], [], [], []

    def classify_regime(d, thr):
        """Dataset-level regime from clean-referenced diagnostics.

        global   : bank-wide correlation rises above the clean reference (rho_bar).
        subgroup : correlation falls / eigenstructure drifts AND removing the
                   vulnerable cluster flips many affirm-decisions (lco_flip).
        weak     : neither (diagnostics ~ clean).
        """
        if d["rho_bar"] - ref["rho_bar"] > thr["rho"]:
            return "global"
        if (d["lco_flip"] - ref["lco_flip"] > thr["lco"]) or (d["eig_overlap"] < thr["eig"]):
            return "subgroup"
        return "weak"

    THR = {"rho": 0.03, "lco": 0.05, "eig": 0.85}   # clean-referenced margins

    for (name, true_regime, O, Md, gold, affirmed) in datasets:
        N = len(gold)
        # small_gold R from a labeled calibration split (held-out eval below excludes it)
        cal_idx = rng.choice(N, size=min(SG_SIZE, N), replace=False)
        eval_mask = np.ones(N, bool); eval_mask[cal_idx] = False
        ei = np.where(eval_mask)[0]
        R_sg = small_gold_R(O, Md, gold, cal_idx)

        Oe, Me, ge = O[ei], Md[ei], gold[ei]
        label = majority_consensus(Oe, Me)
        n_match = max(int((supermajority_consensus(Oe, Me, 0.75) != ABSTAIN).sum()), 1)
        pool = (label == 1) if affirmed else np.ones(len(ge), bool)
        scores = _filter_scores(Oe, Me, label, R_h1, R_sg, cpos_mask)
        feats = _agree_features(Oe, Me, label, cpos_mask, R_h1)

        # dataset diagnostics (on eval split)
        R_O = disagreement_R(Oe, Me)
        d = {"rho_bar": mean_off_diagonal(R_O), "n_eff": effective_size(R_O),
             "eig_overlap": eigenvector_overlap(R_O, R_clean_uf, 3),
             "conc": float(np.nanmean(feats["conc"][pool])), "rho_S": float(np.nanmean(feats["rho"][pool])),
             "disagree": float(np.nanmean(feats["disagree"][pool])),
             "lco_flip": float(np.nanmean(feats["lco_flip"][pool]))}
        diag_rows.append({"dataset": name, "true_regime": true_regime, **{k: round(v, 4) for k, v in d.items()}})

        def keep_by(score):
            s = np.where(pool, score, -np.inf); k, _ = retention_match_threshold(s, n_match); return k & pool

        def add(method, keep):
            m = _metrics(keep, ge); m.update({"dataset": name, "true_regime": true_regime, "method": method})
            rows.append(m)

        for fname, sc in scores.items():
            add(fname, keep_by(sc))
        add("oracle_filter", keep_by(np.where(ge == 1, 1.0, -np.inf)))

        # ---- routers ----
        pred = classify_regime(d, THR)
        conf_rows.append({"dataset": name, "true_regime": true_regime, "pred_regime": pred})
        add("router_dataset", keep_by(scores[REGIME_FILTER[pred]]))
        add("oracle_router", keep_by(scores[REGIME_FILTER[true_regime]]))

        # item-level router: per item choose filter by item signal, rank by percentile.
        # subgroup signal = removing the vulnerable cluster flips this item's affirm-majority;
        # global signal = highly correlated agreeing set.
        pcts = {f: _pct(s) for f, s in scores.items()}
        item_regime = np.where(feats["lco_flip"] > 0.5, "subgroup",
                               np.where(feats["rho"] > ref["rho_S"] + 0.10, "global", "weak"))
        unified = np.array([pcts[REGIME_FILTER[item_regime[i]]][i] for i in range(len(ge))])
        add("router_item", keep_by(unified))

        # hybrid: dataset prior default; item override to subgroup when its affirm-majority
        # is cluster-dependent (leave-cluster-out flips).
        hyb_regime = np.where(feats["lco_flip"] > 0.5, "subgroup", pred)
        unified_h = np.array([pcts[REGIME_FILTER[hyb_regime[i]]][i] for i in range(len(ge))])
        add("router_hybrid", keep_by(unified_h))

        # ablation: sweep dataset-router thresholds (rho deviation x eig-overlap)
        for dr in (0.02, 0.03, 0.05):
            for de in (0.80, 0.85, 0.90):
                pr = classify_regime(d, {"rho": dr, "lco": THR["lco"], "eig": de})
                mm = _metrics(keep_by(scores[REGIME_FILTER[pr]]), ge)
                ablation_rows.append({"dataset": name, "true_regime": true_regime, "rho_thr": dr,
                                      "eig_thr": de, "pred_regime": pr, "precision": mm["precision"]})

    df = pd.DataFrame(rows); df.to_csv(out_dir / "regime_router_results.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out_dir / "regime_diagnostics.csv", index=False)
    conf = pd.DataFrame(conf_rows)
    pd.crosstab(conf.true_regime, conf.pred_regime).to_csv(out_dir / "regime_confusion_matrix.csv")
    pd.DataFrame(ablation_rows).to_csv(out_dir / "router_ablation_results.csv", index=False)
    _figures(df, fig_dir)
    _summary(df, conf, pd.DataFrame(diag_rows), out_dir / "regime_router_summary.md", logical_ids, Cpos, ref)
    print(f"wrote {out_dir}  ({len(df)} rows)")


def _h1gap(lid):
    return {"gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8, "llama-3.1-8b::pairwise": 63.7,
            "llama-3.1-8b::likert": 17.4, "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
            "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9, "qwen-2.5-7b::pairwise": 46.7,
            "qwen-2.5-7b::likert": 51.7}.get(lid, 0.0)


def _figures(df, fig_dir):
    # router vs single filters: mean precision per regime
    methods = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster",
               "router_dataset", "router_item", "router_hybrid", "oracle_router", "oracle_filter"]
    regimes = ["weak", "global", "subgroup"]
    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(methods)); w = 0.26
    for i, reg in enumerate(regimes):
        vals = [df[(df.method == m) & (df.true_regime == reg)]["precision"].mean() for m in methods]
        ax.bar(x + (i - 1) * w, vals, w, label=reg)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("precision (held-out)"); ax.set_title("Router vs single filters, by true regime")
    ax.legend(title="regime"); fig.tight_layout()
    fig.savefig(fig_dir / "router_vs_filters.png", dpi=150); plt.close(fig)


def _summary(df, conf, diag, path, logical_ids, Cpos, ref):
    methods = ["naive_majority", "supermajority_75", "corrfilter_small_gold_R", "bias_cluster",
               "router_dataset", "router_item", "router_hybrid", "oracle_router", "oracle_filter"]
    regimes = ["weak", "global", "subgroup"]
    lines = ["# Regime Router\n"]
    lines.append("Analysis-only, no new inference. Vulnerable cluster = label-free position-sensitivity "
                 f"(top-{CLUSTER_K} by |slot-A rate − 0.5|): {', '.join(sorted(logical_ids[j] for j in Cpos))}. "
                 "Thresholds are clean-bank-referenced (no test-label tuning); small_gold R and held-out "
                 "eval use disjoint splits. Matched retention = supermajority-0.75.\n")
    lines.append(f"Clean reference: { {k: round(v,3) for k,v in ref.items()} }.\n")

    lines.append("## Dataset diagnostics\n")
    lines.append("| dataset | true regime | rho_bar | n_eff | eig-overlap vs clean | conc(Cpos) | within-subset rho |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in diag.iterrows():
        lines.append(f"| {r.dataset} | {r.true_regime} | {r.rho_bar:.3f} | {r.n_eff:.2f} | {r.eig_overlap:.3f} | {r.conc:.3f} | {r.rho_S:.3f} |")
    lines.append("")

    lines.append("## Regime classification (dataset router)\n")
    ct = pd.crosstab(conf.true_regime, conf.pred_regime)
    acc = (conf.true_regime == conf.pred_regime).mean()
    lines.append("```\n" + ct.to_string() + f"\n```\nclassification accuracy = {acc:.0%}\n")

    lines.append("## Precision by method and regime (mean over datasets, held-out)\n")
    lines.append("| method | weak | global | subgroup | overall |")
    lines.append("|---|---|---|---|---|")
    for m in methods:
        cells = [df[(df.method == m) & (df.true_regime == reg)]["precision"].mean() for reg in regimes]
        lines.append(f"| {m} | " + " | ".join(f"{c:.3f}" for c in cells) + f" | {df[df.method==m]['precision'].mean():.3f} |")
    lines.append("")

    # gains: router vs naive, vs best single filter, vs oracle router
    def mean_prec(m, reg=None):
        s = df[df.method == m] if reg is None else df[(df.method == m) & (df.true_regime == reg)]
        return s["precision"].mean()

    single = ["supermajority_75", "corrfilter_small_gold_R", "bias_cluster"]
    lines.append("## Router gains (overall mean precision, all datasets)\n")
    nm = mean_prec("naive_majority"); best_single = max(mean_prec(m) for m in single)
    best_single_name = max(single, key=lambda m: mean_prec(m))
    orouter = mean_prec("oracle_router")
    lines.append(f"- naive_majority: {nm:.3f}")
    lines.append(f"- best fixed single filter: **{best_single_name}** {best_single:.3f}")
    lines.append(f"- oracle_router (knows regime): {orouter:.3f}")
    for r in ["router_dataset", "router_item", "router_hybrid"]:
        v = mean_prec(r)
        lines.append(f"- {r}: {v:.3f}  (vs naive {100*(v-nm):+.1f} pt, vs best-single {100*(v-best_single):+.1f} pt, "
                     f"vs oracle-router {100*(v-orouter):+.1f} pt)")
    lines.append("")

    best_router = max(["router_dataset", "router_item", "router_hybrid"], key=lambda m: mean_prec(m))
    brv = mean_prec(best_router)
    lines.append("## Verdict\n")
    matches_oracle = brv >= orouter - 0.005
    beats_single = brv >= best_single + 0.005
    if matches_oracle and beats_single:
        lines.append(f"**The router works.** `{best_router}` ({brv:.3f}) matches the oracle router "
                     f"({orouter:.3f}) and beats the best fixed single filter ({best_single_name}, "
                     f"{best_single:.3f}) — routing recovers the per-regime best WITHOUT knowing the regime. "
                     "This is the unified-filtering contribution.\n")
    elif beats_single:
        lines.append(f"**Partial.** `{best_router}` ({brv:.3f}) beats the best fixed single filter "
                     f"({best_single_name}, {best_single:.3f}) but trails the oracle router ({orouter:.3f}): "
                     "regime detection is imperfect but routing still helps over any one filter.\n")
    else:
        lines.append(f"**Router does NOT beat the best fixed single filter** (`{best_router}` {brv:.3f} vs "
                     f"{best_single_name} {best_single:.3f}; oracle router {orouter:.3f}). The paper remains a "
                     "**diagnostic taxonomy** of consensus-failure regimes rather than a unified filtering "
                     "method: knowing the regime matters, but inferring it from diagnostics is not reliable "
                     "enough to route correctly.\n")
    lines.append(f"Regime classification accuracy: {acc:.0%}. Best router: `{best_router}`.\n")
    lines.append("## Artefacts\n- `regime_router_results.csv`, `regime_diagnostics.csv`, "
                 "`regime_confusion_matrix.csv`, `router_ablation_results.csv`\n- `figures/router_vs_filters.png`\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

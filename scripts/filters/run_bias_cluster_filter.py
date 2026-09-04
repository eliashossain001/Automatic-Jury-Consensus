"""Cluster-aware filtering prototype (analysis-only, no new inference).

Position-aligned poisoning defeated CorrFilter because the attack splits the bank
into a vulnerable subgroup (that co-affirms the bad label) and a dissenting
remainder, so global pairwise correlation DROPS and alpha_subset rewards the
apparent independence. This script prototypes filters that look for agreement
*concentrated in one judge cluster* even when global correlation is low.

Four cluster-aware scores (all use only calibration-time cluster definitions; no
test labels — so no test-tuning):

  bias_cluster      independent support = #agreeing judges OUTSIDE the H1
                    position-biased cluster (keep items with non-cluster support)
  spectral_cluster  Shannon entropy of the agreeing set over clean-R spectral
                    clusters (keep agreement spread across clusters)
  leave_cluster_out vote fraction for the retained label after removing the
                    dominant agreeing cluster (keep items robust to one cluster)
  hybrid            alpha_subset(clean_R) * (1 - max single-cluster fraction)
                    ("trust high agreement only if spread across clusters")

Compared against naive majority / supermajority-75, CorrFilter (clean_R,
small_gold_R, oracle_variant_R), and the oracle filter, on:
  * position  : position-aligned label poisoning (the failure case)
  * h2b       : content (verbosity/length/polite/style) poisoning
  * cfi:<mech>: CFI prompt-injected biased bank (positive control — co-failure
                here IS internally correlated, where CorrFilter already worked)

Outputs: outputs/cluster_filter/{cluster_filter_results.csv,
cluster_filter_summary.md, figures/}.

Usage: python scripts/filters/run_bias_cluster_filter.py --project-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from corrfilter.cfi.adaptive_r import disagreement_R, oracle_variant_R, small_gold_R  # noqa: E402
from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import consensus_level, majority_consensus, supermajority_consensus, vote_fraction  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

ABSTAIN = -1
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
N_CLUSTERS = 3
SG_SIZE = 100
SG_SEEDS = 10
H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}
CFI_MECHS = ["position_bias_stress_test", "polite_hallucination_preference", "verbosity_bias"]


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


def _spectral_clusters(R, k=N_CLUSTERS):
    D = 1.0 - (R + R.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, k, criterion="maxclust") - 1


def _metrics(keep, gold):
    keep = keep.astype(bool); clean = gold == 1; pois = gold == 0
    n = len(gold); nk = int(keep.sum())
    kc = int((keep & clean).sum()); kp = int((keep & pois).sum())
    prec = kc / nk if nk else 0.0
    rec = kc / int(clean.sum()) if clean.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rem = (int(pois.sum()) - kp) / int(pois.sum()) if pois.sum() else 0.0
    return {"retention_rate": round(nk / n, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "false_retention_rate": round(1 - prec if nk else 0.0, 4),
            "corrupted_removal_rate": round(rem, 4), "n_kept": nk}


def _cluster_scores(O, M, label, cb_mask, clabels, R):
    """Per-item cluster-aware scores (higher = keep)."""
    n, m = O.shape
    Mb = M.astype(bool); Rsym = (R + R.T) / 2.0
    nc = int(clabels.max()) + 1
    indep = np.full(n, -np.inf); entropy = np.full(n, -np.inf)
    lco = np.full(n, -np.inf); hyb = np.full(n, -np.inf); negmax = np.full(n, -np.inf)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        s = int(agree.sum())
        if s == 0:
            continue
        indep[i] = s - int((agree & cb_mask).sum())                 # support outside bias cluster
        counts = np.array([int((agree & (clabels == c)).sum()) for c in range(nc)], float)
        frac = counts / s
        negmax[i] = -frac.max()
        p = frac[frac > 0]
        entropy[i] = float(-(p * np.log(p)).sum())
        d = int(counts.argmax())
        rem = Mb[i] & (clabels != d)
        lco[i] = float((O[i][rem] == label[i]).sum()) / rem.sum() if rem.sum() else 0.5
        sub = Rsym[np.ix_(agree, agree)]; q = float(sub.sum())
        alpha = s / np.sqrt(q) if q > 0 else float(s)
        hyb[i] = alpha * (1.0 + negmax[i])                          # alpha * (1 - maxfrac)
    return {"bias_cluster": indep, "spectral_cluster": entropy,
            "leave_cluster_out": lco, "hybrid_cluster": hyb}


def _eval_attack(name, rate, O, M, gold, R_h1, cb_mask, clabels, sg_R, affirmed_only):
    """Run all methods at matched retention; return list of metric rows."""
    label = majority_consensus(O, M)
    n_match = max(int((supermajority_consensus(O, M, 0.75) != ABSTAIN).sum()), 1)
    pool = (label == 1) if affirmed_only else np.ones(len(gold), bool)

    def keep_top(score):
        s = np.where(pool, score, -np.inf)
        keep, _ = retention_match_threshold(s, n_match)
        return keep & pool

    rows = []

    def add(method, keep):
        m = _metrics(keep, gold); m.update({"attack": name, "corruption_rate": rate, "method": method})
        rows.append(m)

    add("naive_majority", keep_top(np.abs(vote_fraction(O, M) - 0.5)))
    add("naive_supermajority_75", keep_top(consensus_level(O, M)))
    cf_clean = corrfilter_score(O, M, R_h1, label).score
    add("corrfilter_clean_R", keep_top(np.nan_to_num(cf_clean, nan=-np.inf)))
    cf_sg = corrfilter_score(O, M, sg_R, label).score
    add("corrfilter_small_gold_R", keep_top(np.nan_to_num(cf_sg, nan=-np.inf)))
    cf_orc = corrfilter_score(O, M, oracle_variant_R(O, M, gold), label).score
    add("corrfilter_oracle_variant_R", keep_top(np.nan_to_num(cf_orc, nan=-np.inf)))
    cs = _cluster_scores(O, M, label, cb_mask, clabels, R_h1)
    for meth, sc in cs.items():
        add(meth, keep_top(sc))
    add("oracle_filter", keep_top(np.where(gold == 1, 1.0, -np.inf)))
    return rows, n_match


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--uf-manifest", default="outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    ap.add_argument("--uf-votes", default="outputs/synthetic_poisoned_ultrafeedback/judge_votes")
    ap.add_argument("--npz", default="experiments/h1_measurement/results/correlation.npz")
    ap.add_argument("--cfi-config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--out-dir", default="outputs/cluster_filter")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    out_dir = _p(args.out_dir); fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bank_cfg = load_bank_config(str(_p(args.bank)))
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    npz = np.load(_p(args.npz), allow_pickle=True); R_h1 = npz["R"]

    # Calibration-time clusters (no test labels).
    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids])
    cb_idx = set(np.argsort(-gaps)[:5].tolist())                    # top-5 position-biased = bias cluster
    cb_mask = np.array([j in cb_idx for j in range(len(logical_ids))])
    clabels = _spectral_clusters(R_h1, N_CLUSTERS)
    print("bias cluster (H1 top-5 |gap|):", [logical_ids[j] for j in sorted(cb_idx)])
    print("spectral clusters:", {logical_ids[j]: int(clabels[j]) for j in range(len(logical_ids))})

    # --- UF votes (position + h2b attacks) ---
    man = pd.read_csv(_p(args.uf_manifest)); item_ids = man["item_id"].astype(str).tolist()
    cache = VoteCache(_p(args.uf_votes))
    V, M, S = _load_votes_swaps(cache, logical_ids, item_ids)
    n = len(item_ids)
    # position-aligned poison ordering (H1-weighted wrong mass)
    w = gaps / 100.0
    pos_score = ((1 - V) * M * w[None, :]).sum(axis=1)
    pos_order = np.argsort(-pos_score, kind="stable")

    all_rows = []
    rng = np.random.default_rng(20260601)

    def small_gold_mean(O, gold):
        Rs = [small_gold_R(O, M, gold, rng.choice(n, size=min(SG_SIZE, n), replace=False)) for _ in range(SG_SEEDS)]
        return np.mean(Rs, axis=0)

    for tag, rate in RATES.items():
        # position attack
        pmask = np.zeros(n, bool); pmask[pos_order[:int(round(rate * n))]] = True
        gold = np.where(pmask, 0, 1).astype(np.int8); O = V.copy(); O[pmask] = 1 - V[pmask]
        rows, _ = _eval_attack("position", rate, O, M, gold, R_h1, cb_mask, clabels, small_gold_mean(O, gold), True)
        all_rows += rows
        # h2b content attack
        hmask = man[f"poisoned_{tag}"].to_numpy(bool)
        gold = np.where(hmask, 0, 1).astype(np.int8); O = V.copy(); O[hmask] = 1 - V[hmask]
        rows, _ = _eval_attack("h2b", rate, O, M, gold, R_h1, cb_mask, clabels, small_gold_mean(O, gold), True)
        all_rows += rows

    # --- CFI positive control (biased bank, gold=1, keep among all) ---
    try:
        cfg = __import__("yaml").safe_load(_p(args.cfi_config).read_text())
        bank = load_bias_bank(str(_p(args.cfi_config)))
        cal = __import__("yaml").safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())
        cfi_items = load_calibration_set(_p(cal["output"]["manifest_path"]))
        sub = {ln.strip() for ln in (_p("outputs/cfi/cfi_subset_items.txt").read_text().splitlines()) if ln.strip()}
        cfi_items = [it for it in cfi_items if it.item_id in sub]
        clean_views = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)
        for mech in CFI_MECHS:
            bv = load_vote_views(_p("outputs/cfi/votes") / mech, logical_ids)
            if sum(len(x) for x in bv.values()) == 0:
                continue
            var = assemble_variant(cfi_items, logical_ids, bank.by_name(mech), 1.0, bv, clean_views, bank.seed)
            Vc, Mc = var.V, var.M
            # CFI has clean labels but biased judges: an item is "reliable" iff the
            # biased bank's majority still prefers the truly-better response (==1).
            # gold = 1 (majority correct) / 0 (majority flipped by bias). keep-among-all.
            goldc = (majority_consensus(Vc, Mc) == 1).astype(np.int8)
            sgc = np.mean([small_gold_R(Vc, Mc, goldc, np.random.default_rng(s).choice(len(cfi_items), min(SG_SIZE, len(cfi_items)), replace=False)) for s in range(SG_SEEDS)], axis=0)
            rows, _ = _eval_attack(f"cfi:{mech[:10]}", 1.0, Vc, Mc, goldc, R_h1, cb_mask, clabels, sgc, False)
            all_rows += rows
    except Exception as e:  # noqa: BLE001
        print("CFI positive-control skipped:", str(e)[:120])

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "cluster_filter_results.csv", index=False)
    _figures(df, fig_dir)
    _summary(df, out_dir / "cluster_filter_summary.md", logical_ids, cb_idx, clabels)
    print(f"wrote {out_dir}")


def _figures(df, fig_dir):
    methods = ["naive_majority", "naive_supermajority_75", "corrfilter_clean_R",
               "corrfilter_small_gold_R", "corrfilter_oracle_variant_R",
               "bias_cluster", "spectral_cluster", "leave_cluster_out", "hybrid_cluster", "oracle_filter"]
    # position attack precision by method (per rate)
    for attack in ["position", "h2b"]:
        sub = df[df.attack == attack]
        if sub.empty:
            continue
        rates = sorted(sub.corruption_rate.unique())
        fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(methods)); w = 0.25
        for i, r in enumerate(rates):
            vals = [sub[(sub.method == m) & (sub.corruption_rate == r)]["precision"].mean() for m in methods]
            ax.bar(x + (i - 1) * w, vals, w, label=f"{r:.0%}")
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("precision"); ax.set_title(f"{attack}: precision by method"); ax.legend(title="corruption")
        fig.tight_layout(); fig.savefig(fig_dir / f"precision_{attack}.png", dpi=150); plt.close(fig)

    # gain over majority on position attack
    sub = df[df.attack == "position"]
    if not sub.empty:
        rates = sorted(sub.corruption_rate.unique())
        cl = ["corrfilter_clean_R", "corrfilter_small_gold_R", "bias_cluster", "spectral_cluster", "leave_cluster_out", "hybrid_cluster"]
        fig, ax = plt.subplots(figsize=(10, 5)); xr = np.arange(len(rates)); w = 0.13
        for i, m in enumerate(cl):
            g = [100 * (sub[(sub.method == m) & (sub.corruption_rate == r)]["precision"].mean()
                        - sub[(sub.method == "naive_majority") & (sub.corruption_rate == r)]["precision"].mean()) for r in rates]
            ax.bar(xr + (i - 2.5) * w, g, w, label=m)
        ax.axhline(5, ls="--", color="green"); ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xr); ax.set_xticklabels([f"{r:.0%}" for r in rates])
        ax.set_ylabel("precision gain over majority (pts)")
        ax.set_title("Position attack: gain over naive majority (dashed = +5 target)")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout()
        fig.savefig(fig_dir / "gain_over_majority_position.png", dpi=150); plt.close(fig)


def _summary(df, path, logical_ids, cb_idx, clabels):
    methods = ["naive_majority", "naive_supermajority_75", "corrfilter_clean_R",
               "corrfilter_small_gold_R", "corrfilter_oracle_variant_R",
               "bias_cluster", "spectral_cluster", "leave_cluster_out", "hybrid_cluster", "oracle_filter"]
    cluster_methods = ["bias_cluster", "spectral_cluster", "leave_cluster_out", "hybrid_cluster"]
    lines = ["# Cluster-Aware Filtering Prototype\n"]
    lines.append("Analysis-only (no new inference). Cluster definitions are calibration-time only "
                 "(H1 position gaps; clean-R spectral clusters) — no test-label tuning. Matched "
                 "retention = supermajority-0.75.\n")
    lines.append(f"**Bias cluster** (H1 top-5 |gap|): {', '.join(logical_ids[j] for j in sorted(cb_idx))}.")
    lines.append(f"**Spectral clusters** (clean R, k={N_CLUSTERS}): "
                 + ", ".join(f"{logical_ids[j]}={clabels[j]}" for j in range(len(logical_ids))) + ".\n")

    for attack in sorted(df.attack.unique()):
        sub = df[df.attack == attack]
        rates = sorted(sub.corruption_rate.unique())
        lines.append(f"## Attack: {attack}  (precision by method)\n")
        lines.append("| method | " + " | ".join(f"{r:.0%}" for r in rates) + " |")
        lines.append("|" + "---|" * (len(rates) + 1))
        for m in methods:
            cells = [f"{sub[(sub.method==m)&(sub.corruption_rate==r)]['precision'].mean():.3f}" for r in rates]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    # Headline: best cluster method gain over naive on position attack.
    pos = df[df.attack == "position"]
    lines.append("## Headline — position-aligned poisoning: cluster gain over naive majority\n")
    lines.append("| rate | naive maj | best cluster method | gain (pts) | oracle |")
    lines.append("|---|---|---|---|---|")
    best_overall = -1e9; best_name = None
    for r in sorted(pos.corruption_rate.unique()):
        nm = pos[(pos.method == "naive_majority") & (pos.corruption_rate == r)]["precision"].mean()
        orc = pos[(pos.method == "oracle_filter") & (pos.corruption_rate == r)]["precision"].mean()
        cvals = {m: pos[(pos.method == m) & (pos.corruption_rate == r)]["precision"].mean() for m in cluster_methods}
        bm = max(cvals, key=cvals.get); gain = 100 * (cvals[bm] - nm)
        if gain > best_overall:
            best_overall, best_name = gain, bm
        lines.append(f"| {r:.0%} | {nm:.3f} | {bm} ({cvals[bm]:.3f}) | {gain:+.1f} | {orc:.3f} |")
    lines.append("")

    # which cluster penalty works best (avg gain over naive across attacks/rates).
    lines.append("## Which cluster penalty works best (mean precision gain over naive majority)\n")
    lines.append("| method | position | h2b | cfi (mean) |")
    lines.append("|---|---|---|---|")
    for m in cluster_methods:
        def avg_gain(att_prefix):
            s = df[df.attack.str.startswith(att_prefix)]
            if s.empty:
                return float("nan")
            gains = []
            for (a, r), gdf in s.groupby(["attack", "corruption_rate"]):
                nm = gdf[gdf.method == "naive_majority"]["precision"].mean()
                mm = gdf[gdf.method == m]["precision"].mean()
                gains.append(100 * (mm - nm))
            return float(np.nanmean(gains))
        lines.append(f"| {m} | {avg_gain('position'):+.1f} | {avg_gain('h2b'):+.1f} | {avg_gain('cfi'):+.1f} |")
    lines.append("")

    lines.append("## Verdict\n")
    met = best_overall >= 5.0
    if met:
        lines.append(f"**Cluster-aware filtering RECOVERS precision on position-aligned poisoning** — "
                     f"best variant `{best_name}` gains {best_overall:+.1f} pts over naive majority "
                     "(≥5 pt target MET). This is the candidate CorrFilter v2: trust high consensus only "
                     "when it is spread across judge clusters, not concentrated in a vulnerable subgroup.\n")
    else:
        lines.append(f"**Cluster-aware filtering does NOT recover ≥5 pts on position-aligned poisoning** "
                     f"(best variant `{best_name}`, {best_overall:+.1f} pts). Even targeting the known "
                     "vulnerable cluster, the high-consensus errors cannot be separated from clean "
                     "high-consensus items at matched retention. This supports reframing the work as a "
                     "**measurement/diagnostic paper**: consensus fails under a dominant vulnerable "
                     "subgroup, and neither correlation-aware nor cluster-aware filtering reliably "
                     "repairs it without per-item ground truth.\n")
    lines.append(f"Best cluster-aware gain over naive on position attack: **{best_overall:+.1f} pts** "
                 f"(target {'MET' if met else 'NOT met'}).\n")
    lines.append("## Artefacts\n- `cluster_filter_results.csv`\n- `figures/`: precision_position, "
                 "precision_h2b, gain_over_majority_position\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

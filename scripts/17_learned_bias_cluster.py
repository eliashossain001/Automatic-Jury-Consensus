"""Small-Calibration Learned Bias Cluster (analysis-only, no new inference).

Bucket 8 showed an H1-imported position-bias cluster partially fixes
position-aligned poisoning (+4.8 pt peak). This tests deployability: can the
vulnerable cluster be LEARNED from a small labeled calibration set on the
deployment distribution, instead of imported from the H1 study?

Protocol (no leakage): for each calibration size K and seed, sample K labeled
items, learn the vulnerable cluster from them, then evaluate ALL methods on the
HELD-OUT remainder at matched retention. Learning signal = per-judge
poison-affirmation rate (fraction of poisoned calibration items the judge
affirmed = "contribution to retained poisoned labels"); the learned cluster is
the top-5 judges by that score (content-error fallback when a draw contains no
poison). A label-free position-sensitivity cluster (from votes+swap only) is
also reported as a reference for Q5.

Methods (all evaluated on held-out): naive majority, supermajority-75,
CorrFilter clean_R, CorrFilter small_gold_R (R from the K calib items),
bias_cluster (H1-imported), learned_bias_cluster, learned_possens (label-free),
oracle. Datasets: position (primary), h2b, cfi:<mech>.

Outputs: outputs/learned_bias_cluster/{learned_bias_cluster_results.csv,
cluster_overlap_analysis.csv, learned_bias_cluster_summary.md, figures/}.

Usage: python scripts/17_learned_bias_cluster.py --project-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corrfilter.cfi.adaptive_r import small_gold_R  # noqa: E402
from corrfilter.cfi.bias_bank import load_bias_bank  # noqa: E402
from corrfilter.cfi.cfi_votes import assemble_variant, load_vote_views  # noqa: E402
from corrfilter.cfi.consensus import consensus_level, majority_consensus, supermajority_consensus, vote_fraction  # noqa: E402
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

ABSTAIN = -1
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
SIZES = [25, 50, 100, 200]
N_SEEDS = 10
CLUSTER_K = 5
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


def _metrics(keep, gold):
    keep = keep.astype(bool); clean = gold == 1; pois = gold == 0
    n = len(gold); nk = int(keep.sum())
    kc = int((keep & clean).sum())
    prec = kc / nk if nk else 0.0
    rec = kc / int(clean.sum()) if clean.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"retention_rate": round(nk / n, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "false_retention_rate": round(1 - prec if nk else 0.0, 4), "n_kept": nk}


def _indep_support(O, M, label, cluster_mask):
    """#agreeing judges outside the vulnerable cluster (bias-cluster score)."""
    Mb = M.astype(bool); n = O.shape[0]; out = np.full(n, -np.inf)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        if agree.any():
            out[i] = int(agree.sum()) - int((agree & cluster_mask).sum())
    return out


def learn_cluster(O_c, M_c, gold_c, k=CLUSTER_K):
    """Top-k vulnerable judges by poison-affirmation rate on the calibration set."""
    m = O_c.shape[1]; Mb = M_c.astype(bool); pois = gold_c == 0
    if pois.sum() >= 1:
        aff = ((O_c[pois] == 1) & Mb[pois]).sum(0)
        den = Mb[pois].sum(0)
        score = np.where(den > 0, aff / np.maximum(den, 1), 0.0)
    if pois.sum() < 3:  # too few poison examples -> stabilize with content-error rate
        V_c = np.where(gold_c[:, None] == 1, O_c, 1 - O_c)
        cerr = 1 - (np.where(Mb, V_c, 0).sum(0) / np.maximum(Mb.sum(0), 1))
        score = cerr if pois.sum() == 0 else 0.5 * score + 0.5 * cerr
    cluster = set(np.argsort(-score)[:k].tolist())
    return cluster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--uf-manifest", default="outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    ap.add_argument("--uf-votes", default="outputs/synthetic_poisoned_ultrafeedback/judge_votes")
    ap.add_argument("--npz", default="experiments/h1_measurement/results/correlation.npz")
    ap.add_argument("--cfi-config", default="configs/cfi_bias_prompts.yaml")
    ap.add_argument("--out-dir", default="outputs/learned_bias_cluster")
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
    gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in logical_ids])
    C_h1 = set(np.argsort(-gaps)[:CLUSTER_K].tolist())
    mask_h1 = np.array([j in C_h1 for j in range(len(logical_ids))])

    # UF votes
    man = pd.read_csv(_p(args.uf_manifest)); item_ids = man["item_id"].astype(str).tolist()
    V, M, S = _load_votes_swaps(VoteCache(_p(args.uf_votes)), logical_ids, item_ids)
    n = len(item_ids)
    w = gaps / 100.0
    pos_score = ((1 - V) * M * w[None, :]).sum(axis=1)
    pos_order = np.argsort(-pos_score, kind="stable")
    slotA = np.where(S == 1, 1 - V, V)
    possens = np.abs(slotA.sum(0) / np.maximum(M.sum(0), 1) - 0.5)
    C_possens = set(np.argsort(-possens)[:CLUSTER_K].tolist())
    mask_possens = np.array([j in C_possens for j in range(len(logical_ids))])

    # Build the two UF attacks' poison masks per rate.
    attacks = {}
    for tag, rate in RATES.items():
        pm = np.zeros(n, bool); pm[pos_order[:int(round(rate * n))]] = True
        attacks[("position", rate)] = (V, M, pm)
        attacks[("h2b", rate)] = (V, M, man[f"poisoned_{tag}"].to_numpy(bool))

    # CFI attacks (gold = majority-correct; keep-among-all)
    cfi_data = {}
    try:
        import yaml
        cfg = yaml.safe_load(_p(args.cfi_config).read_text())
        bank = load_bias_bank(str(_p(args.cfi_config)))
        cal = yaml.safe_load(_p(cfg["sources"]["h1_calibration_config"]).read_text())
        cfi_items = load_calibration_set(_p(cal["output"]["manifest_path"]))
        sub = {ln.strip() for ln in _p("outputs/cfi/cfi_subset_items.txt").read_text().splitlines() if ln.strip()}
        cfi_items = [it for it in cfi_items if it.item_id in sub]
        clean_views = load_vote_views(_p(cfg["sources"]["h1_votes_dir"]), logical_ids)
        for mech in CFI_MECHS:
            bv = load_vote_views(_p("outputs/cfi/votes") / mech, logical_ids)
            if sum(len(x) for x in bv.values()) == 0:
                continue
            var = assemble_variant(cfi_items, logical_ids, bank.by_name(mech), 1.0, bv, clean_views, bank.seed)
            cfi_data[mech] = (var.V, var.M)
    except Exception as e:  # noqa: BLE001
        print("CFI skipped:", str(e)[:100])

    rows, overlap_rows = [], []
    rng_master = np.random.default_rng(20260601)

    def eval_on(idx_eval, O, M_, gold, R_for_corr, mask_learned, mask_possens_, attack, rate, size, seed, affirmed):
        Oe, Me, ge = O[idx_eval], M_[idx_eval], gold[idx_eval]
        label = majority_consensus(Oe, Me)
        n_match = max(int((supermajority_consensus(Oe, Me, 0.75) != ABSTAIN).sum()), 1)
        pool = (label == 1) if affirmed else np.ones(len(ge), bool)

        def keep_top(score):
            s = np.where(pool, score, -np.inf)
            k, _ = retention_match_threshold(s, n_match)
            return k & pool

        out = []

        def add(method, keep):
            m = _metrics(keep, ge); m.update({"attack": attack, "corruption_rate": rate,
                                              "calibration_size": size, "seed": seed, "method": method})
            out.append(m)

        add("naive_majority", keep_top(np.abs(vote_fraction(Oe, Me) - 0.5)))
        add("naive_supermajority_75", keep_top(consensus_level(Oe, Me)))
        add("corrfilter_clean_R", keep_top(np.nan_to_num(corrfilter_score(Oe, Me, R_h1, label).score, nan=-np.inf)))
        add("corrfilter_small_gold_R", keep_top(np.nan_to_num(corrfilter_score(Oe, Me, R_for_corr, label).score, nan=-np.inf)))
        add("bias_cluster_H1", keep_top(_indep_support(Oe, Me, label, mask_h1)))
        add("learned_bias_cluster", keep_top(_indep_support(Oe, Me, label, mask_learned)))
        add("learned_possens", keep_top(_indep_support(Oe, Me, label, mask_possens_)))
        add("oracle_filter", keep_top(np.where(ge == 1, 1.0, -np.inf)))
        return out

    # UF attacks (position, h2b): poison framework, affirmed-keep, held-out eval.
    for (attack, rate), (Va, Ma, pm) in attacks.items():
        gold = np.where(pm, 0, 1).astype(np.int8)
        O = Va.copy(); O[pm] = 1 - Va[pm]
        for size in SIZES:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(7919 * (int(rate * 100) + 1) + 131 * seed + size)
                cal_idx = rng.choice(n, size=min(size, n), replace=False)
                eval_mask = np.ones(n, bool); eval_mask[cal_idx] = False
                eval_idx = np.where(eval_mask)[0]
                C_learned = learn_cluster(O[cal_idx], Ma[cal_idx], gold[cal_idx])
                mask_learned = np.array([j in C_learned for j in range(len(logical_ids))])
                R_sg = small_gold_R(O, Ma, gold, cal_idx)
                rows += eval_on(eval_idx, O, Ma, gold, R_sg, mask_learned, mask_possens, attack, rate, size, seed, True)
                if attack == "position":
                    overlap_rows.append({"attack": attack, "corruption_rate": rate, "calibration_size": size,
                                         "seed": seed, "n_overlap_with_H1": len(C_learned & C_h1),
                                         "jaccard_with_H1": round(len(C_learned & C_h1) / len(C_learned | C_h1), 3),
                                         "learned_cluster": "|".join(sorted(logical_ids[j] for j in C_learned))})

    # CFI attacks: gold = majority-correct, keep-among-all.
    for mech, (Vc, Mc) in cfi_data.items():
        nc = Vc.shape[0]
        gold = (majority_consensus(Vc, Mc) == 1).astype(np.int8)
        for size in SIZES:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(104729 + 131 * seed + size + sum(map(ord, mech)))
                cal_idx = rng.choice(nc, size=min(size, nc), replace=False)
                eval_mask = np.ones(nc, bool); eval_mask[cal_idx] = False
                eval_idx = np.where(eval_mask)[0]
                C_learned = learn_cluster(Vc[cal_idx], Mc[cal_idx], gold[cal_idx])
                mask_learned = np.array([j in C_learned for j in range(len(logical_ids))])
                R_sg = small_gold_R(Vc, Mc, gold, cal_idx)
                rows += eval_on(eval_idx, Vc, Mc, gold, R_sg, mask_learned, mask_possens,
                                f"cfi:{mech[:10]}", 1.0, size, seed, False)

    df = pd.DataFrame(rows); df.to_csv(out_dir / "learned_bias_cluster_results.csv", index=False)
    ov = pd.DataFrame(overlap_rows); ov.to_csv(out_dir / "cluster_overlap_analysis.csv", index=False)
    _figures(df, fig_dir)
    _summary(df, ov, out_dir / "learned_bias_cluster_summary.md", logical_ids, C_h1, C_possens)
    print(f"wrote {out_dir}  ({len(df)} rows)")


def _figures(df, fig_dir):
    pos = df[df.attack == "position"]
    if pos.empty:
        return
    rates = sorted(pos.corruption_rate.unique())
    # calibration-size curve: learned_bias_cluster precision vs size, per rate, with H1 + naive refs.
    fig, axes = plt.subplots(1, len(rates), figsize=(5 * len(rates), 4.5), squeeze=False)
    for ci, r in enumerate(rates):
        ax = axes[0][ci]; sr = pos[pos.corruption_rate == r]
        for method, style in [("learned_bias_cluster", "o-"), ("learned_possens", "^-")]:
            g = sr[sr.method == method].groupby("calibration_size")["precision"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g.calibration_size, g["mean"], yerr=g["std"], fmt=style, capsize=3, label=method)
        for method, c in [("bias_cluster_H1", "green"), ("naive_majority", "gray"), ("oracle_filter", "black")]:
            v = sr[sr.method == method]["precision"].mean()
            ax.axhline(v, ls="--", color=c, alpha=0.7, label=f"{method} ({v:.3f})")
        ax.set_title(f"position {r:.0%}"); ax.set_xlabel("calibration labels"); ax.set_ylabel("precision (held-out)")
        ax.legend(fontsize=6)
    fig.suptitle("Learned vs imported bias cluster — precision vs calibration size")
    fig.tight_layout(); fig.savefig(fig_dir / "calibration_size_curve.png", dpi=150); plt.close(fig)


def _summary(df, ov, path, logical_ids, C_h1, C_possens):
    methods = ["naive_majority", "naive_supermajority_75", "corrfilter_clean_R",
               "corrfilter_small_gold_R", "bias_cluster_H1", "learned_bias_cluster",
               "learned_possens", "oracle_filter"]
    lines = ["# Small-Calibration Learned Bias Cluster\n"]
    lines.append("Analysis-only, no new inference. The vulnerable cluster is LEARNED from K labeled "
                 "calibration items (poison-affirmation rate) and applied to the HELD-OUT remainder; "
                 "all methods are evaluated on the same held-out split (no leakage). Matched retention "
                 "= supermajority-0.75.\n")
    lines.append(f"H1 cluster: {', '.join(sorted(logical_ids[j] for j in C_h1))}.")
    lines.append(f"Label-free position-sensitivity cluster: {', '.join(sorted(logical_ids[j] for j in C_possens))}.\n")

    pos = df[df.attack == "position"]
    lines.append("## Q2/Q3/Q4 — Position attack: precision by method and calibration size (mean over seeds)\n")
    for r in sorted(pos.corruption_rate.unique()):
        sr = pos[pos.corruption_rate == r]
        nm = sr[sr.method == "naive_majority"]["precision"].mean()
        lines.append(f"### {r:.0%} poisoning (naive majority {nm:.3f}, "
                     f"oracle {sr[sr.method=='oracle_filter']['precision'].mean():.3f})\n")
        lines.append("| size | learned_bias_cluster | gain vs maj | bias_cluster_H1 (gain) | learned_possens (gain) | small_gold |")
        lines.append("|---|---|---|---|---|---|")
        h1v = sr[sr.method == "bias_cluster_H1"]["precision"].mean()
        psv = sr[sr.method == "learned_possens"]["precision"].mean()
        sgv = sr[sr.method == "corrfilter_small_gold_R"]["precision"].mean()
        for size in SIZES:
            lv = sr[(sr.method == "learned_bias_cluster") & (sr.calibration_size == size)]["precision"].mean()
            lines.append(f"| {size} | {lv:.3f} | {100*(lv-nm):+.1f} | {h1v:.3f} ({100*(h1v-nm):+.1f}) | "
                         f"{psv:.3f} ({100*(psv-nm):+.1f}) | {sgv:.3f} |")
        lines.append("")

    # Q5 cluster overlap
    lines.append("## Q5 — Does the learned cluster resemble the H1 position cluster? (overlap, position)\n")
    if not ov.empty:
        ovg = ov.groupby("calibration_size").agg(mean_overlap=("n_overlap_with_H1", "mean"),
                                                 mean_jaccard=("jaccard_with_H1", "mean")).reset_index()
        lines.append("| calib size | mean shared judges (of 5) | mean Jaccard with H1 |")
        lines.append("|---|---|---|")
        for _, rr in ovg.iterrows():
            lines.append(f"| {int(rr.calibration_size)} | {rr.mean_overlap:.2f} | {rr.mean_jaccard:.2f} |")
        lines.append("")

    # secondary attacks summary
    for attack in [a for a in sorted(df.attack.unique()) if a not in ("position",)]:
        sa = df[df.attack == attack]
        lines.append(f"## Secondary — {attack}: precision (mean over sizes/seeds)\n")
        lines.append("| " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * len(methods))
        lines.append("| " + " | ".join(f"{sa[sa.method==m]['precision'].mean():.3f}" for m in methods) + " |")
        lines.append("")

    # Verdict
    best_learned = -1e9; best_cfg = None
    for r in sorted(pos.corruption_rate.unique()):
        sr = pos[pos.corruption_rate == r]; nm = sr[sr.method == "naive_majority"]["precision"].mean()
        for size in SIZES:
            lv = sr[(sr.method == "learned_bias_cluster") & (sr.calibration_size == size)]["precision"].mean()
            g = 100 * (lv - nm)
            if g > best_learned:
                best_learned, best_cfg = g, (r, size)
    h1_best = -1e9
    for r in sorted(pos.corruption_rate.unique()):
        sr = pos[pos.corruption_rate == r]; nm = sr[sr.method == "naive_majority"]["precision"].mean()
        h1_best = max(h1_best, 100 * (sr[sr.method == "bias_cluster_H1"]["precision"].mean() - nm))
    lines.append("## Verdict\n")
    lines.append(f"- **Q1 (learnable?)** Best learned_bias_cluster gain over naive on the position attack: "
                 f"**{best_learned:+.1f} pts** at {best_cfg[0]:.0%} poisoning, {best_cfg[1]} labels.")
    lines.append(f"- **Q2 (recover +4.8?)** H1-imported bias_cluster best gain = {h1_best:+.1f} pts; "
                 f"learned best = {best_learned:+.1f} pts "
                 f"({'recovered' if best_learned >= h1_best - 1.0 else 'did NOT fully recover'}).")
    lines.append(f"- **Q3 (>+5?)** {'YES' if best_learned >= 5.0 else 'NO'} (best {best_learned:+.1f}).")
    lines.append("- **Q4 (labels needed)** see the per-size tables above.")
    lines.append("- **Q5 (resembles H1?)** see overlap table.\n")
    deployable = best_learned >= h1_best - 1.0 and best_learned >= 2.0
    if deployable:
        lines.append("**CONCLUSION: bias-aware filtering is deployable with a small calibration budget** — "
                     "the learned cluster matches the H1-imported cluster's benefit within ~1 pt.\n")
    else:
        lines.append("**CONCLUSION: the bias-cluster approach still depends on prior measurement** — "
                     "the small-calibration learned cluster does not reliably match the H1-imported "
                     "benefit, so it is not yet deployable from in-domain labels alone.\n")
    lines.append("## Artefacts\n- `learned_bias_cluster_results.csv`, `cluster_overlap_analysis.csv`\n"
                 "- `figures/calibration_size_curve.png`\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

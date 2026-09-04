"""Position-Aligned Poisoning Analysis (H2b follow-up, NO new inference).

H2b used content poisoning (verbosity/length/polite/style) and found no
correlated co-failure: CorrFilter ~= naive consensus. Here we instead attack the
bank's empirically dominant shared bias --- position --- reusing the cached
20k-vote UltraFeedback run.

Construction (no inference): the vote cache stores ``position_swapped`` per
judge x item, so we reconstruct each judge's *slot* pick
``slotA_pick = vote XOR swap``. Judges are weighted by their H1-measured
position-bias magnitude w_j = |slot-A/B accuracy gap| / 100. We then poison
items by the position-weighted mass of judges that picked the truly-worse
response, ``score_i = sum_j w_j * (1 - V[i,j])`` --- i.e. items where strongly
position-biased judges co-fail. Flipping the label there makes the position-biased
subset co-affirm the bad label (the realistic correlated-poisoning regime).

We then compare the 7 filtering methods at matched retention and run correlation
diagnostics (drift vs the clean UF bank, eigenvector overlap, mean off-diagonal,
effective ensemble size, per-judge contribution) to test whether attacking a
*real* shared bias changes the H2b conclusion.

Outputs: outputs/position_poisoning/{position_poisoning_results.csv,
position_poisoning_diagnostics.csv, position_poisoning_summary.md, figures/}.

Usage:
    python scripts/regimes/run_position_aligned_poisoning.py --project-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corrfilter.cfi.adaptive_r import (  # noqa: E402
    disagreement_R, eigenvector_overlap, frobenius_distance, hybrid_R,
    oracle_variant_R, small_gold_R,
)
from corrfilter.cfi.consensus import (  # noqa: E402
    consensus_level, majority_consensus, supermajority_consensus, vote_fraction,
)
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold  # noqa: E402
from corrfilter.correlation import effective_size  # noqa: E402
from corrfilter.correlation.effective_size import mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache  # noqa: E402

ABSTAIN = -1
RATES = {"05": 0.05, "10": 0.10, "20": 0.20}
CALIB_SIZES = [25, 50, 100, 200]
N_SEEDS = 10
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HYBRID_SG_SIZE = 100

# H1-measured per-judge position-bias magnitude (|no-swap - swap accuracy gap|, pp)
# from reports/h1_overleaf Table "Per-judge accuracy and position-bias diagnostic".
H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}


def _load_votes_swaps(cache: VoteCache, logical_ids, item_ids):
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


def _keep_topk_affirmed(score, label, n_keep):
    affirm = label == 1
    s = np.where(affirm, score, -np.inf)
    keep, _ = retention_match_threshold(s, n_keep)
    return keep & affirm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--manifest", default="outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
    ap.add_argument("--votes-dir", default="outputs/synthetic_poisoned_ultrafeedback/judge_votes")
    ap.add_argument("--npz", default="experiments/h1_measurement/results/correlation.npz")
    ap.add_argument("--out-dir", default="outputs/position_poisoning")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    out_dir = _p(args.out_dir); fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(_p(args.manifest))
    item_ids = man["item_id"].astype(str).tolist()
    bank_cfg = load_bank_config(str(_p(args.bank)))
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    npz = np.load(_p(args.npz), allow_pickle=True)
    R_h1 = npz["R"]

    cache = VoteCache(_p(args.votes_dir))
    V, M, S = _load_votes_swaps(cache, logical_ids, item_ids)
    if M.mean() < 0.95:
        print(f"WARNING: vote coverage {M.mean():.1%} (<95%); results may be partial.")

    # Reconstruct slot picks and per-judge position-bias weight (H1-measured).
    slotA = np.where(S == 1, 1 - V, V)  # did judge pick slot A?
    w = np.array([H1_POSITION_GAP.get(lid, 0.0) / 100.0 for lid in logical_ids])
    # Position-weighted mass of judges that picked the truly-worse response.
    wrong = (1 - V) * M
    score = (wrong * w[None, :]).sum(axis=1)

    order = np.argsort(-score, kind="stable")
    n = len(item_ids)
    R_clean_uf = disagreement_R(V, M)              # clean UF deployment correlation (baseline)
    mu_clean = mean_off_diagonal(R_clean_uf); neff_clean = effective_size(R_clean_uf)

    rng_global = np.random.default_rng(20260601)
    rows, calib_rows, diag_rows = [], [], []
    methods = ["naive_majority", "naive_supermajority_75", "corrfilter_clean_R",
               "corrfilter_disagreement_R", "corrfilter_small_gold_R",
               "corrfilter_hybrid_R", "corrfilter_oracle_variant_R", "oracle_filter"]

    for tag, rate in RATES.items():
        n_pois = int(round(rate * n))
        poison_idx = set(order[:n_pois].tolist())
        poisoned = np.array([i in poison_idx for i in range(n)])
        gold = np.where(poisoned, 0, 1).astype(np.int8)
        O = V.copy(); O[poisoned] = 1 - V[poisoned]
        label = majority_consensus(O, M)
        n_match = max(int((supermajority_consensus(O, M, 0.75) != ABSTAIN).sum()), 1)

        R_oracle = oracle_variant_R(O, M, gold)
        R_dis = disagreement_R(O, M)

        def cf_keep(R):
            sc = corrfilter_score(O, M, R, label)
            return _keep_topk_affirmed(np.nan_to_num(sc.score, nan=-np.inf), label, n_match)

        def add(method, keep, **extra):
            m = _metrics(keep, gold); m.update({"corruption_rate": rate, "method": method, **extra})
            rows.append(m)

        add("naive_majority", _keep_topk_affirmed(np.abs(vote_fraction(O, M) - 0.5), label, n_match))
        add("naive_supermajority_75", _keep_topk_affirmed(consensus_level(O, M), label, n_match))
        add("corrfilter_clean_R", cf_keep(R_h1))
        add("corrfilter_disagreement_R", cf_keep(R_dis))
        add("corrfilter_oracle_variant_R", cf_keep(R_oracle))
        add("oracle_filter", _keep_topk_affirmed(np.where(gold == 1, 1.0, -np.inf), np.ones_like(label), n_match))

        sg_for_hybrid = []
        for size in CALIB_SIZES:
            eff = min(size, n)
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(7919 * (int(rate * 100) + 1) + 131 * seed + size)
                idx = rng.choice(n, size=eff, replace=False)
                R_sg = small_gold_R(O, M, gold, idx)
                m = _metrics(cf_keep(R_sg), gold)
                m.update({"corruption_rate": rate, "method": "corrfilter_small_gold_R",
                          "calibration_size": size, "seed": seed})
                calib_rows.append(m)
                if size == HYBRID_SG_SIZE:
                    sg_for_hybrid.append(R_sg)
        add("corrfilter_small_gold_R", cf_keep(np.mean(sg_for_hybrid, axis=0)), calibration_size=HYBRID_SG_SIZE)
        for lam in LAMBDAS:
            add("corrfilter_hybrid_R", cf_keep(hybrid_R(R_h1, R_dis, lam)), lam=lam)

        # Diagnostics: correlation drift induced by poisoning (O vs clean UF V).
        mu_p = mean_off_diagonal(R_dis); neff_p = effective_size(R_dis)
        frac_majority_affirm = float(((label == 1) & poisoned).sum() / max(poisoned.sum(), 1))
        diag_rows.append({
            "corruption_rate": rate, "n_poisoned": int(poisoned.sum()),
            "frac_poison_majority_affirmed": round(frac_majority_affirm, 4),
            "mean_offdiag_clean": round(mu_clean, 4), "mean_offdiag_poisoned": round(mu_p, 4),
            "neff_clean": round(neff_clean, 3), "neff_poisoned": round(neff_p, 3),
            "frobenius_drift_O_vs_clean": round(frobenius_distance(R_dis, R_clean_uf), 4),
            "eig_overlap_O_vs_clean": round(eigenvector_overlap(R_dis, R_clean_uf, 3), 4),
            "frobenius_O_vs_oracle": round(frobenius_distance(R_dis, R_oracle), 4),
            "eig_overlap_O_vs_oracle": round(eigenvector_overlap(R_dis, R_oracle, 3), 4),
        })
        print(f"rate {rate:.0%}: poison={int(poisoned.sum())}, maj-affirmed={frac_majority_affirm:.0%}, "
              f"mean_offdiag {mu_clean:.3f}->{mu_p:.3f}, n_eff {neff_clean:.2f}->{neff_p:.2f}, "
              f"eig-overlap(O,clean)={diag_rows[-1]['eig_overlap_O_vs_clean']:.3f}")

    df = pd.DataFrame(rows); df.to_csv(out_dir / "position_poisoning_results.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out_dir / "position_poisoning_calibration.csv", index=False)
    diag = pd.DataFrame(diag_rows); diag.to_csv(out_dir / "position_poisoning_diagnostics.csv", index=False)

    # Judge contributions: position weight + loading on the poisoned (20%) drift eigenvector.
    O20 = V.copy(); p20 = np.array([i in set(order[:int(0.20 * n)]) for i in range(n)]); O20[p20] = 1 - V[p20]
    R20 = disagreement_R(O20, M)
    ev = np.linalg.eigh((R20 + R20.T) / 2.0)[1][:, -1]
    judge_contrib = pd.DataFrame({
        "judge": logical_ids, "h1_position_weight": w,
        "uf_slotA_rate": slotA.sum(0) / np.maximum(M.sum(0), 1),
        "top_eigvec_loading_poisoned20": np.round(ev, 3),
    }).sort_values("h1_position_weight", ascending=False)
    judge_contrib.to_csv(out_dir / "position_poisoning_judge_contrib.csv", index=False)

    _figures(df, diag, judge_contrib, fig_dir)
    _summary(df, diag, judge_contrib, out_dir / "position_poisoning_summary.md", n)
    print(f"wrote results to {out_dir}")


def _figures(df, diag, jc, fig_dir):
    rates = sorted(df.corruption_rate.unique())
    pm = ["naive_majority", "naive_supermajority_75", "corrfilter_clean_R",
          "corrfilter_disagreement_R", "corrfilter_small_gold_R", "corrfilter_oracle_variant_R", "oracle_filter"]
    # precision by method per rate
    fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(pm)); w = 0.25
    for i, r in enumerate(rates):
        vals = [df[(df.method == m) & (df.corruption_rate == r)]["precision"].mean() for m in pm]
        ax.bar(x + (i - 1) * w, vals, w, label=f"{r:.0%}")
    ax.set_xticks(x); ax.set_xticklabels(pm, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("precision"); ax.set_title("Position-aligned poisoning: precision by method"); ax.legend(title="corruption")
    fig.tight_layout(); fig.savefig(fig_dir / "precision_by_method.png", dpi=150); plt.close(fig)

    # gain of small_gold adaptive over naive, per rate
    fig, ax = plt.subplots(figsize=(8, 5))
    gmaj = [100 * (df[(df.method == "corrfilter_small_gold_R") & (df.corruption_rate == r)]["precision"].mean()
                   - df[(df.method == "naive_majority") & (df.corruption_rate == r)]["precision"].mean()) for r in rates]
    gsup = [100 * (df[(df.method == "corrfilter_small_gold_R") & (df.corruption_rate == r)]["precision"].mean()
                   - df[(df.method == "naive_supermajority_75") & (df.corruption_rate == r)]["precision"].mean()) for r in rates]
    xr = np.arange(len(rates))
    ax.bar(xr - 0.2, gmaj, 0.4, label="vs majority"); ax.bar(xr + 0.2, gsup, 0.4, label="vs supermajority")
    ax.axhline(5, ls="--", color="green", label="+5 pt target"); ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xr); ax.set_xticklabels([f"{r:.0%}" for r in rates])
    ax.set_ylabel("precision gain (pts)"); ax.set_title("Adaptive CorrFilter gain over naive k-of-n"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "gain_over_naive.png", dpi=150); plt.close(fig)

    # correlation drift diagnostics
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(diag.corruption_rate, diag.mean_offdiag_poisoned, "o-", label="mean off-diag (poisoned)")
    ax1.axhline(diag.mean_offdiag_clean.iloc[0], ls="--", color="gray", label="mean off-diag (clean)")
    ax1.set_xlabel("corruption rate"); ax1.set_ylabel("mean off-diagonal correlation")
    ax2 = ax1.twinx(); ax2.plot(diag.corruption_rate, diag.eig_overlap_O_vs_clean, "s-", color="#c0392b", label="eig-overlap vs clean")
    ax2.set_ylabel("top-3 eigenvector overlap vs clean", color="#c0392b"); ax2.set_ylim(0, 1.02)
    ax1.legend(loc="center left"); ax1.set_title("Correlation drift induced by position poisoning")
    fig.tight_layout(); fig.savefig(fig_dir / "correlation_drift.png", dpi=150); plt.close(fig)

    # judge position-bias weights
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(jc)), jc["h1_position_weight"], color="#4878a8")
    ax.set_xticks(range(len(jc))); ax.set_xticklabels(jc["judge"], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("H1 position-bias weight |gap|/100"); ax.set_title("Per-judge contribution to position failure mode")
    fig.tight_layout(); fig.savefig(fig_dir / "judge_position_weights.png", dpi=150); plt.close(fig)


def _summary(df, diag, jc, path, n_items):
    rates = sorted(df.corruption_rate.unique())

    def g(m, r, c="precision"):
        s = df[(df.method == m) & (df.corruption_rate == r)]
        return s[c].mean() if len(s) else float("nan")

    lines = ["# Position-Aligned Poisoning Analysis (H2b follow-up)\n"]
    lines.append(f"{n_items} UltraFeedback pairs, **no new inference** (reused 20k-vote cache). "
                 "Corruption targets the bank's H1-measured dominant shared bias (position): items "
                 "are poisoned by position-weighted mass of judges picking the truly-worse response. "
                 "Matched retention = supermajority-0.75.\n")

    lines.append("## Did the attack induce correlated co-failure? (diagnostics)\n")
    lines.append("| rate | poison maj-affirmed | mean off-diag (clean->poisoned) | n_eff (clean->poisoned) | eig-overlap vs clean | Frobenius drift |")
    lines.append("|---|---|---|---|---|---|")
    for _, d in diag.iterrows():
        lines.append(f"| {d.corruption_rate:.0%} | {d.frac_poison_majority_affirmed:.0%} | "
                     f"{d.mean_offdiag_clean:.3f} -> {d.mean_offdiag_poisoned:.3f} | "
                     f"{d.neff_clean:.2f} -> {d.neff_poisoned:.2f} | {d.eig_overlap_O_vs_clean:.3f} | "
                     f"{d.frobenius_drift_O_vs_clean:.3f} |")
    lines.append("\n_(H2b content poisoning, for contrast, drifted negligibly: eig-overlap ~0.99.)_\n")

    lines.append("## Filtering: precision / FRR / F1 by method and rate\n")
    pm = ["naive_majority", "naive_supermajority_75", "corrfilter_clean_R",
          "corrfilter_disagreement_R", "corrfilter_small_gold_R", "corrfilter_hybrid_R",
          "corrfilter_oracle_variant_R", "oracle_filter"]
    lines.append("| method | " + " | ".join(f"{r:.0%} P/FRR/F1" for r in rates) + " |")
    lines.append("|" + "---|" * (len(rates) + 1))
    for m in pm:
        cells = []
        for r in rates:
            sub = df[(df.method == m) & (df.corruption_rate == r)]
            if m == "corrfilter_hybrid_R":
                sub = sub[sub.get("lam") == 0.5]
            if len(sub):
                cells.append(f"{sub.precision.mean():.3f}/{sub.false_retention_rate.mean():.3f}/{sub.f1.mean():.3f}")
            else:
                cells.append("--")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Gain of Small-Calibrated Adaptive CorrFilter over naive k-of-n\n")
    lines.append("| rate | naive maj | naive super | small_gold adaptive | gain vs maj | gain vs super | oracle |")
    lines.append("|---|---|---|---|---|---|---|")
    gains = []
    for r in rates:
        nm, ns, sg, orc = g("naive_majority", r), g("naive_supermajority_75", r), g("corrfilter_small_gold_R", r), g("oracle_filter", r)
        gm, gs = 100 * (sg - nm), 100 * (sg - ns)
        gains.append(max(gm, gs))
        lines.append(f"| {r:.0%} | {nm:.3f} | {ns:.3f} | {sg:.3f} | {gm:+.1f} | {gs:+.1f} | {orc:.3f} |")
    lines.append("")

    best = max(gains) if gains else 0.0
    # Verdict: does CorrFilter help when attacking a real shared bias?
    drift_happened = (diag.mean_offdiag_poisoned > diag.mean_offdiag_clean + 0.02).any() or \
                     (diag.eig_overlap_O_vs_clean < 0.97).any()
    if best >= 5.0:
        verdict = ("**A) CorrFilter helps under correlated failure.** Attacking the measured shared "
                   f"position bias yields a >=5 pt precision gain (best {best:+.1f}). 'Consensus is not "
                   "reliability under dependence' is supported when the dependence is real.")
    elif drift_happened:
        verdict = ("**B-with-drift.** The attack DID induce correlation drift (diagnostics above), but "
                   f"CorrFilter still does not clear +5 pts (best {best:+.1f}). Correlation-aware scoring "
                   "detects the dependence in R yet does not convert it into better label filtering here.")
    else:
        verdict = ("**B) No benefit.** Even attacking the strongest shared bias, the poisoning did not "
                   f"induce exploitable correlated co-failure and CorrFilter ~= naive (best {best:+.1f}).")
    lines.append("## Verdict\n")
    lines.append(verdict + "\n")
    lines.append(f"Best adaptive gain over naive k-of-n across rates: **{best:+.1f} pts** "
                 f"(>=5 pt target {'MET' if best >= 5 else 'NOT met'}).\n")

    lines.append("## Which judges drive the position failure mode\n")
    lines.append("| judge | H1 position weight | UF slot-A rate | loading on poisoned-20% top eigvec |")
    lines.append("|---|---|---|---|")
    for _, j in jc.iterrows():
        lines.append(f"| {j.judge} | {j.h1_position_weight:.3f} | {j.uf_slotA_rate:.3f} | {j.top_eigvec_loading_poisoned20:+.3f} |")
    lines.append("")
    lines.append("## Artefacts\n- `position_poisoning_results.csv`, `position_poisoning_diagnostics.csv`, "
                 "`position_poisoning_calibration.csv`, `position_poisoning_judge_contrib.csv`\n"
                 "- `figures/`: precision_by_method, gain_over_naive, correlation_drift, judge_position_weights\n")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()

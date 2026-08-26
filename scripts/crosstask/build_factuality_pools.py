#!/usr/bin/env python
"""Factuality (VitaminC) regime construction + BLOCKING construct-validity gate.

Builds, from the clean pointwise factuality vote bank:
  pure controlled regimes  weak (clean subsamples), global (shared-shock
                           injection), subgroup (frozen-cluster shock), using
                           the verified task-agnostic generators; injection is
                           restricted to NATIVE gold=0 items (truly
                           unsupported claims), so gold labels are never
                           manufactured, only the error dependence is.
  pure real-mechanism      global-real: instances enriched in native
                           consensus-wrong unsupported claims (votes and gold
                           untouched); subgroup-real: enriched in items where
                           the frozen cluster affirms an unsupported claim
                           while the rest of the bank rejects it. Gold labels
                           are used ONLY for this documented enrichment.
  mixed deployments        pairwise {weak+global, weak+subgroup,
                           global+subgroup} x shares {25,50,75} + equal
                           three-way, over the controlled generators.

The vulnerable cluster is the label-free correlation cluster
(routing.correlation_cluster on the clean disagreement R), frozen before any
contaminated instance is evaluated; stability diagnostics are reported.
Evaluation reuses build_crossed_pool.evaluate verbatim (per-source clean
references; matched retention; small-gold split).

The construct-validity gate is BLOCKING: if a regime family fails its
signature, a FAILED marker is written and downstream routing must not run.

Usage: python scripts/crosstask/build_factuality_pools.py
Outputs -> outputs/router_upgrade/fact_{features,methods}.csv,
fact_mixed_{features,methods}.csv, fact_gate.csv, fact_cluster.json,
fact_instance_manifest.jsonl, gate marker fact_gate_PASSED.txt (or _FAILED)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from build_crossed_pool import Source, evaluate, marginal_params, wrong_probs

from corrfilter.cfi.adaptive_r import disagreement_R
from corrfilter.cfi.consensus import majority_consensus
from corrfilter.data.generic_task import batch_from_cache
from corrfilter.judges import load_bank_config
from corrfilter.routing import agree_features, correlation_cluster
from corrfilter.screening.generators import majority_regime
from corrfilter.voting import VoteCache

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]


FB = ROOT / "outputs/factuality_bank"
TAG = __import__("os").environ.get("FACT_TAG", "")
OUT = ROOT / "outputs/router_upgrade"
BASE = 20270101
INSTANCE_SIZE = 300
RATES = [0.10, 0.20]
N_SEEDS = 8
N_SEEDS_REAL = 8
MIX_SHARES = [0.25, 0.50, 0.75]
MIX_SEEDS = 4
CLUSTER_K = int(__import__("os").environ.get("FACT_CLUSTER_K", "5"))


def load_factuality_source():
    man = pd.read_parquet(FB / "manifest_main.parquet")
    bank = load_bank_config(ROOT / "configs/judge_bank.yaml")
    models = sorted({s.base_id for s in bank.specs})
    lids = [f"{m}::{st}" for m in models for st in ("pw_direct", "pw_analysis")]
    batch = batch_from_cache(man, VoteCache(FB / "votes"), lids)
    V, M, gold = batch.judge_votes, batch.availability, batch.gold_labels
    R_fact = disagreement_R(V, M)
    src = Source("fact", V, M, np.zeros_like(V), man.doc_chars.to_numpy(),
                 man.claim_chars.to_numpy(), lids, R_fact)
    src.cpos = correlation_cluster(R_fact, k=CLUSTER_K)  # frozen, label-free
    lab = majority_consensus(V, M)
    fc = agree_features(V, M, lab, src.cpos, R_fact)
    nj = V.shape[1]
    src.ref = {"rho_bar": float(np.mean(R_fact[~np.eye(nj, dtype=bool)])),
               "rho_S": float(np.nanmean(fc["rho"])),
               "lco_flip": float(np.nanmean(fc["lco_flip"]))}
    return src, R_fact, gold, man, lids


def cluster_stability(V, M, R_fact, seed=BASE):
    """Bootstrap Jaccard of the frozen cluster + k-sensitivity."""
    rng = np.random.default_rng(seed)
    base = set(np.where(correlation_cluster(R_fact, k=CLUSTER_K))[0].tolist())
    jac = []
    for _ in range(50):
        b = rng.integers(0, V.shape[0], V.shape[0])
        Rb = disagreement_R(V[b], M[b])
        cb = set(np.where(correlation_cluster(Rb, k=CLUSTER_K))[0].tolist())
        jac.append(len(base & cb) / len(base | cb))
    ks = {k: sorted(np.where(correlation_cluster(R_fact, k=k))[0].tolist())
          for k in (3, 4, 5)}
    nj = R_fact.shape[0]
    mask = np.zeros(nj, bool)
    mask[list(base)] = True
    off = ~np.eye(nj, dtype=bool)
    within = R_fact[np.ix_(mask, mask)][~np.eye(mask.sum(), dtype=bool)].mean()
    cross = R_fact[np.ix_(mask, ~mask)].mean()
    return {"members": sorted(base), "bootstrap_jaccard_mean": float(np.mean(jac)),
            "bootstrap_jaccard_p10": float(np.quantile(jac, 0.1)),
            "k_sensitivity": ks, "within_corr": float(within),
            "cross_corr": float(cross)}


def make_fact_instance(src, gold_all, regime, rate, counter):
    """Controlled instance: uniform items; inject correlated errors into a
    rate-fraction of the instance drawn from NATIVE gold=0 items."""
    rng = np.random.default_rng(BASE + 7919 * counter)
    idx = np.sort(rng.choice(src.n, INSTANCE_SIZE, replace=False))
    O = src.V[idx].copy()
    Md = src.M[idx]
    gold = gold_all[idx].copy()
    if regime == "weak":
        return idx, O, Md, gold
    nj = O.shape[1]
    zeros = np.where(gold == 0)[0]
    n_bad = min(int(round(rate * INSTANCE_SIZE)), len(zeros))
    bad = rng.choice(zeros, n_bad, replace=False)
    q_rest, p_z = marginal_params(nj, src.cpos)
    for t in bad:
        p = wrong_probs(regime, nj, src.cpos, rng, q_rest, p_z)
        wrong = rng.random(nj) < p           # wrong = AFFIRM the unsupported claim
        O[t] = np.where(Md[t] > 0, np.where(wrong, 1, 0), O[t])
    return idx, O, Md, gold


def make_real_instance(src, gold_all, kind, rate, counter):
    """Real-mechanism instance: enrichment only; votes and gold untouched."""
    rng = np.random.default_rng(BASE + 104729 * counter + 13)
    V, M, gold = src.V, src.M, gold_all
    maj = (np.where(M, V, 0).sum(1) > (M.sum(1) / 2))
    cl = src.cpos
    cl_maj = (np.where(M[:, cl], V[:, cl], 0).sum(1) > (M[:, cl].sum(1) / 2))
    rest = ~cl
    rest_maj = (np.where(M[:, rest], V[:, rest], 0).sum(1) > (M[:, rest].sum(1) / 2))
    if kind == "global_real":
        pool = np.where((gold == 0) & maj)[0]                 # bank-wide blind spot
    else:  # subgroup_real
        pool = np.where((gold == 0) & cl_maj & ~rest_maj)[0]  # cluster-only blind spot
    n_p = min(int(round(rate * INSTANCE_SIZE)), len(pool))
    pi = rng.choice(pool, n_p, replace=False)
    others = np.setdiff1d(np.arange(src.n), pool)
    ci = rng.choice(others, INSTANCE_SIZE - n_p, replace=False)
    idx = np.sort(np.concatenate([pi, ci]))
    return idx, V[idx].copy(), M[idx], gold[idx].copy(), n_p


def make_mixed_fact(src, gold_all, weights, rate, counter):
    rng = np.random.default_rng(BASE + 7919 * counter + 500_000)
    idx = np.sort(rng.choice(src.n, INSTANCE_SIZE, replace=False))
    O = src.V[idx].copy()
    Md = src.M[idx]
    gold = gold_all[idx].copy()
    nj = O.shape[1]
    zeros = np.where(gold == 0)[0]
    n_bad = min(int(round(rate * INSTANCE_SIZE)), len(zeros))
    bad = rng.choice(zeros, n_bad, replace=False)
    regs = sorted(weights)
    counts = {r: int(np.floor(weights[r] * n_bad)) for r in regs}
    rem = n_bad - sum(counts.values())
    for r in sorted(regs, key=lambda r: -(weights[r] * n_bad % 1)):
        if rem <= 0:
            break
        counts[r] += 1
        rem -= 1
    q_rest, p_z = marginal_params(nj, src.cpos)
    pos = 0
    perm = rng.permutation(bad)
    for r in regs:
        for t in perm[pos:pos + counts[r]]:
            if r == "weak":
                continue                      # weak share keeps real votes
            p = wrong_probs(r, nj, src.cpos, rng, q_rest, p_z)
            wrong = rng.random(nj) < p
            O[t] = np.where(Md[t] > 0, np.where(wrong, 1, 0), O[t])
        pos += counts[r]
    return idx, O, Md, gold, counts


def main() -> None:
    src, R_fact, gold_all, man, lids = load_factuality_source()
    nj = src.V.shape[1]
    print(f"factuality source: {src.n} items, {nj} judges, "
          f"clean rho(votes) {src.ref['rho_bar']:.3f}, supported {gold_all.mean():.2f}, "
          f"cluster {sorted(np.where(src.cpos)[0].tolist())}")
    stab = cluster_stability(src.V, src.M, R_fact)
    (OUT / f"fact{TAG}_cluster.json").write_text(json.dumps(
        {"lids": lids, **stab}, indent=2))
    print("cluster stability:", {k: v for k, v in stab.items() if k != "k_sensitivity"})

    manifest_rows, feat_rows, method_rows, sig_rows = [], [], [], []
    counter = 0

    def record(name, cfg, regime, rate, idx, metrics, fvec, pred_hard, d, extra=None):
        for f, m in metrics.items():
            method_rows.append({"instance": name, "true_regime": regime,
                                "source": "fact", "method": f, **m})
        fvec.update({"instance": name, "regime": regime, "source": "fact",
                     "config_id": cfg, "pred_hard": pred_hard, **(extra or {})})
        feat_rows.append(fvec)
        sig_rows.append({"config": cfg, "regime": regime, "rate": rate,
                         **{k: round(v, 4) for k, v in d.items()}})
        manifest_rows.append({"task": "factuality_vitaminc", "instance": name,
                              "config": cfg, "regime": regime, "rate": rate,
                              "item_ids": man.item_id.to_numpy()[idx].tolist()})

    # pure controlled
    for regime in ("weak", "subgroup", "global"):
        for rate in RATES:
            for s in range(N_SEEDS):
                idx, O, Md, gold = make_fact_instance(src, gold_all, regime, rate, counter)
                name = f"fact_{regime}_r{rate:g}_s{s}"
                metrics, fvec, pred_hard, d = evaluate(src, O, Md, gold, counter, R_fact,
                                                       retention_mode="affirmed")
                counter += 1
                record(name, f"fact_{regime}_r{rate:g}", regime, rate, idx,
                       metrics, fvec, pred_hard, d)
    # pure real-mechanism
    for kind in ("global_real", "subgroup_real"):
        base_regime = kind.split("_")[0]
        for rate in RATES:
            for s in range(N_SEEDS_REAL // 2):
                idx, O, Md, gold, n_p = make_real_instance(src, gold_all, kind, rate, counter)
                name = f"fact_{kind}_r{rate:g}_s{s}"
                metrics, fvec, pred_hard, d = evaluate(src, O, Md, gold, counter, R_fact,
                                                       retention_mode="affirmed")
                counter += 1
                record(name, f"fact_{kind}_r{rate:g}", base_regime, rate, idx,
                       metrics, fvec, pred_hard, d, {"variant": "real", "n_enriched": n_p})
    fdf = pd.DataFrame(feat_rows)
    fdf.to_csv(OUT / f"fact{TAG}_features.csv", index=False)
    pd.DataFrame(method_rows).to_csv(OUT / f"fact{TAG}_methods.csv", index=False)

    # mixed (controlled)
    feat_rows, method_rows = [], []
    mixtures = [(f"{a[0]}{b[0]}_{int(100 * sh)}", {a: 1 - sh, b: sh})
                for a, b in [("weak", "global"), ("weak", "subgroup"), ("global", "subgroup")]
                for sh in MIX_SHARES]
    mixtures.append(("3way_eq", {"weak": 1 / 3, "global": 1 / 3, "subgroup": 1 / 3}))
    for mname, weights in mixtures:
        for rate in RATES:
            for s in range(MIX_SEEDS):
                idx, O, Md, gold, counts = make_mixed_fact(src, gold_all, weights, rate, counter)
                name = f"fact_{mname}_r{rate:g}_s{s}"
                metrics, fvec, pred_hard, d = evaluate(src, O, Md, gold, counter, R_fact,
                                                       retention_mode="affirmed")
                counter += 1
                for f, m in metrics.items():
                    method_rows.append({"instance": name, "method": f, **m})
                fvec.update({"instance": name, "config_id": f"fact_{mname}_r{rate:g}",
                             "source": "fact", "mix_name": mname, "rate": rate,
                             "majority_regime": majority_regime(weights),
                             "pred_hard": pred_hard,
                             **{f"w_{r}": round(weights.get(r, 0.0), 4)
                                for r in ("weak", "global", "subgroup")}})
                feat_rows.append(fvec)
                manifest_rows.append({"task": "factuality_vitaminc", "instance": name,
                                      "config": f"fact_{mname}_r{rate:g}",
                                      "regime": "mixed", "rate": rate,
                                      "weights": weights,
                                      "item_ids": man.item_id.to_numpy()[idx].tolist()})
    pd.DataFrame(feat_rows).to_csv(OUT / f"fact{TAG}_mixed_features.csv", index=False)
    pd.DataFrame(method_rows).to_csv(OUT / f"fact{TAG}_mixed_methods.csv", index=False)
    with open(OUT / f"fact{TAG}_instance_manifest.jsonl", "w") as fh:
        for r in manifest_rows:
            fh.write(json.dumps(r) + "\n")

    # ---- BLOCKING construct-validity gate ----
    md = pd.read_csv(OUT / f"fact{TAG}_methods.csv")
    piv = md.pivot_table(index="instance", columns="method", values="precision")
    fdf = pd.read_csv(OUT / f"fact{TAG}_features.csv")
    fdf["variant"] = fdf.get("variant", pd.Series(index=fdf.index, dtype=object)).fillna("ctrl")
    gate_rows, ok_all = [], True

    def fam(regime, variant):
        m = (fdf.regime == regime) & (fdf.variant == variant)
        ii = fdf.instance[m]
        return fdf[m], piv.loc[piv.index.isin(ii)]

    ref = src.ref
    for regime, variant in [("weak", "ctrl"), ("global", "ctrl"), ("subgroup", "ctrl"),
                            ("global", "real"), ("subgroup", "real")]:
        f, p = fam(regime, variant)
        if len(f) == 0:
            continue
        drift = float(f.rho_bar.mean() - ref["rho_bar"])
        lco = float(f.lco_mean.mean())
        eig = float(f.eig_overlap.mean())
        d_cf = float((p["corrfilter_small_gold_R"] - p["naive_majority"]).mean())
        d_bc = float((p["bias_cluster"] - p["naive_majority"]).mean())
        if regime == "weak":
            passed = abs(drift) < 0.05 and eig > 0.75 and max(d_cf, d_bc) < 0.01
        elif regime == "global":
            passed = drift > 0.02 and d_cf > 0
        else:
            passed = lco > ref["lco_flip"] + 0.02 and d_bc > 0 and d_bc > d_cf
        ok_all &= passed
        gate_rows.append({"regime": regime, "variant": variant, "n": len(f),
                          "corr_drift": round(drift, 4), "eig_overlap": round(eig, 3),
                          "lco_rate": round(lco, 4),
                          "naive_precision": round(float(p["naive_majority"].mean()), 4),
                          "corrfilter_delta": round(d_cf, 4),
                          "biascluster_delta": round(d_bc, 4), "gate_passed": passed})
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT / f"fact{TAG}_gate.csv", index=False)
    print("\n== Construct-validity gate (Table B) ==")
    print(gate.to_string(index=False))
    marker = OUT / ("fact_gate_PASSED.txt" if ok_all else "fact_gate_FAILED.txt")
    for old in (OUT / f"fact{TAG}_gate_PASSED.txt", OUT / f"fact{TAG}_gate_FAILED.txt"):
        old.unlink(missing_ok=True)
    marker.write_text(gate.to_string(index=False))
    print(f"\nGATE {'PASSED' if ok_all else 'FAILED'} -> {marker.name}")


if __name__ == "__main__":
    main()

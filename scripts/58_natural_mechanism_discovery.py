#!/usr/bin/env python
"""P0-2 Part 1: natural non-position mechanism discovery (zero inference).

Pre-registration: experiments/natural_regime/PREREG.md
Primary endpoint: diagnostic inversion, i.e. mean error correlation LOWER on
trigger-firing items than on non-firing items while precision on them is worse.
Strict tie convention throughout.

Puts every candidate mechanism on the same measurement scale as the position gap, so the
question "which shared weakness is strongest" becomes answerable.

Outputs -> outputs/natural_regime/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from compute_dependence import estimate_R, load_votes  # noqa: E402
from corrfilter.cfi import triggers as trg  # noqa: E402
from corrfilter.correlation.effective_size import (effective_size,  # noqa: E402
                                                   mean_off_diagonal)
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.evaluation import evaluable_and_correct  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

OUT = ROOT / "outputs/natural_regime"
SEED = 20260706
B = 2000
MECHANISMS = ("verbosity", "sycophancy", "refusal", "polish")


def fire(items, mech):
    cfg = trg.TriggerConfig()
    fn = {"verbosity": trg.detect_verbosity_trigger,
          "sycophancy": trg.detect_sycophancy_trigger,
          "refusal": trg.detect_refusal_trigger,
          "polish": trg.detect_polish_trigger}[mech]
    return np.array([bool(fn(it, cfg)) for it in items])


def rho_neff(V, M, gold, mask):
    if mask.sum() < 40:
        return np.nan, np.nan
    R, _, _, _ = estimate_R(V[mask], M[mask], gold[mask])
    return float(mean_off_diagonal(R)), float(effective_size(R))


def boot_inversion(V, M, gold, on, off, seed=SEED, B=B):
    """Paired-ish bootstrap of rho(on) - rho(off); negative == inversion."""
    ion, ioff = np.where(on)[0], np.where(off)[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        a = rng.choice(ion, len(ion), replace=True)
        b = rng.choice(ioff, len(ioff), replace=True)
        try:
            Ra, _, _, _ = estimate_R(V[a], M[a], gold[a])
            Rb, _, _, _ = estimate_R(V[b], M[b], gold[b])
            vals.append(mean_off_diagonal(Ra) - mean_off_diagonal(Rb))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan, np.nan
    v = np.array(vals)
    return float(v.mean()), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def per_judge_bias(V, M, gold, on):
    """b_j = P(judge errs | trigger fires) - P(judge errs | not), same scale as slot gap."""
    Mb = M.astype(bool)
    out = []
    for j in range(V.shape[1]):
        a = Mb[:, j] & on
        b = Mb[:, j] & ~on
        if a.sum() < 15 or b.sum() < 15:
            out.append(np.nan); continue
        ea = float((V[a, j] != gold[a]).mean())
        eb = float((V[b, j] != gold[b]).mean())
        out.append(ea - eb)
    return np.array(out, float)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = {}
    items = load_calibration_set(ROOT / "experiments/h1_measurement/results/calibration_manifest.parquet")
    ids = [str(i.item_id) for i in items]
    V, M, lids = load_votes(ROOT / "configs/judge_bank.yaml", ids, None)
    datasets["rewardbench"] = (items, V, M, np.ones(len(ids), np.int8), lids)

    try:
        man = pd.read_csv(ROOT / "outputs/synthetic_poisoned_ultrafeedback/poisoned_manifest.csv")
        from corrfilter.data import CalibrationItem
        uids = man.item_id.astype(str).tolist()
        Vu, Mu = load_vote_matrix(
            VoteCache(ROOT / "outputs/synthetic_poisoned_ultrafeedback/judge_votes"), lids, uids)
        uitems = [CalibrationItem(item_id=str(r.item_id), prompt=str(r.get("prompt", "")),
                                  chosen=str(r.get("chosen", "")), rejected=str(r.get("rejected", "")),
                                  subset=str(r.get("subset", "")), category=str(r.get("category", "")),
                                  gold_label=1)
                  for _, r in man.iterrows()]
        datasets["ultrafeedback"] = (uitems, Vu, Mu, np.ones(len(uids), np.int8), lids)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] ultrafeedback unavailable: {exc}")

    rows, prim = [], []
    for dsname, (its, Vd, Md, gold, lid) in datasets.items():
        base_rho, base_neff = rho_neff(Vd, Md, gold, np.ones(len(gold), bool))
        for mech in MECHANISMS:
            try:
                on = fire(its, mech)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {dsname}/{mech}: {exc}")
                continue
            off = ~on
            if on.sum() < 40 or off.sum() < 40:
                rows.append({"dataset": dsname, "mechanism": mech, "n_on": int(on.sum()),
                             "n_off": int(off.sum()), "note": "insufficient support"})
                continue
            r_on, ne_on = rho_neff(Vd, Md, gold, on)
            r_off, ne_off = rho_neff(Vd, Md, gold, off)
            _, ev_on, corr_on = evaluable_and_correct(Vd[on], Md[on], gold[on])
            _, ev_off, corr_off = evaluable_and_correct(Vd[off], Md[off], gold[off])
            prec_on = float(corr_on.sum() / max(ev_on.sum(), 1))
            prec_off = float(corr_off.sum() / max(ev_off.sum(), 1))
            bj = per_judge_bias(Vd, Md, gold, on)
            d, lo, hi = boot_inversion(Vd, Md, gold, on, off)
            # one-sided p for inversion (rho_on < rho_off)
            p = float(np.mean(np.array([d]) >= 0)) if np.isnan(d) else None
            rows.append({
                "dataset": dsname, "mechanism": mech,
                "n_on": int(on.sum()), "n_off": int(off.sum()),
                "rho_on": round(r_on, 4), "rho_off": round(r_off, 4),
                "delta_rho": round(d, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "inversion": bool(hi < 0),
                "n_eff_on": round(ne_on, 3), "n_eff_off": round(ne_off, 3),
                "precision_on": round(prec_on, 4), "precision_off": round(prec_off, 4),
                "precision_worse_on": bool(prec_on < prec_off),
                "max_abs_judge_bias": round(float(np.nanmax(np.abs(bj))), 4),
                "judge_bias_spread": round(float(np.nanstd(bj)), 4),
                "regime_signature": ("inversion (Regime III)" if (hi < 0 and prec_on < prec_off)
                                     else "raised correlation (Regime I)" if (lo > 0)
                                     else "no clear signature"),
            })
            prim.append((dsname, mech, d, lo, hi))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mechanism_scan.csv", index=False)

    pd.set_option("display.width", 240)
    cols = [c for c in ["dataset", "mechanism", "n_on", "n_off", "rho_on", "rho_off",
                        "delta_rho", "ci_low", "ci_high", "inversion", "precision_on",
                        "precision_off", "max_abs_judge_bias", "judge_bias_spread",
                        "regime_signature"] if c in df.columns]
    print("\n=== mechanism scan: trigger-on vs trigger-off ===")
    print(df[cols].to_string(index=False))
    n_inv = int(df.get("inversion", pd.Series(dtype=bool)).sum())
    print(f"\nmechanisms showing inversion (raw CI): {n_inv} of {len(prim)} tested")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

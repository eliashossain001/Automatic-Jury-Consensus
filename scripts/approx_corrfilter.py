"""Bucket D (optional) — low-rank / factor approximation of R for CorrFilter at scale.

For a large bank, storing and inverting the full n x n R is unnecessary: the
dependence is low-rank (a few factors carry most of the correlation). We replace R
with its rank-k eigen-truncation (renormalised to unit diagonal) and re-run the
CorrFilter alpha_subset filter, comparing the false-retention reduction against the
full-R filter. If rank 1-3 recovers the full-R gain, deployment only needs the top few
factors of R. CPU-only.

Outputs outputs/deployment/approx_corrfilter.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from bootstrap_ci import paired_frr_gain  # noqa: E402
from compute_dependence import estimate_R, item_ids_from_manifest, load_votes  # noqa: E402


def low_rank_R(R, k):
    w, v = np.linalg.eigh((R + R.T) / 2)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    w[k:] = 0.0
    Rk = (v * w) @ v.T
    d = np.sqrt(np.clip(np.diag(Rk), 1e-9, None))
    Rk = Rk / (d[:, None] * d[None, :]); np.fill_diagonal(Rk, 1.0)
    return Rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/judge_bank.yaml")
    ap.add_argument("--manifest", default="experiments/h1_measurement/results/calibration_manifest.parquet")
    ap.add_argument("--retention", type=float, default=0.60)
    ap.add_argument("--judge-id", action="append", default=None)
    ap.add_argument("--out", default="outputs/deployment/approx_corrfilter.csv")
    args = ap.parse_args()

    item_ids = item_ids_from_manifest(args.manifest)
    V, M, lids = load_votes(args.bank, item_ids, args.judge_id)
    gold = np.ones(len(item_ids), dtype=np.int8)
    R, _, _, _ = estimate_R(V, M, gold)
    m = R.shape[0]

    rows = []
    full = paired_frr_gain(V, M, R, gold, retention=args.retention)
    rows.append({"R_variant": "full", "rank": m, "gain_pts": full["gain_pts"],
                 "ci_low": full["ci_low_pts"], "ci_high": full["ci_high_pts"]})
    for k in [1, 2, 3, min(5, m)]:
        Rk = low_rank_R(R, k)
        g = paired_frr_gain(V, M, Rk, gold, retention=args.retention)
        rows.append({"R_variant": f"rank-{k}", "rank": k, "gain_pts": g["gain_pts"],
                     "ci_low": g["ci_low_pts"], "ci_high": g["ci_high_pts"]})
    # identity (no correlation correction) baseline
    gi = paired_frr_gain(V, M, np.eye(m), gold, retention=args.retention)
    rows.append({"R_variant": "identity (no correction)", "rank": 0, "gain_pts": gi["gain_pts"],
                 "ci_low": gi["ci_low_pts"], "ci_high": gi["ci_high_pts"]})

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

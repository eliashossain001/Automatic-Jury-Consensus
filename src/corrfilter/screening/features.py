"""Label-free deployment features for regime identification.

The 35-feature extractor consumed by the learned regime routers. Every feature
is computable at deployment time from votes alone: no gold labels are used
anywhere in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from corrfilter.cfi.adaptive_r import eigenvector_overlap
from corrfilter.cfi.consensus import majority_consensus


def extract_features(Oe, Me, label, feats, R_O, R_clean_uf, R_h1, ref, cpos_mask):
    """Label-free deployment features from the eval split (no gold anywhere)."""
    Mb = Me.astype(bool)
    n_j = Oe.shape[1]
    frac = np.where(Mb.sum(1) > 0, ((Oe == 1) & Mb).sum(1) / np.maximum(Mb.sum(1), 1), 0.5)
    margin = np.abs(frac - 0.5)
    pf = np.clip(frac, 1e-6, 1 - 1e-6)
    entropy = -(pf * np.log2(pf) + (1 - pf) * np.log2(1 - pf))

    # pairwise judge disagreement rates
    pdis = []
    for i in range(n_j):
        for j in range(i + 1, n_j):
            both = Mb[:, i] & Mb[:, j]
            if both.sum() >= 20:
                pdis.append(float((Oe[both, i] != Oe[both, j]).mean()))
    pdis = np.array(pdis) if pdis else np.array([0.0])

    off = R_O[~np.eye(n_j, dtype=bool)]
    ev = np.sort(eigh((R_O + R_O.T) / 2, eigvals_only=True))[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum() if ev.sum() > 0 else 1.0

    # spectral 2-partition of |R| graph: Fiedler value + cross/within contrast
    W = np.abs((R_O + R_O.T) / 2)
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    Lap = np.diag(d) - W
    lev, lvec = eigh(Lap)
    fiedler = float(lev[1]) if len(lev) > 1 else 0.0
    part = lvec[:, 1] >= 0
    if 0 < part.sum() < n_j:
        within = np.concatenate([R_O[np.ix_(part, part)][~np.eye(part.sum(), dtype=bool)],
                                 R_O[np.ix_(~part, ~part)][~np.eye((~part).sum(), dtype=bool)]])
        cross = R_O[np.ix_(part, ~part)].ravel()
        part_contrast = float(np.mean(within) - np.mean(cross)) if len(within) else 0.0
    else:
        part_contrast = 0.0

    # leave-one-judge-out majority flip rate, per judge -> max/mean
    lojo = []
    for j in range(n_j):
        keep = np.ones(n_j, bool)
        keep[j] = False
        Ok, Mk = Oe[:, keep], Me[:, keep]
        lab_k = majority_consensus(Ok, Mk)
        lojo.append(float((lab_k != label).mean()))
    lojo = np.array(lojo)

    return {
        # base (heuristic router's information set)
        "rho_bar": float(np.mean(off)),
        "n_eff": float(n_j / (1 + (n_j - 1) * np.mean(off))),
        "eig_overlap": float(eigenvector_overlap(R_O, R_clean_uf, 3)),
        "lco_mean": float(np.nanmean(feats["lco_flip"])),
        "conc_mean": float(np.nanmean(feats["conc"])),
        "rho_S_mean": float(np.nanmean(feats["rho"])),
        "disagree_mean": float(np.nanmean(feats["disagree"])),
        # vote-distribution features
        "margin_mean": float(margin.mean()), "margin_std": float(margin.std()),
        "margin_q25": float(np.quantile(margin, 0.25)),
        "entropy_mean": float(entropy.mean()),
        "near_tie_frac": float((margin < 0.1).mean()),
        "unanimous_frac": float((margin > 0.45).mean()),
        # pairwise disagreement
        "pdis_mean": float(pdis.mean()), "pdis_std": float(pdis.std()),
        "pdis_min": float(pdis.min()), "pdis_max": float(pdis.max()),
        # correlation-matrix shape
        "off_std": float(off.std()), "off_min": float(off.min()), "off_max": float(off.max()),
        "neg_frac": float((off < 0).mean()),
        "neg_mean": float(off[off < 0].mean()) if (off < 0).any() else 0.0,
        "top1_share": float(ev[0] / tot), "top2_share": float(ev[:2].sum() / tot),
        "spec_gap": float((ev[0] - ev[1]) / max(ev[0], 1e-9)) if len(ev) > 1 else 0.0,
        "eig_rank_ratio": float((tot ** 2) / max((ev ** 2).sum(), 1e-9) / n_j),
        "frob_drift_clean": float(np.linalg.norm(R_O - R_clean_uf)),
        "frob_drift_h1": float(np.linalg.norm(R_O - R_h1)),
        "rho_dev": float((np.mean(off) - ref["rho_bar"]) / max(ref["rho_bar"], 1e-6)),
        # graph
        "fiedler": fiedler, "part_contrast": part_contrast,
        # stability / cluster consistency
        "lojo_max": float(lojo.max()), "lojo_mean": float(lojo.mean()),
        "conc_std": float(np.nanstd(feats["conc"])),
        "lco_std": float(np.nanstd(feats["lco_flip"])),
    }

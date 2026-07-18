"""Compute R, n_eff, eigenspectrum, dendrogram, heatmap from cached votes.

Usage:
    python scripts/03_compute_correlation.py
        --bank configs/judge_bank.yaml
        --calibration configs/calibration.yaml
        [--bootstrap 1000]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from corrfilter.analysis import compute_agreement
from corrfilter.correlation import (
    bootstrap_correlation,
    compute_error_matrix,
    compute_error_matrix_pairwise,
    correlation_shrunk,
    correlation_shrunk_pairwise,
    effective_eig_rank,
    effective_size,
    hierarchical_order,
)
from corrfilter.correlation.effective_size import mean_off_diagonal
from corrfilter.data import load_calibration_set
from corrfilter.judges import load_bank_config
from corrfilter.viz import (
    plot_correlation_heatmap,
    plot_dendrogram,
    plot_eigenspectrum,
    plot_neff_collapse,
)
from corrfilter.voting import VoteCache, load_vote_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="configs/judge_bank.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--results-dir", default="experiments/h1_measurement/results")
    parser.add_argument("--figures-dir", default="experiments/h1_measurement/figures")
    parser.add_argument(
        "--listwise-retention-floor",
        type=float,
        default=0.70,
        help="Use Ledoit-Wolf on complete data when listwise retention is at "
        "or above this fraction; otherwise fall back to pairwise-complete + "
        "shrinkage on the available entries.",
    )
    args = parser.parse_args()

    cal_cfg = yaml.safe_load(Path(args.calibration).read_text())
    items = load_calibration_set(cal_cfg["output"]["manifest_path"])
    item_ids = [it.item_id for it in items]
    # Gold label is always 1 by construction (chosen > rejected) for our manifest.
    gold = np.ones(len(items), dtype=np.int8)

    bank_cfg = load_bank_config(args.bank)
    logical_ids = [s.logical_id for s in bank_cfg.specs]
    cache = VoteCache(bank_cfg.votes_dir)

    V, M = load_vote_matrix(cache, logical_ids, item_ids)
    judges_with_votes = M.sum(axis=0) > 0
    if not judges_with_votes.all():
        missing = [logical_ids[i] for i, has in enumerate(judges_with_votes) if not has]
        raise RuntimeError(f"no cached votes for judges: {missing}")

    E_listwise, complete_mask = compute_error_matrix(V, M, gold)
    n_items_kept = int(complete_mask.sum())
    n_items_total = int(complete_mask.size)
    n_items_dropped = n_items_total - n_items_kept
    listwise_retention = n_items_kept / max(n_items_total, 1)
    print(
        f"listwise retention: {n_items_kept}/{n_items_total} "
        f"= {listwise_retention:.1%} (dropped {n_items_dropped} items with any abstention)"
    )

    if listwise_retention >= args.listwise_retention_floor:
        estimator = "listwise_ledoit_wolf"
        print(f"using {estimator} (retention {listwise_retention:.1%} ≥ floor {args.listwise_retention_floor:.0%})")
        E = E_listwise
        R, shrinkage = correlation_shrunk(E)
        boot = bootstrap_correlation(E, n_boot=args.bootstrap, seed=20260601, shrunk=True)
        n_pairs = np.full(R.shape, n_items_kept, dtype=np.int64)
    else:
        estimator = "pairwise_complete_shrunk"
        E_full, avail = compute_error_matrix_pairwise(V, M, gold)
        E = E_full
        print(
            f"using {estimator} (listwise retention {listwise_retention:.1%} < floor "
            f"{args.listwise_retention_floor:.0%})"
        )
        R, n_pairs, shrinkage = correlation_shrunk_pairwise(E_full, avail)
        # Bootstrap on the pairwise estimator: resample items with replacement
        # of E_full and avail jointly.
        rng = np.random.default_rng(20260601)
        boot_samples = np.empty((args.bootstrap, R.shape[0], R.shape[1]), dtype=np.float64)
        for b in range(args.bootstrap):
            idx = rng.integers(0, E_full.shape[0], size=E_full.shape[0])
            R_b, _, _ = correlation_shrunk_pairwise(E_full[idx], avail[idx])
            boot_samples[b] = R_b
        from corrfilter.correlation.estimator import BootstrapResult

        boot = BootstrapResult(
            point=R,
            samples=boot_samples,
            lower=np.quantile(boot_samples, 0.025, axis=0),
            upper=np.quantile(boot_samples, 0.975, axis=0),
        )

    rho_bar = mean_off_diagonal(R)
    n_eff = effective_size(R)
    n_eff_eig = effective_eig_rank(R)

    # Proposal §3.4 supplementary stats: Cohen's κ per judge pair and
    # Krippendorff's α over the bank. Operate on the raw votes V plus the
    # availability mask M, independent of the gold label (these measure
    # inter-judge consistency rather than error-rate correlation).
    agreement = compute_agreement(V, M)
    rho_bar_off = agreement.kappa[~np.eye(len(logical_ids), dtype=bool)]
    kappa_mean = float(rho_bar_off.mean())
    print(
        f"agreement: Krippendorff α = {agreement.krippendorff_alpha:.4f}, "
        f"mean pairwise Cohen κ = {kappa_mean:.4f}"
    )

    # Hierarchical clustering and reordered R.
    Z, order = hierarchical_order(R)
    R_ordered = R[order][:, order]
    ordered_labels = [logical_ids[i] for i in order]

    # Eigenspectrum of the original (un-reordered) R.
    eigvals = np.sort(np.linalg.eigvalsh((R + R.T) / 2.0))[::-1]

    # Persist artefacts.
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        results_dir / "correlation.npz",
        R=R,
        R_ordered=R_ordered,
        order=order,
        eigvals=eigvals,
        boot_lower=boot.lower,
        boot_upper=boot.upper,
        boot_mean=boot.mean,
        boot_std=boot.std,
        E=E,
        n_pairs=n_pairs,
        kappa=agreement.kappa,
        krippendorff_alpha=np.array(agreement.krippendorff_alpha),
        logical_ids=np.array(logical_ids),
        estimator=np.array(estimator),
    )
    summary = {
        "n_judges": len(logical_ids),
        "n_items_total": n_items_total,
        "n_items_kept_listwise": n_items_kept,
        "n_items_dropped_listwise": n_items_dropped,
        "listwise_retention": listwise_retention,
        "estimator": estimator,
        "rho_bar": rho_bar,
        "n_eff": n_eff,
        "n_eff_eig": n_eff_eig,
        "shrinkage": shrinkage,
        "min_pairs_off_diag": int(n_pairs[~np.eye(len(logical_ids), dtype=bool)].min()),
        "max_pairs_off_diag": int(n_pairs[~np.eye(len(logical_ids), dtype=bool)].max()),
        "krippendorff_alpha": agreement.krippendorff_alpha,
        "mean_pairwise_cohen_kappa": kappa_mean,
        "eigvals_top5": eigvals[:5].tolist(),
        "logical_ids": logical_ids,
        "hierarchical_order": order.tolist(),
    }
    (results_dir / "correlation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    plot_correlation_heatmap(
        R_ordered,
        ordered_labels,
        figures_dir / "R_heatmap_reordered.png",
        title=f"R reordered (n_judges={len(logical_ids)}, n_items={n_items_kept}, ρ̄={rho_bar:.3f})",
    )
    plot_correlation_heatmap(
        R,
        logical_ids,
        figures_dir / "R_heatmap_raw.png",
        title="R in bank declaration order",
    )
    kappa_ordered = agreement.kappa[order][:, order]
    plot_correlation_heatmap(
        kappa_ordered,
        ordered_labels,
        figures_dir / "kappa_heatmap_reordered.png",
        title=f"Cohen's κ reordered (α = {agreement.krippendorff_alpha:.3f})",
        vmin=-0.2,
        vmax=1.0,
    )
    plot_dendrogram(Z, logical_ids, figures_dir / "R_dendrogram.png")
    plot_eigenspectrum(eigvals, n_eff_eig, figures_dir / "R_eigenspectrum.png")
    plot_neff_collapse(rho_bar, len(logical_ids), figures_dir / "neff_collapse.png")
    print(f"figures → {figures_dir}")


if __name__ == "__main__":
    main()

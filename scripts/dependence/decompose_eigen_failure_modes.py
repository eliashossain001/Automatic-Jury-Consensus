"""Eigenvector failure-mode decomposition (Bucket 2).

Convert the H1 error-correlation eigenspectrum into interpretable, item-level
failure modes. For each leading eigenvector v1..vk of R we:

  * project every calibration item's error pattern onto the direction
    (loading = E @ v, reusing ``analysis.top_eigen_directions``),
  * rank items by |loading| and take the top-N per direction,
  * attach prompt / candidate / gold / per-judge vote+error / consensus /
    position-swap metadata and a deterministic heuristic failure tag,
  * score how strongly each direction aligns with the candidate failure
    families (position, common-difficulty, prompt-divergence, cross-family,
    safety/refusal, verbosity/style, hallucination/factuality).

The script is analysis-only and deterministic — it reuses the cached H1 votes
and the ``correlation.npz`` produced by ``scripts/dependence/compute_error_correlation.py``
and runs no inference.

Usage:
    python scripts/dependence/decompose_eigen_failure_modes.py \
        --project-root /path/to/corrfilter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import yaml  # noqa: E402

from corrfilter.analysis import top_eigen_directions  # noqa: E402
from corrfilter.analysis.eigen_failure import (  # noqa: E402
    ALIGNMENT_FAMILIES,
    ALL_TAGS,
    alignment_scores,
    compute_item_signals,
    describe_judge_loading,
    heuristic_tag,
    orient_directions,
)
from corrfilter.correlation import compute_error_matrix  # noqa: E402
from corrfilter.data import load_calibration_set  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402

PROMPT_TRUNC = 600
CAND_TRUNC = 1500


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("\r", " ").strip()
    return text if len(text) <= n else text[:n] + f" …[+{len(text) - n} chars]"


def _build_swap_matrix(
    cache: VoteCache, logical_ids: list[str], item_ids: list[str]
) -> np.ndarray:
    """``(N × n)`` per-judge position-swap flag aligned to ``item_ids``/``logical_ids``."""
    n_items, n_judges = len(item_ids), len(logical_ids)
    swap = np.zeros((n_items, n_judges), dtype=np.int8)
    item_index = {iid: k for k, iid in enumerate(item_ids)}
    for i, lid in enumerate(logical_ids):
        df = cache.load(lid)
        if df is None or "position_swapped" not in df.columns:
            continue
        for row in df.itertuples(index=False):
            iid = str(getattr(row, "item_id"))
            j = item_index.get(iid)
            if j is not None:
                swap[j, i] = int(bool(getattr(row, "position_swapped")))
    return swap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--bank", default="configs/judge_bank.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument(
        "--npz", default="experiments/h1_measurement/results/correlation.npz"
    )
    parser.add_argument("--out-dir", default="outputs/eigen_failure_modes")
    parser.add_argument("--top-k", type=int, default=3, help="number of eigenvectors")
    parser.add_argument("--top-n", type=int, default=50, help="top items per eigenvector")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    def _p(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    out_dir = _p(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load cached H1 artefacts (no inference) ---------------------------
    data = np.load(_p(args.npz), allow_pickle=True)
    R = data["R"]
    E_stored = data["E"]
    eigvals_stored = data["eigvals"]
    logical_ids = [str(x) for x in data["logical_ids"]]
    n_judges = len(logical_ids)

    cal_cfg = yaml.safe_load(_p(args.calibration).read_text())
    manifest_path = _p(cal_cfg["output"]["manifest_path"])
    all_items = load_calibration_set(manifest_path)
    all_item_ids = [it.item_id for it in all_items]
    gold = np.ones(len(all_items), dtype=np.int8)

    bank_cfg = load_bank_config(str(_p(args.bank)))
    bank_logical_ids = [s.logical_id for s in bank_cfg.specs]
    if bank_logical_ids != logical_ids:
        raise RuntimeError(
            "bank logical_ids do not match correlation.npz; refusing to misalign columns"
        )
    cache = VoteCache(_p(bank_cfg.votes_dir) if not Path(bank_cfg.votes_dir).is_absolute() else bank_cfg.votes_dir)

    V, M = load_vote_matrix(cache, logical_ids, all_item_ids)
    E, complete = compute_error_matrix(V, M, gold)
    if E.shape != E_stored.shape or not np.allclose(E, E_stored):
        raise RuntimeError(
            f"recomputed E {E.shape} does not match stored E {E_stored.shape}; "
            "votes or manifest have drifted from correlation.npz"
        )

    # Restrict every item-aligned array to the listwise-complete rows that E uses.
    items = [it for it, keep in zip(all_items, complete) if keep]
    item_ids = [it.item_id for it in items]
    M_c = M[complete]
    swap_full = _build_swap_matrix(cache, logical_ids, all_item_ids)
    swap_c = swap_full[complete]

    print(f"loaded R {R.shape}, E {E.shape} over {len(items)} complete items, "
          f"{n_judges} judges")

    # --- Eigen directions + per-item signals ------------------------------
    directions = orient_directions(top_eigen_directions(R, E, k=args.top_k))
    signals = compute_item_signals(items, E, M_c, swap_c, logical_ids)

    # Per-judge error matrix for record bookkeeping.
    err = (E.astype(bool) & M_c.astype(bool)).astype(int)

    manual_rows: list[dict] = []
    per_eigen_records: dict[int, list[dict]] = {}
    alignment_per_eigen: dict[int, dict[str, float]] = {}
    tag_counts: dict[int, dict[str, int]] = {}

    total_var = float(np.clip(eigvals_stored, 0, None).sum())

    for d in directions:
        rank = d.rank
        loading = d.item_loading
        order = np.argsort(-np.abs(loading), kind="stable")[: args.top_n]
        alignment_per_eigen[rank] = alignment_scores(loading, signals)
        tag_counts[rank] = {t: 0 for t in ALL_TAGS}
        records = []
        for pos, idx in enumerate(order, start=1):
            it = items[idx]
            tag = heuristic_tag(it, signals, idx, n_judges)
            tag_counts[rank][tag] += 1
            judge_votes = {logical_ids[i]: int(V[complete][idx, i]) for i in range(n_judges)}
            judge_errors = {logical_ids[i]: int(err[idx, i]) for i in range(n_judges)}
            swapped_judges = [logical_ids[i] for i in range(n_judges) if swap_c[idx, i]]
            rec = {
                "item_id": it.item_id,
                "eigenvector": f"v{rank}",
                "rank": pos,
                "loading_score": round(float(loading[idx]), 6),
                "abs_loading": round(float(abs(loading[idx])), 6),
                "subset": it.subset,
                "category": it.category,
                "prompt": _truncate(it.prompt, PROMPT_TRUNC),
                "candidate_a": _truncate(it.chosen, CAND_TRUNC),
                "candidate_b": _truncate(it.rejected, CAND_TRUNC),
                "gold_label": "A",  # candidate_a == chosen ≻ rejected by construction
                "judge_votes": json.dumps(judge_votes),
                "judge_errors": json.dumps(judge_errors),
                "n_judges_erred": int(signals.error_count[idx]),
                "consensus_ratio": round(float(signals.consensus_ratio[idx]), 4),
                "position_swap_rate": round(float(signals.position_swap_rate[idx]), 4),
                "position_swapped_judges": json.dumps(swapped_judges),
                "verbosity_signal": round(float(signals.verbosity_signal[idx]), 4),
                "prompt_divergence_score": round(float(signals.prompt_divergence_score[idx]), 4),
                "cross_family_score": round(float(signals.cross_family_score[idx]), 4),
                "heuristic_tag": tag,
            }
            records.append(rec)
            manual_rows.append(
                {
                    "item_id": rec["item_id"],
                    "eigenvector": rec["eigenvector"],
                    "rank": rec["rank"],
                    "loading_score": rec["loading_score"],
                    "prompt": rec["prompt"],
                    "candidate_a": rec["candidate_a"],
                    "candidate_b": rec["candidate_b"],
                    "gold_label": rec["gold_label"],
                    "judge_votes": rec["judge_votes"],
                    "heuristic_tag": rec["heuristic_tag"],
                    "manual_failure_label": "",
                    "notes": "",
                }
            )
        per_eigen_records[rank] = records
        pd.DataFrame(records).to_csv(out_dir / f"eigen_v{rank}_top_items.csv", index=False)

    manual_df = pd.DataFrame(manual_rows)
    manual_df.to_csv(out_dir / "eigen_top_items_for_manual_labeling.csv", index=False)

    # --- Figures -----------------------------------------------------------
    _plot_item_loadings(directions, args.top_n, eigvals_stored, total_var, out_dir)
    _plot_tag_by_eigenvector(tag_counts, out_dir)

    # --- Markdown summary --------------------------------------------------
    _write_summary(
        out_dir / "eigen_failure_summary.md",
        R=R,
        eigvals=eigvals_stored,
        total_var=total_var,
        directions=directions,
        logical_ids=logical_ids,
        alignment_per_eigen=alignment_per_eigen,
        per_eigen_records=per_eigen_records,
        tag_counts=tag_counts,
        n_items=len(items),
        top_n=args.top_n,
    )

    print(f"wrote outputs to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print("  ", f.name)


def _plot_item_loadings(directions, top_n, eigvals, total_var, out_dir: Path) -> None:
    for d in directions:
        loading = d.item_loading
        order = np.argsort(-loading, kind="stable")
        sorted_load = loading[order]
        abs_rank = np.argsort(-np.abs(loading), kind="stable")[:top_n]
        top_mask = np.zeros(loading.shape[0], dtype=bool)
        top_mask[abs_rank] = True
        top_mask_sorted = top_mask[order]
        x = np.arange(sorted_load.shape[0])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(x[~top_mask_sorted], sorted_load[~top_mask_sorted], s=6,
                   c="#9bb3d4", label="items")
        ax.scatter(x[top_mask_sorted], sorted_load[top_mask_sorted], s=14,
                   c="#c0392b", label=f"top {top_n} |loading|")
        ax.axhline(0.0, color="black", lw=0.8)
        ve = d.eigenvalue / total_var if total_var > 0 else 0.0
        ax.set_title(f"Item loadings on v{d.rank} "
                     f"(λ={d.eigenvalue:.3f}, {ve:.1%} variance)")
        ax.set_xlabel("item (sorted by loading)")
        ax.set_ylabel(f"loading  E·v{d.rank}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"eigen_item_loading_v{d.rank}.png", dpi=150)
        plt.close(fig)


def _plot_tag_by_eigenvector(tag_counts: dict[int, dict[str, int]], out_dir: Path) -> None:
    ranks = sorted(tag_counts)
    tags = list(ALL_TAGS)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(ranks))
    x = np.arange(len(ranks))
    for ti, tag in enumerate(tags):
        vals = np.array([tag_counts[r].get(tag, 0) for r in ranks], dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottoms, label=tag, color=cmap(ti % 10))
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"v{r}" for r in ranks])
    ax.set_ylabel("count among top items")
    ax.set_title("Heuristic failure tag composition by eigenvector")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "failure_tag_by_eigenvector.png", dpi=150)
    plt.close(fig)


def _write_summary(
    path: Path,
    *,
    R,
    eigvals,
    total_var,
    directions,
    logical_ids,
    alignment_per_eigen,
    per_eigen_records,
    tag_counts,
    n_items,
    top_n,
) -> None:
    lines: list[str] = []
    lines.append("# Eigenvector Failure-Mode Decomposition (Bucket 2)\n")
    lines.append(
        f"Source: `experiments/h1_measurement/results/correlation.npz` — "
        f"R over {len(logical_ids)} judges and {n_items} listwise-complete "
        f"calibration items. Analysis-only, deterministic (no inference).\n"
    )

    lines.append("## Top eigenvalues and explained variance\n")
    lines.append("| rank | eigenvalue | variance explained | cumulative |")
    lines.append("|---|---|---|---|")
    cum = 0.0
    for i, lam in enumerate(eigvals[: max(5, len(directions))], start=1):
        ve = lam / total_var if total_var > 0 else 0.0
        cum += ve
        lines.append(f"| {i} | {lam:.4f} | {ve:.1%} | {cum:.1%} |")
    lines.append("")
    lines.append(
        f"Trace(R) = {total_var:.2f} (= n_judges). The leading direction carries "
        f"{eigvals[0] / total_var:.1%} of the bank's error variance — far above the "
        f"{1.0 / len(logical_ids):.1%} an independent bank would give per direction.\n"
    )

    lines.append("## Interpretation of v1, v2, v3\n")
    for d in directions:
        rank = d.rank
        ve = d.eigenvalue / total_var if total_var > 0 else 0.0
        judge_desc = describe_judge_loading(d.judge_loading, logical_ids)
        align = alignment_per_eigen[rank]
        ranked = sorted(align.items(), key=lambda kv: -abs(kv[1]))
        top_family, top_corr = ranked[0]
        loadings_str = ", ".join(
            f"{lid.split('::')[0]}/{lid.split('::')[1][:2]}={w:+.2f}"
            for lid, w in zip(logical_ids, d.judge_loading)
        )
        lines.append(f"### v{rank} — λ={d.eigenvalue:.3f} ({ve:.1%} variance)\n")
        lines.append(f"- **Judge-loading pattern:** {judge_desc}.")
        lines.append(f"- **Judge loadings:** {loadings_str}")
        lines.append(
            f"- **Strongest item-side alignment:** `{top_family}` "
            f"(r={top_corr:+.2f} between item loading and the {top_family} signal)."
        )
        # dominant tag among top items
        tc = tag_counts[rank]
        dom_tag = max(tc, key=lambda t: tc[t])
        lines.append(
            f"- **Dominant heuristic tag among top {top_n} items:** "
            f"`{dom_tag}` ({tc[dom_tag]}/{top_n})."
        )
        lines.append("")

    lines.append("## Failure-family alignment per eigenvector\n")
    families = [f for f, _ in ALIGNMENT_FAMILIES]
    header = "| eigenvector | " + " | ".join(families) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(families) + 1))
    for d in directions:
        align = alignment_per_eigen[d.rank]
        cells = " | ".join(f"{align[f]:+.2f}" for f in families)
        lines.append(f"| v{d.rank} | {cells} |")
    lines.append("")
    lines.append(
        "_Values are Pearson r between the per-item eigenvector loading and a "
        "heuristic signal for each failure family. |r|≥0.30 is a notable "
        "alignment; the sign indicates whether the failure side of the "
        "direction (positive loading) carries more of that signal._\n"
    )

    for d in directions:
        rank = d.rank
        lines.append(f"## Top 10 representative items for v{rank}\n")
        lines.append(
            "| rank | item_id | subset | loading | #err | consensus | tag | prompt (truncated) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for rec in per_eigen_records[rank][:10]:
            prompt_cell = rec["prompt"][:90].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {rec['rank']} | {rec['item_id']} | {rec['subset']} | "
                f"{rec['loading_score']:+.3f} | {rec['n_judges_erred']} | "
                f"{rec['consensus_ratio']:.2f} | {rec['heuristic_tag']} | {prompt_cell} |"
            )
        lines.append("")

    lines.append("## Artefacts\n")
    lines.append("- `eigen_top_items_for_manual_labeling.csv` — combined manual-labeling template.")
    for d in directions:
        lines.append(f"- `eigen_v{d.rank}_top_items.csv` — full per-item records for v{d.rank}.")
    for d in directions:
        lines.append(f"- `eigen_item_loading_v{d.rank}.png` — item-loading scatter for v{d.rank}.")
    lines.append("- `failure_tag_by_eigenvector.png` — heuristic tag composition.\n")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

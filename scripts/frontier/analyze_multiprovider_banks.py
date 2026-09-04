#!/usr/bin/env python
"""Full multi-provider analysis: OpenRouter + Gemini + open banks on identical items.

Datasets: rewardbench (400 stratified items), pku (300 stratified items); all
votes replayed from caches, no API calls.

Sections:
  per-judge     accuracy (overall + by RewardBench subset / PKU subset), slot
                accuracies, position gap, coverage, abstention
  banks         openrouter_3, gemini_3, frontier_6, open_pairwise_5,
                open_all_10, mixed_11: R, rho_bar (+bootstrap CI), n_eff,
                conditional co-failure (+lift), co-success, majority accuracy,
                jointly-wrong, disagreement matrices
  blocks        pair classes on the mixed_11 R: within-provider frontier
                (Gemini pairs), cross-provider frontier, frontier-open,
                open-open; paired item-bootstrap CIs on block differences
  difficulty    external difficulty = #open judges wrong (0-10); residual
                frontier-frontier correlation within difficulty strata;
                leave-pair-out variant; jointly-missed frontier items
  truncation    (pku) items whose rendered prompt exceeds the open judges'
                1024-token truncation; accuracy/correlation on fitting vs
                exceeding subsets (matched-input sensitivity, no new calls)

Usage: python scripts/frontier/analyze_multiprovider_banks.py [--dataset rewardbench pku]
Outputs -> outputs/openrouter_bank/analysis_full_<dataset>/
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from compute_dependence import dependence_metrics  # noqa: E402

from corrfilter.correlation import compute_error_matrix, correlation_shrunk  # noqa: E402
from corrfilter.correlation.effective_size import mean_off_diagonal  # noqa: E402
from corrfilter.judges import load_bank_config  # noqa: E402
from corrfilter.judges.prompts import PairwisePrompt  # noqa: E402
from corrfilter.data import CalibrationItem  # noqa: E402
from corrfilter.voting import VoteCache, load_vote_matrix  # noqa: E402


def _load_script(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s34 = _load_script("gba", "frontier/analyze_gemini_bank.py")
_s39 = _load_script("orp", "frontier/analyze_openrouter_pilot.py")

BOOT_SEED = 20260706
OR_JUDGES = _s39.OR_JUDGES
GEM_JUDGES = _s39.GEM_JUDGES
PROVIDER = dict(_s39.PROVIDER)

OPEN_BANK_YAML = {"rewardbench": "configs/judge_bank.yaml", "pku": "configs/judge_bank_pku.yaml"}
MANIFEST = {"rewardbench": "experiments/h1_measurement/results/calibration_manifest.parquet",
            "pku": "outputs/pku_saferlhf/pku_manifest.parquet"}
TRUNC_TOKENS = 1024   # open judges' PKU max_length


def pair_class(lids, i, j):
    a, b = PROVIDER.get(lids[i], "open"), PROVIDER.get(lids[j], "open")
    if a == "open" and b == "open":
        return "open_open"
    if a == "open" or b == "open":
        return "frontier_open"
    if a == b:
        return "within_provider_frontier"
    return "cross_provider_frontier"


def block_means(R, lids):
    out = {}
    n = len(lids)
    for i in range(n):
        for j in range(i + 1, n):
            out.setdefault(pair_class(lids, i, j), []).append(R[i, j])
    return {k: (round(float(np.mean(v)), 4), len(v)) for k, v in out.items()}


def block_diff_ci(V, M, gold, lids, n_boot, seed=BOOT_SEED):
    """Item bootstrap of block means and their key differences."""
    E, _ = compute_error_matrix(V, M, gold)
    rng = np.random.default_rng(seed)
    keys = ["within_provider_frontier", "cross_provider_frontier", "frontier_open", "open_open"]
    draws = {k: [] for k in keys}
    dif = {"cross_frontier_minus_frontier_open": [], "cross_frontier_minus_open_open": [],
           "within_minus_cross_frontier": []}
    for _ in range(n_boot):
        R_b, _ = correlation_shrunk(E[rng.integers(0, len(E), len(E))])
        bm = {k: v[0] for k, v in block_means(R_b, lids).items()}
        for k in keys:
            draws[k].append(bm.get(k, np.nan))
        dif["cross_frontier_minus_frontier_open"].append(bm["cross_provider_frontier"] - bm["frontier_open"])
        dif["cross_frontier_minus_open_open"].append(bm["cross_provider_frontier"] - bm["open_open"])
        dif["within_minus_cross_frontier"].append(
            bm.get("within_provider_frontier", np.nan) - bm["cross_provider_frontier"])
    q = lambda a: (round(float(np.nanquantile(a, 0.025)), 4), round(float(np.nanquantile(a, 0.975)), 4))
    point_R, _ = correlation_shrunk(E)
    pm = {k: v[0] for k, v in block_means(point_R, lids).items()}
    rows = []
    for k in keys:
        rows.append({"quantity": k, "point": pm.get(k), "ci95": q(draws[k])})
    for k, v in dif.items():
        pt = {"cross_frontier_minus_frontier_open": pm["cross_provider_frontier"] - pm["frontier_open"],
              "cross_frontier_minus_open_open": pm["cross_provider_frontier"] - pm["open_open"],
              "within_minus_cross_frontier": pm.get("within_provider_frontier", np.nan) - pm["cross_provider_frontier"]}[k]
        lo, hi = q(v)
        rows.append({"quantity": k, "point": round(float(pt), 4), "ci95": (lo, hi),
                     "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows), int(E.shape[0])


def difficulty_analysis(Ef, Mf_b, lids_f, open_wrong, n_open, out_dir, tag):
    """Residual frontier-frontier correlation conditioned on difficulty."""
    # external strata by number of open judges wrong
    edges = [(0, 1), (2, 3), (4, 6), (7, n_open)]
    rows = []
    nf = Ef.shape[1]
    raw, resid = [], []
    for i in range(nf):
        for j in range(i + 1, nf):
            both = Mf_b[:, i] & Mf_b[:, j]
            ei, ej = Ef[both, i], Ef[both, j]
            if ei.std() < 1e-9 or ej.std() < 1e-9:
                continue
            raw.append(float(np.corrcoef(ei, ej)[0, 1]))
            ow = open_wrong[both]
            rs, ws = [], []
            for lo, hi in edges:
                sel = (ow >= lo) & (ow <= hi)
                if sel.sum() < 20:
                    continue
                si, sj = ei[sel], ej[sel]
                if si.std() < 1e-9 or sj.std() < 1e-9:
                    continue
                rs.append(float(np.corrcoef(si, sj)[0, 1]))
                ws.append(int(sel.sum()))
            if rs:
                resid.append(float(np.average(rs, weights=ws)))
    rows.append({"conditioner": "open_bank_difficulty(4 strata)",
                 "raw_mean_pair_corr": round(float(np.mean(raw)), 4),
                 "residual_mean_pair_corr": round(float(np.mean(resid)), 4) if resid else np.nan,
                 "n_pairs": len(raw), "n_pairs_residual": len(resid)})
    # leave-pair-out difficulty within the frontier bank
    Cf = 1 - Ef
    raw2, resid2 = [], []
    edges2 = np.linspace(0, 1, 4)
    edges2[-1] += 1e-9
    for i in range(nf):
        for j in range(i + 1, nf):
            both = Mf_b[:, i] & Mf_b[:, j]
            ei, ej = Ef[both, i], Ef[both, j]
            if ei.std() < 1e-9 or ej.std() < 1e-9:
                continue
            raw2.append(float(np.corrcoef(ei, ej)[0, 1]))
            others = [k for k in range(nf) if k not in (i, j)]
            Mo = Mf_b[both][:, others]
            Co = Cf[both][:, others] * Mo
            cnt = Mo.sum(1)
            diff = np.where(cnt > 0, Co.sum(1) / np.maximum(cnt, 1), 0.5)
            rs, ws = [], []
            for b in range(3):
                sel = (diff >= edges2[b]) & (diff < edges2[b + 1])
                if sel.sum() < 20:
                    continue
                si, sj = ei[sel], ej[sel]
                if si.std() < 1e-9 or sj.std() < 1e-9:
                    continue
                rs.append(float(np.corrcoef(si, sj)[0, 1]))
                ws.append(int(sel.sum()))
            if rs:
                resid2.append(float(np.average(rs, weights=ws)))
    rows.append({"conditioner": "leave_pair_out_frontier(3 strata)",
                 "raw_mean_pair_corr": round(float(np.mean(raw2)), 4),
                 "residual_mean_pair_corr": round(float(np.mean(resid2)), 4) if resid2 else np.nan,
                 "n_pairs": len(raw2), "n_pairs_residual": len(resid2)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"difficulty_residual_{tag}.csv", index=False)
    return df


def analyze(ds: str, root: Path, n_boot: int) -> None:
    out_dir = root / "outputs" / "openrouter_bank" / f"analysis_full_{ds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    item_ids = [l.strip() for l in (root / f"outputs/gemini_bank/items_{ds}.txt").read_text().splitlines() if l.strip()]
    n_items = len(item_ids)
    gold = np.ones(n_items, dtype=np.int8)
    man = pd.read_parquet(root / MANIFEST[ds])
    man = man[man.item_id.astype(str).isin(set(item_ids))].drop_duplicates("item_id", keep="last")
    man = man.set_index(man.item_id.astype(str)).loc[item_ids]

    or_cache = VoteCache(root / "outputs/openrouter_bank/votes/frontier" / ds)
    gem_cache = VoteCache(root / "outputs/gemini_bank/votes" / ds)
    open_bank = load_bank_config(root / OPEN_BANK_YAML[ds])
    open_cache = VoteCache(open_bank.votes_dir if Path(open_bank.votes_dir).is_absolute()
                           else root / open_bank.votes_dir)
    open_all = [s.logical_id for s in open_bank.specs]
    open_pw = [l for l in open_all if l.endswith("::pairwise")]

    # ---- per-judge, incl. per-subset accuracy ----
    rows, subset_rows = [], []
    for cache, lids, prov in ((or_cache, OR_JUDGES, "openrouter"), (gem_cache, GEM_JUDGES, "gemini"),
                              (open_cache, open_all, "open")):
        for lid in lids:
            df = cache.load(lid)
            if df is None:
                continue
            df = df[df["item_id"].astype(str).isin(set(item_ids))].drop_duplicates("item_id", keep="last")
            rows.append({**_s34.judge_row(lid, df, n_items), "source": prov})
            dfi = df.set_index(df["item_id"].astype(str))
            for subset in sorted(man["subset"].unique()):
                ids_s = [i for i in man.index[man["subset"] == subset] if i in dfi.index]
                comm = dfi.loc[ids_s]
                comm = comm[comm["vote"] != -1]
                if len(comm):
                    subset_rows.append({"judge": lid, "source": prov, "subset": subset,
                                        "n": len(comm), "accuracy": round(float((comm["vote"] == 1).mean()), 4)})
    pd.DataFrame(rows).to_csv(out_dir / "judge_metrics.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(out_dir / "judge_subset_accuracy.csv", index=False)

    # ---- banks ----
    Vor, Mor = load_vote_matrix(or_cache, OR_JUDGES, item_ids)
    Vg, Mg = load_vote_matrix(gem_cache, GEM_JUDGES, item_ids)
    Vp, Mp = load_vote_matrix(open_cache, open_pw, item_ids)
    Va, Ma = load_vote_matrix(open_cache, open_all, item_ids)
    banks = {
        "openrouter_3": (Vor, Mor, OR_JUDGES),
        "gemini_3": (Vg, Mg, GEM_JUDGES),
        "frontier_6": (np.hstack([Vg, Vor]), np.hstack([Mg, Mor]), GEM_JUDGES + OR_JUDGES),
        "open_pairwise_5": (Vp, Mp, open_pw),
        "open_all_10": (Va, Ma, open_all),
        "mixed_11": (np.hstack([Vp, Vg, Vor]), np.hstack([Mp, Mg, Mor]), open_pw + GEM_JUDGES + OR_JUDGES),
    }
    bank_rows = []
    for name, (V, M, lids) in banks.items():
        row, R = dependence_metrics(V, M, gold, lids, label=name)
        row.update(_s34.majority_accuracy(V, M, gold))
        row["cond_co_success"] = round(_s39.cond_cosuccess(V, M, gold), 4)
        row.update({f"boot_{k}": v for k, v in _s34.bootstrap_bank(V, M, gold, n_boot).items()})
        bank_rows.append(row)
        pd.DataFrame(R, index=lids, columns=lids).to_csv(out_dir / f"R_{name}.csv")
        _s39.disagreement_matrix(V, M, lids).to_csv(out_dir / f"disagreement_{name}.csv")
    bank_df = pd.DataFrame(bank_rows)
    bank_df.to_csv(out_dir / "bank_metrics.csv", index=False)

    # ---- paired rho comparisons (same as pilot, on this item set) ----
    comps = []
    for a, b, note in (
        ("gemini_3", "openrouter_3", "single-provider frontier minus cross-provider frontier"),
        ("open_pairwise_5", "openrouter_3", "open 5 minus cross-provider frontier 3"),
        ("gemini_3", "frontier_6", "gemini-only minus all-frontier"),
        ("open_all_10", "mixed_11", "open 10 minus mixed 11"),
    ):
        Va_, Ma_, _ = banks[a]
        Vb_, Mb_, _ = banks[b]
        point, lo, hi, n_shared = _s39.paired_rho_diff_ci(Va_, Ma_, Vb_, Mb_, gold, n_boot)
        comps.append({"comparison": f"rho({a}) - rho({b})", "note": note, "diff": point,
                      "ci95_low": lo, "ci95_high": hi, "n_shared_items": n_shared,
                      "significant": bool(lo > 0 or hi < 0)})
    pd.DataFrame(comps).to_csv(out_dir / "comparisons_ci.csv", index=False)

    # ---- block analysis on mixed_11 ----
    Vm, Mm, lids_m = banks["mixed_11"]
    blocks_df, n_lw = block_diff_ci(Vm, Mm, gold, lids_m, n_boot)
    blocks_df.to_csv(out_dir / "block_analysis.csv", index=False)

    # ---- difficulty analysis on frontier_6, conditioned on open-bank difficulty ----
    Vf, Mf, lids_f = banks["frontier_6"]
    Mf_b = Mf.astype(bool)
    Ef = ((Vf != gold[:, None]) & Mf_b).astype(float)
    Ma_b = Ma.astype(bool)
    open_wrong = (((Va != gold[:, None]) & Ma_b).sum(1)).astype(int)
    diff_df = difficulty_analysis(Ef, Mf_b, lids_f, open_wrong, len(open_all), out_dir, "frontier_6")

    # jointly-missed frontier items (>=5 of 6 wrong among committed)
    wrong_f = ((Vf != gold[:, None]) & Mf_b).sum(1)
    comm_f = Mf_b.sum(1)
    jm = (wrong_f >= 5) & (comm_f == 6)
    jm_df = pd.DataFrame({"item_id": np.array(item_ids)[jm], "subset": man.loc[np.array(item_ids)[jm], "subset"].values,
                          "n_frontier_wrong": wrong_f[jm], "n_open_wrong": open_wrong[jm],
                          "prompt_chars": man.loc[np.array(item_ids)[jm], "prompt"].str.len().values})
    jm_df.to_csv(out_dir / "jointly_missed_frontier.csv", index=False)

    # ---- PKU truncation analysis ----
    trunc_df = None
    if ds == "pku":
        prompt = PairwisePrompt()
        est_tokens = []
        for iid in item_ids:
            r = man.loc[iid]
            it = CalibrationItem(item_id=iid, prompt=r["prompt"], chosen=r["chosen"],
                                 rejected=r["rejected"], subset=r["subset"], category=r["category"])
            est_tokens.append(len(prompt.render(it, False).text) / 4.0)
        est_tokens = np.array(est_tokens)
        fits = est_tokens <= TRUNC_TOKENS
        rows_t = [{"group": "all", "n": n_items, "frac": 1.0}]
        for gname, mask in (("fits_1024", fits), ("exceeds_1024", ~fits)):
            row = {"group": gname, "n": int(mask.sum()), "frac": round(float(mask.mean()), 3)}
            for bname in ("openrouter_3", "gemini_3", "open_pairwise_5", "open_all_10"):
                V, M, lids = banks[bname]
                Mb = M.astype(bool)
                acc = float(((V == 1) & Mb)[mask].sum() / max(Mb[mask].sum(), 1))
                row[f"acc_{bname}"] = round(acc, 4)
                maj = _s34.majority_accuracy(V[mask], M[mask], gold[mask])
                row[f"majacc_{bname}"] = maj["majority_accuracy"]
                E_m, comp_m = compute_error_matrix(V[mask], M[mask], gold[mask])
                if E_m.shape[0] >= 30:
                    R_m, _ = correlation_shrunk(E_m)
                    row[f"rho_{bname}"] = round(mean_off_diagonal(R_m), 4)
            rows_t.append(row)
        trunc_df = pd.DataFrame(rows_t)
        trunc_df.to_csv(out_dir / "truncation_sensitivity.csv", index=False)

    # ---- summary ----
    cols = ["label", "n_judges", "rho_bar", "n_eff", "conditional_cofailure", "conditional_lift",
            "cond_co_success", "majority_accuracy", "majority_jointly_wrong"]
    with open(out_dir / "summary.md", "w") as f:
        f.write(f"# Multi-provider full analysis — {ds} ({n_items} items)\n\n")
        f.write(bank_df[cols].to_markdown(index=False))
        f.write("\n\n## Paired rho comparisons\n\n")
        f.write(pd.DataFrame(comps).to_markdown(index=False))
        f.write("\n\n## Block analysis (mixed_11)\n\n")
        f.write(blocks_df.to_markdown(index=False))
        f.write("\n\n## Difficulty-conditioned frontier correlation\n\n")
        f.write(diff_df.to_markdown(index=False))
        f.write(f"\n\nJointly-missed frontier items (>=5/6 wrong): {int(jm.sum())}; by subset: "
                f"{man.loc[np.array(item_ids)[jm], 'subset'].value_counts().to_dict()}\n")
        if trunc_df is not None:
            f.write("\n## PKU truncation sensitivity\n\n")
            f.write(trunc_df.to_markdown(index=False))
            f.write("\n")
    print(f"[40] {ds}: wrote {out_dir}")
    print(bank_df[cols].to_string(index=False))
    print(blocks_df.to_string(index=False))
    print(diff_df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["rewardbench", "pku"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    for ds in args.dataset:
        analyze(ds, Path(args.project_root), args.n_boot)


if __name__ == "__main__":
    main()

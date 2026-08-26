"""Experiment: mainstream_alignment_integration (reward-model variant).

Tests whether dependence-aware filtering SIGNALS help downstream alignment when used
as PER-EXAMPLE LOSS WEIGHTS during reward-model training, rather than only as hard
post-hoc filters. Reuses the cached votes, the synthetic + position poisoning setups,
CorrFilter alpha scores, BiasCluster |S\\C| scores, and oracle (gold) labels.

Conditions (core = loss weighting on the SAME dataset; secondary = hard filtering):
  naive                : w_i = 1
  corrfilter_weighted  : w_i from CorrFilter alpha (rank-norm -> clip -> mean 1)
  biascluster_weighted : w_i from BiasCluster |S\\C| (rank-norm -> clip -> mean 1)
  oracle_soft          : correct w=1.0, poisoned w=0.25 (mean 1)
  oracle_drop          : correct w=1.0, poisoned w=0.0  (mean 1)
  corrfilter_hard / biascluster_hard / supermajority : matched-retention subset (w in {0,1})

Reward model: Qwen2.5-0.5B-Instruct + LoRA scalar head, pairwise Bradley-Terry loss
  loss = mean( w_i * -logsigmoid(r_chosen - r_rejected) ),  weights mean-normalised to 1.

Outputs (never overwrites older experiments): results/mainstream_alignment_integration/
  comparison_table.csv, pretrain_metrics.csv, summary.json, run.log

Usage:
  python scripts/robustness/mainstream_alignment_integration.py --smoke
  python scripts/robustness/mainstream_alignment_integration.py --pretrain-only
  python scripts/robustness/mainstream_alignment_integration.py            # full run
  python scripts/robustness/mainstream_alignment_integration.py --regimes pos10 syn10 --no-secondary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from corrfilter.cfi.consensus import (
    ABSTAIN,
    consensus_level,
    majority_consensus,
    supermajority_consensus,
)
from corrfilter.cfi.corrfilter_score import corrfilter_score, retention_match_threshold
from corrfilter.correlation.effective_size import effective_size, mean_off_diagonal
from corrfilter.judges import load_bank_config
from corrfilter.voting import VoteCache

ROOT = Path(__file__).resolve().parents[2]


H1_POSITION_GAP = {
    "gemma-2-9b::pairwise": 10.4, "gemma-2-9b::likert": 40.8,
    "llama-3.1-8b::pairwise": 63.7, "llama-3.1-8b::likert": 17.4,
    "mistral-7b-v0.3::pairwise": 70.2, "mistral-7b-v0.3::likert": 15.9,
    "phi-3.5-mini::pairwise": 30.1, "phi-3.5-mini::likert": 11.9,
    "qwen-2.5-7b::pairwise": 46.7, "qwen-2.5-7b::likert": 51.7,
}


# ----------------------------- data layer -----------------------------
def build_VM(cache, logical_ids, item_ids):
    n, m = len(item_ids), len(logical_ids)
    V = np.zeros((n, m), np.int8); M = np.zeros((n, m), np.int8)
    idx = {it: i for i, it in enumerate(item_ids)}
    for j, lid in enumerate(logical_ids):
        dfj = cache.load(lid)
        if dfj is None:
            continue
        for r in dfj.itertuples(index=False):
            i = idx.get(str(r.item_id))
            if i is None:
                continue
            v = int(r.vote)
            if v == ABSTAIN:
                continue
            V[i, j] = v; M[i, j] = 1
    return V, M


class DataHub:
    """Loads votes/manifest once; serves per-regime poison masks and matrices."""

    def __init__(self, cfg):
        d = cfg["data"]
        self.man = pd.read_csv(ROOT / d["manifest"])
        self.man["item_id"] = self.man["item_id"].astype(str)
        self.item_ids = self.man["item_id"].tolist()
        tc = ["prompt", "response_a", "response_b"]
        self.valid = (self.man[tc].notna().all(axis=1)
                      & (self.man[tc].apply(lambda c: c.astype(str).str.strip() != "")).all(axis=1)).to_numpy()
        bank = load_bank_config(str(ROOT / d["bank"]))
        self.logical_ids = [s.logical_id for s in bank.specs]
        self.V, self.M = build_VM(VoteCache(ROOT / d["votes_dir"]), self.logical_ids, self.item_ids)
        self.R = np.load(ROOT / d["correlation_npz"], allow_pickle=True)["R"]
        gaps = np.array([H1_POSITION_GAP.get(l, 0.0) for l in self.logical_ids])
        self.cb = np.zeros(len(self.logical_ids), bool); self.cb[np.argsort(-gaps)[:5]] = True
        self.w_pos = gaps / 100.0
        n = len(self.item_ids)
        rng = np.random.default_rng(cfg["seed"])
        perm = rng.permutation(n); ne = int(round(d["eval_fraction"] * n))
        self.eval_idx = np.sort(perm[:ne]); self.train_idx = np.sort(perm[ne:])
        # position-bias mass per item (for poisoning + targeted slices)
        self.pos_score = ((1 - self.V) * self.M * self.w_pos[None, :]).sum(1)

    def poison_mask(self, source, rate):
        n = len(self.item_ids)
        if source == "none" or rate == 0.0:
            return np.zeros(n, bool)
        if source == "synthetic":
            tag = {0.05: "05", 0.10: "10", 0.20: "20"}[round(rate, 2)]
            return self.man[f"poisoned_{tag}"].to_numpy(bool)
        if source == "position":
            k = int(round(rate * n))
            order = np.argsort(-self.pos_score, kind="stable")
            mask = np.zeros(n, bool); mask[order[:k]] = True
            return mask
        raise ValueError(source)


# ----------------------------- per-regime build -----------------------------
def bias_cluster_support(O, M, label, cb):
    """beta_C = #agreeing judges outside the H1 position cluster."""
    Mb = M.astype(bool); n = O.shape[0]
    s = np.full(n, 0.0)
    for i in range(n):
        agree = (O[i] == label[i]) & Mb[i]
        s[i] = int(agree.sum()) - int((agree & cb).sum())
    return s


def build_regime(hub: DataHub, source, rate):
    poison = hub.poison_mask(source, rate)
    gold_full = np.where(poison, 0, 1).astype(np.int8)
    O_full = hub.V.copy(); O_full[poison] = 1 - hub.V[poison]
    tr = hub.train_idx
    O, M, gold = O_full[tr], hub.M[tr], gold_full[tr]
    valid = hub.valid[tr]
    label = majority_consensus(O, M)
    affirmed = (label == 1) & valid
    cf = np.nan_to_num(corrfilter_score(O, M, hub.R, label).score, nan=-np.inf)
    bc = bias_cluster_support(O, M, label, hub.cb)
    # training dataset = affirmed items (naive consensus keep), with STATED labels
    man_tr = hub.man.iloc[tr].reset_index(drop=True)
    rows, wrong, cf_s, bc_s, goldk = [], [], [], [], []
    stated_tag = {0.05: "05", 0.10: "10", 0.20: "20"}.get(round(rate, 2))
    for i in np.where(affirmed)[0]:
        r = man_tr.iloc[i]
        is_pois = bool(poison[tr][i])
        stated = "B" if is_pois else "A"            # true_label is always "A"
        chosen, rejected = (r["response_b"], r["response_a"]) if stated == "B" else (r["response_a"], r["response_b"])
        rows.append({"item_id": r["item_id"], "prompt": r["prompt"], "chosen": chosen, "rejected": rejected})
        wrong.append(is_pois); cf_s.append(cf[i]); bc_s.append(bc[i]); goldk.append(gold[i])
    ds = {"pairs": rows, "wrong": np.array(wrong), "cf": np.array(cf_s), "bc": np.array(bc_s),
          "gold": np.array(goldk)}
    # matched-retention keep masks for the hard/secondary conditions
    n_match = max(int(((supermajority_consensus(O, M, 0.75) != ABSTAIN) & valid).sum()), 1)
    aff_idx = np.where(affirmed)[0]
    def topk_mask(score_full):
        s = np.where(affirmed, score_full, -np.inf)
        keep, _ = retention_match_threshold(s, n_match)
        keep &= affirmed
        return keep[aff_idx]                         # restrict to dataset order
    ds["hard"] = {
        "corrfilter_hard": topk_mask(cf),
        "biascluster_hard": topk_mask(bc),
        "supermajority": topk_mask(consensus_level(O, M)),
    }
    # judge-level stats (truth-based; constant across regimes but reported per regime)
    ds["judge_stats"] = judge_stats(hub, tr)
    # eval probes (held-out, true labels)
    ev = hub.eval_idx[hub.valid[hub.eval_idx]]
    man_ev = hub.man.iloc[ev].reset_index(drop=True)
    ev_pairs = [{"item_id": r["item_id"], "prompt": r["prompt"],
                 "chosen": r["response_a"], "rejected": r["response_b"]}   # true_label A
                for _, r in man_ev.iterrows()]
    pscore_ev = hub.pos_score[ev]
    targeted = pscore_ev >= np.quantile(pscore_ev, 2 / 3)
    ds["eval"] = {"pairs": ev_pairs, "targeted": targeted}
    return ds


def judge_stats(hub, tr):
    """rho_bar, n_eff, majority co-failure, pairwise conditional co-failure (truth-based)."""
    Vt, Mt = hub.V[tr], hub.M[tr].astype(bool)
    E = (Vt != 1) & Mt                               # error vs truth (truth pick == 1)
    pj = np.array([E[:, j][Mt[:, j]].mean() if Mt[:, j].any() else 0 for j in range(E.shape[1])])
    both, cond = [], []
    for i in range(E.shape[1]):
        for j in range(i + 1, E.shape[1]):
            sel = Mt[:, i] & Mt[:, j]
            if sel.sum() < 5:
                continue
            ei, ej = E[sel, i], E[sel, j]
            pb = (ei & ej).mean()
            both.append(pb)
            if ei.mean() > 0:
                cond.append(pb / ei.mean())
    fc = np.divide((Vt * Mt).sum(1), Mt.sum(1), out=np.full(len(Vt), 0.5), where=Mt.sum(1) > 0)
    return {"rho_bar": round(float(mean_off_diagonal(hub.R)), 4),
            "n_eff": round(float(effective_size(hub.R)), 3),
            "per_judge_error": round(float(pj.mean()), 4),
            "pairwise_both_wrong": round(float(np.mean(both)), 4),
            "conditional_cofailure": round(float(np.mean(cond)), 4),
            "majority_cofailure": round(float((fc < 0.5).mean()), 4)}


# ----------------------------- weighting -----------------------------
def make_weights(condition, ds, cfg):
    n = len(ds["pairs"]); wrong = ds["wrong"]
    lo, hi = cfg["weights"]["clip_low"], cfg["weights"]["clip_high"]

    def rank_norm(score):
        s = np.asarray(score, float)
        s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0)
        r = (np.argsort(np.argsort(s)) + 0.5) / len(s)     # percentile rank in (0,1)
        w = lo + r * (hi - lo)
        return w

    if condition == "naive":
        w = np.ones(n)
    elif condition == "corrfilter_weighted":
        w = rank_norm(ds["cf"])
    elif condition == "biascluster_weighted":
        w = rank_norm(ds["bc"])
    elif condition == "oracle_soft":
        w = np.where(wrong, cfg["weights"]["oracle_soft_incorrect"], 1.0)
    elif condition == "oracle_drop":
        w = np.where(wrong, 0.0, 1.0)
    elif condition in ds["hard"]:
        w = ds["hard"][condition].astype(float)
    else:
        raise ValueError(condition)
    w = np.clip(w, 0.0, hi)
    m = w.mean()
    return w / m if m > 0 else w                          # mean-normalise to 1.0


def weight_summary(w):
    q = np.percentile(w, [25, 50, 75])
    return {"min": round(float(w.min()), 3), "max": round(float(w.max()), 3),
            "mean": round(float(w.mean()), 3), "std": round(float(w.std()), 3),
            "q25": round(float(q[0]), 3), "median": round(float(q[1]), 3), "q75": round(float(q[2]), 3),
            "n_zero": int((w == 0).sum())}


def pretrain_metrics(condition, ds, w):
    wrong = ds["wrong"].astype(float)
    raw = float(wrong.mean())
    eff = float((w * wrong).sum() / w.sum()) if w.sum() else 0.0
    return {"n_train": len(wrong), "raw_label_error": round(raw, 4),
            "weighted_effective_error": round(eff, 4),
            "effective_error_reduction_pct": round(100 * (raw - eff) / raw, 1) if raw > 0 else 0.0,
            "weights": weight_summary(w)}


# ----------------------------- reward model -----------------------------
def train_and_eval_rm(ds, w, cfg, seed, device):
    import torch
    import torch.nn.functional as Fnn
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t = cfg["rm_train"]
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(t["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if t["fp16"] else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(t["base_model"], num_labels=1, dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    if t["use_lora"]:
        lc = LoraConfig(task_type=TaskType.SEQ_CLS, r=t["lora_r"], lora_alpha=t["lora_alpha"],
                        lora_dropout=t["lora_dropout"], bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        modules_to_save=["score"])
        model = get_peft_model(model, lc)
    if t.get("grad_checkpointing"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(t["learning_rate"]))
    L = t["max_length"]; bs = t["batch_pairs"]

    def fmt(prompt, resp):
        return tok.apply_chat_template([{"role": "user", "content": prompt},
                                        {"role": "assistant", "content": resp}], tokenize=False)

    def reward(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=L).to(device)
        return model(**enc).logits.squeeze(-1)

    pairs = ds["pairs"]
    keep = np.where(w > 0)[0]                              # drop zero-weight (oracle_drop / hard)
    rng = np.random.default_rng(seed)
    for ep in range(t["epochs"]):
        order = rng.permutation(keep)
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            ch = [fmt(pairs[i]["prompt"], pairs[i]["chosen"]) for i in b]
            rj = [fmt(pairs[i]["prompt"], pairs[i]["rejected"]) for i in b]
            wb = torch.tensor(w[b], dtype=torch.float32, device=device)
            rc, rr = reward(ch), reward(rj)
            loss = (wb * -Fnn.logsigmoid((rc - rr).float())).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], t["grad_clip"])
            opt.step()

    # ---- eval ----
    model.eval()
    def acc_loss(eval_pairs):
        rc, rr = [], []
        with torch.no_grad():
            for s in range(0, len(eval_pairs), bs):
                bp = eval_pairs[s:s + bs]
                rc.append(reward([fmt(p["prompt"], p["chosen"]) for p in bp]).float().cpu())
                rr.append(reward([fmt(p["prompt"], p["rejected"]) for p in bp]).float().cpu())
        rc = torch.cat(rc); rr = torch.cat(rr)
        acc = float((rc > rr).float().mean())
        loss = float((-Fnn.logsigmoid(rc - rr)).mean())
        return acc, loss, (rc - rr).numpy()
    ev = ds["eval"]
    acc, vloss, margin = acc_loss(ev["pairs"])
    tmask = ev["targeted"]
    tacc = float((margin[tmask] > 0).mean()) if tmask.any() else None
    del model; torch.cuda.empty_cache()
    return {"val_pref_acc": round(acc, 4), "val_loss": round(vloss, 4),
            "val_pref_acc_targeted": round(tacc, 4) if tacc is not None else None,
            "mean_eval_margin": round(float(margin.mean()), 4)}


# ----------------------------- orchestration -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mainstream_alignment_integration.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-regime/2-condition sanity run")
    ap.add_argument("--pretrain-only", action="store_true", help="compute label/weight metrics, no training")
    ap.add_argument("--regimes", nargs="+", default=None)
    ap.add_argument("--no-secondary", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    out = ROOT / cfg["out_dir"]; out.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hub = DataHub(cfg)

    regimes = cfg["regimes"]
    if args.regimes:
        regimes = [r for r in regimes if r["name"] in args.regimes]
    conditions = list(cfg["core_conditions"])
    if not args.no_secondary and not args.smoke:
        conditions += list(cfg["secondary_conditions"])
    if args.smoke:
        regimes = [r for r in cfg["regimes"] if r["name"] == "pos10"]
        conditions = ["naive", "biascluster_weighted"]
        cfg["rm_train"]["epochs"] = 1

    pre_rows, post_rows = [], []
    for rg in regimes:
        ds = build_regime(hub, rg["source"], rg["rate"])
        js = ds["judge_stats"]
        print(f"\n=== regime {rg['name']} (n_train={len(ds['pairs'])}, "
              f"raw_err={ds['wrong'].mean():.3f}, maj_cofail={js['majority_cofailure']}) ===")
        if args.smoke:
            ds["pairs"] = ds["pairs"][:48]; ds["wrong"] = ds["wrong"][:48]
            ds["cf"] = ds["cf"][:48]; ds["bc"] = ds["bc"][:48]; ds["gold"] = ds["gold"][:48]
            for k in ds["hard"]:
                ds["hard"][k] = ds["hard"][k][:48]
            ds["eval"]["pairs"] = ds["eval"]["pairs"][:48]
            ds["eval"]["targeted"] = ds["eval"]["targeted"][:48]
        for cond in conditions:
            if cond in cfg["secondary_conditions"] and cond not in ds["hard"]:
                continue
            w = make_weights(cond, ds, cfg)
            pm = pretrain_metrics(cond, ds, w)
            base = {"regime": rg["name"], "source": rg["source"], "contamination": rg["rate"],
                    "condition": cond, **{k: js[k] for k in ("rho_bar", "n_eff", "majority_cofailure",
                                                             "conditional_cofailure")},
                    "n_train": pm["n_train"], "raw_label_error": pm["raw_label_error"],
                    "weighted_effective_error": pm["weighted_effective_error"],
                    "eff_error_reduction_pct": pm["effective_error_reduction_pct"]}
            pre_rows.append({**base, **{f"w_{k}": v for k, v in pm["weights"].items()}})
            print(f"  {cond:22s} raw_err={pm['raw_label_error']:.3f} "
                  f"weff_err={pm['weighted_effective_error']:.3f} "
                  f"(-{pm['effective_error_reduction_pct']:.0f}%) wmean={pm['weights']['mean']}")
            if not args.pretrain_only:
                ev = train_and_eval_rm(ds, w, cfg, cfg["seed"], device)
                post_rows.append({**base, **ev})
                print(f"      -> val_acc={ev['val_pref_acc']} targeted={ev['val_pref_acc_targeted']} "
                      f"loss={ev['val_loss']}")

    pre = pd.DataFrame(pre_rows)
    pre.to_csv(out / "pretrain_metrics.csv", index=False)
    if post_rows:
        post = pd.DataFrame(post_rows)
        comp = pre.merge(post[["regime", "condition", "val_pref_acc", "val_pref_acc_targeted",
                               "val_loss", "mean_eval_margin"]], on=["regime", "condition"], how="left")
        comp.to_csv(out / "comparison_table.csv", index=False)
        write_summary(comp, out, cfg)
    else:
        pre.to_csv(out / "comparison_table.csv", index=False)
        print(f"\n[pretrain-only] wrote {out/'pretrain_metrics.csv'}")
    print(f"\nwrote outputs under {out}")


def write_summary(df, out, cfg):
    thr = 2 * (0.25 / max(int(cfg["data"]["eval_fraction"] * 2000), 1)) ** 0.5
    def g(regime, cond, col):
        r = df[(df.regime == regime) & (df.condition == cond)][col]
        return float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else None
    regimes = list(df.regime.unique())
    summ = {"experiment": "mainstream_alignment_integration", "noise_floor_winrate": round(thr, 4),
            "regimes": regimes, "per_regime": {}}
    cf_helps = bc_helps = oracle_helps = eff_improves = False
    pos_only = True
    for rg in regimes:
        nv = g(rg, "naive", "val_pref_acc")
        row = {"naive_val_acc": nv}
        for c in ["corrfilter_weighted", "biascluster_weighted", "oracle_soft", "oracle_drop"]:
            a = g(rg, c, "val_pref_acc")
            row[c + "_val_acc"] = a
            row[c + "_delta"] = round(a - nv, 4) if (a is not None and nv is not None) else None
        row["weighted_eff_err"] = {c: g(rg, c, "weighted_effective_error")
                                   for c in ["naive", "corrfilter_weighted", "biascluster_weighted", "oracle_drop"]}
        # did weighting reduce effective error?
        for c in ["corrfilter_weighted", "biascluster_weighted"]:
            if (g(rg, c, "weighted_effective_error") or 1) < (g(rg, "naive", "weighted_effective_error") or 0) - 0.005:
                eff_improves = True
        if (row.get("corrfilter_weighted_delta") or 0) > thr: cf_helps = True
        if (row.get("biascluster_weighted_delta") or 0) > thr:
            bc_helps = True
            if not rg.startswith("pos"): pos_only = False
        if max((row.get("oracle_soft_delta") or -1), (row.get("oracle_drop_delta") or -1)) > thr:
            oracle_helps = True
        summ["per_regime"][rg] = row
    summ["answers"] = {
        "1_weights_reduce_effective_label_error": bool(eff_improves),
        "2_weights_reduce_majority_cofailure_exposure": bool(eff_improves),
        "3_rm_improves_over_naive": bool(cf_helps or bc_helps),
        "4_oracle_meaningfully_better": bool(oracle_helps),
        "5_filtering_signal_too_weak_if_oracle_helps_but_filters_dont":
            bool(oracle_helps and not (cf_helps or bc_helps)),
        "6_alignment_insensitive_if_eff_error_improves_but_downstream_doesnt":
            bool(eff_improves and not (cf_helps or bc_helps or oracle_helps)),
        "7_benefit_regime_specific_position_only": bool(bc_helps and pos_only),
    }
    summ["verdict"] = (
        "Dependence-aware weighting did not produce a beyond-noise downstream reward-model gain over "
        "naive in any tested regime." if not (cf_helps or bc_helps) else
        "A downstream gain appeared; see per-regime deltas.") + (
        " Even the oracle upper bound did not clear the noise floor, indicating the alignment signal is "
        "insensitive at this scale." if not oracle_helps else
        " The oracle upper bound did clear the floor, so the ceiling exists but practical filter signals "
        "are weaker than it." if not (cf_helps or bc_helps) else "")
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print("\n=== summary.json ===\n" + json.dumps(summ["answers"], indent=2))
    print("verdict:", summ["verdict"])


if __name__ == "__main__":
    main()

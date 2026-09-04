"""Write a run manifest.json: provenance for a strengthening-phase run.

Records git commit (or a content hash if not a repo), model IDs from the configs used,
dataset/manifest versions (path + sha1 + row count), seeds, and output paths. Called at
the end of run_all so every run is reproducible. No heavy deps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha1(path, cap=8_000_000):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha1()
    if p.is_file():
        h.update(p.read_bytes()[:cap])
    else:
        for f in sorted(p.rglob("*"))[:200]:
            if f.is_file():
                h.update(f.name.encode())
    return h.hexdigest()[:12]


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _models_from_configs(configs):
    import yaml
    ids = {}
    for c in configs:
        p = Path(c)
        if not p.exists():
            continue
        try:
            d = yaml.safe_load(p.read_text())
            ids[c] = [{"hf_model": e.get("hf_model"), "adapter_path": e.get("adapter_path")}
                      for e in d.get("bank", [])]
        except Exception:
            pass
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[], help="config/bank yaml used (repeatable)")
    ap.add_argument("--dataset", action="append", default=[], help="dataset/manifest path (repeatable)")
    ap.add_argument("--seed", action="append", default=[], help="seed key=value (repeatable)")
    ap.add_argument("--output", action="append", default=[], help="output path produced (repeatable)")
    ap.add_argument("--stage", default="run_all")
    args = ap.parse_args()

    manifest = {
        "stage": args.stage,
        "git_commit": _git(),
        "configs": {c: _sha1(c) for c in args.config},
        "models": _models_from_configs(args.config),
        "datasets": {d: {"sha1": _sha1(d), "exists": Path(d).exists()} for d in args.dataset},
        "seeds": dict(s.split("=", 1) for s in args.seed if "=" in s),
        "outputs": {o: {"exists": Path(o).exists(), "sha1": _sha1(o)} for o in args.output},
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[write_manifest] git={manifest['git_commit']} -> {out}")


if __name__ == "__main__":
    main()

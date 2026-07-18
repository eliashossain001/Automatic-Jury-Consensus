"""Judge bank construction: parse YAML, cross with prompt styles, materialise specs.

A bank entry in the YAML config becomes ``len(prompt_styles)`` logical judges,
one per (base_judge, prompt_style) pair. The registry returns the materialised
``JudgeSpec`` list plus any runtime defaults so the voting runner can iterate
the bank one logical judge at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from corrfilter.judges.base import JudgeSpec


@dataclass(frozen=True)
class BankConfig:
    """Resolved judge-bank configuration ready for execution."""

    specs: list[JudgeSpec]
    batch_size: int
    max_new_tokens: int
    device: str
    shuffle_position: bool
    position_seed: int
    votes_dir: Path
    max_length: int = 4096


def load_bank_config(path: str | Path) -> BankConfig:
    cfg = yaml.safe_load(Path(path).read_text())
    bank_yaml = cfg["bank"]
    styles: list[str] = cfg["prompt_styles"]
    gen = cfg["generation"]
    rt = cfg["runtime"]
    out = cfg["output"]

    specs: list[JudgeSpec] = []
    for entry in bank_yaml:
        for style in styles:
            specs.append(
                JudgeSpec(
                    base_id=entry["id"],
                    family=entry["family"],
                    scale=str(entry["scale"]),
                    prompt_style=style,
                    hf_model=entry["hf_model"],
                    hf_revision=entry.get("hf_revision"),
                    quantization=entry.get("quantization", "nf4"),
                    dtype=entry.get("dtype", "bfloat16"),
                    adapter_path=entry.get("adapter_path"),
                    trust_remote_code=bool(entry.get("trust_remote_code", False)),
                )
            )

    return BankConfig(
        specs=specs,
        batch_size=int(rt.get("batch_size", 4)),
        max_new_tokens=int(gen.get("max_new_tokens", 8)),
        device=str(rt.get("device", "cuda")),
        shuffle_position=bool(rt.get("shuffle_position", True)),
        position_seed=int(rt.get("position_seed", 20260601)),
        votes_dir=Path(out["votes_dir"]),
        max_length=int(rt.get("max_length", 4096)),
    )


def build_bank(config_path: str | Path):
    """Yield ``(spec, factory)`` pairs so the runner can lazily instantiate judges.

    Lazy factories prevent eager construction of five 7-9B HuggingFace models,
    which would not fit in GPU memory simultaneously.
    """
    from corrfilter.judges.hf_judge import HFJudge

    bank = load_bank_config(config_path)
    for spec in bank.specs:
        def factory(spec=spec, bank=bank):
            return HFJudge(
                spec=spec,
                batch_size=bank.batch_size,
                max_new_tokens=bank.max_new_tokens,
                device=bank.device,
                position_seed=bank.position_seed,
                shuffle_position=bank.shuffle_position,
                max_length=bank.max_length,
            )

        yield spec, factory

# CorrFilter

Measuring and mitigating structured co-failure in LLM judge banks for alignment-data filtering.

## Layout

```
corrfilter/
  main.tex, references.bib     ICLR proposal source
  src/corrfilter/              Python package
    data/                      RewardBench v2 calibration set
    judges/                    Judge abstraction + HF local-weights judges + prompts
    voting/                    Calibration + filtering runners with on-disk vote cache
    correlation/               R estimation, n_eff, eigenspectrum, bootstrap
    analysis/                  Hypothesis tests and failure-mode decomposition
    viz/                       Heatmap, dendrogram, eigenspectrum, n_eff collapse
  configs/                     YAML configs for judge banks and calibration sets
  scripts/                     Numbered driver scripts (01_build_calibration_set.py ...)
  experiments/                 Per-experiment artefacts (votes/ figures/ results/)
  reports/                     Per-experiment Markdown writeups
  tests/                       Unit tests
```

## H1 entry point

```bash
pip install -e .
python scripts/01_build_calibration_set.py
python scripts/02_run_judges.py --config configs/judge_bank.yaml
python scripts/03_compute_correlation.py
python scripts/04_h1_analysis.py
```

Outputs land in `experiments/h1_measurement/` and the writeup in `reports/h1_measurement.md`.

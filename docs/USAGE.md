# Usage

This document gives the minimal commands needed to inspect, test, and reuse the SGC-VRPTW repository.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional LLM dependencies:

```bash
pip install -e ".[llm]"
```

## Smoke Tests

```bash
python -m pytest tests/test_toy_instance.py tests/test_ortools_smoke.py tests/test_llm_loop_smoke.py -q
```

The tests use the toy JSON instance and the mock LLM client, so they run without configuring an external LLM backend.

## Baseline Solver

```bash
python -m src.cli solve ortools data/toy/vrptw_tiny.json --time-limit 5
```

This runs the deterministic OR-Tools routing baseline on the toy instance.

## SGC-VRPTW Demo

```bash
python -m src.cli sgc data/toy/vrptw_tiny.json \
  --llm mock \
  --max-iterations 2 \
  --time-limit 20 \
  --output-dir results/demo/sgc_tiny
```

The `sgc` command is the public command name for the generated-solver workflow. The `droc` command is kept as a backward-compatible alias because older scripts and result fields used that implementation name.

## External LLM Backend

Use an OpenAI-compatible backend through environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"

python -m src.cli ping --llm openai
python -m src.cli sgc data/toy/vrptw_tiny.json --llm openai --max-iterations 4 --time-limit 60
```

The manuscript experiments describe the evaluated backend as a GPT-5.4-class advanced LLM model. For exact reproduction, use the same model family, solver versions, random seeds, and time budgets recorded with the experiment.

## Batch Experiments

Batch scripts expect benchmark instances under the layout described in `docs/DATA.md`.

Static OR-Tools baseline:

```bash
python scripts/experiments/batch_ortools.py --help
```

SGC-VRPTW batch comparison:

```bash
python scripts/experiments/run_batch_droc.py --help
```

Dynamic dispatch:

```bash
python -m src.cli dynamic-dispatch --help
python -m src.cli dynamic-compare --help
```

## Paper Evidence

The manuscript-level analyses use compact processed records:

```text
results/official/
evidence/figure_sources/
evidence/tables/
```

Final figure exports are stored at:

```text
paper/manuscript/fig/final/
```

Selected plotting scripts:

```bash
python scripts/figures/redraw_main_data_figures_seaborn.py
python scripts/figures/build_discussion_figures.py
```

If you move figure output paths, keep the manuscript paths in `paper/manuscript/main.tex` synchronized.

## Manuscript

The repository includes the current manuscript source and PDF:

```text
paper/manuscript/main.tex
paper/manuscript/main.pdf
```

Compilation depends on a LaTeX environment with the MDPI template requirements available in the manuscript package.

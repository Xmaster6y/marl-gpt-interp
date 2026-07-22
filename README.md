# marl-gpt-interp

[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![python versions](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org/downloads/)
[![LaTeX](https://img.shields.io/badge/latex-grey.svg?logo=latex)](https://www.latex-project.org/)
![ci](https://github.com/Xmaster6y/marl-gpt-interp/actions/workflows/ci.yml/badge.svg)

Research code and project memory for tracing action and value computations in the frozen MARL-GPT policy. The canonical method uses separate full-path actor and critic cross-layer transcoders (CLTs), prompt-local attribution graphs, and interventions in the original model.

## Layout

- `src/` contains reusable Python package code.
- `scripts/` contains independent experiment entrypoints.
- `configs/` contains one config folder per script.
- `docs/` contains project status, questions, claims, decisions, reviews, literature, experiment conclusions, presentations, and Slurm launch artifacts.
- `latex/` contains the paper.
- `results/` is untracked scratch space for logs, outputs, checkpoints, and generated artifacts.
- `AGENTS.md` contains the operational guidelines.

## Commands

Install and validate the local environment:

```bash
uv sync
just grf-install
just checks
just tests
```

The claim-bearing workflow is config-driven:

```bash
uv run -m scripts.experiments.circuit_tracing.build_balanced_dataset --config-name 2026-07-22-training-pilot
uv run -m scripts.experiments.circuit_tracing.audit_balanced_dataset --config-name 2026-07-22-training-pilot
uv run -m scripts.experiments.circuit_tracing.collect_corpus --config-name 2026-07-22-training-pilot
uv run -m scripts.experiments.circuit_tracing.train_clt --config-name 2026-07-22-actor-pilot
uv run -m scripts.experiments.circuit_tracing.train_clt --config-name 2026-07-22-critic-pilot
uv run -m scripts.experiments.circuit_tracing.evaluate_replacement --config-name 2026-07-22-pilot
uv run -m scripts.experiments.circuit_tracing.audit_clt_suite --config-name 2026-07-22-pilot
uv run -m scripts.experiments.circuit_tracing.build_graph --config-name 2026-07-22-actor-example
uv run -m scripts.experiments.circuit_tracing.build_graph --config-name 2026-07-22-critic-example
uv run -m scripts.experiments.circuit_tracing.evaluate_intervention --config-name 2026-07-22-actor-example
uv run -m scripts.experiments.circuit_tracing.evaluate_intervention --config-name 2026-07-22-critic-example
```

The frozen checkpoint is expected at `results/marl-gpt-main.pt`. Large datasets and outputs are untracked. The Jean Zay runtime and scratch paths are summarized in [docs/2026-07-02-grf-jz-setup.md](docs/2026-07-02-grf-jz-setup.md); the frozen scientific contract and evidence status live in [docs/README.md](docs/README.md).

## LaTeX

The paper lives in `latex/`. The LaTeX Workshop output directory is `%WORKSPACE_FOLDER%/latex/build/`.

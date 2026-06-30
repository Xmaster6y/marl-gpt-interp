# marl-gpt-interp

[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![python versions](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org/downloads/)
[![LaTeX](https://img.shields.io/badge/latex-grey.svg?logo=latex)](https://www.latex-project.org/)
![ci](https://github.com/Xmaster6y/marl-gpt-interp/actions/workflows/ci.yml/badge.svg)

Research code and project memory for interpreting MARL-GPT football behavior in Google Research Football.

## Layout

- `src/` contains reusable Python package code.
- `scripts/` contains experiment entrypoints.
- `configs/` contains experiment configs.
- `docs/` contains project status, questions, claims, decisions, reviews, literature, experiment conclusions, and Slurm launch artifacts.
- `latex/` contains the paper.
- `results/` is untracked scratch space for logs, outputs, checkpoints, and generated artifacts.
- `AGENTS.md` contains the operational guidelines.

## Commands

Python environment and package commands:

```bash
uv sync
just grf-install
uv add <package>
uv run -m scripts.run_experiment grf_rollout_stats=2026-06-30-smoke
```

Reusable recipes:

```bash
uv tool install rust-just
just install
just grf-install
just checks
just tests
just run grf_rollout_stats=2026-06-30-smoke
```

The first GRF experiment expects the MARL-GPT checkpoint at `results/marl-gpt-main.pt`.
The checkpoint is untracked; put it there manually or set `grf_rollout_stats.download_checkpoint=true`.
Installing `gfootball` requires system CMake and Boost.Python libraries matching the Python used for the virtual environment.
Use `just grf-install` after loading the matching system Python and Boost.Python stack; it disables uv-managed Python so the interpreter, CMake, and Boost.Python agree.

Slurm launch recipes:

```bash
bash docs/experiments/to-launch/2026-06-30-grf-rollout-statistics-v100.sh
just launch-all --dry-run
just retrieve jz
just clean
```

## LaTeX

The paper lives in `latex/`. The LaTeX Workshop output directory is `%WORKSPACE_FOLDER%/latex/build/`.

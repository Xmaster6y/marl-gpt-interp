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
- `scripts/` contains independent experiment entrypoints.
- `configs/` contains one config folder per script.
- `docs/` contains project status, questions, claims, decisions, reviews, literature, experiment conclusions, presentations, and Slurm launch artifacts.
- `latex/` contains the paper.
- `results/` is untracked scratch space for logs, outputs, checkpoints, and generated artifacts.
- `AGENTS.md` contains the operational guidelines.

## Commands

Python environment and package commands:

```bash
uv sync
just grf-install
uv add <package>
uv run -m scripts.grf_rollout_stats --config-name 2026-06-30-smoke
```

Reusable recipes:

```bash
uv tool install rust-just
just install
just grf-install
just checks
just tests
just run grf_rollout_stats 2026-06-30-smoke
```

The first GRF experiment expects the MARL-GPT checkpoint at `results/marl-gpt-main.pt`.
The checkpoint is untracked; put it there manually or pass `download_checkpoint=true` to the GRF rollout script.
Installing `gfootball` requires system CMake and native GRF engine libraries in addition to Python packages.
Use `just grf-install` after loading CMake and native GRF build dependencies; it uses uv-managed Python and passes that interpreter into the GRF CMake build.
The current JZ setup path is documented in `docs/2026-07-02-grf-jz-setup.md`.
Pass an explicit version when needed, for example `just grf-install 3.13` on local macOS if the available Boost.Python is `boost_python313`.

Slurm launch recipes:

```bash
bash docs/experiments/to-launch/2026-06-30-grf-rollout-statistics-v100.sh
just launch-all --dry-run
just retrieve jz
just clean
```

## LaTeX

The paper lives in `latex/`. The LaTeX Workshop output directory is `%WORKSPACE_FOLDER%/latex/build/`.

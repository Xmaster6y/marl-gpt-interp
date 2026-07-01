# GRF Rollout Statistics

## Status

Blocked on JZ as of 2026-06-30.

## Question

Can a fresh clone run MARL-GPT on GRF and produce simple, inspectable football statistics before representation probing or steering?

## Hypothesis

A short config-driven rollout should produce stable infrastructure evidence: checkpoint loading, environment stepping, action distributions, reward and score summaries, and basic position-derived statistics when raw GRF observations expose positions.

## Config Or Command

Local smoke:

```bash
just grf-install
uv run -m scripts.run_experiment grf_rollout_stats=2026-06-30-smoke
```

JZ small run:

```bash
just run grf_rollout_stats=2026-06-30-v100-small
```

Launch artifact:

- [`to-launch/2026-06-30-grf-rollout-statistics-v100.sh`](to-launch/2026-06-30-grf-rollout-statistics-v100.sh)

## Metrics

- Episode length, termination, truncation, reward, score, and goal difference.
- Action counts and action entropy.
- Possession fraction when raw observations expose `ball_owned_team`.
- Distance to goal, nearest defender distance, attacking team width and depth, and defensive compactness when raw positions are available.
- Set-piece counters from the GRF wrapper info when present.

## Baseline Or Comparison

No scientific baseline. This is an infrastructure and descriptive-statistics gate.

## Expected Result

The run writes per-step JSONL, per-episode JSONL, aggregate JSON, and aggregate CSV under `results/experiments/2026-06-30-grf-rollout-statistics/`. The output should be enough to decide whether GRF is ready for probes.

## Verification Notes

Base unit tests and checks do not require GRF. The `gfootball` dependency is pinned to a fork commit branched from `b3a0768` that passes the uv-managed build interpreter into GRF's CMake engine build. On the local macOS machine, uv-managed Python 3.13.5 plus Homebrew Boost.Python 3.13 builds, links against uv's `libpython3.13.dylib`, and passes a direct `gfootball.env.create_environment` reset and one-step smoke test. On JZ, `just grf-install` uses uv-managed Python 3.12 and project-local uv cache/temp directories under `results/`; this avoids the raw shell's `XDG_CACHE_HOME=/.cache` failure. A direct JZ test with `gcc/11.5.0`, `cmake/3.26.6`, and `boost/1.86.0` reached the native CMake build with uv's CPython 3.12.3, then failed because `SDL2Config.cmake` was unavailable. The JZ `boost/1.86.0` GCC variants also did not expose `libboost_python*` libraries in the module lib directory during inspection. So plain uv is not sufficient for GRF on a raw JZ platform; JZ still needs native SDL2, SDL2_image, SDL2_ttf, SDL2_gfx, OpenGL/EGL, and Boost.Python for the chosen Python minor, or a separate native-dependency bootstrap/wheel build.

JZ Slurm job `1126349` failed before running Python because the launch script sourced missing `./secret-env.sh`. The retrieved log is `results/slurm/grf-stats-1126349.err`, and no `results/experiments/` or `results/hydra/` directory existed remotely after the failure. The active launch script and Slurm templates now source `secret-env.sh` only when present, and `just retrieve jz` skips missing remote result folders after retrieving `results/slurm/`. The next JZ run still needs the checkpoint at `results/marl-gpt-main.pt` or `grf_rollout_stats.download_checkpoint=true`.

JZ Slurm job `1154231` reached Hydra setup but failed before rollout with `No module named 'gymnasium'`. The checkpoint was present at `results/marl-gpt-main.pt`, but the launch script used plain `uv run`, which installed only the default dependency groups. The active launch script now runs `uv run --python 3.12 --group grf` with project-local uv cache and temp directories under `results/`.

JZ Slurm job `1154677` failed before Hydra setup while uv was resolving the `gfootball` Git dependency. The Slurm log reported `Git executable not found` when fetching `https://github.com/Xmaster6y/football.git`. The active launch script now prepends `/usr/bin:/bin` to `PATH` and prints the resolved `git` executable before running uv.

JZ compute nodes do not have internet access, so Slurm jobs must not let uv sync or resolve missing dependencies. The stable JZ path uses the prebuilt wheel at `results/wheels/gfootball-2.10.3-cp312-cp312-linux_x86_64.whl`, the native prefix at `results/grf-native/py3.12`, and uv-managed CPython `3.12.11` under `results/uv-python/`. Prepare or update the environment on a login node first with `just grf-install-jz`, then submit the Slurm launch script. `just grf-install-jz` uses the populated JZ user uv cache when available, syncs the GRF dependency group with `--no-install-package gfootball --no-install-package wandb --inexact`, installs the prebuilt wheel with `uv pip install --no-deps`, and verifies imports without rebuilding GRF. The active launch script exports the matching `LD_LIBRARY_PATH`, `UV_PYTHON_INSTALL_DIR`, and runs `uv run --no-sync --python 3.12.11 --group grf` so dependency installation happens only during the login-node preparation step. This setup passed on the JZ login node and printed `jz grf env ok`.

## Decision Rule

If the fresh-env run works on local and JZ, move to activation capture and first concept probes. If GRF runtime or checkpoint access fails, document the blocker and use offline checkpoint inspection or dataset batches before claiming any probing result.

## Links

- [Start with GRF rollout statistics](../decisions/2026-06-30-start-with-grf-rollout-statistics.md)
- [Pretrained weights smoke test](2026-06-30-pretrained-weights-smoke-test.md)
- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)

# GRF On JZ Setup

## Status

Ready on the JZ login-node preparation path as of 2026-07-02; Slurm rollout completion still depends on the active GRF rollout experiment.

## Purpose

This note records the durable setup path for running MARL-GPT GRF experiments on JZ. It replaces the setup history that was previously embedded inside the GRF rollout statistics experiment note.

## Current Stable Path

Prepare or update the environment on a JZ login node before submitting Slurm jobs:

```bash
just grf-install-jz
```

Then submit the launch artifact for the experiment:

```bash
bash docs/experiments/to-launch/2026-06-30-grf-rollout-statistics-v100.sh
```

For SAE work, stage reusable datasets on the fast semi-permanent SCRATCH filesystem before submission:

```bash
just jz-stage-data
```

Long Hugging Face downloads must use a login shell so the IDRIS HTTP proxy is exported, or run through the `prepost`
partition. The balanced offline-corpus workflow uses the latter:

```bash
bash docs/experiments/to-launch/2026-07-20-balanced-offline-corpus-prepost.sh \
  2026-07-20-core-small
```

It keeps immutable revision-pinned files in one SCRATCH cache and creates config-specific balanced symlink views, so
larger mixtures reuse prior downloads without duplicating data.

If the JZ login node cannot reach PyPI's file host, copy the locked, platform-independent wheels into
`results/wheels/sae-jz/` and install the cached-activation SAE runtime without network access:

```bash
just sae-install-jz-offline
```

The JZ path uses:

- uv-managed CPython `3.12.11` under `results/uv-python/`.
- Reusable uv, Hugging Face, and Torch caches under
  `/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/.cache/` on SCRATCH.
- Reusable MARL-GPT datasets under `/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/dataset/` on SCRATCH.
- Per-job `TMPDIR` and transient W&B files under `$JOBSCRATCH`, with a SCRATCH fallback for nonstandard shells.
- Source, the Python environment, checkpoints, manifests, and retained experiment outputs under the WORK checkout.
- Prebuilt GRF wheel at `results/wheels/gfootball-2.10.3-cp312-cp312-linux_x86_64.whl`.
- Offline cached-activation SAE wheels under `results/wheels/sae-jz/`; the runtime uses
  `dictionary-learning`'s upstream TopK trainer while avoiding its unused language-model activation-buffer imports.
- Native GRF dependency prefix at `results/grf-native/py3.12`.
- Cached `torch==2.8.0`, which produced `torch 2.8.0+cu128` and supports V100 `sm_70`.
- `uv run --no-sync --python 3.12.11 --group grf --group sae` inside SAE Slurm jobs so compute nodes do not attempt
  dependency resolution.

The setup passed on the JZ login node with the import check output `jz grf env ok`.

IDRIS documents WORK for source and persistent inputs/outputs, SCRATCH for high-throughput data reused over several jobs
with a 30-day inactive-file lifetime, and JOBSCRATCH for job-lifetime temporary files that are deleted automatically.
Large caches and datasets therefore do not belong under HOME or the WORK checkout. Important outputs must never remain
only in JOBSCRATCH.

## Why Login-Node Preparation Is Required

JZ compute nodes do not have internet access, so Slurm jobs must not let uv sync or resolve missing dependencies. Dependency installation and wheel availability need to be prepared on the login node first.

Plain uv installation is not enough on a raw JZ shell. A direct test with `gcc/11.5.0`, `cmake/3.26.6`, and `boost/1.86.0` reached the native CMake build with uv CPython `3.12.3`, then failed because `SDL2Config.cmake` was unavailable. The inspected JZ `boost/1.86.0` GCC variants also did not expose `libboost_python*` libraries in the module lib directory.

## Resolved Failure Modes

- Job `1126349`: failed before Python because the launch script sourced missing `./secret-env.sh`. Launch scripts and Slurm templates now source `secret-env.sh` only when present.
- Job `1154231`: failed with `No module named 'gymnasium'` because the launch script used plain `uv run`. The launch path now uses the GRF dependency group.
- Job `1154677`: failed while resolving the `gfootball` Git dependency with `Git executable not found`. The launch script now prepends `/usr/bin:/bin` to `PATH` and prints the resolved `git` executable.
- Job `1157646`: failed with `No module named 'envs'` because the `marl-gpt` submodule was uninitialized on JZ. Run `git submodule update --init marl-gpt` on the login node when setting up a fresh clone.
- Job `1158389`: failed on V100 with `CUDA error: no kernel image is available for execution on the device` because the installed Torch wheel did not support compute capability `7.0`. Cached `torch==2.8.0` fixed this; probe job `1158897` verified CUDA availability, device capability `(7, 0)`, an arch list including `sm_70`, and successful CUDA tensor allocation.
- Job `1164968`: failed with exit `127` because the batch shell did not define the `module` command used by `module purge`. The active launch script now guards module usage with `command -v module`.

## Related Artifacts

- Experiment note: [GRF rollout statistics](experiments/2026-06-30-grf-rollout-statistics.md)
- Launch script: [GRF rollout statistics V100](experiments/to-launch/2026-06-30-grf-rollout-statistics-v100.sh)
- Raw Slurm logs: `results/slurm/`

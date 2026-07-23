# Jean Zay Runtime and Storage

## Runtime Contract

The canonical cluster environment uses Python 3.12.11, PyTorch 2.8, the repository's pinned GRF wheel, and the `grf` dependency group. The [runtime setup job](experiments/to-launch/2026-07-23-jhr-a100-runtime-setup.slurm) runs `just grf-install-jz` on `prepost`; do not perform the metadata-heavy installation on a shared login node. The primary GPU target is `jhr@a100`; every GPU job loads `arch/a100` before using the project environment. CLT training depends only on PyTorch and no longer installs the former SAE/dictionary-learning stack.

The Git checkout and installed environment remain on `$WORK`:

```text
/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp
```

The claim-bearing jobs inject the implementation commit through `EXPERIMENT_GIT_COMMIT`, so run manifests retain the submitted code revision even if documentation is subsequently fast-forwarded on `$WORK`.

The local `jz` Git remote now targets this `jhr` checkout. Before the former `nwq` WORK checkout was removed on 2026-07-23, its durable `results/experiments/`, `results/hydra/`, and `results/slurm/` trees were copied without overwriting newer files and verified against the destination by file count, byte count, and checksum dry-run. The frozen source checkpoint and reusable runtime assets were also staged on `jhr`; runtime setup job `83591` remains responsible for completing and validating the installed environment. Only the old checkout at `/lustre/fswork/projects/rech/nwq/uim47nr/marl-gpt-interp` was deleted; the broader `nwq` project storage and SCRATCH namespace were not touched.

## WORK/SCRATCH Placement Policy

WORK is the canonical location for code, runtime dependencies, and compact or durable experiment artifacts:

```text
/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp
```

SCRATCH is reserved for bulky, reconstructible datasets, activation corpora, downloads, caches, and temporary files:

```text
/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp
```

The suite sets both roots explicitly rather than relying on the login shell's default project variables.

| Durable or compact content | WORK location |
|---|---|
| Git checkout and installed environment | repository root and `.venv/` |
| Frozen source checkpoint | `results/marl-gpt-main.pt` |
| Trained CLTs and final checkpoints | `results/experiments/` |
| Metrics, audits, graphs, and interventions | `results/experiments/` |
| Hydra configuration and run metadata | `results/hydra/` |
| Slurm standard output and errors | `results/slurm/` |
| A100 preflight evidence | `results/preflight/` |

| Bulky or reconstructible content | SCRATCH location |
|---|---|
| Balanced dataset view | `balanced-datasets/2026-07-22-clt-training-pilot/` |
| Pinned Hugging Face source cache | `hf-cache/` |
| Combined actor/critic tensor corpus | `clt-corpora/2026-07-22-training-pilot/` |
| Shared package/model caches | `.cache/` |
| Per-job temporary and offline logging data | `jobs/` |

Do not place full datasets, Hugging Face downloads, activation tensors, CLT corpora, or other large regenerable intermediates on WORK. Compact metadata colocated with a SCRATCH dataset, such as its manifest and structural audit, may remain beside that dataset when doing so preserves an atomic data gate. The verified SHA-256 of the WORK source checkpoint is:

```text
c3deaeb67f679657b27e9d3373e42e4104cc9370be6dba60ab5fd0efe7b1ce5a
```

Claim-bearing artifacts are intentionally written to WORK so they survive SCRATCH cleanup and are easy to retrieve. The local-only smoke config remains relative to `results/`; the A100 preflight also writes its small smoke artifact and commit marker under WORK `results/preflight/`.

## Capacity and Retention

The `jhr` capacity check on 2026-07-23 reported a 5 TB WORK limit with effectively no project usage. Its SCRATCH path resides on a 400 TB filesystem with only 12 KB attributed to `jhr` and no enforced group quota. The estimated float16 tensor payload for the corpus is 152.9 GB (142.4 GiB), before JSONL metadata and filesystem overhead, so capacity is not a launch blocker.

SCRATCH is not backed up. Under the [IDRIS storage policy](https://www.idris.fr/static/intro/doc_nouvel_utilisateur-eng.html), files that have not been read or modified for 30 days may be purged. Active access resets the inactivity window; the bulky dataset and activation corpus are therefore regenerable inputs, while durable claim-bearing artifacts stay on WORK.

## A100 Launch Contract

The canonical launcher is [`docs/experiments/to-launch/2026-07-22-actor-critic-clt-suite.sh`](experiments/to-launch/2026-07-22-actor-critic-clt-suite.sh). It submits four dependency-linked records under `jhr@a100` only when the [A100 preflight](experiments/to-launch/2026-07-23-clt-a100-preflight.slurm) has recorded a passing runtime marker for the exact current Git commit:

1. balanced-dataset materialization and structural audit on `prepost`;
2. A100 corpus collection after the dataset audit succeeds;
3. independent actor and critic A100 training tasks after corpus collection succeeds;
4. replacement evaluation, the hard CLT eligibility audit, both example graphs, and both example interventions after both training tasks succeed.

The GPU records use `-C a100`, `qos_gpu_a100-t3`, one 80 GB A100, eight CPUs, approximately 58.5 GB of proportional host RAM, and a 20-hour limit. The data record uses `prepost`, two CPUs, the corresponding Jean Zay allocation of 60 GB RAM, and a 12-hour limit. `afterok` dependencies prevent training from running on an invalid corpus and prevent graph or intervention analysis from running when either CLT or the replacement audit fails.

Runtime setup job `83591` was submitted under `jhr@a100` on 2026-07-23 and is priority-pending on `prepost`. No A100 preflight or claim-bearing job has been submitted yet.

## Cancelled V100 History

The records `36790`, `36791`, `36792_[0-1]`, `36793`, `53452`, `53453`, `53454_[0-1]`, and `53455` were all cancelled at zero runtime. The first chain exposed Jean Zay's implicit 30 GB-per-prepost-CPU policy; the corrected second chain remained priority- or dependency-pending until the project migration. Neither chain produced logs, manifests, corpora, checkpoints, metrics, graphs, interventions, or scientific evidence.

## H100 Escalation

`jhr@h100` is available as a later, separate runtime target. A100 and H100 both provide 80 GB GPU memory, so H100 does not solve a GPU-memory requirement above 80 GB. It does provide greater compute throughput, approximately twice the proportional host RAM per GPU (about 117 GB with 24 CPUs), and access to a 100-hour QoS. Escalate only for measured runtime, host-memory, or wall-time pressure, and build a separate `arch/h100` environment plus preflight before submitting H100 jobs.

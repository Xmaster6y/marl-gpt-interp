# Jean Zay Runtime and Storage

## Runtime Contract

The canonical cluster environment uses Python 3.12.11, PyTorch 2.8, the repository's pinned GRF wheel, and the `grf` dependency group. `just grf-install-jz` prepares the runtime. The primary GPU target is `jhr@a100`; every GPU job loads `arch/a100` before using the project environment. CLT training depends only on PyTorch and no longer installs the former SAE/dictionary-learning stack.

The Git checkout and installed environment remain on `$WORK`:

```text
/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp
```

The claim-bearing jobs inject the implementation commit through `EXPERIMENT_GIT_COMMIT`, so run manifests retain the submitted code revision even if documentation is subsequently fast-forwarded on `$WORK`.

## SCRATCH Contract

The suite sets its SCRATCH root explicitly rather than relying on the login shell's default project variables:

```text
/lustre/fsn1/projects/rech/jhr/uim47nr
```

The project root is therefore:

```text
/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp
```

All generated files from the CLT suite use this root:

| Content | SCRATCH location |
|---|---|
| Staged frozen model | `checkpoints/marl-gpt-main.pt` |
| Balanced dataset view | `balanced-datasets/2026-07-22-clt-training-pilot/` |
| Pinned Hugging Face source cache | `hf-cache/` |
| Combined actor/critic tensor corpus | `clt-corpora/2026-07-22-training-pilot/` |
| Trained CLTs, metrics, audits, graphs, and interventions | `experiments/` |
| Hydra run state | `hydra/` |
| Slurm standard output and errors | `slurm/` |
| Shared package/model caches | `.cache/` |
| Per-job temporary and offline logging data | `jobs/` |

The Git checkout, installed runtime, native GRF libraries, and original source checkpoint remain on `$WORK`. The launcher copies the checkpoint to SCRATCH before submission and refuses to proceed if an existing staged copy differs. The verified SHA-256 of both copies is:

```text
c3deaeb67f679657b27e9d3373e42e4104cc9370be6dba60ab5fd0efe7b1ce5a
```

No generated claim-bearing artifact is written to `$WORK` by the submitted suite. The local-only smoke config remains intentionally relative to `results/` and is not used by Jean Zay jobs.

## Capacity and Retention

The `jhr` capacity check on 2026-07-23 reported a 5 TB WORK limit with effectively no project usage. Its SCRATCH path resides on a 400 TB filesystem with only 12 KB attributed to `jhr` and no enforced group quota. The estimated float16 tensor payload for the corpus is 152.9 GB (142.4 GiB), before JSONL metadata and filesystem overhead, so capacity is not a launch blocker.

SCRATCH is not backed up. Under the [IDRIS storage policy](https://www.idris.fr/static/intro/doc_nouvel_utilisateur-eng.html), files that have not been read or modified for 30 days may be purged. Active access resets the inactivity window; selected manifests, metrics, graphs, and final checkpoints must therefore be archived after the experiment rather than treated as permanent SCRATCH storage.

## A100 Launch Contract

The canonical launcher is [`docs/experiments/to-launch/2026-07-22-actor-critic-clt-suite.sh`](experiments/to-launch/2026-07-22-actor-critic-clt-suite.sh). It submits four dependency-linked records under `jhr@a100` only when the [A100 preflight](experiments/to-launch/2026-07-23-clt-a100-preflight.slurm) has recorded a passing runtime marker for the exact current Git commit:

1. balanced-dataset materialization and structural audit on `prepost`;
2. A100 corpus collection after the dataset audit succeeds;
3. independent actor and critic A100 training tasks after corpus collection succeeds;
4. replacement evaluation, the hard CLT eligibility audit, both example graphs, and both example interventions after both training tasks succeed.

The GPU records use `-C a100`, `qos_gpu_a100-t3`, one 80 GB A100, eight CPUs, approximately 58.5 GB of proportional host RAM, and a 20-hour limit. The data record uses `prepost`, two CPUs, the corresponding Jean Zay allocation of 60 GB RAM, and a 12-hour limit. `afterok` dependencies prevent training from running on an invalid corpus and prevent graph or intervention analysis from running when either CLT or the replacement audit fails.

## Cancelled V100 History

The records `36790`, `36791`, `36792_[0-1]`, `36793`, `53452`, `53453`, `53454_[0-1]`, and `53455` were all cancelled at zero runtime. The first chain exposed Jean Zay's implicit 30 GB-per-prepost-CPU policy; the corrected second chain remained priority- or dependency-pending until the project migration. Neither chain produced logs, manifests, corpora, checkpoints, metrics, graphs, interventions, or scientific evidence.

## H100 Escalation

`jhr@h100` is available as a later, separate runtime target. A100 and H100 both provide 80 GB GPU memory, so H100 does not solve a GPU-memory requirement above 80 GB. It does provide greater compute throughput, approximately twice the proportional host RAM per GPU (about 117 GB with 24 CPUs), and access to a 100-hour QoS. Escalate only for measured runtime, host-memory, or wall-time pressure, and build a separate `arch/h100` environment plus preflight before submitting H100 jobs.

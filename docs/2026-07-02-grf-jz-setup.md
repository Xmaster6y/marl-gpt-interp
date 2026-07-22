# Jean Zay Runtime and Storage

## Runtime Contract

The canonical cluster environment uses Python 3.12.11, PyTorch 2.8, the repository's pinned GRF wheel, and the `grf` dependency group. `just grf-install-jz` prepares the runtime. CLT training depends only on PyTorch and no longer installs the former SAE/dictionary-learning stack.

The Git checkout and installed environment remain on `$WORK`:

```text
/lustre/fswork/projects/rech/nwq/uim47nr/marl-gpt-interp
```

The claim-bearing jobs inject the implementation commit through `EXPERIMENT_GIT_COMMIT`, so run manifests retain the submitted code revision even if documentation is subsequently fast-forwarded on `$WORK`.

## SCRATCH Contract

`$SCRATCH` and `$CCFRSCRATCH` were verified on 2026-07-22 to resolve to:

```text
/lustre/fsn1/projects/rech/nwq/uim47nr
```

The project root is therefore:

```text
/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp
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

The live project quota check on 2026-07-22 reported a 400 TB byte limit with approximately 95.83 GB used, and 96,644 inodes used out of 150,000,000. The estimated float16 tensor payload for the corpus is 152.9 GB (142.4 GiB), before JSONL metadata and filesystem overhead, so capacity is not a launch constraint.

SCRATCH is not backed up. Under the [IDRIS storage policy](https://www.idris.fr/static/intro/doc_nouvel_utilisateur-eng.html), files that have not been read or modified for 30 days may be purged. Active access resets the inactivity window; selected manifests, metrics, graphs, and final checkpoints must therefore be archived after the experiment rather than treated as permanent SCRATCH storage.

## Submitted Suite

The canonical launcher is [`docs/experiments/to-launch/2026-07-22-actor-critic-clt-suite.sh`](experiments/to-launch/2026-07-22-actor-critic-clt-suite.sh). It submits four dependency-linked Slurm records from launch commit `b4e1deb9a7cab19ae84176f4aa79c1c897b4a69b`:

1. `53452`: balanced-dataset materialization and structural audit on `prepost`;
2. `53453`: V100 corpus collection after `53452` succeeds;
3. `53454_[0-1]`: independent actor and critic V100 training tasks after `53453` succeeds;
4. `53455`: replacement evaluation, the hard CLT eligibility audit, both example graphs, and both example interventions after both training tasks succeed.

The GPU records use `gpu_p13`, standard `qos_gpu-t3`, one V100 each, and a 20-hour limit. The data record uses `prepost`, two CPUs, the corresponding Jean Zay allocation of 60 GB RAM, and a 12-hour limit. `afterok` dependencies prevent training from running on an invalid corpus and prevent graph or intervention analysis from running when either CLT or the replacement audit fails.

The initial records `36790`, `36791`, `36792_[0-1]`, and `36793` were cancelled at zero runtime after Slurm revealed that eight preprocessing CPUs implicitly requested 240 GB RAM. Jean Zay rejects explicit memory directives, so the preprocessing request was corrected to two CPUs, which allocates 60 GB. The authoritative status refresh on 2026-07-22 found replacement job `53452` pending for priority and all downstream records pending on their declared dependencies. This is a completed launch process, not a scientific result.

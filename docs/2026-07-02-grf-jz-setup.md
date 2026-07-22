# Jean Zay Runtime

The canonical cluster environment uses Python 3.12.11, PyTorch 2.8, the repository's pinned GRF wheel, and the `grf` dependency group. `just grf-install-jz` prepares the runtime. CLT training depends only on PyTorch and no longer installs the former SAE/dictionary-learning stack.

Large datasets and the approximately 153 GB CLT activation corpus live under `/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp`, the resolved `$SCRATCH` path. The corpus config uses an absolute scratch output and copies only its manifests back to `results/` in the repository on `$WORK`. Trained CLTs, metrics, and graphs remain on `$WORK`. No claim-bearing job has yet been launched for the CLT workflow.

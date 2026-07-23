#!/usr/bin/env bash
#SBATCH --job-name=clt-data
#SBATCH --account=jhr@a100
#SBATCH --cpus-per-task=2
#SBATCH --partition=prepost
#SBATCH --output=/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp/results/slurm/%x-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp/results/slurm/%x-%j.err
#SBATCH --time=12:00:00

set -euo pipefail

work_repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
scratch_root="/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp"
cd "$work_repo"
if [[ -f ./secret-env.sh ]]; then
    source ./secret-env.sh
fi
export HF_HOME="$scratch_root/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$scratch_root/.cache"
export UV_CACHE_DIR="$scratch_root/.cache/uv"
export TMPDIR="$scratch_root/jobs/$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR"

.venv/bin/python -m scripts.experiments.circuit_tracing.build_balanced_dataset \
    --config-name 2026-07-22-training-pilot
.venv/bin/python -m scripts.experiments.circuit_tracing.audit_balanced_dataset \
    --config-name 2026-07-22-training-pilot

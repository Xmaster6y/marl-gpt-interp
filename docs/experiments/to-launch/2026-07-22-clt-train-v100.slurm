#!/usr/bin/env bash
#SBATCH --job-name=clt-train
#SBATCH --account=nwq@v100
#SBATCH --array=0-1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/slurm/%x-%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --time=20:00:00

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID:?training requires the two-task array}" in
    0) config_name=2026-07-22-actor-pilot ;;
    1) config_name=2026-07-22-critic-pilot ;;
    *) exit 2 ;;
esac
work_repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
scratch_root="/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp"
job_root="$scratch_root/jobs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$work_repo"
if command -v module >/dev/null 2>&1; then
    module purge
fi
mkdir -p "$job_root/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export TMPDIR="$job_root/tmp"
export TORCH_HOME="$scratch_root/.cache/torch"
export XDG_CACHE_HOME="$scratch_root/.cache"
export UV_CACHE_DIR="$scratch_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$work_repo/results/uv-python"
export UV_MANAGED_PYTHON=1
export PATH="/usr/bin:/bin:${PATH}"

uv run --no-sync --python 3.12.11 --group grf \
    -m scripts.experiments.circuit_tracing.train_clt \
    --config-name "$config_name"

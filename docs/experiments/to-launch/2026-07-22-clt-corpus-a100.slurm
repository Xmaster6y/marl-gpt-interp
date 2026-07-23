#!/usr/bin/env bash
#SBATCH --job-name=clt-corpus
#SBATCH --account=jhr@a100
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --output=/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp/results/slurm/%x-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp/results/slurm/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --time=20:00:00

set -euo pipefail

work_repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
scratch_root="/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp"
job_root="$scratch_root/jobs/$SLURM_JOB_ID"
cd "$work_repo"
if command -v module >/dev/null 2>&1; then
    module purge
    module load arch/a100
fi
if [[ -f ./secret-env.sh ]]; then
    source ./secret-env.sh
fi
mkdir -p "$job_root/tmp" "$job_root/wandb" "$job_root/wandb-cache"
export PREFIX="$work_repo/results/grf-native/py3.12"
export PYROOT="$work_repo/results/uv-python/cpython-3.12.11-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$PYROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$job_root/tmp"
export HF_HOME="$scratch_root/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$scratch_root/.cache/torch"
export XDG_CACHE_HOME="$scratch_root/.cache"
export UV_CACHE_DIR="$scratch_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$work_repo/results/uv-python"
export WANDB_DIR="$job_root/wandb"
export WANDB_CACHE_DIR="$job_root/wandb-cache"
export WANDB_MODE=offline
export UV_MANAGED_PYTHON=1
export PATH="/usr/bin:/bin:${PATH}"

uv run --no-sync --python 3.12.11 --group grf \
    -m scripts.experiments.circuit_tracing.collect_corpus \
    --config-name 2026-07-22-training-pilot

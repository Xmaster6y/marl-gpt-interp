#!/usr/bin/env bash
#SBATCH --job-name=sae-l03-smoke
#SBATCH --account=nwq@v100
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=02:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p results/slurm
    git_commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    exec sbatch --export="ALL,EXPERIMENT_GIT_COMMIT=$git_commit" "$0"
fi

if command -v module >/dev/null 2>&1; then
    module purge
fi

if [[ -f ./secret-env.sh ]]; then
    source ./secret-env.sh
fi

work_repo="${SLURM_SUBMIT_DIR:-$PWD}"
user_scratch="${SCRATCH:-/lustre/fsn1/projects/rech/nwq/uim47nr}"
scratch_root="$user_scratch/marl-gpt-interp"
job_root="${JOBSCRATCH:-$scratch_root/jobs/$SLURM_JOB_ID}"

cd "$work_repo"
mkdir -p results/experiments results/hydra results/slurm
mkdir -p "$job_root/tmp" "$job_root/wandb" "$job_root/wandb-cache"
mkdir -p "$scratch_root/.cache/huggingface/hub" "$scratch_root/.cache/huggingface/datasets"
mkdir -p "$scratch_root/.cache/torch" "$scratch_root/.cache/uv"

export PREFIX="$work_repo/results/grf-native/py3.12"
export PYROOT="$work_repo/results/uv-python/cpython-3.12.11-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$PYROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
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

test -f results/marl-gpt-main.pt
test -f "$scratch_root/dataset/zerg_5_vs_5/chunk_0_part_0.pt"
test -f "$scratch_root/dataset/dataset_grf/trajectories/academy_corner/chunk_1_part_0.pt"
test -f "$scratch_root/dataset/dataset_pogema_ll/random/part_0_0.arrow"

echo "work_repo=$work_repo"
echo "scratch_root=$scratch_root"
echo "job_root=$job_root"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.collect_activations \
    --config-name 2026-07-20-jz-smoke \
    hydra.run.dir="results/hydra/sparse_marl_gpt/collect_activations/${SLURM_JOB_ID}"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.train_dictionary \
    --config-name 2026-07-20-jz-smoke \
    hydra.run.dir="results/hydra/sparse_marl_gpt/train_dictionary/${SLURM_JOB_ID}"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.evaluate_dictionary \
    --config-name 2026-07-20-jz-smoke \
    hydra.run.dir="results/hydra/sparse_marl_gpt/evaluate_dictionary/${SLURM_JOB_ID}"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.analyze_features \
    --config-name 2026-07-20-jz-smoke \
    hydra.run.dir="results/hydra/sparse_marl_gpt/analyze_features/${SLURM_JOB_ID}"

#!/usr/bin/env bash
#SBATCH --job-name=repr-geometry
#SBATCH --account=nwq@v100
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p results/slurm
    exec sbatch "$0"
fi

if command -v module >/dev/null 2>&1; then
    module purge
fi

if [[ -f ./secret-env.sh ]]; then
    source ./secret-env.sh
fi

mkdir -p results/experiments results/slurm
mkdir -p results/uv-cache results/tmp

export PREFIX="$PWD/results/grf-native/py3.12"
export PYROOT="$PWD/results/uv-python/cpython-3.12.11-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$PYROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export UV_CACHE_DIR="$PWD/results/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/results/uv-python"
export TMPDIR="$PWD/results/tmp"
export UV_MANAGED_PYTHON=1
export PATH="/usr/bin:/bin:${PATH}"

echo "Using native prefix: $PREFIX"

uv run --no-sync --python 3.12.11 --group grf -m scripts.internal_representation_geometry \
    --config-name 2026-07-06-jz-small \
    hydra.run.dir="results/hydra/2026-07-06-internal-representation-geometry/${SLURM_JOB_ID}"

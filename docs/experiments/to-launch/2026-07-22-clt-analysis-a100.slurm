#!/usr/bin/env bash
#SBATCH --job-name=clt-analysis
#SBATCH --account=jhr@a100
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --output=/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp/slurm/%x-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp/slurm/%x-%j.err
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
mkdir -p "$job_root/tmp"
export PREFIX="$work_repo/results/grf-native/py3.12"
export PYROOT="$work_repo/results/uv-python/cpython-3.12.11-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$PYROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$job_root/tmp"
export TORCH_HOME="$scratch_root/.cache/torch"
export XDG_CACHE_HOME="$scratch_root/.cache"
export UV_CACHE_DIR="$scratch_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$work_repo/results/uv-python"
export UV_MANAGED_PYTHON=1
export PATH="/usr/bin:/bin:${PATH}"

run() {
    uv run --no-sync --python 3.12.11 --group grf -m "$1" --config-name "$2"
}

run scripts.experiments.circuit_tracing.evaluate_replacement 2026-07-22-pilot
run scripts.experiments.circuit_tracing.audit_clt_suite 2026-07-22-pilot
run scripts.experiments.circuit_tracing.build_graph 2026-07-22-actor-example
run scripts.experiments.circuit_tracing.evaluate_intervention 2026-07-22-actor-example
run scripts.experiments.circuit_tracing.build_graph 2026-07-22-critic-example
run scripts.experiments.circuit_tracing.evaluate_intervention 2026-07-22-critic-example

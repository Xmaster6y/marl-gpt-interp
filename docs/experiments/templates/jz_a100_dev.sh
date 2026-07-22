#!/usr/bin/env bash
#SBATCH --job-name=<job_name>
#SBATCH --account=jhr@a100
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=2:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p results/slurm
    exec sbatch "$0"
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load arch/a100
fi

if [[ -f ./secret-env.sh ]]; then
    source ./secret-env.sh
fi

mkdir -p results/experiments results/slurm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

<experiment_command>

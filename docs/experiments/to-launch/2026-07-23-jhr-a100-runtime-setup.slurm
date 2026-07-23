#!/usr/bin/env bash
#SBATCH --job-name=jhr-a100-runtime
#SBATCH --account=jhr@a100
#SBATCH --cpus-per-task=2
#SBATCH --partition=prepost
#SBATCH --output=/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp/results/slurm/%x-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp/results/slurm/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --time=02:00:00

set -euo pipefail

work_repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$work_repo"
mkdir -p results/slurm
export UV_NO_PROGRESS=1
just grf-install-jz

#!/usr/bin/env bash
#SBATCH --job-name=balanced-corpus
#SBATCH --account=nwq@v100
#SBATCH --cpus-per-task=4
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --mem=8G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --partition=prepost
#SBATCH --time=12:00:00

set -euo pipefail

config_name="${1:-2026-07-20-core-small}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p results/slurm
    git_commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    exec sbatch --export="ALL,EXPERIMENT_GIT_COMMIT=$git_commit" "$0" "$config_name"
fi

work_repo="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$work_repo"
export PATH="/usr/bin:/bin:${PATH}"
export XDG_CACHE_HOME="/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/.cache"

echo "config_name=$config_name"
echo "work_repo=$work_repo"
.venv/bin/python -m scripts.experiments.sparse_marl_gpt.build_balanced_dataset --config-name "$config_name"

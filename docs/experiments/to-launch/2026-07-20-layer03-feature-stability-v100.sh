#!/usr/bin/env bash
#SBATCH --job-name=sae-l03-stability
#SBATCH --account=nwq@v100
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH --error=results/slurm/%x-%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=02:00:00

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?submit this launcher as a Slurm array}"
case "$task_id" in
    0) left_seed=0; right_seed=1 ;;
    1) left_seed=0; right_seed=2 ;;
    2) left_seed=1; right_seed=2 ;;
    *) echo "unsupported stability comparison task $task_id" >&2; exit 2 ;;
esac

work_repo="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$work_repo"
user_scratch="${SCRATCH:-/lustre/fsn1/projects/rech/nwq/uim47nr}"
scratch_root="$user_scratch/marl-gpt-interp"
job_root="${JOBSCRATCH:-$scratch_root/jobs/$SLURM_JOB_ID}"
mkdir -p results/experiments results/hydra results/slurm "$job_root/tmp" "$scratch_root/.cache/uv"
export TMPDIR="$job_root/tmp"
export XDG_CACHE_HOME="$scratch_root/.cache"
export UV_CACHE_DIR="$scratch_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$work_repo/results/uv-python"
export UV_MANAGED_PYTHON=1
export PATH="/usr/bin:/bin:${PATH}"

model_name() {
    if [[ "$1" == 0 ]]; then
        echo 2026-07-20-layer03-balanced-training-small-topk-w2048-k16-seed0
    else
        echo "2026-07-20-layer03-posthealth-natural-w2048-k16-seed$1"
    fi
}
left_model="results/experiments/$(model_name "$left_seed")"
right_model="results/experiments/$(model_name "$right_seed")"
output_dir="results/experiments/2026-07-20-layer03-natural-w2048-seed${left_seed}-seed${right_seed}-stability"
test -f "$left_model/model.pt"
test -f "$right_model/model.pt"
test ! -e "$output_dir"

uv run --no-sync --python 3.12.11 --group sae \
    -m scripts.experiments.sparse_marl_gpt.compare_features \
    --config-name 2026-07-20-example \
    "model_dirs=[$left_model,$right_model]" "output_dir=$output_dir" \
    "hydra.run.dir=results/hydra/sparse_marl_gpt/compare_features/${SLURM_ARRAY_JOB_ID}_${task_id}"

#!/usr/bin/env bash
#SBATCH --job-name=sae-l03-dossiers
#SBATCH --account=nwq@v100
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --error=results/slurm/%x-%A_%a.err
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --time=04:00:00

set -euo pipefail

branch="${1:?expected primary, stability, or diagnostic branch}"
task_id="${SLURM_ARRAY_TASK_ID:?submit this launcher as a Slurm array}"
case "$branch:$task_id" in
    primary:0) seed=0; run_name=2026-07-20-layer03-balanced-training-small-topk-w2048-k16-seed0 ;;
    stability:0) seed=1; run_name=2026-07-20-layer03-posthealth-natural-w2048-k16-seed1 ;;
    stability:1) seed=2; run_name=2026-07-20-layer03-posthealth-natural-w2048-k16-seed2 ;;
    diagnostic:0) seed=0; run_name=2026-07-20-layer03-posthealth-natural-w512-k16-seed0 ;;
    diagnostic:1) seed=0; run_name=2026-07-20-layer03-posthealth-per-domain-center-rms-w512-k16-seed0 ;;
    diagnostic:2) seed=0; run_name=2026-07-20-layer03-posthealth-per-domain-center-rms-w2048-k16-seed0 ;;
    *) echo "unsupported dossier task $branch:$task_id" >&2; exit 2 ;;
esac

work_repo="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$work_repo"
export PATH="/usr/bin:/bin:${PATH}"
export XDG_CACHE_HOME="/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/.cache"

cache_dir="results/experiments/2026-07-20-layer03-balanced-training-small-cache"
feature_dir="results/experiments/$run_name-features"
output_dir="results/experiments/$run_name-dossiers"
test -f "$feature_dir/feature_summary.jsonl"
test ! -e "$output_dir"

.venv/bin/python -m scripts.experiments.sparse_marl_gpt.build_feature_dossiers \
    --config-name 2026-07-20-layer03-balanced-training-small \
    "cache_dir=$cache_dir" "feature_dir=$feature_dir" "output_dir=$output_dir" "seed=$seed" \
    "hydra.run.dir=results/hydra/sparse_marl_gpt/build_feature_dossiers/${SLURM_ARRAY_JOB_ID}_${task_id}"

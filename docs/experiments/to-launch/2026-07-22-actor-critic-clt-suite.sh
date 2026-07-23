#!/usr/bin/env bash
set -euo pipefail

work_repo="$(git rev-parse --show-toplevel)"
scratch_root="/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp"
artifact_root="$work_repo/results"
launch_root="$work_repo/docs/experiments/to-launch"
source_checkpoint="$work_repo/results/marl-gpt-main.pt"
git_commit="$(git -C "$work_repo" rev-parse HEAD)"
preflight_commit="$artifact_root/preflight/a100-runtime-pass.commit"

test -f "$source_checkpoint"
test -f "$preflight_commit"
test "$(tr -d '\n' < "$preflight_commit")" = "$git_commit"
mkdir -p \
    "$artifact_root/slurm" \
    "$artifact_root/experiments" \
    "$artifact_root/hydra" \
    "$scratch_root/jobs" \
    "$scratch_root/.cache"

for output in \
    "$scratch_root/clt-corpora/2026-07-22-training-pilot" \
    "$artifact_root/experiments/2026-07-22-actor-clt-pilot" \
    "$artifact_root/experiments/2026-07-22-critic-clt-pilot" \
    "$artifact_root/experiments/2026-07-22-clt-replacement-pilot" \
    "$artifact_root/experiments/2026-07-22-clt-suite-audit" \
    "$artifact_root/experiments/2026-07-22-actor-graph-example" \
    "$artifact_root/experiments/2026-07-22-critic-graph-example" \
    "$artifact_root/experiments/2026-07-22-actor-intervention-example" \
    "$artifact_root/experiments/2026-07-22-critic-intervention-example"
do
    if [[ -e "$output" ]]; then
        echo "refusing to overwrite existing output: $output" >&2
        exit 1
    fi
done

cd "$work_repo"
data_job="$(sbatch --parsable --export="ALL,EXPERIMENT_GIT_COMMIT=$git_commit" \
    "$launch_root/2026-07-22-clt-data-prepost.slurm")"
data_job="${data_job%%;*}"
corpus_job="$(sbatch --parsable --dependency="afterok:$data_job" \
    --export="ALL,EXPERIMENT_GIT_COMMIT=$git_commit" \
    "$launch_root/2026-07-22-clt-corpus-a100.slurm")"
corpus_job="${corpus_job%%;*}"
training_job="$(sbatch --parsable --dependency="afterok:$corpus_job" \
    --export="ALL,EXPERIMENT_GIT_COMMIT=$git_commit" \
    "$launch_root/2026-07-22-clt-train-a100.slurm")"
training_job="${training_job%%;*}"
analysis_job="$(sbatch --parsable --dependency="afterok:$training_job" \
    --export="ALL,EXPERIMENT_GIT_COMMIT=$git_commit" \
    "$launch_root/2026-07-22-clt-analysis-a100.slurm")"
analysis_job="${analysis_job%%;*}"

printf 'data_job=%s\ncorpus_job=%s\ntraining_array=%s\nanalysis_job=%s\n' \
    "$data_job" "$corpus_job" "$training_job" "$analysis_job"

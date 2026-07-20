#!/usr/bin/env bash
#SBATCH --job-name=sae-l03-posthealth
#SBATCH --account=nwq@v100
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH --error=results/slurm/%x-%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00

set -euo pipefail

branch="${1:?expected stability or diagnostic branch}"
task_id="${SLURM_ARRAY_TASK_ID:?submit this launcher as a Slurm array}"
case "$branch:$task_id" in
    stability:0) width=2048; seed=1; preprocessing=natural ;;
    stability:1) width=2048; seed=2; preprocessing=natural ;;
    diagnostic:0) width=512; seed=0; preprocessing=natural ;;
    diagnostic:1) width=512; seed=0; preprocessing=per_domain_center_rms ;;
    diagnostic:2) width=2048; seed=0; preprocessing=per_domain_center_rms ;;
    *) echo "unsupported post-health task $branch:$task_id" >&2; exit 2 ;;
esac

work_repo="${SLURM_SUBMIT_DIR:-$PWD}"
primary_audit="$work_repo/results/experiments/2026-07-20-layer03-balanced-training-small-suite-audit/suite_audit.json"
cd "$work_repo"
.venv/bin/python - "$primary_audit" "$branch" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
branch = sys.argv[2]
if branch == "stability":
    assert audit["status"] == "passed", audit
else:
    assert audit["status"] == "failed", audit
    assert all(audit["structural_checks"].values()), audit["structural_checks"]
    assert audit["health"]["passed"] is False, audit["health"]
PY

if command -v module >/dev/null 2>&1; then
    module purge
fi
if [[ -f ./secret-env.sh ]]; then
    source ./secret-env.sh
fi

user_scratch="${SCRATCH:-/lustre/fsn1/projects/rech/nwq/uim47nr}"
scratch_root="$user_scratch/marl-gpt-interp"
job_root="${JOBSCRATCH:-$scratch_root/jobs/$SLURM_JOB_ID}"
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

mode_slug="${preprocessing//_/-}"
run_name="2026-07-20-layer03-posthealth-${mode_slug}-w${width}-k16-seed${seed}"
cache_dir="results/experiments/2026-07-20-layer03-balanced-training-small-cache"
model_dir="results/experiments/$run_name"
evaluation_dir="results/experiments/$run_name-evaluation"
feature_dir="results/experiments/$run_name-features"
audit_dir="results/experiments/$run_name-suite-audit"
strict=false
if [[ "$branch" == stability ]]; then
    strict=true
fi

test ! -e "$model_dir"
echo "branch=$branch task_id=$task_id width=$width seed=$seed preprocessing=$preprocessing"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.train_dictionary \
    --config-name 2026-07-20-layer03-balanced-training-small \
    "model.width=$width" "seed=$seed" "preprocessing.mode=$preprocessing" \
    "output_dir=$model_dir" observability.wandb.enabled=false \
    "hydra.run.dir=results/hydra/sparse_marl_gpt/train_dictionary/${SLURM_ARRAY_JOB_ID}_${task_id}"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.evaluate_dictionary \
    --config-name 2026-07-20-layer03-balanced-training-small \
    "model_dir=$model_dir" "output_dir=$evaluation_dir" "seed=$seed" \
    "hydra.run.dir=results/hydra/sparse_marl_gpt/evaluate_dictionary/${SLURM_ARRAY_JOB_ID}_${task_id}"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.analyze_features \
    --config-name 2026-07-20-layer03-balanced-training-small \
    "model_dir=$model_dir" "output_dir=$feature_dir" "seed=$seed" \
    "hydra.run.dir=results/hydra/sparse_marl_gpt/analyze_features/${SLURM_ARRAY_JOB_ID}_${task_id}"

uv run --no-sync --python 3.12.11 --group grf --group sae \
    -m scripts.experiments.sparse_marl_gpt.audit_sae_suite \
    --config-name 2026-07-20-layer03-balanced-training-small \
    "model_dir=$model_dir" "evaluation_dir=$evaluation_dir" "feature_dir=$feature_dir" \
    "output_dir=$audit_dir" "strict=$strict" "seed=$seed" \
    "hydra.run.dir=results/hydra/sparse_marl_gpt/audit_sae_suite/${SLURM_ARRAY_JOB_ID}_${task_id}"

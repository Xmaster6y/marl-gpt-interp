#!/usr/bin/env bash
#SBATCH --job-name=sae-l03-balanced
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

config_name="${1:-2026-07-20-layer03-balanced-core-small}"
case "$config_name" in
    2026-07-20-layer03-balanced-core-small)
        dataset_manifest="/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/balanced-datasets/2026-07-20-core-balanced-small/balanced_dataset_manifest.json"
        ;;
    2026-07-20-layer03-balanced-training-small)
        dataset_manifest="/lustre/fsn1/projects/rech/nwq/uim47nr/marl-gpt-interp/balanced-datasets/2026-07-20-training-small/balanced_dataset_manifest.json"
        ;;
    *)
        echo "unsupported suite config: $config_name" >&2
        exit 2
        ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p results/slurm
    git_commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    exec sbatch --export="ALL,EXPERIMENT_GIT_COMMIT=$git_commit" "$0" "$config_name"
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
.venv/bin/python - "$dataset_manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["status"] == "audited_balanced_pending_provenance", manifest["status"]
assert manifest["structural_balance_passed"] is True
PY

echo "config_name=$config_name"
echo "work_repo=$work_repo"
echo "scratch_root=$scratch_root"
echo "job_root=$job_root"

for stage in collect_activations train_dictionary evaluate_dictionary analyze_features audit_sae_suite; do
    uv run --no-sync --python 3.12.11 --group grf --group sae \
        -m "scripts.experiments.sparse_marl_gpt.${stage}" \
        --config-name "$config_name" \
        hydra.run.dir="results/hydra/sparse_marl_gpt/${stage}/${SLURM_JOB_ID}"
done

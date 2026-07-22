#!/usr/bin/env bash
#SBATCH --job-name=clt-a100-preflight
#SBATCH --account=jhr@a100
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --output=/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp/slurm/%x-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp/slurm/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --time=00:20:00

set -euo pipefail

work_repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
scratch_root="/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp"
job_root="$scratch_root/jobs/$SLURM_JOB_ID"
preflight_root="$scratch_root/preflight/a100-$SLURM_JOB_ID"
cd "$work_repo"

if command -v module >/dev/null 2>&1; then
    module purge
    module load arch/a100
fi
mkdir -p "$job_root/tmp" "$preflight_root" "$scratch_root/slurm"

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

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
uv run --no-sync --python 3.12.11 --group grf python -c \
    "import gfootball, torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'sum': float(x.sum())})"
uv run --no-sync --python 3.12.11 --group grf \
    -m scripts.experiments.circuit_tracing.collect_corpus \
    --config-name 2026-07-22-local-smoke \
    device=cuda:0 \
    output_dir="$preflight_root/corpus-smoke" \
    hydra.run.dir="$preflight_root/hydra"

git_commit="$(git -C "$work_repo" rev-parse HEAD)"
printf '%s\n' "$git_commit" > "$preflight_root/runtime-pass.commit"
cp "$preflight_root/runtime-pass.commit" "$scratch_root/preflight/a100-runtime-pass.commit"

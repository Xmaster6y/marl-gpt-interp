cluster_host_cv := "cv"
cluster_host_jz := "jz"
cluster_repo_cv := "~/work/marl-gpt-interp"
cluster_repo_jz := "/lustre/fswork/projects/rech/nwq/uim47nr/marl-gpt-interp"
result_folders := "slurm experiments hydra"
grf_python := "3.12"
grf_jz_python := "3.12.11"
grf_jz_native_prefix := "results/grf-native/py3.12"
grf_jz_python_install_dir := "results/uv-python"
grf_jz_wheel := "results/wheels/gfootball-2.10.3-cp312-cp312-linux_x86_64.whl"

install:
	uv run pre-commit install
	uv sync

grf-install python=grf_python:
	mkdir -p results/uv-cache results/tmp
	UV_CACHE_DIR="$PWD/results/uv-cache" TMPDIR="$PWD/results/tmp" UV_MANAGED_PYTHON=1 uv sync --python {{python}} --group grf

grf-install-jz python=grf_jz_python:
	#!/usr/bin/env bash
	set -euo pipefail
	wheel="{{grf_jz_wheel}}"
	prefix="$PWD/{{grf_jz_native_prefix}}"
	pyroot="$PWD/{{grf_jz_python_install_dir}}/cpython-{{python}}-linux-x86_64-gnu"
	cache_dir="$PWD/results/uv-cache"
	if [[ -d "$HOME/.cache/uv" ]]; then
		cache_dir="$HOME/.cache/uv"
	fi
	test -f "$wheel"
	test -d "$prefix"
	test -d "$pyroot"
	mkdir -p results/uv-cache results/tmp
	export PREFIX="$prefix"
	export PYROOT="$pyroot"
	export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$PYROOT/lib:${LD_LIBRARY_PATH:-}"
	export UV_MANAGED_PYTHON=1
	export UV_PYTHON_INSTALL_DIR="$PWD/{{grf_jz_python_install_dir}}"
	export UV_CACHE_DIR="$cache_dir"
	export TMPDIR="$PWD/results/tmp"
	uv sync --python {{python}} --group grf --no-install-package gfootball --no-install-package wandb --inexact
	uv pip install --python .venv/bin/python --no-deps "$wheel"
	uv run --no-sync --python {{python}} --group grf python -c "import gymnasium, gfootball, torch; print('jz grf env ok')"

grf-reinstall-gfootball python=grf_python:
	mkdir -p results/uv-cache results/tmp
	UV_CACHE_DIR="$PWD/results/uv-cache" TMPDIR="$PWD/results/tmp" UV_MANAGED_PYTHON=1 uv sync --python {{python}} --group grf --reinstall-package gfootball

checks:
	uv run pre-commit run --all-files

test-assets:
	@echo "No test assets to resolve"

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

run *args:
	uv run -m scripts.run_experiment {{args}}

launch-all dry_run="":
	#!/usr/bin/env bash
	set -euo pipefail
	shopt -s nullglob
	for job in docs/experiments/to-launch/*.sh; do
		if [ "{{dry_run}}" = "--dry-run" ] || [ "{{dry_run}}" = "dry-run" ]; then
			echo "bash ${job}"
		else
			bash "${job}"
		fi
	done
	shopt -u nullglob

wandb-sync clean="":
	export WANDB__SERVICE_WAIT=300; \
	uv run wandb sync results/experiments/*/wandb/offline-run-*; \
	if [ "{{clean}}" = "clean" ]; then \
		rm -r results/experiments/*/wandb/offline-run-*; \
	fi

retrieve cluster folder="":
	#!/usr/bin/env bash
	set -euo pipefail
	case "{{cluster}}" in
		cv)
			host="{{cluster_host_cv}}"
			repo="{{cluster_repo_cv}}"
			;;
		jz)
			host="{{cluster_host_jz}}"
			repo="{{cluster_repo_jz}}"
			;;
		*)
			echo "unknown cluster: {{cluster}} (expected cv or jz)" >&2
			exit 1
			;;
	esac
	folders="{{result_folders}}"
	if [ -n "{{folder}}" ]; then
		folders="$folders {{folder}}"
	fi
	for folder in $folders; do
		folder="${folder#results/}"
		folder="${folder#/}"
		folder="${folder%/}"
		case "$folder" in
			""|.|..|*/*)
				echo "refusing to retrieve unsafe result folder: $folder" >&2
				exit 1
				;;
		esac
		mkdir -p "./results/$folder"
		echo "Syncing results/$folder/ from {{cluster}}..."
		if ssh -q "$host" "test -d '$repo/results/$folder'"; then
			rsync -a "$host:$repo/results/$folder/" "./results/$folder/"
		else
			echo "Skipping missing remote folder: $repo/results/$folder" >&2
		fi
	done

clean folder="":
	#!/usr/bin/env bash
	set -euo pipefail
	shopt -s dotglob nullglob
	folders="{{result_folders}}"
	if [ -n "{{folder}}" ]; then
		folders="$folders {{folder}}"
	fi
	for folder in $folders; do
		folder="${folder#results/}"
		folder="${folder#/}"
		folder="${folder%/}"
		case "$folder" in
			""|.|..|*/*)
				echo "refusing to clean unsafe result folder: $folder" >&2
				exit 1
				;;
		esac
		mkdir -p "./results/$folder"
		echo "Cleaning results/$folder/*..."
		rm -rf "./results/$folder"/*
	done

sync-to cluster:
	#!/usr/bin/env bash
	set -euo pipefail
	case "{{cluster}}" in
		cv)
			repo='{{cluster_repo_cv}}'
			;;
		jz)
			repo='{{cluster_repo_jz}}'
			;;
		*)
			echo "unknown cluster: {{cluster}} (expected cv or jz)" >&2
			exit 1
			;;
	esac
	echo "Updating local branch tr from main..."
	git fetch -q . main:tr
	echo "Pushing tr to {{cluster}}..."
	git push -q "{{cluster}}" tr
	echo "Fast-forward main from tr on {{cluster}}..."
	ssh -q "{{cluster}}" "cd $repo && git checkout -q main && git merge -q tr --ff-only"

sync-from cluster:
	#!/usr/bin/env bash
	set -euo pipefail
	case "{{cluster}}" in
		cv)
			repo='{{cluster_repo_cv}}'
			;;
		jz)
			repo='{{cluster_repo_jz}}'
			;;
		*)
			echo "unknown cluster: {{cluster}} (expected cv or jz)" >&2
			exit 1
			;;
	esac
	echo "Fetching main:tr on {{cluster}}..."
	ssh -q "{{cluster}}" "cd $repo && git fetch -q . main:tr"
	echo "Fetching tr from {{cluster}}..."
	git fetch -q "{{cluster}}" tr:tr
	echo "Merging tr --ff-only locally..."
	git merge -q tr --ff-only

cluster_host_cv := "cv"
cluster_host_jz := "jz"
cluster_repo_cv := "~/work/marl-gpt-interp"
cluster_repo_jz := "/lustre/fswork/projects/rech/jhr/uim47nr/marl-gpt-interp"
result_folders := "slurm experiments hydra"
grf_python := "3.12"
grf_jz_python := "3.12.11"
grf_jz_native_prefix := "results/grf-native/py3.12"
grf_jz_python_install_dir := "results/uv-python"
grf_jz_wheel := "results/wheels/gfootball-2.10.3-cp312-cp312-linux_x86_64.whl"
grf_jz_torch := "2.8.0"
jz_scratch_root := "/lustre/fsn1/projects/rech/jhr/uim47nr/marl-gpt-interp"

install:
	uv run pre-commit install
	uv sync

grf-install python=grf_python:
	mkdir -p results/uv-cache results/tmp
	UV_CACHE_DIR="$PWD/results/uv-cache" TMPDIR="$PWD/results/tmp" UV_MANAGED_PYTHON=1 uv sync --python {{python}} --group grf

grf-install-jz python=grf_jz_python:
	#!/usr/bin/env bash
	set -euo pipefail
	if command -v module >/dev/null 2>&1; then
		module purge
		module load arch/a100
	fi
	wheel="{{grf_jz_wheel}}"
	prefix="$PWD/{{grf_jz_native_prefix}}"
	pyroot="$PWD/{{grf_jz_python_install_dir}}/cpython-{{python}}-linux-x86_64-gnu"
	cache_dir="{{jz_scratch_root}}/.cache/uv"
	tmp_dir="{{jz_scratch_root}}/tmp/setup"
	test -f "$wheel"
	test -d "$prefix"
	test -d "$pyroot"
	mkdir -p "$cache_dir" "$tmp_dir"
	export PREFIX="$prefix"
	export PYROOT="$pyroot"
	export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$PYROOT/lib:${LD_LIBRARY_PATH:-}"
	export UV_MANAGED_PYTHON=1
	export UV_PYTHON_INSTALL_DIR="$PWD/{{grf_jz_python_install_dir}}"
	export UV_CACHE_DIR="$cache_dir"
	export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
	export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
	export TMPDIR="$tmp_dir"
	uv sync --python {{python}} --group grf --no-install-package gfootball --no-install-package torch --inexact
	uv pip install --python .venv/bin/python "torch=={{grf_jz_torch}}"
	uv pip install --python .venv/bin/python --no-deps "$wheel"
	uv run --no-sync --python {{python}} --group grf python -c "import gymnasium, gfootball, torch; print('jz grf+clt env ok')"

jz-stage-data:
	#!/usr/bin/env bash
	set -euo pipefail
	source_dir="$PWD/dataset"
	target_dir="{{jz_scratch_root}}/dataset"
	test -d "$source_dir"
	mkdir -p "$target_dir"
	rsync -a --ignore-existing "$source_dir/" "$target_dir/"
	find "$target_dir" -type f -print

grf-reinstall-gfootball python=grf_python:
	mkdir -p results/uv-cache results/tmp
	UV_CACHE_DIR="$PWD/results/uv-cache" TMPDIR="$PWD/results/tmp" UV_MANAGED_PYTHON=1 uv sync --python {{python}} --group grf --reinstall-package gfootball

checks:
	uv run pre-commit run --all-files

test-assets:
	@echo "No test assets to resolve"

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

run script config *args:
	uv run -m scripts.{{script}} --config-name {{config}} {{args}}

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

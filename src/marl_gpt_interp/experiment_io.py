"""Small provenance and artifact helpers shared by experiment workflows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    injected = os.environ.get("EXPERIMENT_GIT_COMMIT")
    if injected:
        return injected
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_run_manifest(
    path: Path,
    *,
    root: Path,
    run_id: str,
    config: Mapping[str, Any],
    seed: int,
    status: str,
    artifacts: Mapping[str, str] | None = None,
    hashes: Mapping[str, str] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    environment_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config_json = json.dumps(dict(config), sort_keys=True, default=str)
    payload = {
        "format_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "config": dict(config),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "seed": seed,
        "status": status,
        "artifacts": dict(artifacts or {}),
        "hashes": dict(hashes or {}),
        "split_manifest": dict(split_manifest or {}),
        "environment_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            **dict(environment_versions or {}),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload


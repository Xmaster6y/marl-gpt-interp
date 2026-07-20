"""Deterministic, resumable construction of balanced offline dataset views."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class RemoteFile:
    source_path: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class SelectedFile:
    environment: str
    component: str
    source_path: str
    destination_path: str
    size: int
    sha256: str | None = None


def allocate_component_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Use largest remainders while retaining every positive component when possible."""

    positive = {name: float(weight) for name, weight in weights.items() if float(weight) > 0}
    if total <= 0 or not positive:
        raise ValueError("total and at least one component weight must be positive")
    if total < len(positive):
        raise ValueError("total must be at least the number of positive mixture components")
    denominator = sum(positive.values())
    exact = {name: total * weight / denominator for name, weight in positive.items()}
    counts = {name: int(value) for name, value in exact.items()}
    for name in sorted(positive):
        if counts[name] == 0:
            donor = max(counts, key=lambda candidate: (counts[candidate], exact[candidate], candidate))
            if counts[donor] <= 1:
                raise ValueError("cannot retain all positive mixture components")
            counts[donor] -= 1
            counts[name] = 1
    remaining = total - sum(counts.values())
    order = sorted(positive, key=lambda name: (exact[name] - int(exact[name]), name), reverse=True)
    for index in range(remaining):
        counts[order[index % len(order)]] += 1
    return counts


def deterministic_select(files: Sequence[RemoteFile], count: int, *, seed: int, namespace: str) -> list[RemoteFile]:
    if count > len(files):
        raise ValueError(f"requested {count} files from {namespace}, but only {len(files)} are available")
    ranked = sorted(
        files,
        key=lambda item: hashlib.sha256(f"{seed}:{namespace}:{item.source_path}".encode()).hexdigest(),
    )
    return ranked[:count]


def _lfs_sha256(item: Mapping[str, Any]) -> str | None:
    lfs = item.get("lfs") or {}
    oid = str(lfs.get("oid", ""))
    return oid.removeprefix("sha256:") or None


def list_huggingface_files(repo_id: str, revision: str, prefix: str) -> list[RemoteFile]:
    encoded_prefix = urllib.parse.quote(prefix.strip("/"), safe="/")
    query = urllib.parse.urlencode({"recursive": "true", "expand": "true", "limit": 1000})
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{encoded_prefix}?{query}"
    with urllib.request.urlopen(url, timeout=120) as response:
        items = json.load(response)
    files = [
        RemoteFile(source_path=str(item["path"]), size=int(item.get("size", 0)), sha256=_lfs_sha256(item))
        for item in items
        if item.get("type") == "file"
    ]
    if not files:
        raise ValueError(f"Hugging Face prefix contains no files: {prefix}")
    return files


def build_selection(
    environments: Mapping[str, Any],
    *,
    seed: int,
    catalog: Callable[[str], Sequence[RemoteFile]],
) -> list[SelectedFile]:
    selected: list[SelectedFile] = []
    for environment, environment_cfg in environments.items():
        components = list(environment_cfg["components"])
        counts = allocate_component_counts(
            int(environment_cfg["source_files"]),
            {str(component["name"]): float(component["weight"]) for component in components},
        )
        for component in components:
            name = str(component["name"])
            source_prefix = str(component["source_prefix"]).strip("/")
            destination_prefix = str(component["destination_prefix"]).strip("/")
            chosen = deterministic_select(catalog(source_prefix), counts[name], seed=seed, namespace=source_prefix)
            for remote in chosen:
                selected.append(
                    SelectedFile(
                        environment=str(environment),
                        component=name,
                        source_path=remote.source_path,
                        destination_path=f"{destination_prefix}/{Path(remote.source_path).name}",
                        size=remote.size,
                        sha256=remote.sha256,
                    )
                )
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(
    item: SelectedFile,
    *,
    repo_id: str,
    revision: str,
    cache_root: Path,
    retries: int,
    verify_sha256: bool,
) -> Path:
    destination = cache_root / item.source_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != item.size:
            raise ValueError(f"existing file has the wrong size: {destination}")
        if verify_sha256 and item.sha256 and _sha256(destination) != item.sha256:
            raise ValueError(f"existing file has the wrong SHA-256: {destination}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and partial.stat().st_size > item.size:
        raise ValueError(f"partial file exceeds expected size: {partial}")
    encoded_path = urllib.parse.quote(item.source_path, safe="/")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded_path}"
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    if partial.stat().st_size != item.size:
        raise ValueError(f"downloaded file has the wrong size: {partial}")
    if verify_sha256 and item.sha256 and _sha256(partial) != item.sha256:
        raise ValueError(f"downloaded file has the wrong SHA-256: {partial}")
    os.replace(partial, destination)
    return destination


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def materialize_balanced_view(cfg: Mapping[str, Any]) -> dict[str, Any]:
    repo_id = str(cfg["repo_id"])
    revision = str(cfg["revision"])
    seed = int(cfg["seed"])
    cache_root = Path(str(cfg["cache_root"])).expanduser()
    view_root = Path(str(cfg["view_root"])).expanduser()
    environments = dict(cfg["environments"])
    catalog_cache: dict[str, Sequence[RemoteFile]] = {}

    def catalog(prefix: str) -> Sequence[RemoteFile]:
        if prefix not in catalog_cache:
            catalog_cache[prefix] = list_huggingface_files(repo_id, revision, prefix)
        return catalog_cache[prefix]

    selected = build_selection(environments, seed=seed, catalog=catalog)
    counts = {
        environment: sum(item.environment == environment for item in selected) for environment in environments
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"environment source-file counts are not balanced: {counts}")
    manifest_path = view_root / "balanced_dataset_manifest.json"
    payload: dict[str, Any] = {
        "format_version": 1,
        "status": "planned",
        "claim_bearing": False,
        "claim_blocker": "episode-boundary and shard-family audit pending",
        "repo_id": repo_id,
        "revision": revision,
        "seed": seed,
        "cache_root": str(cache_root),
        "view_root": str(view_root),
        "source_files_per_environment": counts,
        "max_rows_per_source": int(cfg["max_rows_per_source"]),
        "total_expected_bytes": sum(item.size for item in selected),
        "files": [asdict(item) for item in selected],
    }
    _write_manifest(manifest_path, payload)
    if bool(cfg.get("plan_only", False)):
        return payload

    payload["status"] = "downloading"
    _write_manifest(manifest_path, payload)
    with ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", 2))) as executor:
        cached_paths = list(
            executor.map(
                lambda item: _download_one(
                    item,
                    repo_id=repo_id,
                    revision=revision,
                    cache_root=cache_root,
                    retries=int(cfg.get("retries", 8)),
                    verify_sha256=bool(cfg.get("verify_sha256", True)),
                ),
                selected,
            )
        )
    for item, cached_path in zip(selected, cached_paths, strict=True):
        link = view_root / item.destination_path
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() == cached_path.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"balanced view path already exists with a different target: {link}")
        link.symlink_to(cached_path)
    payload["status"] = "materialized_pending_audit"
    _write_manifest(manifest_path, payload)
    return payload

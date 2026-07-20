"""Deterministic, resumable construction of balanced offline dataset views."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
    source_group: str
    source_path: str
    destination_path: str
    size: int
    sha256: str | None = None


def _path_aliases(path: Path) -> set[str]:
    aliases = {Path(os.path.abspath(path)).as_posix()}
    if path.exists():
        aliases.add(path.resolve().as_posix())
    return aliases


def audited_source_records(manifest_path: Path, source_paths: Mapping[int, str]) -> dict[int, dict[str, str]]:
    """Map native loader source ids to structurally audited dataset groups."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "audited_balanced_pending_provenance":
        raise ValueError(f"dataset manifest is not structurally audited: {manifest.get('status')}")
    if manifest.get("structural_balance_passed") is not True:
        raise ValueError("dataset manifest does not pass structural balance")
    audit_path = Path(str(manifest.get("audit_path", "")))
    if not audit_path.is_file():
        raise ValueError(f"dataset audit artifact is missing: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != manifest["status"] or audit.get("structural_errors"):
        raise ValueError("dataset audit artifact disagrees with the manifest")

    view_root = Path(str(manifest["view_root"]))
    by_alias: dict[str, dict[str, str]] = {}
    expected_groups = set()
    for item in manifest["files"]:
        record = {
            "environment": str(item["environment"]),
            "component": str(item["component"]),
            "source_group": str(item["source_group"]),
        }
        expected_groups.add(record["source_group"])
        for alias in _path_aliases(view_root / str(item["destination_path"])):
            if alias in by_alias and by_alias[alias] != record:
                raise ValueError(f"dataset source alias maps to conflicting groups: {alias}")
            by_alias[alias] = record

    records: dict[int, dict[str, str]] = {}
    for source_id, source_path in source_paths.items():
        matches = {tuple(sorted(by_alias[alias].items())) for alias in _path_aliases(Path(source_path)) if alias in by_alias}
        if len(matches) != 1:
            raise ValueError(f"loader source is absent or ambiguous in the audited manifest: {source_path}")
        records[int(source_id)] = dict(matches.pop())
    observed_groups = {record["source_group"] for record in records.values()}
    if observed_groups != expected_groups:
        missing = sorted(expected_groups - observed_groups)
        extra = sorted(observed_groups - expected_groups)
        raise ValueError(f"loader and audited manifest source groups differ: missing={missing}, extra={extra}")
    return records


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
    query = urllib.parse.urlencode({"recursive": "true", "expand": "false", "limit": 1000})
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
            int(environment_cfg["source_groups"]),
            {str(component["name"]): float(component["weight"]) for component in components},
        )
        for component in components:
            name = str(component["name"])
            source_prefix = str(component["source_prefix"]).strip("/")
            destination_prefix = str(component["destination_prefix"]).strip("/")
            available = list(catalog(source_prefix))
            group_pattern = component.get("group_pattern")
            representative_policy = str(component.get("representative_policy", "largest"))
            minimum_representative_bytes = int(component.get("minimum_representative_bytes", 0))
            if representative_policy not in {"largest", "smallest_above_minimum"}:
                raise ValueError(f"unknown representative policy: {representative_policy}")
            grouped: dict[str, list[RemoteFile]] = {}
            for remote in available:
                if group_pattern:
                    match = re.search(str(group_pattern), Path(remote.source_path).name)
                    if not match:
                        raise ValueError(f"group pattern does not match {remote.source_path}: {group_pattern}")
                    group = f"{source_prefix}/{match.group(1)}"
                else:
                    group = remote.source_path
                grouped.setdefault(group, []).append(remote)
            if representative_policy == "smallest_above_minimum":
                grouped = {
                    group: [member for member in members if member.size >= minimum_representative_bytes]
                    for group, members in grouped.items()
                    if any(member.size >= minimum_representative_bytes for member in members)
                }
            group_candidates = [RemoteFile(group, max(item.size for item in members)) for group, members in grouped.items()]
            chosen_groups = deterministic_select(
                group_candidates, counts[name], seed=seed, namespace=f"{source_prefix}:groups"
            )
            for chosen_group in chosen_groups:
                members = grouped[chosen_group.source_path]
                def rank(item: RemoteFile) -> tuple[int, str]:
                    return (
                        item.size,
                        hashlib.sha256(
                            f"{seed}:{chosen_group.source_path}:{item.source_path}".encode()
                        ).hexdigest(),
                    )

                remote = (
                    min(members, key=rank)
                    if representative_policy == "smallest_above_minimum"
                    else max(members, key=rank)
                )
                selected.append(
                    SelectedFile(
                        environment=str(environment),
                        component=name,
                        source_group=chosen_group.source_path,
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
        raise ValueError(f"environment source-group counts are not balanced: {counts}")
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
        "source_groups_per_environment": counts,
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


def shard_family(source_path: str) -> str:
    """Return a conservative candidate family for multipart source filenames."""

    stem = Path(source_path).stem
    match = re.match(r"^(chunk_\d+)_part_\d+$", stem)
    if match:
        return f"{Path(source_path).parent.as_posix()}/{match.group(1)}"
    return source_path


def audit_balanced_view(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Inspect row/terminal counts without treating candidate shard families as ground truth."""

    import pyarrow as pa
    import torch

    manifest = json.loads(manifest_path.read_text())
    view_root = Path(manifest["view_root"])
    cap = int(manifest["max_rows_per_source"])
    rows = []
    for item in manifest["files"]:
        path = view_root / item["destination_path"]
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".pt":
            tensors = torch.load(path, map_location="cpu", weights_only=True)
            count = int(tensors["actions"].shape[0])
            terminals = int(tensors["dones"].to(torch.bool).sum()) if "dones" in tensors else None
        elif suffix == ".arrow":
            with pa.memory_map(str(path)) as source:
                reader = pa.ipc.open_file(source)
                count = sum(reader.get_batch(index).num_rows for index in range(reader.num_record_batches))
            terminals = count // 256
        else:
            raise ValueError(f"unsupported balanced dataset file: {path}")
        rows.append(
            {
                **item,
                "rows": count,
                "terminal_count": terminals,
                "accepted_row_cap": min(count, cap),
                "meets_row_cap": count >= cap,
                "candidate_shard_family": shard_family(item["source_path"]),
            }
        )
    environment_summary = {}
    for environment in sorted({row["environment"] for row in rows}):
        selected = [row for row in rows if row["environment"] == environment]
        environment_summary[environment] = {
            "physical_files": len(selected),
            "source_groups": len(
                {row.get("source_group", row["candidate_shard_family"]) for row in selected}
            ),
            "candidate_shard_families": len({row["candidate_shard_family"] for row in selected}),
            "raw_rows": sum(row["rows"] for row in selected),
            "accepted_row_cap": sum(row["accepted_row_cap"] for row in selected),
            "files_below_cap": sum(not row["meets_row_cap"] for row in selected),
        }
    structural_errors = []
    accepted_counts = {summary["accepted_row_cap"] for summary in environment_summary.values()}
    for environment, summary in environment_summary.items():
        if summary["files_below_cap"]:
            structural_errors.append(f"{environment} has {summary['files_below_cap']} files below the row cap")
        if summary["source_groups"] != summary["physical_files"]:
            structural_errors.append(f"{environment} repeats a configured source group")
    if len(accepted_counts) != 1:
        structural_errors.append("accepted row budgets differ across environments")
    audit_status = "rejected_structural_imbalance" if structural_errors else "audited_balanced_pending_provenance"
    payload = {
        "format_version": 1,
        "manifest": str(manifest_path),
        "status": audit_status,
        "claim_bearing": False,
        "structural_errors": structural_errors,
        "blockers": [
            "candidate multipart shard families require provenance validation",
            "POGEMA terminal boundaries are synthesized at 256 rows by the native loader",
        ],
        "environment_summary": environment_summary,
        "files": rows,
    }
    _write_manifest(output_path, payload)
    manifest["status"] = audit_status
    manifest["audit_path"] = str(output_path)
    manifest["structural_balance_passed"] = not structural_errors
    _write_manifest(manifest_path, manifest)
    if structural_errors:
        raise ValueError("balanced dataset audit rejected: " + "; ".join(structural_errors))
    return payload

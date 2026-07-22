"""Sharded, token-level activation corpus for MARL-GPT cross-layer transcoders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from marl_gpt_interp.experiment_io import file_sha256


PATHS = ("shared", "actor", "critic")
TENSORS = (
    "shared_residual_inputs",
    "shared_mlp_outputs",
    "actor_residual_input",
    "actor_mlp_output",
    "critic_residual_input",
    "critic_mlp_output",
)


def grouped_split(
    groups: Sequence[str], *, seed: int = 0, fractions: Sequence[float] = (0.7, 0.15, 0.15)
) -> list[str]:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must contain train/validation/test values summing to one")
    unique = sorted(set(groups), key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    train_end = round(len(unique) * fractions[0])
    validation_end = train_end + round(len(unique) * fractions[1])
    assignments = {
        group: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, group in enumerate(unique)
    }
    return [assignments[group] for group in groups]


def stratified_grouped_split(
    groups: Sequence[str], strata: Sequence[str], *, seed: int = 0
) -> list[str]:
    if len(groups) != len(strata):
        raise ValueError("groups and strata must have the same length")
    output = [""] * len(groups)
    for stratum in sorted(set(strata)):
        indices = [index for index, value in enumerate(strata) if value == stratum]
        local = grouped_split([groups[index] for index in indices], seed=seed)
        for index, split in zip(indices, local, strict=True):
            output[index] = split
    return output


def validate_clt_tensors(tensors: Mapping[str, torch.Tensor], metadata: Sequence[Mapping[str, Any]]) -> None:
    missing = set(TENSORS) - tensors.keys()
    if missing:
        raise ValueError(f"CLT shard is missing tensors: {sorted(missing)}")
    rows = {int(tensors[name].shape[0]) for name in TENSORS}
    if len(rows) != 1 or rows.pop() != len(metadata):
        raise ValueError("all CLT tensors and metadata must have the same row count")
    shared_shape = tensors["shared_residual_inputs"].shape
    if len(shared_shape) != 3 or tensors["shared_mlp_outputs"].shape != shared_shape:
        raise ValueError("shared tensors must have shape [tokens, layers, d_model]")
    for branch in ("actor", "critic"):
        residual = tensors[f"{branch}_residual_input"]
        output = tensors[f"{branch}_mlp_output"]
        if residual.shape != output.shape or residual.shape != (shared_shape[0], shared_shape[2]):
            raise ValueError(f"{branch} tensors must have shape [tokens, d_model]")
    required = {
        "environment",
        "split_group",
        "source_file_id",
        "source_row_index",
        "sample_index",
        "token_index",
        "is_output_token",
        "split",
    }
    for index, row in enumerate(metadata):
        absent = required - row.keys()
        if absent:
            raise ValueError(f"metadata row {index} is missing fields: {sorted(absent)}")


class CLTCorpusWriter:
    """Write bounded tensor shards without assembling the full corpus in RAM."""

    def __init__(self, directory: Path, *, rows_per_shard: int, manifest: Mapping[str, Any]) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.base_manifest = dict(manifest)
        self.tensor_parts: dict[str, list[torch.Tensor]] = {name: [] for name in TENSORS}
        self.metadata: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.rows = 0

    def add(self, tensors: Mapping[str, torch.Tensor], metadata: Sequence[Mapping[str, Any]]) -> None:
        validate_clt_tensors(tensors, metadata)
        offset = 0
        while offset < len(metadata):
            capacity = self.rows_per_shard - len(self.metadata)
            count = min(capacity, len(metadata) - offset)
            for name in TENSORS:
                self.tensor_parts[name].append(tensors[name][offset : offset + count].detach().cpu())
            self.metadata.extend(dict(row) for row in metadata[offset : offset + count])
            offset += count
            if len(self.metadata) == self.rows_per_shard:
                self._flush()

    def _flush(self) -> None:
        if not self.metadata:
            return
        index = len(self.shards)
        tensor_path = self.directory / f"tensors-{index:05d}.pt"
        metadata_path = self.directory / f"metadata-{index:05d}.jsonl"
        payload = {name: torch.cat(parts) for name, parts in self.tensor_parts.items()}
        torch.save(payload, tensor_path)
        with metadata_path.open("w") as handle:
            for row in self.metadata:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        row_count = len(self.metadata)
        self.shards.append(
            {
                "tensors": tensor_path.name,
                "tensors_sha256": file_sha256(tensor_path),
                "metadata": metadata_path.name,
                "metadata_sha256": file_sha256(metadata_path),
                "rows": row_count,
            }
        )
        self.rows += row_count
        self.tensor_parts = {name: [] for name in TENSORS}
        self.metadata = []

    def close(self) -> dict[str, Any]:
        self._flush()
        manifest = {
            **self.base_manifest,
            "format": "marl-gpt-clt-corpus",
            "format_version": 1,
            "rows": self.rows,
            "shards": self.shards,
            "tensor_schema": {
                "shared_residual_inputs": "[tokens, shared_layers, d_model]",
                "shared_mlp_outputs": "[tokens, shared_layers, d_model]",
                "actor_residual_input": "[tokens, d_model]",
                "actor_mlp_output": "[tokens, d_model]",
                "critic_residual_input": "[tokens, d_model]",
                "critic_mlp_output": "[tokens, d_model]",
            },
        }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def load_corpus_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "marl-gpt-clt-corpus" or manifest.get("format_version") != 1:
        raise ValueError("unsupported CLT corpus format")
    return manifest


def iter_clt_shards(
    directory: Path, *, split: str | None = None, verify_hashes: bool = True
) -> Iterator[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]]:
    manifest = load_corpus_manifest(directory)
    for shard in manifest["shards"]:
        tensor_path = directory / shard["tensors"]
        metadata_path = directory / shard["metadata"]
        if verify_hashes:
            if file_sha256(tensor_path) != shard["tensors_sha256"]:
                raise ValueError(f"tensor hash mismatch: {tensor_path}")
            if file_sha256(metadata_path) != shard["metadata_sha256"]:
                raise ValueError(f"metadata hash mismatch: {metadata_path}")
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line]
        validate_clt_tensors(tensors, metadata)
        if split is not None:
            retained = [index for index, row in enumerate(metadata) if row["split"] == split]
            if not retained:
                continue
            indices = torch.tensor(retained, dtype=torch.long)
            tensors = {name: value[indices] for name, value in tensors.items()}
            metadata = [metadata[index] for index in retained]
        yield tensors, metadata


def branch_batch(tensors: Mapping[str, torch.Tensor], branch: str) -> tuple[torch.Tensor, torch.Tensor]:
    if branch not in {"actor", "critic"}:
        raise ValueError("branch must be actor or critic")
    residuals = torch.cat(
        [tensors["shared_residual_inputs"], tensors[f"{branch}_residual_input"].unsqueeze(1)], dim=1
    )
    outputs = torch.cat(
        [tensors["shared_mlp_outputs"], tensors[f"{branch}_mlp_output"].unsqueeze(1)], dim=1
    )
    return residuals, outputs

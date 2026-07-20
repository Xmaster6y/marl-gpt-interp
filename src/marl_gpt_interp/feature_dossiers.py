"""Safe source-row rehydration for sparse-feature interpretation dossiers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def summarize_tensor_row(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu()
    flat = value.reshape(-1)
    summary: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if flat.numel() <= 32:
        summary["values"] = flat.tolist()
        return summary
    numeric = flat.float()
    finite = torch.isfinite(numeric)
    finite_values = numeric[finite]
    summary["finite_fraction"] = float(finite.float().mean())
    summary["nonzero_fraction"] = float((numeric != 0).float().mean())
    if len(finite_values):
        summary.update(
            {
                "minimum": float(finite_values.min()),
                "maximum": float(finite_values.max()),
                "mean": float(finite_values.mean()),
                "standard_deviation": float(finite_values.std(unbiased=False)),
                "l2_norm": float(torch.linalg.vector_norm(finite_values)),
            }
        )
    return summary


def rehydrate_references(
    references: Iterable[dict[str, Any]],
    source_files: dict[int, str],
    *,
    history_length: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Load each tensor-only source once and summarize referenced rows."""

    by_source: dict[int, set[int]] = defaultdict(set)
    for reference in references:
        source_id = reference.get("source_file_id")
        row_index = reference.get("source_row_index")
        if source_id is not None and row_index is not None:
            by_source[int(source_id)].add(int(row_index))
    hydrated = {}
    for source_id, row_indices in by_source.items():
        if source_id not in source_files:
            raise ValueError(f"Feature example references unknown source_file_id {source_id}")
        path = Path(source_files[source_id])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected tensor dictionary in {path}")
        for row_index in sorted(row_indices):
            fields = {}
            for key, tensor in sorted(payload.items()):
                if isinstance(tensor, torch.Tensor) and tensor.ndim and row_index < len(tensor):
                    fields[key] = summarize_tensor_row(tensor[row_index])
            hydrated[(source_id, row_index)] = {
                "source_file": str(path),
                "source_row_index": row_index,
                "history_row_start": max(0, row_index - history_length + 1),
                "history_row_end": row_index,
                "fields": fields,
            }
        del payload
    return hydrated

"""Provider-neutral soccer data loading and normalization helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


EVENT_FIELDS = [
    "dataset",
    "match_id",
    "possession_id",
    "team_id",
    "player_id",
    "timestamp",
    "event_type",
    "outcome",
    "x",
    "y",
    "end_x",
    "end_y",
]

TRACKING_FIELDS = [
    "dataset",
    "match_id",
    "frame_id",
    "timestamp",
    "team_id",
    "player_id",
    "x",
    "y",
    "vx",
    "vy",
    "is_ball",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV, JSON, or JSONL rows from path."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload["rows"]
    raise ValueError(f"Unsupported data file format: {path}")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "ball"}


def normalize_value(value: Any, *, field: str) -> Any:
    if field in {"x", "y", "end_x", "end_y", "timestamp", "vx", "vy"}:
        return to_float(value)
    if field == "is_ball":
        return to_bool(value)
    return None if value == "" else value


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    kind: str,
    column_map: Mapping[str, str],
    constants: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map provider-specific rows into the common event or tracking schema."""

    fields = EVENT_FIELDS if kind == "events" else TRACKING_FIELDS
    constants = constants or {}
    normalized = []
    for source in rows:
        row = {field: None for field in fields}
        row["dataset"] = dataset
        for target, source_key in column_map.items():
            if target not in row:
                continue
            row[target] = normalize_value(source.get(source_key), field=target)
        for target, value in constants.items():
            if target in row:
                row[target] = normalize_value(value, field=target)
        normalized.append(row)
    return normalized

"""Comparable soccer statistics over normalized event and tracking rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import sqrt
from typing import Any

from marl_gpt_interp.soccer_schema import to_float

PASS_TYPES = {"pass", "short_pass", "long_pass", "cross"}
SHOT_TYPES = {"shot", "goal", "penalty_shot"}
SUCCESS_OUTCOMES = {"1", "true", "success", "successful", "complete", "completed", "goal"}


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _event_type(row: Mapping[str, Any]) -> str:
    return str(row.get("event_type") or "").strip().lower()


def _is_success(row: Mapping[str, Any]) -> bool:
    return str(row.get("outcome") or "").strip().lower() in SUCCESS_OUTCOMES


def _point(row: Mapping[str, Any], x_key: str = "x", y_key: str = "y") -> tuple[float, float] | None:
    x = to_float(row.get(x_key))
    y = to_float(row.get(y_key))
    if x is None or y is None:
        return None
    return x, y


def _mean(values: Iterable[float]) -> float | None:
    present = list(values)
    if not present:
        return None
    return sum(present) / len(present)


def event_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize normalized event rows for one dataset."""

    passes = [row for row in rows if _event_type(row) in PASS_TYPES]
    shots = [row for row in rows if _event_type(row) in SHOT_TYPES]
    carries = [row for row in rows if _event_type(row) == "carry"]
    event_counts = Counter(_event_type(row) for row in rows if _event_type(row))

    pass_lengths = []
    for row in passes:
        start = _point(row)
        end = _point(row, "end_x", "end_y")
        if start is not None and end is not None:
            pass_lengths.append(_distance(start, end))

    return {
        "events": len(rows),
        "passes": len(passes),
        "completed_passes": sum(1 for row in passes if _is_success(row)),
        "pass_completion_rate": sum(1 for row in passes if _is_success(row)) / len(passes) if passes else None,
        "mean_pass_length": _mean(pass_lengths),
        "shots": len(shots),
        "goals": event_counts["goal"] + sum(1 for row in shots if _is_success(row) and _event_type(row) != "goal"),
        "carries": len(carries),
        "event_type_counts": dict(sorted(event_counts.items())),
    }


def _tracking_frames(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, Any], list[Mapping[str, Any]]]:
    frames: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        frames[(row.get("match_id"), row.get("frame_id"))].append(row)
    return frames


def _team_points(frame_rows: Sequence[Mapping[str, Any]]) -> dict[Any, list[tuple[float, float]]]:
    teams: dict[Any, list[tuple[float, float]]] = defaultdict(list)
    for row in frame_rows:
        if row.get("is_ball"):
            continue
        point = _point(row)
        team_id = row.get("team_id")
        if point is not None and team_id not in (None, ""):
            teams[team_id].append(point)
    return teams


def pitch_control_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    grid_x: int = 12,
    grid_y: int = 8,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> dict[str, Any]:
    """Approximate pitch control as nearest-player ownership of grid cells."""

    frames = _tracking_frames(rows)
    team_cell_counts: Counter[Any] = Counter()
    contested_cells = 0
    total_cells = 0
    frame_team_counts: Counter[Any] = Counter()
    frame_count = 0

    for frame_rows in frames.values():
        teams = _team_points(frame_rows)
        if len(teams) < 2:
            continue
        frame_count += 1
        for team_id in teams:
            frame_team_counts[team_id] += 1
        for ix in range(grid_x):
            x = pitch_length * (ix + 0.5) / grid_x
            for iy in range(grid_y):
                y = pitch_width * (iy + 0.5) / grid_y
                nearest = sorted(
                    (min(_distance((x, y), point) for point in points), team_id)
                    for team_id, points in teams.items()
                    if points
                )
                if len(nearest) < 2:
                    continue
                total_cells += 1
                if abs(nearest[0][0] - nearest[1][0]) < 1e-6:
                    contested_cells += 1
                else:
                    team_cell_counts[nearest[0][1]] += 1

    stats: dict[str, Any] = {
        "tracking_frames": frame_count,
        "pitch_control_grid_cells": total_cells,
        "contested_pitch_control_fraction": contested_cells / total_cells if total_cells else None,
    }
    for team_id in sorted(frame_team_counts, key=str):
        controlled = team_cell_counts[team_id]
        stats[f"pitch_control_fraction_team_{team_id}"] = controlled / total_cells if total_cells else None
    return stats


def compare_datasets(
    *,
    event_rows: Sequence[Mapping[str, Any]] = (),
    tracking_rows: Sequence[Mapping[str, Any]] = (),
    grid_x: int = 12,
    grid_y: int = 8,
) -> list[dict[str, Any]]:
    """Return one flat statistics row per dataset."""

    datasets = sorted(
        {str(row.get("dataset")) for row in [*event_rows, *tracking_rows] if row.get("dataset") not in (None, "")}
    )
    comparison = []
    for dataset in datasets:
        dataset_events = [row for row in event_rows if row.get("dataset") == dataset]
        dataset_tracking = [row for row in tracking_rows if row.get("dataset") == dataset]
        row = {"dataset": dataset}
        row.update(event_statistics(dataset_events))
        row.update(pitch_control_statistics(dataset_tracking, grid_x=grid_x, grid_y=grid_y))
        comparison.append(row)
    return comparison

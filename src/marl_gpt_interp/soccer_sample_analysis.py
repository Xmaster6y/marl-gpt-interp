"""Diagnostics for the bounded Fuji human and RoboCup soccer samples."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Any


def _range(values) -> dict[str, list[float]]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return {"min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist()}


def _velocity_diagnostics(positions, velocities, dt: float) -> dict[str, Any]:
    import numpy as np

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    observed = np.diff(positions, axis=0) / dt
    predicted = velocities[:-1]
    residual = observed - predicted
    correlations = []
    for axis in range(observed.shape[1]):
        if np.std(observed[:, axis]) == 0 or np.std(predicted[:, axis]) == 0:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(observed[:, axis], predicted[:, axis])[0, 1]))
    return {
        "pairs": int(len(observed)),
        "rmse": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "median_absolute_error": np.median(np.abs(residual), axis=0).tolist(),
        "correlation": correlations,
    }


def analyze_laliga_arrow(path: Path) -> dict[str, Any]:
    """Analyze a nested Hugging Face Arrow sample without exposing player names."""

    import numpy as np
    import pyarrow.ipc as ipc

    table = ipc.open_stream(path).read_all()
    rows = table.to_pylist()
    player_positions = []
    player_velocities = []
    ball_positions = []
    ball_velocities = []
    player_actions: Counter[str] = Counter()
    sequence_summaries = []
    player_velocity_residuals = []
    ball_velocity_residuals = []

    for row in rows:
        events = row["events"]
        start = float(row["sequence_start_frame"])
        end = float(row["sequence_end_frame"])
        dt = (end - start) / (len(events) - 1) if len(events) > 1 else None
        sequence_summaries.append(
            {
                "game_id": row["game_id"],
                "half": row["half"],
                "sequence_id": int(row["sequence_id"]),
                "start": start,
                "end": end,
                "events": len(events),
                "implied_dt": dt,
            }
        )

        player_tracks: dict[Any, tuple[list[list[float]], list[list[float]]]] = {}
        sequence_ball_positions = []
        sequence_ball_velocities = []
        for event in events:
            raw = event["state"]["raw_state"]
            ball = raw["ball"]
            ball_position = [ball["position"]["x"], ball["position"]["y"]]
            ball_velocity = [ball["velocity"]["x"], ball["velocity"]["y"]]
            ball_positions.append(ball_position)
            ball_velocities.append(ball_velocity)
            sequence_ball_positions.append(ball_position)
            sequence_ball_velocities.append(ball_velocity)
            for player in raw["players"]:
                position = [player["position"]["x"], player["position"]["y"]]
                velocity = [player["velocity"]["x"], player["velocity"]["y"]]
                player_positions.append(position)
                player_velocities.append(velocity)
                player_actions[str(player["action"])] += 1
                track = player_tracks.setdefault(player["player_id"], ([], []))
                track[0].append(position)
                track[1].append(velocity)

        if dt is not None and dt > 0:
            ball_diag = _velocity_diagnostics(sequence_ball_positions, sequence_ball_velocities, dt)
            ball_velocity_residuals.append(ball_diag)
            for positions, velocities in player_tracks.values():
                if len(positions) == len(events):
                    player_velocity_residuals.append(_velocity_diagnostics(positions, velocities, dt))

    def aggregate_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {}
        return {
            "tracks": len(items),
            "pairs": sum(item["pairs"] for item in items),
            "mean_rmse": np.mean([item["rmse"] for item in items], axis=0).tolist(),
            "median_absolute_error": np.median([item["median_absolute_error"] for item in items], axis=0).tolist(),
            "mean_correlation": np.nanmean(
                [[np.nan if value is None else value for value in item["correlation"]] for item in items], axis=0
            ).tolist(),
        }

    event_count = sum(len(row["events"]) for row in rows)
    player_counts = [len(event["state"]["raw_state"]["players"]) for row in rows for event in row["events"]]
    implied_dt = [item["implied_dt"] for item in sequence_summaries if item["implied_dt"] is not None]
    return {
        "path": str(path),
        "rows": table.num_rows,
        "columns": table.column_names,
        "sequences": sequence_summaries,
        "events": event_count,
        "players_per_event": {"min": min(player_counts), "max": max(player_counts)},
        "player_state_rows": len(player_positions),
        "implied_sampling": {
            "dt_min": min(implied_dt),
            "dt_max": max(implied_dt),
            "dt_mean": float(np.mean(implied_dt)),
            "hz_mean": float(1.0 / np.mean(implied_dt)),
        },
        "player_position_range": _range(player_positions),
        "player_velocity_range": _range(player_velocities),
        "ball_position_range": _range(ball_positions),
        "ball_velocity_range": _range(ball_velocities),
        "player_actions": dict(sorted(player_actions.items())),
        "player_velocity_consistency": aggregate_diagnostics(player_velocity_residuals),
        "ball_velocity_consistency": aggregate_diagnostics(ball_velocity_residuals),
    }


def _physical_mask(states):
    import numpy as np

    states = np.asarray(states)
    return (
        (np.abs(states[..., 0]) <= 53)
        & (np.abs(states[..., 1]) <= 35)
        & (np.abs(states[..., 2]) <= 10)
        & (np.abs(states[..., 3]) <= 10)
    )


def _robocup_ball_diagnostics(states) -> dict[str, Any]:
    import numpy as np

    positions = states[:, :2].astype(float)
    velocities = states[:, 2:4].astype(float)
    displacement = np.diff(positions, axis=0)
    predicted = velocities[:-1]
    residual = displacement - predicted
    correlations = []
    for axis in range(2):
        if np.std(displacement[:, axis]) == 0 or np.std(predicted[:, axis]) == 0:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(displacement[:, axis], predicted[:, axis])[0, 1]))
    return {
        "position_range": _range(positions),
        "velocity_range": _range(velocities),
        "per_cycle_rmse": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "per_cycle_correlation": correlations,
    }


def analyze_robocup_pair(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Test documented and grouped interpretations of one 92-value RoboCup pair."""

    import numpy as np

    left = np.load(left_path, mmap_mode="r", allow_pickle=False)
    right = np.load(right_path, mmap_mode="r", allow_pickle=False)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 92:
        raise ValueError(f"Expected equal (timesteps, 92) arrays, got {left.shape} and {right.shape}")

    timesteps = left.shape[0]
    interleaved = left[:, 4:].reshape(timesteps, 22, 4)
    grouped = np.concatenate(
        [
            left[:, 4:48].reshape(timesteps, 4, 11).transpose(0, 2, 1),
            left[:, 48:92].reshape(timesteps, 4, 11).transpose(0, 2, 1),
        ],
        axis=1,
    )
    interleaved_mask = _physical_mask(interleaved)
    grouped_mask = _physical_mask(grouped)

    return {
        "left_path": str(left_path),
        "right_path": str(right_path),
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
        "ball_blocks_equal": bool(np.array_equal(left[:, :4], right[:, :4])),
        "team_blocks_swap": bool(
            np.array_equal(left[:, 4:48], right[:, 48:92]) and np.array_equal(left[:, 48:92], right[:, 4:48])
        ),
        "exact_8000_values": int(np.count_nonzero(left[:, 4:] == 8000)),
        "interleaved_physical_fraction": float(interleaved_mask.mean()),
        "interleaved_physical_players_per_timestep": {
            "min": int(interleaved_mask.sum(axis=1).min()),
            "max": int(interleaved_mask.sum(axis=1).max()),
            "complete_timesteps": int(np.count_nonzero(interleaved_mask.sum(axis=1) == 22)),
        },
        "grouped_physical_fraction": float(grouped_mask.mean()),
        "grouped_physical_players_per_timestep": {
            "min": int(grouped_mask.sum(axis=1).min()),
            "max": int(grouped_mask.sum(axis=1).max()),
            "complete_timesteps": int(np.count_nonzero(grouped_mask.sum(axis=1) == 22)),
        },
        "ball": _robocup_ball_diagnostics(left[:, :4]),
    }


def analyze_stp_tracking_csv(path: Path, derived_left_path: Path | None = None) -> dict[str, Any]:
    """Validate STP named physical columns and audit a derived 92-column array."""

    import numpy as np

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []

    agents = ["b", *(f"l{i}" for i in range(1, 12)), *(f"r{i}" for i in range(1, 12))]
    features = ["x", "y", "vx", "vy"]
    physical_columns = [f"{agent}_{feature}" for agent in agents for feature in features]
    missing = [column for column in physical_columns if column not in columns]
    if missing:
        raise ValueError(f"Missing STP physical columns: {missing}")

    physical = np.asarray([[float(row[column]) for column in physical_columns] for row in rows], dtype=np.float32)
    result: dict[str, Any] = {
        "path": str(path),
        "rows": len(rows),
        "columns": len(columns),
        "physical_columns": len(physical_columns),
        "all_physical_values_finite": bool(np.isfinite(physical).all()),
        "physical_feature_range": _range(physical.reshape(-1, 4)),
    }

    if derived_left_path is not None:
        derived = np.load(derived_left_path, mmap_mode="r", allow_pickle=False)[: len(rows)]
        if derived.shape != physical.shape:
            raise ValueError(f"Expected matching sample shapes, got {derived.shape} and {physical.shape}")

        # The sampled derived arrays match an obsolete eight-column player stride exactly:
        # four ball fields, then four-column windows advanced by eight fields for 22 players.
        # Current tracking CSVs contain nine fields per player, including ``vwidth``.
        obsolete_indices = [*range(10, 14)]
        for player_index in range(22):
            start = 15 + 8 * player_index
            obsolete_indices.extend(range(start, start + 4))
        obsolete_columns = [columns[index] for index in obsolete_indices]
        obsolete_selection = np.asarray(
            [[float(row[column]) for column in obsolete_columns] for row in rows], dtype=np.float32
        )
        nonphysical = [column for column in obsolete_columns[4:] if column.rsplit("_", maxsplit=1)[-1] not in features]
        result["derived_audit"] = {
            "path": str(derived_left_path),
            "matches_named_physical_state": bool(np.array_equal(derived, physical)),
            "matches_obsolete_eight_column_stride": bool(np.array_equal(derived, obsolete_selection)),
            "assumed_player_stride": 8,
            "actual_player_stride": 9,
            "selected_nonphysical_columns": len(nonphysical),
            "selected_nonphysical_suffix_counts": dict(
                sorted(Counter(column.rsplit("_", maxsplit=1)[-1] for column in nonphysical).items())
            ),
            "first_derived_columns": obsolete_columns[:16],
            "last_derived_columns": obsolete_columns[-8:],
        }
    return result

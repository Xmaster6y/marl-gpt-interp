"""Small, dependency-light statistics helpers for GRF raw observations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log2, sqrt
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point2(value: Any) -> tuple[float, float] | None:
    try:
        if value is None or len(value) < 2:
            return None
    except TypeError:
        return None
    x = _as_float(value[0])
    y = _as_float(value[1])
    if x is None or y is None:
        return None
    return x, y


def _team_points(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    points = []
    for item in value:
        point = _point2(item)
        if point is not None:
            points.append(point)
    return points


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _extent(points: Sequence[tuple[float, float]], axis: int) -> float | None:
    if not points:
        return None
    values = [point[axis] for point in points]
    return max(values) - min(values)


def _mean_pairwise_distance(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    total = 0.0
    count = 0
    for i, first in enumerate(points):
        for second in points[i + 1 :]:
            total += _distance(first, second)
            count += 1
    return total / count if count else None


def _nearest_distance(point: tuple[float, float] | None, points: Sequence[tuple[float, float]]) -> float | None:
    if point is None or not points:
        return None
    return min(_distance(point, candidate) for candidate in points)


def _owned_player_position(
    observation: dict[str, Any],
    left_team: Sequence[tuple[float, float]],
    right_team: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    owned_team = observation.get("ball_owned_team")
    owned_player = observation.get("ball_owned_player")
    if owned_team not in (0, 1) or owned_player is None or owned_player < 0:
        return None
    owned_player = int(owned_player)
    team = left_team if owned_team == 0 else right_team
    if owned_player >= len(team):
        return None
    return team[owned_player]


def _get_index(value: Any, index: int) -> Any:
    try:
        if value is not None and len(value) > index:
            return value[index]
    except TypeError:
        return None
    return None


def extract_raw_observation_stats(raw_observations: Any) -> dict[str, float | int | None]:
    """Extract simple football statistics from one GRF raw observation list.

    GRF returns one raw dict per controlled player. Most global fields are repeated, so
    this function reads the first dict and tolerates missing fields by returning None.
    """

    if not raw_observations:
        return {}
    observation = raw_observations[0] if isinstance(raw_observations, list) else raw_observations
    if not isinstance(observation, dict):
        return {}

    left_team = _team_points(observation.get("left_team"))
    right_team = _team_points(observation.get("right_team"))
    ball = _point2(observation.get("ball"))
    owned_team = observation.get("ball_owned_team")
    carrier = _owned_player_position(observation, left_team, right_team) or ball
    defenders = right_team if owned_team == 0 else left_team if owned_team == 1 else right_team

    score = observation.get("score")
    left_score = _get_index(score, 0)
    right_score = _get_index(score, 1)
    goal_difference = None
    if left_score is not None and right_score is not None:
        goal_difference = left_score - right_score

    return {
        "ball_owned_team": owned_team if owned_team in (-1, 0, 1) else None,
        "left_score": left_score,
        "right_score": right_score,
        "goal_difference": goal_difference,
        "game_mode": observation.get("game_mode"),
        "distance_to_goal": _distance(ball, (1.0, 0.0)) if ball is not None else None,
        "nearest_defender_distance": _nearest_distance(carrier, defenders),
        "left_team_width": _extent(left_team, axis=1),
        "left_team_depth": _extent(left_team, axis=0),
        "right_team_width": _extent(right_team, axis=1),
        "right_team_depth": _extent(right_team, axis=0),
        "defensive_compactness": _mean_pairwise_distance(defenders),
    }


def count_actions(actions: Iterable[int], action_size: int) -> list[int]:
    counts = [0 for _ in range(action_size)]
    for action in actions:
        if 0 <= action < action_size:
            counts[action] += 1
    return counts


def action_entropy(action_counts: Sequence[int]) -> float:
    total = sum(action_counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in action_counts:
        if count:
            probability = count / total
            entropy -= probability * log2(probability)
    return entropy


def average_present(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)

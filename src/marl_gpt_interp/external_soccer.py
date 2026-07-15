"""Reusable external-soccer adapters and GRF simple115v2 encoding."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


GRF_OBSERVATION_SIZE = 115
GRF_ACTIONS = 19
GRF_GAME_MODES = 7


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    team_id: str
    index: int
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class BallState:
    x: float
    y: float
    vx: float
    vy: float
    z: float = 0.0
    vz: float = 0.0


@dataclass(frozen=True)
class SoccerFrame:
    source: str
    match_id: str
    sequence_id: str
    step_index: int
    frame_id: str
    timestamp: float | None
    teams: tuple[str, str]
    positive_x_team: str
    players: tuple[PlayerState, ...]
    ball: BallState
    possession_team: str | None
    game_mode: int
    imputed_fields: frozenset[str] = field(default_factory=frozenset)

    def validate(self) -> None:
        if len(self.teams) != 2 or len(set(self.teams)) != 2:
            raise ValueError(f"Expected two distinct teams, got {self.teams}")
        if self.positive_x_team not in self.teams:
            raise ValueError(f"positive_x_team {self.positive_x_team!r} is not in {self.teams}")
        if self.possession_team is not None and self.possession_team not in self.teams:
            raise ValueError(f"possession_team {self.possession_team!r} is not in {self.teams}")
        if not 0 <= self.game_mode < GRF_GAME_MODES:
            raise ValueError(f"GRF game mode must be in [0, 6], got {self.game_mode}")
        by_team = Counter(player.team_id for player in self.players)
        expected = {team: 11 for team in self.teams}
        if by_team != expected:
            raise ValueError(f"Expected 11 players per team, got {dict(by_team)}")
        for team in self.teams:
            indices = sorted(player.index for player in self.players if player.team_id == team)
            if indices != list(range(11)):
                raise ValueError(f"Expected stable indices 0..10 for {team!r}, got {indices}")
        values = [
            *(value for player in self.players for value in (player.x, player.y, player.vx, player.vy)),
            self.ball.x,
            self.ball.y,
            self.ball.vx,
            self.ball.vy,
            self.ball.z,
            self.ball.vz,
        ]
        if not np.isfinite(values).all():
            raise ValueError(f"Frame {self.frame_id!r} contains non-finite physical values")


@dataclass(frozen=True)
class GRFEncodingConfig:
    source_x_half: float = 52.5
    source_y_half: float = 34.0
    target_y_half: float = 1.0 / 2.25
    direction_mode: Literal["finite_difference", "stored_velocity"] = "finite_difference"
    velocity_dt: float = 0.1
    history_len: int = 6
    history_step: int = 1
    block_size: int = 700
    action_size: int = 20
    environment_id: int = 3
    environment_attr: int = 128

    def validate(self) -> None:
        if self.source_x_half <= 0 or self.source_y_half <= 0 or self.target_y_half <= 0:
            raise ValueError("Pitch and target half-extents must be positive")
        if self.history_len <= 0 or self.history_step <= 0:
            raise ValueError("history_len and history_step must be positive")
        token_count = self.history_len * GRF_OBSERVATION_SIZE
        if self.block_size < token_count + 1:
            raise ValueError(f"block_size {self.block_size} cannot hold {token_count} observation tokens and env token")
        if self.action_size < GRF_ACTIONS:
            raise ValueError(f"action_size must be at least {GRF_ACTIONS}")


@dataclass(frozen=True)
class EncodedObservation:
    vector: np.ndarray
    source: str
    match_id: str
    sequence_id: str
    step_index: int
    frame_id: str
    focal_team: str
    focal_player: str
    focal_index: int
    rotated: bool
    imputed_fields: frozenset[str]

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return self.source, self.match_id, self.sequence_id, self.focal_team, self.focal_player


@dataclass(frozen=True)
class EncodedHistory:
    observations: np.ndarray
    current: EncodedObservation


@dataclass(frozen=True)
class ModelInputBatch:
    arrays: dict[str, np.ndarray]
    histories: tuple[EncodedHistory, ...]


def _iter_laliga_sequences(path: Path) -> Iterator[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".arrow":
        import pyarrow.ipc as ipc

        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            for batch in reader:
                yield from batch.to_pylist()
        return
    if suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    raise ValueError(f"LaLiga input must be Arrow or JSONL, got {path}")


def iter_laliga_frames(path: Path, *, max_sequences: int = 0, max_frames: int = 0) -> Iterator[SoccerFrame]:
    """Stream complete LaLiga event states without retaining player names."""

    emitted = 0
    for sequence_number, row in enumerate(_iter_laliga_sequences(path)):
        if max_sequences > 0 and sequence_number >= max_sequences:
            break
        events = row.get("events") or []
        if not events:
            continue
        attack_team = str(row["team_name_attack"])
        defense_team = str(row["team_name_defense"])
        start = float(row["sequence_start_frame"])
        end = float(row["sequence_end_frame"])
        dt = (end - start) / (len(events) - 1) if len(events) > 1 else None
        sequence_id = str(row["sequence_id"])
        match_id = str(row["game_id"])
        half = str(row.get("half", ""))
        for event_index, event in enumerate(events):
            if max_frames > 0 and emitted >= max_frames:
                return
            raw = event["state"]["raw_state"]
            players: list[PlayerState] = []
            for team_id, key in ((attack_team, "attack_players"), (defense_team, "defense_players")):
                source_players = sorted(raw[key], key=lambda player: int(player["index"]))
                if len(source_players) != 11:
                    raise ValueError(f"Expected 11 {key}, got {len(source_players)}")
                for index, player in enumerate(source_players):
                    position = player["position"]
                    velocity = player["velocity"]
                    players.append(
                        PlayerState(
                            player_id=str(player["player_id"]),
                            team_id=team_id,
                            index=index,
                            x=float(position["x"]),
                            y=float(position["y"]),
                            vx=float(velocity["x"]),
                            vy=float(velocity["y"]),
                        )
                    )
            ball = raw["ball"]
            timestamp = start + event_index * dt if dt is not None else None
            frame = SoccerFrame(
                source="laliga",
                match_id=match_id,
                sequence_id=f"{half}:{sequence_id}",
                step_index=event_index,
                frame_id=f"{match_id}:{half}:{sequence_id}:{event_index}",
                timestamp=timestamp,
                teams=(attack_team, defense_team),
                positive_x_team=attack_team,
                players=tuple(players),
                ball=BallState(
                    x=float(ball["position"]["x"]),
                    y=float(ball["position"]["y"]),
                    vx=float(ball["velocity"]["x"]),
                    vy=float(ball["velocity"]["y"]),
                ),
                possession_team=attack_team,
                game_mode=0,
                imputed_fields=frozenset({"ball_z", "ball_vz", "game_mode", "possession_team"}),
            )
            frame.validate()
            yield frame
            emitted += 1


def robocup_playmode_to_grf(playmode: str) -> tuple[int, bool]:
    mode = playmode.strip().lower()
    if mode == "play_on":
        return 0, False
    prefixes = (
        (("before_kick_off", "kick_off", "after_goal"), 1),
        (("goal_kick",), 2),
        (("free_kick", "foul", "back_pass", "indirect_free_kick"), 3),
        (("corner_kick",), 4),
        (("kick_in",), 5),
        (("penalty",), 6),
    )
    for names, grf_mode in prefixes:
        if mode.startswith(names):
            return grf_mode, False
    return 0, True


def iter_robocup_frames(path: Path, *, max_frames: int = 0) -> Iterator[SoccerFrame]:
    """Stream named physical columns from a raw STP tracking CSV."""

    if path.suffix.lower() == ".npy":
        raise ValueError("Derived RoboCup .npy states are unsupported; use the raw named .tracking.csv columns")
    if not path.name.endswith(".tracking.csv"):
        raise ValueError(f"RoboCup input must end in .tracking.csv, got {path}")

    required = [
        "cycle",
        "playmode",
        "l_name",
        "r_name",
        *(f"{agent}_{feature}" for agent in ("b", *(f"l{i}" for i in range(1, 12)), *(f"r{i}" for i in range(1, 12))) for feature in ("x", "y", "vx", "vy")),
    ]
    match_id = path.name.removesuffix(".tracking.csv")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing STP columns: {missing}")
        previous_cycle: int | None = None
        segment = 0
        segment_step = 0
        for row_index, row in enumerate(reader):
            if max_frames > 0 and row_index >= max_frames:
                break
            cycle = int(float(row["cycle"]))
            if previous_cycle is not None and cycle != previous_cycle + 1:
                segment += 1
                segment_step = 0
            left = str(row["l_name"])
            right = str(row["r_name"])
            players = []
            for side, team_id in (("l", left), ("r", right)):
                for index in range(11):
                    agent = f"{side}{index + 1}"
                    players.append(
                        PlayerState(
                            player_id=agent,
                            team_id=team_id,
                            index=index,
                            x=float(row[f"{agent}_x"]),
                            y=float(row[f"{agent}_y"]),
                            vx=float(row[f"{agent}_vx"]),
                            vy=float(row[f"{agent}_vy"]),
                        )
                    )
            game_mode, imputed_mode = robocup_playmode_to_grf(row["playmode"])
            imputed = {"ball_z", "ball_vz", "possession_team"}
            if imputed_mode:
                imputed.add("game_mode")
            frame = SoccerFrame(
                source="robocup",
                match_id=match_id,
                sequence_id=str(segment),
                step_index=segment_step,
                frame_id=f"{match_id}:{cycle}",
                timestamp=float(cycle),
                teams=(left, right),
                positive_x_team=left,
                players=tuple(players),
                ball=BallState(
                    x=float(row["b_x"]),
                    y=float(row["b_y"]),
                    vx=float(row["b_vx"]),
                    vy=float(row["b_vy"]),
                ),
                possession_team=None,
                game_mode=game_mode,
                imputed_fields=frozenset(imputed),
            )
            frame.validate()
            yield frame
            previous_cycle = cycle
            segment_step += 1


class Simple115V2Encoder:
    """Encode canonical soccer frames exactly in GRF simple115v2 field order."""

    def __init__(self, config: GRFEncodingConfig | None = None):
        self.config = config or GRFEncodingConfig()
        self.config.validate()

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        return x / self.config.source_x_half, y / self.config.source_y_half * self.config.target_y_half

    def _direction(self, current: tuple[float, float], previous: tuple[float, float] | None) -> tuple[float, float]:
        if self.config.direction_mode == "finite_difference":
            if previous is None:
                raise ValueError("finite_difference encoding requires a previous frame")
            current_xy = self._xy(*current)
            previous_xy = self._xy(*previous)
            return current_xy[0] - previous_xy[0], current_xy[1] - previous_xy[1]
        vx, vy = current
        return self._xy(vx * self.config.velocity_dt, vy * self.config.velocity_dt)

    @staticmethod
    def _player_map(frame: SoccerFrame) -> dict[tuple[str, str], PlayerState]:
        return {(player.team_id, player.player_id): player for player in frame.players}

    def encode_frame(self, frame: SoccerFrame, previous: SoccerFrame | None = None) -> tuple[EncodedObservation, ...]:
        frame.validate()
        if previous is not None:
            previous.validate()
            if (frame.source, frame.match_id, frame.sequence_id) != (
                previous.source,
                previous.match_id,
                previous.sequence_id,
            ) or frame.step_index != previous.step_index + 1:
                raise ValueError("Previous frame is not contiguous with current frame")
        if self.config.direction_mode == "finite_difference" and previous is None:
            raise ValueError("A previous contiguous frame is required for finite-difference directions")

        previous_players = self._player_map(previous) if previous is not None else {}
        team_players = {
            team: sorted((player for player in frame.players if player.team_id == team), key=lambda item: item.index)
            for team in frame.teams
        }
        encoded = []
        for focal_team in frame.teams:
            opponent_team = frame.teams[1] if focal_team == frame.teams[0] else frame.teams[0]
            rotated = focal_team != frame.positive_x_team
            sign = -1.0 if rotated else 1.0
            positions: dict[str, list[tuple[float, float]]] = {focal_team: [], opponent_team: []}
            directions: dict[str, list[tuple[float, float]]] = {focal_team: [], opponent_team: []}
            for team in (focal_team, opponent_team):
                for player in team_players[team]:
                    positions[team].append(tuple(sign * value for value in self._xy(player.x, player.y)))
                    previous_player = previous_players.get((team, player.player_id))
                    direction_input = (player.vx, player.vy)
                    previous_input = (
                        (previous_player.x, previous_player.y) if previous_player is not None else None
                    )
                    directions[team].append(
                        tuple(sign * value for value in self._direction(direction_input if self.config.direction_mode == "stored_velocity" else (player.x, player.y), previous_input))
                    )

            ball_xy = tuple(sign * value for value in self._xy(frame.ball.x, frame.ball.y))
            previous_ball = (previous.ball.x, previous.ball.y) if previous is not None else None
            ball_direction_input = (
                (frame.ball.vx, frame.ball.vy)
                if self.config.direction_mode == "stored_velocity"
                else (frame.ball.x, frame.ball.y)
            )
            ball_direction = tuple(sign * value for value in self._direction(ball_direction_input, previous_ball))
            ball_z = frame.ball.z
            ball_vz = frame.ball.vz * self.config.velocity_dt if self.config.direction_mode == "stored_velocity" else (
                frame.ball.z - previous.ball.z if previous is not None else 0.0
            )
            ownership = -1 if frame.possession_team is None else (0 if frame.possession_team == focal_team else 1)

            shared = []
            shared.extend(value for point in positions[focal_team] for value in point)
            shared.extend(value for point in directions[focal_team] for value in point)
            shared.extend(value for point in positions[opponent_team] for value in point)
            shared.extend(value for point in directions[opponent_team] for value in point)
            shared.extend([*ball_xy, ball_z])
            shared.extend([*ball_direction, ball_vz])
            shared.extend([1.0, 0.0, 0.0] if ownership == -1 else ([0.0, 1.0, 0.0] if ownership == 0 else [0.0, 0.0, 1.0]))
            if len(shared) != 97:
                raise AssertionError(f"Expected 97 shared features, got {len(shared)}")

            for focal in team_players[focal_team]:
                active = [0.0] * 11
                active[focal.index] = 1.0
                game_mode = [0.0] * GRF_GAME_MODES
                game_mode[frame.game_mode] = 1.0
                vector = np.asarray([*shared, *active, *game_mode], dtype=np.float32)
                if vector.shape != (GRF_OBSERVATION_SIZE,):
                    raise AssertionError(f"Expected {(GRF_OBSERVATION_SIZE,)}, got {vector.shape}")
                encoded.append(
                    EncodedObservation(
                        vector=vector,
                        source=frame.source,
                        match_id=frame.match_id,
                        sequence_id=frame.sequence_id,
                        step_index=frame.step_index,
                        frame_id=frame.frame_id,
                        focal_team=focal_team,
                        focal_player=focal.player_id,
                        focal_index=focal.index,
                        rotated=rotated,
                        imputed_fields=frame.imputed_fields,
                    )
                )
        return tuple(encoded)


def encode_histories(
    frames: Iterable[SoccerFrame], encoder: Simple115V2Encoder
) -> tuple[tuple[EncodedHistory, ...], dict[str, Any]]:
    """Encode newest-first histories without crossing source sequence boundaries."""

    histories: list[EncodedHistory] = []
    buffers: dict[tuple[str, str, str, str, str], deque[EncodedObservation]] = defaultdict(deque)
    previous: SoccerFrame | None = None
    invalid_directions = 0
    encoded_frames = 0
    max_buffer = (encoder.config.history_len - 1) * encoder.config.history_step + 1
    for frame in frames:
        same_sequence = previous is not None and (
            frame.source,
            frame.match_id,
            frame.sequence_id,
            frame.step_index,
        ) == (
            previous.source,
            previous.match_id,
            previous.sequence_id,
            previous.step_index + 1,
        )
        if not same_sequence:
            buffers.clear()
            previous = frame
            invalid_directions += 1
            continue
        observations = encoder.encode_frame(frame, previous)
        encoded_frames += 1
        for observation in observations:
            buffer = buffers[observation.identity]
            buffer.append(observation)
            while len(buffer) > max_buffer:
                buffer.popleft()
            if len(buffer) < max_buffer:
                continue
            selected = [buffer[-1 - offset] for offset in range(0, max_buffer, encoder.config.history_step)]
            histories.append(
                EncodedHistory(
                    observations=np.stack([item.vector for item in selected]),
                    current=observation,
                )
            )
        previous = frame
    return tuple(histories), {
        "encoded_frames": encoded_frames,
        "histories": len(histories),
        "sequence_starts_dropped": invalid_directions,
        "incomplete_histories_dropped": encoded_frames * 22 - len(histories),
    }


def grf_position_templates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GRF group, agent, and attribute templates for 115 scalar tokens."""

    group = np.zeros(GRF_OBSERVATION_SIZE, dtype=np.int64)
    agent = np.zeros(GRF_OBSERVATION_SIZE, dtype=np.int64)
    attr = np.zeros(GRF_OBSERVATION_SIZE, dtype=np.int64)
    index = 0
    for team in range(2):
        for attribute_pair in ((0, 1), (2, 3)):
            for player in range(11):
                for attribute in attribute_pair:
                    group[index] = team
                    agent[index] = player + 1
                    attr[index] = attribute
                    index += 1
    for attribute in range(4, 13):
        attr[index] = attribute
        index += 1
    for player in range(11):
        agent[index] = player + 1
        attr[index] = 13
        index += 1
    for attribute in range(14, 21):
        attr[index] = attribute
        index += 1
    if index != GRF_OBSERVATION_SIZE:
        raise AssertionError(f"Built {index} positional tokens")
    return group, agent, attr


def build_model_inputs(histories: Sequence[EncodedHistory], config: GRFEncodingConfig) -> ModelInputBatch:
    """Create padded MARL-GPT arrays with the GRF environment token."""

    config.validate()
    if not histories:
        raise ValueError("At least one encoded history is required")
    for history in histories:
        if history.observations.shape != (config.history_len, GRF_OBSERVATION_SIZE):
            raise ValueError(
                f"Expected history {(config.history_len, GRF_OBSERVATION_SIZE)}, got {history.observations.shape}"
            )
    batch = len(histories)
    group_template, agent_template, attr_template = grf_position_templates()
    token_count = config.history_len * GRF_OBSERVATION_SIZE
    arrays = {
        "obs": np.zeros((batch, config.block_size), dtype=np.float32),
        "group_pos": np.zeros((batch, config.block_size), dtype=np.int64),
        "agent_pos": np.zeros((batch, config.block_size), dtype=np.int64),
        "time_pos": np.zeros((batch, config.block_size), dtype=np.int64),
        "attr_pos": np.zeros((batch, config.block_size), dtype=np.int64),
        "action_mask": np.zeros((batch, config.action_size), dtype=np.int64),
    }
    arrays["obs"][:, :token_count] = np.stack([history.observations.reshape(-1) for history in histories])
    arrays["group_pos"][:, :token_count] = np.tile(group_template, config.history_len)
    arrays["agent_pos"][:, :token_count] = np.tile(agent_template, config.history_len)
    arrays["attr_pos"][:, :token_count] = np.tile(attr_template, config.history_len)
    arrays["time_pos"][:, :token_count] = np.repeat(np.arange(config.history_len), GRF_OBSERVATION_SIZE)
    arrays["obs"][:, -1] = config.environment_id
    arrays["attr_pos"][:, -1] = config.environment_attr
    arrays["action_mask"][:, :GRF_ACTIONS] = 1
    return ModelInputBatch(arrays=arrays, histories=tuple(histories))


def audit_histories(histories: Sequence[EncodedHistory]) -> dict[str, Any]:
    if not histories:
        return {"histories": 0}
    values = np.stack([history.observations for history in histories])
    imputed = Counter(field for history in histories for field in history.current.imputed_fields)
    return {
        "histories": len(histories),
        "finite": bool(np.isfinite(values).all()),
        "shape": list(values.shape),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "feature_min": values.min(axis=(0, 1)).tolist(),
        "feature_max": values.max(axis=(0, 1)).tolist(),
        "imputed_history_counts": dict(sorted(imputed.items())),
        "rotated_histories": sum(history.current.rotated for history in histories),
    }

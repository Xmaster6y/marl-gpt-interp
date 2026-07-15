import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from marl_gpt_interp.external_soccer import (
    BallState,
    GRFEncodingConfig,
    PlayerState,
    Simple115V2Encoder,
    SoccerFrame,
    build_model_inputs,
    encode_histories,
    grf_position_templates,
    iter_laliga_frames,
    iter_robocup_frames,
    robocup_playmode_to_grf,
)


def _frame(step: int, *, possession="left", game_mode=0) -> SoccerFrame:
    players = []
    for team, offset in (("left", -0.2), ("right", 0.2)):
        for index in range(11):
            players.append(
                PlayerState(
                    player_id=f"{team}-{index}",
                    team_id=team,
                    index=index,
                    x=offset + index * 0.01 + step * 0.001,
                    y=-0.1 + index * 0.01,
                    vx=0.001,
                    vy=0.0,
                )
            )
    return SoccerFrame(
        source="synthetic",
        match_id="match",
        sequence_id="sequence",
        step_index=step,
        frame_id=str(step),
        timestamp=float(step),
        teams=("left", "right"),
        positive_x_team="left",
        players=tuple(players),
        ball=BallState(x=step * 0.001, y=0.0, vx=0.001, vy=0.0),
        possession_team=possession,
        game_mode=game_mode,
    )


def test_encoder_matches_official_simple115v2_ordering():
    wrappers = pytest.importorskip("gfootball.env.wrappers")
    frame = _frame(0, game_mode=3)
    config = GRFEncodingConfig(
        source_x_half=1.0,
        source_y_half=1.0,
        target_y_half=1.0,
        direction_mode="stored_velocity",
        velocity_dt=1.0,
    )
    encoded = Simple115V2Encoder(config).encode_frame(frame)[0].vector
    left = sorted((player for player in frame.players if player.team_id == "left"), key=lambda item: item.index)
    right = sorted((player for player in frame.players if player.team_id == "right"), key=lambda item: item.index)
    raw = {
        "left_team": np.asarray([[player.x, player.y] for player in left]),
        "left_team_direction": np.asarray([[player.vx, player.vy] for player in left]),
        "right_team": np.asarray([[player.x, player.y] for player in right]),
        "right_team_direction": np.asarray([[player.vx, player.vy] for player in right]),
        "ball": np.asarray([frame.ball.x, frame.ball.y, frame.ball.z]),
        "ball_direction": np.asarray([frame.ball.vx, frame.ball.vy, frame.ball.vz]),
        "ball_owned_team": 0,
        "active": 0,
        "game_mode": 3,
    }
    official = wrappers.Simple115StateWrapper.convert_observation([raw], True)[0]
    np.testing.assert_array_equal(encoded, official)


def test_encoder_generates_22_perspectives_and_rotates_right_team():
    encoder = Simple115V2Encoder(
        GRFEncodingConfig(source_x_half=1.0, source_y_half=1.0, target_y_half=1.0)
    )
    encoded = encoder.encode_frame(_frame(1), _frame(0))

    assert len(encoded) == 22
    assert sum(item.rotated for item in encoded) == 11
    left = encoded[0].vector
    right = encoded[11].vector
    assert left[97] == 1
    assert right[97] == 1
    np.testing.assert_allclose(right[:22], -left[44:66])
    np.testing.assert_allclose(right[44:66], -left[:22])
    assert left[94:97].tolist() == [0, 1, 0]
    assert right[94:97].tolist() == [0, 0, 1]


def test_histories_are_newest_first_and_model_inputs_match_checkpoint_contract():
    config = GRFEncodingConfig(
        source_x_half=1.0,
        source_y_half=1.0,
        target_y_half=1.0,
        history_len=6,
        block_size=700,
    )
    histories, diagnostics = encode_histories((_frame(step) for step in range(8)), Simple115V2Encoder(config))
    assert len(histories) == 44
    assert diagnostics["sequence_starts_dropped"] == 1
    first = histories[0]
    assert first.current.step_index == 6
    assert first.observations[:, 88].tolist() == pytest.approx([0.006, 0.005, 0.004, 0.003, 0.002, 0.001])

    batch = build_model_inputs(histories[:2], config)
    assert batch.arrays["obs"].shape == (2, 700)
    assert batch.arrays["obs"][0, -1] == 3
    assert batch.arrays["attr_pos"][0, -1] == 128
    assert batch.arrays["action_mask"][0].tolist() == [1] * 19 + [0]
    assert batch.arrays["time_pos"][0, :690].tolist() == [value for value in range(6) for _ in range(115)]


def test_histories_reset_after_a_frame_gap():
    config = GRFEncodingConfig(
        source_x_half=1.0,
        source_y_half=1.0,
        target_y_half=1.0,
        history_len=6,
    )
    frames = [_frame(step) for step in (0, 1, 3, 4, 5, 6, 7, 8, 9)]

    histories, diagnostics = encode_histories(frames, Simple115V2Encoder(config))

    assert len(histories) == 22
    assert diagnostics["sequence_starts_dropped"] == 2
    assert histories[0].observations[:, 88].tolist() == pytest.approx([0.009, 0.008, 0.007, 0.006, 0.005, 0.004])


def test_position_templates_match_grf_tokenizer():
    marl_gpt_path = str(Path(__file__).resolve().parents[2] / "marl-gpt")
    sys.path.insert(0, marl_gpt_path)
    try:
        from envs.grf_env.obs_tokenizer import grfTokenizer

        expected = grfTokenizer("grf")
        group, agent, attr = grf_position_templates()
        np.testing.assert_array_equal(group, expected.group_pos_template.numpy())
        np.testing.assert_array_equal(agent, expected.agent_pos_template.numpy())
        np.testing.assert_array_equal(attr, expected.attr_pos_template.numpy())
    finally:
        sys.path.remove(marl_gpt_path)


def test_laliga_jsonl_adapter_omits_names_and_preserves_sequence(tmp_path):
    def player(index, team):
        return {
            "player_id": 1000 + index,
            "player_name": f"private-{index}",
            "index": index,
            "position": {"x": float(index), "y": 0.0},
            "velocity": {"x": 0.1, "y": 0.0},
            "team_name": team,
        }

    attack = [player(index, "attack") for index in range(11)]
    defense = [player(index + 11, "defense") for index in range(11)]
    event = {
        "state": {
            "raw_state": {
                "attack_players": attack,
                "defense_players": defense,
                "players": attack + defense,
                "ball": {"position": {"x": 0.0, "y": 0.0}, "velocity": {"x": 0.0, "y": 0.0}},
            }
        }
    }
    reversed_event = {
        "state": {
            "raw_state": {
                "attack_players": list(reversed(attack)),
                "defense_players": list(reversed(defense)),
                "players": list(reversed(attack + defense)),
                "ball": {"position": {"x": 0.0, "y": 0.0}, "velocity": {"x": 0.0, "y": 0.0}},
            }
        }
    }
    payload = {
        "game_id": 7,
        "half": 1,
        "sequence_id": 2,
        "sequence_start_frame": 0.0,
        "sequence_end_frame": 0.1,
        "team_name_attack": "attack",
        "team_name_defense": "defense",
        "events": [event, reversed_event],
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(payload) + "\n")

    frames = list(iter_laliga_frames(path))

    assert len(frames) == 2
    assert frames[1].timestamp == pytest.approx(0.1)
    assert frames[0].possession_team == "attack"
    assert [player.player_id for player in frames[0].players] == [player.player_id for player in frames[1].players]
    assert all("private" not in player.player_id for player in frames[0].players)
    assert frames[0].imputed_fields == frozenset({"ball_z", "ball_vz", "game_mode", "possession_team"})


def test_robocup_adapter_uses_named_physical_columns_and_rejects_npy(tmp_path):
    metadata = ["cycle", "playmode", "l_name", "r_name"]
    physical = [
        f"{agent}_{feature}"
        for agent in ("b", *(f"l{i}" for i in range(1, 12)), *(f"r{i}" for i in range(1, 12)))
        for feature in ("x", "y", "vx", "vy")
    ]
    path = tmp_path / "sample.tracking.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*metadata, *physical])
        writer.writeheader()
        for cycle in (1, 2):
            row = {column: 0.0 for column in physical}
            row.update({"cycle": cycle, "playmode": "free_kick_l", "l_name": "left", "r_name": "right"})
            writer.writerow(row)

    frames = list(iter_robocup_frames(path))

    assert len(frames) == 2
    assert frames[0].game_mode == 3
    assert frames[0].possession_team is None
    assert robocup_playmode_to_grf("unknown_mode") == (0, True)
    with pytest.raises(ValueError, match="unsupported"):
        list(iter_robocup_frames(tmp_path / "broken.npy"))

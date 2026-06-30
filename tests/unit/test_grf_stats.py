from marl_gpt_interp.grf_stats import (
    action_entropy,
    average_present,
    count_actions,
    extract_raw_observation_stats,
)


def test_extract_raw_observation_stats_from_synthetic_grf_raw_obs():
    raw_observations = [
        {
            "left_team": [[0.0, 0.0], [0.2, 0.3], [0.4, -0.2]],
            "right_team": [[0.5, 0.0], [0.7, 0.4], [0.9, -0.4]],
            "ball": [0.2, 0.1, 0.0],
            "ball_owned_team": 0,
            "ball_owned_player": 1,
            "score": [2, 1],
            "game_mode": 0,
        }
    ]

    stats = extract_raw_observation_stats(raw_observations)

    assert stats["ball_owned_team"] == 0
    assert stats["left_score"] == 2
    assert stats["right_score"] == 1
    assert stats["goal_difference"] == 1
    assert stats["left_team_width"] == 0.5
    assert stats["left_team_depth"] == 0.4
    assert stats["right_team_width"] == 0.8
    assert stats["right_team_depth"] == 0.4
    assert round(stats["distance_to_goal"], 3) == 0.806
    assert round(stats["nearest_defender_distance"], 3) == 0.424
    assert round(stats["defensive_compactness"], 3) == 0.613


def test_action_summaries_are_dependency_free():
    counts = count_actions([0, 1, 1, 19, -1], action_size=4)

    assert counts == [1, 2, 0, 0]
    assert round(action_entropy(counts), 3) == 0.918
    assert average_present([1, None, 3]) == 2.0

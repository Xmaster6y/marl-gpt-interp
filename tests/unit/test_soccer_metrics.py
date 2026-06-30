from marl_gpt_interp.soccer_metrics import compare_datasets, event_statistics, pitch_control_statistics
from marl_gpt_interp.soccer_schema import normalize_rows


def test_normalize_rows_maps_provider_columns_to_event_schema():
    rows = [
        {
            "game": "m1",
            "team": "home",
            "kind": "pass",
            "result": "complete",
            "sx": "10",
            "sy": "20",
            "tx": "30",
            "ty": "25",
        }
    ]

    normalized = normalize_rows(
        rows,
        dataset="human",
        kind="events",
        column_map={
            "match_id": "game",
            "team_id": "team",
            "event_type": "kind",
            "outcome": "result",
            "x": "sx",
            "y": "sy",
            "end_x": "tx",
            "end_y": "ty",
        },
    )

    assert normalized == [
        {
            "dataset": "human",
            "match_id": "m1",
            "possession_id": None,
            "team_id": "home",
            "player_id": None,
            "timestamp": None,
            "event_type": "pass",
            "outcome": "complete",
            "x": 10.0,
            "y": 20.0,
            "end_x": 30.0,
            "end_y": 25.0,
        }
    ]


def test_event_statistics_counts_passes_shots_and_pass_lengths():
    rows = [
        {"event_type": "pass", "outcome": "complete", "x": 0, "y": 0, "end_x": 3, "end_y": 4},
        {"event_type": "pass", "outcome": "failed", "x": 0, "y": 0, "end_x": 0, "end_y": 6},
        {"event_type": "shot", "outcome": "goal"},
    ]

    stats = event_statistics(rows)

    assert stats["passes"] == 2
    assert stats["completed_passes"] == 1
    assert stats["pass_completion_rate"] == 0.5
    assert stats["mean_pass_length"] == 5.5
    assert stats["shots"] == 1
    assert stats["goals"] == 1


def test_pitch_control_statistics_uses_nearest_team_on_grid():
    rows = [
        {"dataset": "toy", "match_id": "m1", "frame_id": 1, "team_id": "left", "x": 10, "y": 34},
        {"dataset": "toy", "match_id": "m1", "frame_id": 1, "team_id": "right", "x": 95, "y": 34},
        {"dataset": "toy", "match_id": "m1", "frame_id": 2, "team_id": "left", "x": 10, "y": 34},
        {"dataset": "toy", "match_id": "m1", "frame_id": 2, "team_id": "right", "x": 95, "y": 34},
    ]

    stats = pitch_control_statistics(rows, grid_x=2, grid_y=1, pitch_length=100, pitch_width=68)

    assert stats["tracking_frames"] == 2
    assert stats["pitch_control_grid_cells"] == 4
    assert stats["pitch_control_fraction_team_left"] == 0.5
    assert stats["pitch_control_fraction_team_right"] == 0.5


def test_compare_datasets_returns_one_flat_row_per_dataset():
    rows = [
        {"dataset": "human", "event_type": "pass", "outcome": "complete"},
        {"dataset": "grf", "event_type": "shot", "outcome": "failed"},
    ]

    comparison = compare_datasets(event_rows=rows)

    assert [row["dataset"] for row in comparison] == ["grf", "human"]
    assert comparison[0]["shots"] == 1
    assert comparison[1]["passes"] == 1

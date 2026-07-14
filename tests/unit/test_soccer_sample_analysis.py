import csv

import numpy as np

from marl_gpt_interp.soccer_sample_analysis import analyze_robocup_pair, analyze_stp_tracking_csv


def test_robocup_pair_detects_documented_block_swap(tmp_path):
    left = np.zeros((3, 92), dtype=np.float32)
    left[:, 4:48] = 1.0
    left[:, 48:92] = 2.0
    right = left.copy()
    right[:, 4:48] = left[:, 48:92]
    right[:, 48:92] = left[:, 4:48]
    left_path = tmp_path / "sample.left.state.npy"
    right_path = tmp_path / "sample.right.state.npy"
    np.save(left_path, left)
    np.save(right_path, right)

    result = analyze_robocup_pair(left_path, right_path)

    assert result["shape"] == [3, 92]
    assert result["ball_blocks_equal"]
    assert result["team_blocks_swap"]
    assert result["interleaved_physical_fraction"] == 1.0
    assert result["grouped_physical_fraction"] == 1.0


def test_stp_tracking_audit_detects_obsolete_player_stride(tmp_path):
    metadata = [
        "#",
        "cycle",
        "stopped",
        "playmode",
        "l_name",
        "l_score",
        "l_pen_score",
        "r_name",
        "r_score",
        "r_pen_score",
    ]
    columns = [*metadata, "b_x", "b_y", "b_vx", "b_vy"]
    for side in ("l", "r"):
        for player in range(1, 12):
            columns.extend(
                f"{side}{player}_{suffix}"
                for suffix in ("t", "x", "y", "vx", "vy", "body", "neck", "vwidth", "stamina")
            )

    csv_path = tmp_path / "sample.tracking.csv"
    numeric = {column: float(index) for index, column in enumerate(columns)}
    numeric.update({"playmode": "play_on", "l_name": "left", "r_name": "right"})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(numeric)

    obsolete_indices = [*range(10, 14)]
    for player_index in range(22):
        obsolete_indices.extend(range(15 + 8 * player_index, 19 + 8 * player_index))
    derived_path = tmp_path / "sample.left.state.npy"
    np.save(
        derived_path, np.asarray([[float(numeric[columns[index]]) for index in obsolete_indices]], dtype=np.float32)
    )

    result = analyze_stp_tracking_csv(csv_path, derived_path)

    assert result["physical_columns"] == 92
    assert not result["derived_audit"]["matches_named_physical_state"]
    assert result["derived_audit"]["matches_obsolete_eight_column_stride"]
    assert result["derived_audit"]["actual_player_stride"] == 9

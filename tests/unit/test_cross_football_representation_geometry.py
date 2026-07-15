import pytest

torch = pytest.importorskip("torch")

from scripts.cross_football_representation_geometry import (  # noqa: E402
    _collect_source_latents,
    build_analysis_units,
    select_analysis_frame_groups,
)


def test_build_analysis_units_balances_perspectives_and_frame_means():
    source_features = {}
    source_metadata = {}
    perspective_counts = {"laliga": 22, "robocup": 22, "grf": 11}
    for source_index, (source, perspectives) in enumerate(perspective_counts.items()):
        rows = []
        metadata = []
        for frame in range(2):
            for perspective in range(perspectives):
                rows.append([float(source_index), float(frame * 100 + perspective)])
                metadata.append({"frame_id": f"{source}-{frame}"})
        source_features[source] = {"layer:mean": torch.tensor(rows)}
        source_metadata[source] = metadata

    units = build_analysis_units(source_features, source_metadata, max_frames=2)

    perspective_features, perspective_labels = units["perspective"]
    frame_features, frame_labels = units["frame_mean"]
    assert perspective_features["layer:mean"].shape == (66, 2)
    assert [(perspective_labels == label).sum().item() for label in (1, 2, 3)] == [22, 22, 22]
    assert frame_features["layer:mean"].shape == (6, 2)
    assert [(frame_labels == label).sum().item() for label in (1, 2, 3)] == [2, 2, 2]
    assert frame_features["layer:mean"][0, 1] == pytest.approx(10.5)


def test_latent_collection_reuses_a_shape_compatible_cache(tmp_path):
    cached = {"layer:mean": torch.ones((3, 2))}
    torch.save(cached, tmp_path / "laliga.pt")
    batches = {"laliga": {"obs": torch.zeros((3, 4))}}

    result = _collect_source_latents(None, batches, type("Config", (), {"batch_size": 1})(), tmp_path)

    torch.testing.assert_close(result["laliga"]["layer:mean"], cached["layer:mean"])


def test_random_frame_sampling_is_deterministic_complete_and_temporally_spaced():
    source_metadata = {}
    perspective_counts = {"laliga": 22, "robocup": 22, "grf": 11}
    for source, perspectives in perspective_counts.items():
        source_metadata[source] = [
            {
                "frame_id": f"{source}-{step}",
                "match_id": f"{source}-match",
                "sequence_id": "sequence-0",
                "step_index": step,
            }
            for step in range(12)
            for _perspective in range(perspectives)
        ]

    selected, audit = select_analysis_frame_groups(
        source_metadata,
        4,
        mode="random",
        seed=17,
        min_step_gap=2,
    )
    repeated, repeated_audit = select_analysis_frame_groups(
        source_metadata,
        4,
        mode="random",
        seed=17,
        min_step_gap=2,
    )

    assert selected == repeated
    assert audit == repeated_audit
    for source, perspectives in perspective_counts.items():
        assert all(len(group) == perspectives for group in selected[source])
        steps = audit[source]["step_indices"]
        assert all(abs(left - right) >= 2 for index, left in enumerate(steps) for right in steps[index + 1 :])

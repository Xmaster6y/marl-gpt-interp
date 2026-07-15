from __future__ import annotations

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from marl_gpt_interp.marl_gpt_tools import (  # noqa: E402
    activation_subspace_similarity_rows,
    asymmetric_subspace_rows,
    pooled_activations,
    representation_proximity_rows,
    representation_separation_rows,
    self_subspace_similarity_rows,
)


def _cfg():
    return OmegaConf.create(
        {
            "max_pairwise_examples_per_env": 16,
            "max_pca_rank": 8,
            "asymmetric_top_ks": [1, 2],
        }
    )


def test_pooled_activations_can_exclude_final_environment_token_from_mean():
    captured = {
        "layer_00": torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [100.0, 200.0],
                ]
            ]
        )
    }

    pooled = pooled_activations(captured, exclude_final_token_from_mean=True)

    assert torch.equal(pooled["layer_00:mean"], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(pooled["layer_00:final"], torch.tensor([[100.0, 200.0]]))


def test_pooled_activations_reject_excluding_an_only_token():
    with pytest.raises(ValueError, match="only token"):
        pooled_activations(
            {"layer_00": torch.ones(1, 1, 2)},
            exclude_final_token_from_mean=True,
        )


def test_representation_proximity_rows_measure_internal_spread():
    features = {
        "layer:mean": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [10.0, 10.0],
                [11.0, 10.0],
                [10.0, 11.0],
            ]
        )
    }
    labels = torch.tensor([1, 1, 1, 2, 2, 2])

    rows = representation_proximity_rows(features, labels, cfg=_cfg())

    assert len(rows) == 2
    assert {row["env"] for row in rows} == {"smac", "pogema"}
    assert all(row["mean_pairwise_l2"] > 0 for row in rows)
    assert all(row["pca_components_90"] >= 1 for row in rows)


def test_representation_helpers_accept_cross_football_source_names():
    features = {
        "layer:mean": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [10.0, 10.0],
                [11.0, 10.0],
                [20.0, 20.0],
                [21.0, 20.0],
            ]
        )
    }
    labels = torch.tensor([1, 1, 2, 2, 3, 3])
    names = {1: "laliga", 2: "robocup", 3: "grf"}

    proximity = representation_proximity_rows(features, labels, cfg=_cfg(), label_names=names)
    cka = activation_subspace_similarity_rows(features, labels, label_names=names)

    assert {row["env"] for row in proximity} == {"laliga", "robocup", "grf"}
    assert {row["env_pair"] for row in cka} == {
        "laliga_vs_robocup",
        "laliga_vs_grf",
        "robocup_vs_grf",
    }


def test_representation_separation_rows_normalize_by_within_spread():
    compact_features = {
        "layer:mean": torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [10.0, 10.0],
                [10.1, 10.0],
                [10.0, 10.1],
            ]
        )
    }
    labels = torch.tensor([1, 1, 1, 2, 2, 2])

    rows = representation_separation_rows(compact_features, labels, cfg=_cfg())

    assert len(rows) == 1
    row = rows[0]
    assert row["env_pair"] == "smac_vs_pogema"
    assert row["same_env_nearest_neighbor_fraction"] == pytest.approx(1.0)
    assert row["normalized_centroid_l2"] > 10
    assert row["silhouette_l2"] > 0.9


def test_asymmetric_subspace_rows_are_directional():
    features = {
        "layer:mean": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
    }
    labels = torch.tensor([1, 1, 1, 2, 2, 2])

    rows = asymmetric_subspace_rows(features, labels, cfg=_cfg())

    assert {row["direction"] for row in rows} == {"smac_to_pogema", "pogema_to_smac"}
    assert {row["requested_rank"] for row in rows} == {1, 2}
    assert all(0.0 <= row["target_variance_explained"] <= 1.0 for row in rows)


def test_self_subspace_similarity_rows_compare_split_halves():
    features = {
        "layer:mean": torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [1.0, 0.1],
                [2.0, 0.1],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.1, 1.0],
                [0.1, 2.0],
            ]
        )
    }
    labels = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])

    rows = self_subspace_similarity_rows(features, labels)

    assert len(rows) == 2
    assert {row["env"] for row in rows} == {"smac", "pogema"}
    assert all(row["n_examples_per_split"] == 2 for row in rows)
    assert all(0.0 <= row["linear_cka"] <= 1.0 for row in rows)

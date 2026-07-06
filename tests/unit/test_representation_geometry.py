from __future__ import annotations

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from marl_gpt_interp.marl_gpt_tools import (  # noqa: E402
    asymmetric_subspace_rows,
    representation_proximity_rows,
    representation_separation_rows,
)


def _cfg():
    return OmegaConf.create(
        {
            "max_pairwise_examples_per_env": 16,
            "max_pca_rank": 8,
            "asymmetric_top_ks": [1, 2],
        }
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

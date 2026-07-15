from __future__ import annotations

import pytest

from scripts.make_cross_domain_representation_figures import layer_series


def test_layer_series_extracts_all_transformer_layers_in_order():
    rows = [
        {
            "env_pair": "left_vs_right",
            "feature": f"layer_{layer:02d}:mean",
            "linear_cka": str(layer / 10),
        }
        for layer in reversed(range(7))
    ]

    values = layer_series(
        rows,
        group_column="env_pair",
        group="left_vs_right",
        pool="mean",
        value_column="linear_cka",
    )

    assert values == pytest.approx([layer / 10 for layer in range(7)])


def test_layer_series_rejects_missing_layer_rows():
    with pytest.raises(ValueError, match="Expected one"):
        layer_series(
            [],
            group_column="env_pair",
            group="left_vs_right",
            pool="mean",
            value_column="linear_cka",
        )

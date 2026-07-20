from __future__ import annotations

from marl_gpt_interp.balanced_dataset import (
    RemoteFile,
    allocate_component_counts,
    build_selection,
    deterministic_select,
)


def test_component_allocation_is_exact_and_preserves_positive_components():
    assert allocate_component_counts(12, {"mazes": 9, "random": 1}) == {"mazes": 11, "random": 1}
    assert allocate_component_counts(12, {str(index): 1 for index in range(6)}) == {
        str(index): 2 for index in range(6)
    }


def test_deterministic_selection_is_order_independent():
    files = [RemoteFile(f"root/file-{index}.pt", index + 1) for index in range(10)]
    expected = deterministic_select(files, 4, seed=7, namespace="root")
    assert deterministic_select(list(reversed(files)), 4, seed=7, namespace="root") == expected
    assert len({item.source_path for item in expected}) == 4


def test_selection_balances_source_files_across_environments():
    environments = {
        environment: {
            "source_files": 6,
            "components": [
                {
                    "name": "task",
                    "weight": 1,
                    "source_prefix": f"remote/{environment}",
                    "destination_prefix": f"dataset/{environment}",
                }
            ],
        }
        for environment in ("smac", "pogema", "grf")
    }

    def catalog(prefix: str):
        return [RemoteFile(f"{prefix}/file-{index}.pt", 100) for index in range(10)]

    selected = build_selection(environments, seed=0, catalog=catalog)
    assert {environment: sum(item.environment == environment for item in selected) for environment in environments} == {
        "smac": 6,
        "pogema": 6,
        "grf": 6,
    }
    assert all(item.destination_path.startswith(f"dataset/{item.environment}/") for item in selected)

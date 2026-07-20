from __future__ import annotations

import json

import pytest
import torch

from marl_gpt_interp.balanced_dataset import (
    RemoteFile,
    allocate_component_counts,
    audit_balanced_view,
    build_selection,
    deterministic_select,
    shard_family,
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


def test_selection_balances_source_groups_across_environments():
    environments = {
        environment: {
            "source_groups": 6,
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


def test_selection_uses_one_largest_file_from_each_distinct_group():
    environments = {
        "grf": {
            "source_groups": 2,
            "components": [
                {
                    "name": "task",
                    "weight": 1,
                    "source_prefix": "remote/grf",
                    "destination_prefix": "dataset/grf",
                    "group_pattern": r"(chunk_\d+)",
                }
            ],
        }
    }
    files = [
        RemoteFile("remote/grf/chunk_0_part_0.pt", 10),
        RemoteFile("remote/grf/chunk_0_part_1.pt", 20),
        RemoteFile("remote/grf/chunk_1_part_0.pt", 30),
    ]
    selected = build_selection(environments, seed=0, catalog=lambda _prefix: files)
    assert {item.source_group for item in selected} == {"remote/grf/chunk_0", "remote/grf/chunk_1"}
    assert {item.source_path for item in selected} == {
        "remote/grf/chunk_0_part_1.pt",
        "remote/grf/chunk_1_part_0.pt",
    }


def test_shard_family_marks_multipart_chunks_without_assuming_arrow_families():
    assert shard_family("dataset-GRF/task/chunk_3_part_4.pt") == "dataset-GRF/task/chunk_3"
    assert shard_family("dataset-LMAPF/random/part_2_7.arrow") == "dataset-LMAPF/random/part_2_7.arrow"


def test_audit_rejects_unequal_accepted_environment_budgets(tmp_path):
    files = []
    for environment, count in (("smac", 4), ("pogema", 4), ("grf", 3)):
        relative_path = f"dataset/{environment}/chunk_0.pt"
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actions": torch.zeros(count), "dones": torch.zeros(count)}, path)
        files.append(
            {
                "environment": environment,
                "source_group": f"remote/{environment}/chunk_0",
                "source_path": f"remote/{environment}/chunk_0.pt",
                "destination_path": relative_path,
            }
        )
    manifest_path = tmp_path / "balanced_dataset_manifest.json"
    audit_path = tmp_path / "balanced_dataset_audit.json"
    manifest_path.write_text(json.dumps({"view_root": str(tmp_path), "max_rows_per_source": 4, "files": files}))

    with pytest.raises(ValueError, match="audit rejected"):
        audit_balanced_view(manifest_path, audit_path)

    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "rejected_structural_imbalance"
    assert audit["environment_summary"]["grf"]["accepted_row_cap"] == 3
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "rejected_structural_imbalance"
    assert manifest["structural_balance_passed"] is False

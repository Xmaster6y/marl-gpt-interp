"""Audit split integrity and predeclared held-out health gates for one SAE suite."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import hydra
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import evaluate_sae_health, file_sha256, write_run_manifest


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/audit_sae_suite", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = as_path(root, str(cfg.cache_dir))
    model_dir = as_path(root, str(cfg.model_dir))
    evaluation_dir = as_path(root, str(cfg.evaluation_dir))
    feature_dir = as_path(root, str(cfg.feature_dir))

    cache_manifest = json.loads((cache_dir / "manifest.json").read_text())
    metadata_path = cache_dir / cache_manifest["metadata"]["path"]
    if file_sha256(metadata_path) != cache_manifest["metadata"]["sha256"]:
        raise ValueError("activation metadata hash mismatch")
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line]
    group_splits: dict[str, set[str]] = defaultdict(set)
    rows_per_environment = Counter()
    groups_per_environment: dict[str, set[str]] = defaultdict(set)
    split_coverage: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        group_splits[row["trajectory_group"]].add(row["split"])
        rows_per_environment[row["environment"]] += 1
        groups_per_environment[row["environment"]].add(row["trajectory_group"])
        split_coverage[row["environment"]].add(row["split"])
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    domains = list(cache_manifest["environments"])
    structural_checks = {
        "no_group_leakage": not leaking_groups,
        "equal_environment_rows": len(set(rows_per_environment.values())) == 1,
        "minimum_groups": all(
            len(groups_per_environment[domain]) >= int(cfg.minimum_groups_per_environment) for domain in domains
        ),
        "complete_split_coverage": all(
            split_coverage[domain] == {"train", "validation", "test"} for domain in domains
        ),
        "audited_dataset_source_groups": cache_manifest.get("grouping_mode") == "dataset_source_group",
    }
    upstream_manifests = {
        "collection": json.loads((cache_dir / "run_manifest.json").read_text()),
        "training": json.loads((model_dir / "run_manifest.json").read_text()),
        "evaluation": json.loads((evaluation_dir / "run_manifest.json").read_text()),
        "features": json.loads((feature_dir / "run_manifest.json").read_text()),
    }
    structural_checks["upstream_runs_completed"] = all(
        manifest.get("status") == "completed" for manifest in upstream_manifests.values()
    )
    metrics = json.loads((evaluation_dir / "evaluation_metrics.json").read_text())
    model_spec = json.loads((model_dir / "model_spec.json").read_text())
    feature_summary = json.loads((feature_dir / "summary.json").read_text())
    health = evaluate_sae_health(
        metrics,
        domains,
        expected_l0=float(model_spec["model"]["k"]),
        max_normalized_mse=float(cfg.gates.max_normalized_mse),
        max_domain_normalized_mse=float(cfg.gates.max_domain_normalized_mse),
        max_dead_feature_fraction=float(cfg.gates.max_dead_feature_fraction),
        l0_tolerance=float(cfg.gates.l0_tolerance),
    )
    passed = all(structural_checks.values()) and health["passed"]
    payload = {
        "status": "passed" if passed else "failed",
        "strict": bool(cfg.strict),
        "structural_checks": structural_checks,
        "leaking_groups": leaking_groups,
        "rows_per_environment": dict(rows_per_environment),
        "groups_per_environment": {key: len(value) for key, value in groups_per_environment.items()},
        "split_coverage": {key: sorted(value) for key, value in split_coverage.items()},
        "health": health,
        "evaluation_metrics": metrics,
        "feature_summary": feature_summary,
    }
    write_json(output_dir / "suite_audit.json", payload)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status=payload["status"],
        artifacts={"audit": "suite_audit.json"},
        hashes={"activation_metadata": cache_manifest["metadata"]["sha256"]},
        split_manifest={split: sum(row["split"] == split for row in metadata) for split in ("train", "validation", "test")},
        environment_versions={"cache_schema": str(cache_manifest["format_version"])},
    )
    if bool(cfg.strict) and not passed:
        raise ValueError(f"SAE suite audit failed: {payload}")
    return payload


if __name__ == "__main__":
    main()

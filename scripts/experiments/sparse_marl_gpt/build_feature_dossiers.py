"""Rehydrate top and random-active SAE examples into interpretation dossiers."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from marl_gpt_interp.feature_dossiers import rehydrate_references
from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import file_sha256, write_run_manifest


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/build_feature_dossiers", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    cache_dir = as_path(root, str(cfg.cache_dir))
    feature_dir = as_path(root, str(cfg.feature_dir))
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text())
    source_files = {int(key): value for key, value in cache_manifest["source_files"].items()}
    feature_rows = [
        json.loads(line)
        for line in (feature_dir / "feature_summary.jsonl").read_text().splitlines()
        if line
    ]
    eligible = [row for row in feature_rows if float(row["activation_rate"]) >= float(cfg.minimum_activation_rate)]
    eligible.sort(key=lambda row: (-float(row["activation_rate"]), int(row["feature"])))
    selected = eligible[: int(cfg.maximum_features)]
    references = [
        example
        for row in selected
        for field in ("top_examples", "random_active_examples")
        for example in row.get(field, [])
    ]
    hydrated = rehydrate_references(references, source_files, history_length=int(cfg.history_length))
    dossier_path = output_dir / "feature_dossiers.jsonl"
    with dossier_path.open("w") as handle:
        for row in selected:
            dossier = dict(row)
            for field in ("top_examples", "random_active_examples"):
                dossier[field] = [
                    {
                        **example,
                        "source_context": hydrated.get(
                            (int(example["source_file_id"]), int(example["source_row_index"]))
                        ),
                    }
                    for example in row.get(field, [])
                    if example.get("source_file_id") is not None and example.get("source_row_index") is not None
                ]
            dossier["semantic_status"] = "unlabeled_requires_human_review"
            handle.write(json.dumps(dossier, sort_keys=True) + "\n")
    summary = {
        "eligible_features": len(eligible),
        "dossiers": len(selected),
        "rehydrated_source_rows": len(hydrated),
        "minimum_activation_rate": float(cfg.minimum_activation_rate),
        "warning": "Tensor summaries support interpretation but do not establish semantic or causal universality.",
    }
    write_json(output_dir / "summary.json", summary)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"dossiers": dossier_path.name, "summary": "summary.json"},
        hashes={"feature_summary": file_sha256(feature_dir / "feature_summary.jsonl")},
    )
    return summary


if __name__ == "__main__":
    main()

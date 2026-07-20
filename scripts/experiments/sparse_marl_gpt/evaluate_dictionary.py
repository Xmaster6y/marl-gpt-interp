"""Evaluate a trained sparse dictionary on held-out cached activations."""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import file_sha256, load_activation_cache, sparse_metrics, write_run_manifest
from scripts.experiments.sparse_marl_gpt.train_dictionary import build_model


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/evaluate_dictionary", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = as_path(root, str(cfg.model_dir))
    spec = json.loads((model_dir / "model_spec.json").read_text())
    tensors, metadata, cache_manifest = load_activation_cache(as_path(root, str(cfg.cache_dir)))
    x = tensors[spec["activation_location"]].float()
    domains = list(spec["domains"])
    labels = torch.tensor([{domain: i for i, domain in enumerate(domains)}[row["environment"]] for row in metadata])
    mask = torch.tensor([row["split"] == str(cfg.split) for row in metadata])
    model_cfg = DictConfig({"model": spec["model"]})
    model = build_model(model_cfg, int(spec["input_dim"]), domains)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        reconstruction, codes = model(x[mask], labels[mask])
    metrics = sparse_metrics(x[mask], reconstruction, codes)
    metrics["examples"] = int(mask.sum())
    selected_labels = labels[mask]
    for label, domain in enumerate(domains):
        domain_mask = selected_labels == label
        if domain_mask.any():
            domain_metrics = sparse_metrics(x[mask][domain_mask], reconstruction[domain_mask], codes[domain_mask])
            metrics.update({f"{domain}/{key}": value for key, value in domain_metrics.items()})
    write_json(output_dir / "evaluation_metrics.json", metrics)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"metrics": "evaluation_metrics.json"},
        hashes={
            "activation_cache": cache_manifest["shards"][0]["sha256"],
            "source_checkpoint": cache_manifest["checkpoint_sha256"],
            "sparse_checkpoint": file_sha256(model_dir / "model.pt"),
        },
        split_manifest={str(cfg.split): int(mask.sum())},
        environment_versions={"cache_schema": str(cache_manifest["format_version"])},
    )
    return metrics


if __name__ == "__main__":
    main()

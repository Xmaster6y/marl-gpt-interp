"""Train a flat, independent, or domain-lattice sparse dictionary from a cache."""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import (
    DomainLatticeSAE,
    IndependentSAEs,
    SparseAutoencoder,
    file_sha256,
    load_activation_cache,
    sparse_metrics,
    train_sparse_model,
    write_run_manifest,
)


def build_model(cfg, input_dim: int, domains: list[str]):
    method = str(cfg.model.method)
    if method == "flat":
        return SparseAutoencoder(input_dim, int(cfg.model.width), int(cfg.model.k), str(cfg.model.activation))
    if method == "independent":
        return IndependentSAEs(input_dim, domains, int(cfg.model.width_per_domain), int(cfg.model.k))
    if method == "lattice":
        return DomainLatticeSAE(
            input_dim, domains, int(cfg.model.width_per_support), int(cfg.model.k), str(cfg.model.activation)
        )
    raise ValueError(f"Unknown method {method!r}")


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/train_dictionary", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors, metadata, cache_manifest = load_activation_cache(as_path(root, str(cfg.cache_dir)))
    location = str(cfg.activation_location)
    x = tensors[location].float()
    domains = list(cache_manifest["environments"])
    domain_to_label = {domain: index for index, domain in enumerate(domains)}
    labels = torch.tensor([domain_to_label[row["environment"]] for row in metadata])
    train_mask = torch.tensor([row["split"] == "train" for row in metadata])
    model = build_model(cfg, x.shape[1], domains)
    losses = train_sparse_model(
        model,
        x[train_mask],
        labels[train_mask],
        steps=int(cfg.training.steps),
        batch_size=int(cfg.training.batch_size),
        learning_rate=float(cfg.training.learning_rate),
        seed=int(cfg.seed),
    )
    model.eval()
    with torch.no_grad():
        reconstruction, codes = model(x[train_mask], labels[train_mask])
    metrics = sparse_metrics(x[train_mask], reconstruction, codes)
    torch.save(model.state_dict(), output_dir / "model.pt")
    spec = {
        "model": to_plain_config(cfg.model),
        "input_dim": x.shape[1],
        "domains": domains,
        "activation_location": location,
        "cache_manifest_sha256": cache_manifest["shards"][0]["sha256"],
    }
    (output_dir / "model_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    write_json(output_dir / "training_metrics.json", {**metrics, "final_loss": losses[-1]})
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"model": "model.pt", "spec": "model_spec.json", "metrics": "training_metrics.json"},
        hashes={
            "activation_cache": cache_manifest["shards"][0]["sha256"],
            "source_checkpoint": cache_manifest["checkpoint_sha256"],
            "sparse_checkpoint": file_sha256(output_dir / "model.pt"),
        },
        split_manifest={
            split: sum(row["split"] == split for row in metadata) for split in ("train", "validation", "test")
        },
        environment_versions={"cache_schema": str(cache_manifest["format_version"])},
    )
    return metrics


if __name__ == "__main__":
    main()

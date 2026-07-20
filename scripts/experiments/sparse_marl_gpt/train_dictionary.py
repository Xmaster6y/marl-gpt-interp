"""Train a flat, independent, or domain-lattice sparse dictionary from a cache."""

from __future__ import annotations

import json
from importlib.metadata import version

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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
from marl_gpt_interp.sae_training import (
    DictionaryLearningTopK,
    apply_activation_preprocessing,
    fit_activation_preprocessing,
    train_topk_sae,
)


def build_model(cfg, input_dim: int, domains: list[str]):
    method = str(cfg.model.method)
    if method == "flat":
        if str(OmegaConf.select(cfg, "model.backend", default="local")) == "dictionary_learning":
            return DictionaryLearningTopK(input_dim, int(cfg.model.width), int(cfg.model.k))
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
    preprocessing = fit_activation_preprocessing(
        x[train_mask],
        labels[train_mask],
        domains,
        str(OmegaConf.select(cfg, "preprocessing.mode", default="natural")),
    )
    model_x = apply_activation_preprocessing(x, labels, preprocessing)
    backend = str(OmegaConf.select(cfg, "model.backend", default="local"))
    model = build_model(cfg, x.shape[1], domains)
    if backend == "dictionary_learning":
        if str(cfg.model.method) != "flat":
            raise ValueError("dictionary_learning backend currently supports the flat TopK baseline only")
        validation_mask = torch.tensor([row["split"] == "validation" for row in metadata])
        if not validation_mask.any():
            raise ValueError("Training requires a non-empty leakage-safe validation split")
        resume_value = OmegaConf.select(cfg, "training.resume_from", default=None)
        result = train_topk_sae(
            model_x[train_mask],
            labels[train_mask],
            domains,
            model_x[validation_mask],
            labels[validation_mask],
            output_dir=output_dir,
            width=int(cfg.model.width),
            k=int(cfg.model.k),
            steps=int(cfg.training.steps),
            batch_size=int(cfg.training.batch_size),
            learning_rate=(
                None
                if OmegaConf.select(cfg, "training.learning_rate", default=None) is None
                else float(cfg.training.learning_rate)
            ),
            warmup_steps=int(cfg.training.warmup_steps),
            decay_start=(
                None
                if OmegaConf.select(cfg, "training.decay_start", default=None) is None
                else int(cfg.training.decay_start)
            ),
            auxk_alpha=float(cfg.training.auxk_alpha),
            log_every=int(cfg.observability.log_every),
            checkpoint_every=int(cfg.observability.checkpoint_every),
            device=str(cfg.device),
            seed=int(cfg.seed),
            resume_from=None if resume_value is None else as_path(root, str(resume_value)),
            wandb_cfg={
                **to_plain_config(cfg.observability.wandb),
                "config": to_plain_config(cfg),
            },
        )
        model = result.model.cpu()
        metrics = result.metrics
        final_loss = metrics["train/loss"]
    else:
        losses = train_sparse_model(
            model,
            model_x[train_mask],
            labels[train_mask],
            steps=int(cfg.training.steps),
            batch_size=int(cfg.training.batch_size),
            learning_rate=float(cfg.training.learning_rate),
            seed=int(cfg.seed),
        )
        final_loss = losses[-1]
        metrics = {}
    model.eval()
    if backend != "dictionary_learning":
        with torch.no_grad():
            reconstruction, codes = model(model_x[train_mask], labels[train_mask])
        metrics = {
            **metrics,
            **{
                f"train_final/{key}": value
                for key, value in sparse_metrics(model_x[train_mask], reconstruction, codes).items()
            },
        }
    torch.save(model.state_dict(), output_dir / "model.pt")
    spec = {
        "model": to_plain_config(cfg.model),
        "input_dim": x.shape[1],
        "domains": domains,
        "activation_location": location,
        "backend": backend,
        "cache_manifest_sha256": cache_manifest["shards"][0]["sha256"],
        "preprocessing": preprocessing,
    }
    (output_dir / "model_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    write_json(output_dir / "training_metrics.json", {**metrics, "final_loss": final_loss})
    artifacts = {"model": "model.pt", "spec": "model_spec.json", "metrics": "training_metrics.json"}
    environment_versions = {"cache_schema": str(cache_manifest["format_version"])}
    if backend == "dictionary_learning":
        artifacts.update({"metric_history": "training_metrics.jsonl", "latest_checkpoint": "checkpoints/latest.pt"})
        environment_versions["dictionary_learning"] = version("dictionary-learning")
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts=artifacts,
        hashes={
            "activation_cache": cache_manifest["shards"][0]["sha256"],
            "source_checkpoint": cache_manifest["checkpoint_sha256"],
            "sparse_checkpoint": file_sha256(output_dir / "model.pt"),
        },
        split_manifest={
            split: sum(row["split"] == split for row in metadata) for split in ("train", "validation", "test")
        },
        environment_versions=environment_versions,
    )
    return metrics


if __name__ == "__main__":
    main()

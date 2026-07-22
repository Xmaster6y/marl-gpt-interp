"""Train one full actor or critic cross-layer transcoder."""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.clt import CLTConfig
from marl_gpt_interp.clt_data import load_corpus_manifest
from marl_gpt_interp.clt_training import train_clt
from marl_gpt_interp.experiment_io import file_sha256, write_run_manifest
from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config


@hydra.main(config_path="../../../configs/experiments/circuit_tracing/train_clt", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    corpus_dir = as_path(root, str(cfg.corpus_dir))
    corpus_manifest = load_corpus_manifest(corpus_dir)
    path_layers = int(corpus_manifest["model"]["path_layers"])
    widths = tuple(int(value) for value in cfg.model.features_per_layer)
    if len(widths) != path_layers:
        raise ValueError(f"features_per_layer must contain {path_layers} entries")
    clt_config = CLTConfig(
        input_dim=int(corpus_manifest["model"]["d_model"]),
        features_per_layer=widths,
        initial_threshold=float(cfg.model.initial_threshold),
        jump_relu_bandwidth=float(cfg.model.jump_relu_bandwidth),
    )
    resume_value = cfg.training.get("resume_from")
    result = train_clt(
        corpus_dir,
        branch=str(cfg.branch),
        config=clt_config,
        output_dir=output_dir,
        steps=int(cfg.training.steps),
        batch_size=int(cfg.training.batch_size),
        learning_rate=float(cfg.training.learning_rate),
        sparsity_coefficient=float(cfg.training.sparsity_coefficient),
        sparsity_tanh_scale=float(cfg.training.sparsity_tanh_scale),
        log_every=int(cfg.observability.log_every),
        checkpoint_every=int(cfg.observability.checkpoint_every),
        validation_rows=int(cfg.evaluation.validation_rows),
        gradient_clip=float(cfg.training.gradient_clip),
        device=str(cfg.device),
        seed=int(cfg.seed),
        resume_from=None if resume_value is None else as_path(root, str(resume_value)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), output_dir / "model.pt")
    spec = {
        "format": "marl-gpt-cross-layer-transcoder",
        "format_version": 1,
        "branch": str(cfg.branch),
        "clt": clt_config.to_dict(),
        "source_checkpoint_sha256": corpus_manifest["checkpoint_sha256"],
        "corpus_manifest_sha256": file_sha256(corpus_dir / "manifest.json"),
        "preprocessing": "natural-residual-stream",
    }
    (output_dir / "model_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (output_dir / "final_metrics.json").write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"model": "model.pt", "spec": "model_spec.json", "metrics": "final_metrics.json"},
        hashes={
            "corpus_manifest": spec["corpus_manifest_sha256"],
            "source_checkpoint": spec["source_checkpoint_sha256"],
            "clt": file_sha256(output_dir / "model.pt"),
        },
        environment_versions={"clt_schema": "1"},
    )
    return result.metrics


if __name__ == "__main__":
    main()

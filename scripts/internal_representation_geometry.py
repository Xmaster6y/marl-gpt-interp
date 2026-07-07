"""Analyze within-env and asymmetric MARL-GPT representation geometry."""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.marl_gpt_tools import (
    activation_centroid_cosine_similarity_rows,
    activation_pairwise_cosine_distance_rows,
    activation_hooks,
    activation_subspace_similarity_rows,
    asymmetric_subspace_rows,
    as_path,
    build_loader,
    enabled_envs,
    env_labels_for_batch,
    inspect_dataset_files,
    load_model,
    load_torch,
    marl_gpt_cwd,
    pooled_activations,
    repo_root,
    representation_proximity_rows,
    representation_separation_rows,
    resolve_dataset_config,
    self_subspace_similarity_rows,
    to_plain_config,
    write_csv,
    write_json,
)


def collect_natural_activations(root: Path, dataset_config: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    torch = load_torch()
    torch.manual_seed(int(OmegaConf.select(cfg, "seed", default=0)))

    loader = build_loader(root, dataset_config, cfg)
    iterator = iter(loader)
    activation_tables: dict[str, list[Any]] = defaultdict(list)
    activation_labels = []
    behavior_rows = []

    with marl_gpt_cwd(root):
        model, _model_config = load_model(root, cfg)

    hooks = []
    try:
        for batch_index in range(int(cfg.num_batches)):
            with marl_gpt_cwd(root):
                batch_obs, _target, _mask_target, _batch_obs_next, _batch_info = next(iterator)
            env_labels = env_labels_for_batch(loader)
            if env_labels.shape[0] != batch_obs["obs"].shape[0]:
                raise RuntimeError(
                    f"Label count {env_labels.shape[0]} does not match batch size {batch_obs['obs'].shape[0]}"
                )

            captured: dict[str, Any] = {}
            hooks = activation_hooks(model, captured)
            with marl_gpt_cwd(root), torch.no_grad():
                act_logits, val_logits, _loss, _info = model(batch_obs)
            for active_hook in hooks:
                active_hook.remove()
            hooks = []

            for name, features in pooled_activations(captured).items():
                activation_tables[name].append(features.detach().cpu())
            activation_labels.append(env_labels.detach().cpu())
            behavior_rows.append(
                {
                    "batch": batch_index,
                    "condition": "natural",
                    "mean_entropy": float(
                        torch.distributions.Categorical(logits=act_logits).entropy().mean().item()
                    ),
                    "mean_value_logit": float(val_logits.detach().float().mean().item()),
                }
            )
            logger.info(f"Collected natural activation batch {batch_index + 1}/{int(cfg.num_batches)}")
    finally:
        for active_hook in hooks:
            active_hook.remove()

    return {
        "activation_features": {name: torch.cat(chunks, dim=0) for name, chunks in activation_tables.items()},
        "activation_labels": torch.cat(activation_labels, dim=0) if activation_labels else None,
        "behavior_rows": behavior_rows,
    }


def tdhook_status() -> dict[str, Any]:
    spec = importlib.util.find_spec("tdhook")
    return {
        "available": spec is not None,
        "used": False,
        "note": "This run uses built-in PCA subspace containment; tdhook can be wired in once its API is available.",
    }


def main(cfg: DictConfig) -> dict[str, Any]:
    script_cfg = cfg.internal_representation_geometry
    root = repo_root()
    output_dir = as_path(root, str(script_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = resolve_dataset_config(root, script_cfg)
    requested_envs = list(OmegaConf.select(script_cfg, "envs", default=["smac", "pogema", "grf"]))
    active_envs = enabled_envs(dataset_config, requested_envs)
    dataset_config = {env: dataset_config[env] for env in active_envs}

    inspection = inspect_dataset_files(root, dataset_config, script_cfg)
    write_json(output_dir / "dataset_inspection.json", inspection)

    collected = collect_natural_activations(root, dataset_config, script_cfg)
    proximity_rows = representation_proximity_rows(
        collected["activation_features"],
        collected["activation_labels"],
        cfg=script_cfg,
    )
    separation_rows = representation_separation_rows(
        collected["activation_features"],
        collected["activation_labels"],
        cfg=script_cfg,
    )
    centroid_cosine_rows = activation_centroid_cosine_similarity_rows(
        collected["activation_features"],
        collected["activation_labels"],
        cfg=script_cfg,
    )
    pairwise_cosine_rows = activation_pairwise_cosine_distance_rows(
        collected["activation_features"],
        collected["activation_labels"],
        cfg=script_cfg,
    )
    asymmetric_rows = asymmetric_subspace_rows(
        collected["activation_features"],
        collected["activation_labels"],
        cfg=script_cfg,
    )
    cka_rows = activation_subspace_similarity_rows(
        collected["activation_features"],
        collected["activation_labels"],
    )
    self_cka_rows = self_subspace_similarity_rows(
        collected["activation_features"],
        collected["activation_labels"],
    )
    status = tdhook_status()

    write_json(output_dir / "internal_representation_proximity.json", proximity_rows)
    write_csv(output_dir / "internal_representation_proximity.csv", proximity_rows)
    write_json(output_dir / "representation_separation.json", separation_rows)
    write_csv(output_dir / "representation_separation.csv", separation_rows)
    write_json(output_dir / "activation_centroid_cosine_similarity.json", centroid_cosine_rows)
    write_csv(output_dir / "activation_centroid_cosine_similarity.csv", centroid_cosine_rows)
    write_json(output_dir / "activation_pairwise_cosine_distance.json", pairwise_cosine_rows)
    write_csv(output_dir / "activation_pairwise_cosine_distance.csv", pairwise_cosine_rows)
    write_json(output_dir / "asymmetric_representation_analysis.json", asymmetric_rows)
    write_csv(output_dir / "asymmetric_representation_analysis.csv", asymmetric_rows)
    write_json(output_dir / "activation_subspace_similarity.json", cka_rows)
    write_csv(output_dir / "activation_subspace_similarity.csv", cka_rows)
    write_json(output_dir / "self_subspace_similarity.json", self_cka_rows)
    write_csv(output_dir / "self_subspace_similarity.csv", self_cka_rows)
    write_csv(output_dir / "natural_behavior.csv", collected["behavior_rows"])
    write_json(
        output_dir / "summary.json",
        {
            "active_envs": active_envs,
            "num_activation_examples": int(collected["activation_labels"].shape[0])
            if collected["activation_labels"] is not None
            else 0,
            "proximity_rows": len(proximity_rows),
            "separation_rows": len(separation_rows),
            "centroid_cosine_rows": len(centroid_cosine_rows),
            "pairwise_cosine_rows": len(pairwise_cosine_rows),
            "asymmetric_rows": len(asymmetric_rows),
            "cka_rows": len(cka_rows),
            "self_cka_rows": len(self_cka_rows),
            "behavior_rows": len(collected["behavior_rows"]),
            "tdhook": status,
            "config": to_plain_config(cfg),
        },
    )
    logger.info(f"Wrote internal representation geometry outputs to {output_dir}")
    return {
        "proximity_rows": proximity_rows,
        "separation_rows": separation_rows,
        "centroid_cosine_rows": centroid_cosine_rows,
        "pairwise_cosine_rows": pairwise_cosine_rows,
        "asymmetric_rows": asymmetric_rows,
        "cka_rows": cka_rows,
        "self_cka_rows": self_cka_rows,
    }

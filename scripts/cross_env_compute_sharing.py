"""Analyze shared versus environment-specific MARL-GPT computation under natural inference."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.marl_gpt_tools import (
    activation_direction_rows,
    activation_hooks,
    activation_subspace_similarity_rows,
    as_path,
    build_loader,
    collect_parameter_gradients_by_env,
    enabled_envs,
    env_labels_for_batch,
    feature_groups,
    inspect_dataset_files,
    load_model,
    load_torch,
    marl_gpt_cwd,
    parameter_gradient_cosine_rows,
    pooled_activations,
    repo_root,
    resolve_dataset_config,
    to_plain_config,
    train_linear_probe,
    write_csv,
    write_json,
)


def collect_natural_batches(root: Path, dataset_config: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    torch = load_torch()
    torch.manual_seed(int(OmegaConf.select(cfg, "seed", default=0)))

    loader = build_loader(root, dataset_config, cfg)
    iterator = iter(loader)
    input_tables: dict[str, list[Any]] = defaultdict(list)
    input_labels = []
    activation_tables: dict[str, list[Any]] = defaultdict(list)
    activation_labels = []
    behavior_rows = []
    parameter_rows = []
    gradient_sums: dict[str, dict[int, Any]] = defaultdict(dict)
    gradient_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with marl_gpt_cwd(root):
        model, _model_config = load_model(root, cfg)

    hooks = []
    try:
        for batch_index in range(int(cfg.num_batches)):
            with marl_gpt_cwd(root):
                batch_obs, target, mask_target, batch_obs_next, batch_info = next(iterator)
            env_labels = env_labels_for_batch(loader)
            if env_labels.shape[0] != batch_obs["obs"].shape[0]:
                raise RuntimeError(
                    f"Label count {env_labels.shape[0]} does not match batch size {batch_obs['obs'].shape[0]}"
                )

            for name, features in feature_groups(batch_obs, cfg).items():
                input_tables[name].append(features.detach().cpu())
            input_labels.append(env_labels.detach().cpu())

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

            parameter_cfg = OmegaConf.select(cfg, "parameter_sensitivity", default={})
            max_gradient_batches = int(OmegaConf.select(parameter_cfg, "max_batches", default=1))
            if bool(OmegaConf.select(parameter_cfg, "enabled", default=False)) and batch_index < max_gradient_batches:
                collect_parameter_gradients_by_env(
                    root=root,
                    model=model,
                    batch_index=batch_index,
                    batch_obs=batch_obs,
                    target=target,
                    mask_target=mask_target,
                    batch_obs_next=batch_obs_next,
                    batch_info=batch_info,
                    env_labels=env_labels,
                    parameter_rows=parameter_rows,
                    gradient_sums=gradient_sums,
                    gradient_counts=gradient_counts,
                )

            logger.info(f"Collected natural batch {batch_index + 1}/{int(cfg.num_batches)}")
    finally:
        for active_hook in hooks:
            active_hook.remove()

    return {
        "input_features": {name: torch.cat(chunks, dim=0) for name, chunks in input_tables.items()},
        "input_labels": torch.cat(input_labels, dim=0) if input_labels else None,
        "activation_features": {name: torch.cat(chunks, dim=0) for name, chunks in activation_tables.items()},
        "activation_labels": torch.cat(activation_labels, dim=0) if activation_labels else None,
        "behavior_rows": behavior_rows,
        "parameter_rows": parameter_rows,
        "gradient_sums": gradient_sums,
        "gradient_counts": gradient_counts,
    }


def input_probe_rows(collected: dict[str, Any], cfg: DictConfig) -> list[dict[str, Any]]:
    rows = []
    labels = collected.get("input_labels")
    if labels is None:
        return rows
    for name, features in collected["input_features"].items():
        result = train_linear_probe(features.to(str(cfg.device)), labels.to(str(cfg.device)), cfg)
        result.update({"source": "input", "feature": name, "condition": "natural"})
        rows.append(result)
    return rows


def main(cfg: DictConfig) -> dict[str, Any]:
    script_cfg = cfg.cross_env_compute_sharing
    root = repo_root()
    output_dir = as_path(root, str(script_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = resolve_dataset_config(root, script_cfg)
    requested_envs = list(OmegaConf.select(script_cfg, "envs", default=["smac", "pogema", "grf"]))
    active_envs = enabled_envs(dataset_config, requested_envs)
    dataset_config = {env: dataset_config[env] for env in active_envs}

    inspection = inspect_dataset_files(root, dataset_config, script_cfg)
    write_json(output_dir / "dataset_inspection.json", inspection)

    collected = collect_natural_batches(root, dataset_config, script_cfg)
    probe_rows = input_probe_rows(collected, script_cfg)
    direction_rows = activation_direction_rows(
        collected["activation_features"],
        collected["activation_labels"],
        cfg=script_cfg,
    )
    subspace_rows = activation_subspace_similarity_rows(
        collected["activation_features"],
        collected["activation_labels"],
    )
    parameter_gradient_rows = parameter_gradient_cosine_rows(
        collected["gradient_sums"],
        collected["gradient_counts"],
    )

    write_json(output_dir / "input_probe_results.json", probe_rows)
    write_csv(output_dir / "input_probe_results.csv", probe_rows)
    write_json(output_dir / "activation_geometry.json", direction_rows)
    write_csv(output_dir / "activation_geometry.csv", direction_rows)
    write_json(output_dir / "activation_subspace_similarity.json", subspace_rows)
    write_csv(output_dir / "activation_subspace_similarity.csv", subspace_rows)
    write_csv(output_dir / "natural_behavior.csv", collected["behavior_rows"])
    write_json(output_dir / "parameter_gradients.json", collected["parameter_rows"])
    write_csv(output_dir / "parameter_gradients.csv", collected["parameter_rows"])
    write_json(output_dir / "parameter_gradient_overlap.json", parameter_gradient_rows)
    write_csv(output_dir / "parameter_gradient_overlap.csv", parameter_gradient_rows)
    write_json(
        output_dir / "summary.json",
        {
            "active_envs": active_envs,
            "num_input_examples": int(collected["input_labels"].shape[0])
            if collected["input_labels"] is not None
            else 0,
            "num_activation_examples": int(collected["activation_labels"].shape[0])
            if collected["activation_labels"] is not None
            else 0,
            "input_probe_rows": len(probe_rows),
            "activation_geometry_rows": len(direction_rows),
            "activation_subspace_similarity_rows": len(subspace_rows),
            "behavior_rows": len(collected["behavior_rows"]),
            "parameter_gradient_rows": len(collected["parameter_rows"]),
            "parameter_gradient_overlap_rows": len(parameter_gradient_rows),
            "config": to_plain_config(cfg),
        },
    )
    logger.info(f"Wrote cross-env compute-sharing outputs to {output_dir}")
    return {
        "input_probe_rows": probe_rows,
        "activation_geometry_rows": direction_rows,
        "activation_subspace_similarity_rows": subspace_rows,
        "parameter_gradient_rows": collected["parameter_rows"],
        "parameter_gradient_overlap_rows": parameter_gradient_rows,
    }

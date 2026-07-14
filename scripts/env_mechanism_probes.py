"""Run environment identity probes and counterfactual-token analyses for MARL-GPT."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.marl_gpt_tools import (
    activation_direction_rows,
    activation_hooks,
    as_path,
    build_loader,
    condition_mask,
    collect_parameter_gradients_by_env,
    copy_obs,
    enabled_envs,
    entropy,
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


def _wrong_prompt_ids(true_env_ids: Any) -> Any:
    torch = load_torch()
    choices = torch.tensor([1, 2, 3], device=true_env_ids.device)
    wrong = []
    for label in true_env_ids.tolist():
        candidates = choices[choices != int(label)]
        wrong.append(int(candidates[random.randrange(len(candidates))].item()))
    return torch.tensor(wrong, dtype=torch.long, device=true_env_ids.device)


def collect_batches(root: Path, dataset_config: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    torch = load_torch()
    torch.manual_seed(int(OmegaConf.select(cfg, "seed", default=0)))
    random.seed(int(OmegaConf.select(cfg, "seed", default=0)))

    loader = build_loader(root, dataset_config, cfg)
    iterator = iter(loader)
    true_feature_tables: dict[str, list[Any]] = defaultdict(list)
    true_labels = []
    activation_tables: dict[str, list[Any]] = defaultdict(list)
    activation_true_labels = []
    activation_prompted_labels = []
    activation_conditions = []
    behavior_rows = []
    parameter_rows = []
    gradient_sums: dict[str, dict[int, Any]] = defaultdict(dict)
    gradient_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    model = None
    model_config = None
    hooks = []
    if bool(OmegaConf.select(cfg, "run_model", default=False)):
        with marl_gpt_cwd(root):
            model, model_config = load_model(root, cfg)
            last_attr_pos = model_config.n_attr - 1
    else:
        last_attr_pos = None

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
                true_feature_tables[name].append(features.detach().cpu())
            true_labels.append(env_labels.detach().cpu())

            if model is None:
                continue

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

            prompt_sets = {"correct": env_labels, "wrong": _wrong_prompt_ids(env_labels)}
            for env_id in (1, 2, 3):
                prompt_sets[f"prompt_{env_id}"] = torch.full_like(env_labels, env_id)

            correct_logits = None
            for condition, prompt_ids in prompt_sets.items():
                prompted_obs = copy_obs(batch_obs)
                prompted_obs["obs"][:, -1] = prompt_ids
                if last_attr_pos is not None:
                    prompted_obs["attr_pos"][:, -1] = last_attr_pos

                captured: dict[str, Any] = {}
                hooks = activation_hooks(model, captured)
                with marl_gpt_cwd(root), torch.no_grad():
                    act_logits, val_logits, _loss, _info = model(prompted_obs)
                for active_hook in hooks:
                    active_hook.remove()
                hooks = []

                if condition == "correct":
                    correct_logits = act_logits.detach()
                for name, features in pooled_activations(captured).items():
                    activation_tables[name].append(features.detach().cpu())
                activation_true_labels.append(env_labels.detach().cpu())
                activation_prompted_labels.append(prompt_ids.detach().cpu())
                activation_conditions.extend([condition] * int(env_labels.shape[0]))

                if correct_logits is not None:
                    delta = (act_logits - correct_logits).detach()
                    behavior_rows.append(
                        {
                            "batch": batch_index,
                            "condition": condition,
                            "mean_abs_action_logit_delta": float(delta.abs().mean().item()),
                            "mean_entropy": float(entropy(act_logits).mean().item()),
                            "selected_action_change_fraction": float(
                                (act_logits.argmax(dim=1) != correct_logits.argmax(dim=1)).float().mean().item()
                            ),
                            "mean_value_logit": float(val_logits.detach().float().mean().item()),
                        }
                    )
            logger.info(f"Collected batch {batch_index + 1}/{int(cfg.num_batches)}")
    finally:
        for active_hook in hooks:
            active_hook.remove()

    return {
        "input_features": {name: torch.cat(chunks, dim=0) for name, chunks in true_feature_tables.items()},
        "input_true_labels": torch.cat(true_labels, dim=0) if true_labels else None,
        "activation_features": {name: torch.cat(chunks, dim=0) for name, chunks in activation_tables.items()},
        "activation_true_labels": torch.cat(activation_true_labels, dim=0) if activation_true_labels else None,
        "activation_prompted_labels": torch.cat(activation_prompted_labels, dim=0)
        if activation_prompted_labels
        else None,
        "activation_conditions": activation_conditions,
        "behavior_rows": behavior_rows,
        "parameter_rows": parameter_rows,
        "gradient_sums": gradient_sums,
        "gradient_counts": gradient_counts,
    }


def run_probes(collected: dict[str, Any], cfg: DictConfig) -> list[dict[str, Any]]:
    rows = []
    activation_probe_conditions = list(OmegaConf.select(cfg, "activation_probe_conditions", default=[]))
    activation_probe_label_targets = set(
        OmegaConf.select(cfg, "activation_probe_label_targets", default=["true_env", "prompted_env"])
    )
    activation_mask = condition_mask(collected.get("activation_conditions", []), activation_probe_conditions)
    probe_specs = [("input", "input_features", "input_true_labels")]
    if "true_env" in activation_probe_label_targets:
        probe_specs.append(("activation_true_env", "activation_features", "activation_true_labels"))
    if "prompted_env" in activation_probe_label_targets:
        probe_specs.append(("activation_prompted_env", "activation_features", "activation_prompted_labels"))
    for source, feature_key, label_key in probe_specs:
        labels = collected.get(label_key)
        if labels is None:
            continue
        for name, features in collected[feature_key].items():
            probe_features = features
            probe_labels = labels
            condition_label = "all"
            if feature_key == "activation_features" and activation_mask is not None:
                probe_features = features[activation_mask]
                probe_labels = labels[activation_mask]
                condition_label = ",".join(activation_probe_conditions)
            result = train_linear_probe(
                probe_features.to(str(cfg.device)),
                probe_labels.to(str(cfg.device)),
                cfg,
            )
            result.update({"source": source, "feature": name, "condition": condition_label})
            rows.append(result)
    return rows


@hydra.main(config_path="../configs/env_mechanism_probes", version_base=None)
def main(cfg: DictConfig) -> dict[str, Any]:
    script_cfg = cfg
    root = repo_root()
    output_dir = as_path(root, str(script_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = resolve_dataset_config(root, script_cfg)
    requested_envs = list(OmegaConf.select(script_cfg, "envs", default=["smac", "pogema", "grf"]))
    active_envs = enabled_envs(dataset_config, requested_envs)
    dataset_config = {env: dataset_config[env] for env in active_envs}

    inspection = inspect_dataset_files(root, dataset_config, script_cfg)
    write_json(output_dir / "dataset_inspection.json", inspection)

    collected = collect_batches(root, dataset_config, script_cfg)
    probe_rows = run_probes(collected, script_cfg)
    direction_rows = activation_direction_rows(
        collected["activation_features"],
        collected["activation_true_labels"],
        cfg=script_cfg,
        conditions=collected["activation_conditions"],
        requested_conditions=list(OmegaConf.select(script_cfg, "direction_conditions", default=["wrong"])),
    )
    parameter_gradient_rows = parameter_gradient_cosine_rows(
        collected["gradient_sums"],
        collected["gradient_counts"],
    )
    write_json(output_dir / "probe_results.json", probe_rows)
    write_csv(output_dir / "probe_results.csv", probe_rows)
    write_json(output_dir / "activation_directions.json", direction_rows)
    write_csv(output_dir / "activation_directions.csv", direction_rows)
    write_csv(output_dir / "token_swap_behavior.csv", collected["behavior_rows"])
    write_json(output_dir / "parameter_sensitivity.json", collected["parameter_rows"])
    write_csv(output_dir / "parameter_sensitivity.csv", collected["parameter_rows"])
    write_json(output_dir / "parameter_gradient_cosines.json", parameter_gradient_rows)
    write_csv(output_dir / "parameter_gradient_cosines.csv", parameter_gradient_rows)
    write_json(
        output_dir / "summary.json",
        {
            "active_envs": active_envs,
            "num_input_examples": int(collected["input_true_labels"].shape[0])
            if collected["input_true_labels"] is not None
            else 0,
            "num_activation_examples": int(collected["activation_true_labels"].shape[0])
            if collected["activation_true_labels"] is not None
            else 0,
            "activation_condition_counts": {
                condition: collected["activation_conditions"].count(condition)
                for condition in sorted(set(collected["activation_conditions"]))
            },
            "probe_rows": len(probe_rows),
            "direction_rows": len(direction_rows),
            "behavior_rows": len(collected["behavior_rows"]),
            "parameter_sensitivity_rows": len(collected["parameter_rows"]),
            "parameter_gradient_cosine_rows": len(parameter_gradient_rows),
            "config": to_plain_config(cfg),
        },
    )
    logger.info(f"Wrote environment mechanism probe outputs to {output_dir}")
    return {
        "probe_rows": probe_rows,
        "direction_rows": direction_rows,
        "behavior_rows": collected["behavior_rows"],
        "parameter_rows": collected["parameter_rows"],
        "parameter_gradient_rows": parameter_gradient_rows,
    }


if __name__ == "__main__":
    main()

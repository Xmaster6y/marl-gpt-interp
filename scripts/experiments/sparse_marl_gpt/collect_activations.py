"""Collect tensor-only MARL-GPT activation caches for sparse feature training."""

from __future__ import annotations

from collections import defaultdict

import hydra
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import (
    ID_TO_ENV,
    activation_hooks,
    as_path,
    build_loader,
    enabled_envs,
    env_labels_for_batch,
    load_model,
    load_torch,
    marl_gpt_cwd,
    repo_root,
    resolve_dataset_config,
    to_plain_config,
)
from marl_gpt_interp.sparse_features import file_sha256, grouped_split, write_activation_cache, write_run_manifest


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/collect_activations", version_base=None)
def main(cfg: DictConfig) -> dict:
    torch = load_torch()
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    dataset_config = resolve_dataset_config(root, cfg)
    active_envs = enabled_envs(dataset_config, list(cfg.envs))
    dataset_config = {env: dataset_config[env] for env in active_envs}
    loader = build_loader(root, dataset_config, cfg)
    with marl_gpt_cwd(root):
        model, model_config = load_model(root, cfg)
    requested = set(str(value) for value in cfg.activation_locations)
    checkpoint_sha256 = file_sha256(as_path(root, str(cfg.checkpoint)))
    preprocessing_identity = "marl-gpt-native-loader-v1"
    tensors: dict[str, list] = defaultdict(list)
    metadata = []
    iterator = iter(loader)
    hooks = []
    try:
        for batch_index in range(int(cfg.num_batches)):
            with marl_gpt_cwd(root):
                batch_obs, _target, _mask, _next_obs, batch_info = next(iterator)
            labels = env_labels_for_batch(loader)
            captured = {}
            hooks = activation_hooks(model, captured)
            with torch.no_grad(), marl_gpt_cwd(root):
                model(batch_obs)
            for hook in hooks:
                hook.remove()
            hooks = []
            selected = {f"{name}:final": value[:, -1, :].detach().cpu() for name, value in captured.items()}
            for location in requested:
                if location not in selected:
                    raise KeyError(f"Unknown activation location {location!r}; available: {sorted(selected)}")
                tensors[location].append(selected[location])
            group_field = str(cfg.get("trajectory_group_field", "trajectory_id"))
            raw_groups = batch_info.get(group_field)
            if raw_groups is not None:
                group_values = raw_groups.detach().cpu().reshape(-1).tolist()
                if len(group_values) != len(labels):
                    raise ValueError(f"{group_field} must have one value per sample")
            elif str(cfg.grouping_mode) == "batch_schema_smoke":
                group_values = [f"batch-{batch_index:06d}"] * len(labels)
            else:
                raise ValueError(
                    f"Claim-bearing collection requires batch_info[{group_field!r}]; "
                    "batch grouping is allowed only for a schema smoke"
                )
            for sample_index, label in enumerate(labels.tolist()):
                domain = ID_TO_ENV.get(int(label), str(label))
                group = f"{domain}:{group_values[sample_index]}"
                metadata.append(
                    {
                        "environment": domain,
                        "environment_label": int(label),
                        "trajectory_group": group,
                        "grouping_mode": str(cfg.grouping_mode),
                        "sample_index": len(metadata),
                        "batch_index": batch_index,
                        "batch_sample_index": sample_index,
                        "activation_location": sorted(requested),
                        "token_selector": "final",
                        "checkpoint_sha256": checkpoint_sha256,
                        "preprocessing_identity": preprocessing_identity,
                    }
                )
    finally:
        for hook in hooks:
            hook.remove()
    splits = grouped_split([row["trajectory_group"] for row in metadata], seed=int(cfg.seed))
    for row, split in zip(metadata, splits, strict=True):
        row["split"] = split
    cache = write_activation_cache(
        output_dir,
        {name: torch.cat(parts) for name, parts in tensors.items()},
        metadata,
        {
            "checkpoint": str(cfg.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "model": {"n_layer": int(model_config.n_layer), "n_embd": int(model_config.n_embd)},
            "environments": active_envs,
            "activation_locations": sorted(requested),
            "preprocessing_identity": preprocessing_identity,
            "grouping_mode": str(cfg.grouping_mode),
            "claim_bearing": str(cfg.grouping_mode) != "batch_schema_smoke",
        },
    )
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"cache_manifest": "manifest.json"},
        hashes={"checkpoint": checkpoint_sha256},
        split_manifest={split: splits.count(split) for split in ("train", "validation", "test")},
        environment_versions={"cache_schema": "1", "preprocessing": preprocessing_identity},
    )
    return cache


if __name__ == "__main__":
    main()

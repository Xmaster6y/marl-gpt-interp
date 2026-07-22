"""Collect a sharded, token-level corpus for actor and critic CLTs."""

from __future__ import annotations

import shutil
from collections import Counter

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.balanced_dataset import audited_source_records
from marl_gpt_interp.clt_data import CLTCorpusWriter, iter_clt_shards, stratified_grouped_split
from marl_gpt_interp.experiment_io import file_sha256, write_run_manifest
from marl_gpt_interp.marl_gpt_clt import PathCapture
from marl_gpt_interp.marl_gpt_tools import (
    ID_TO_ENV,
    as_path,
    build_loader,
    enable_sample_identity,
    enabled_envs,
    env_labels_for_batch,
    load_model,
    marl_gpt_cwd,
    repo_root,
    resolve_dataset_config,
    to_plain_config,
)


def sampled_token_coordinates(
    batch_size: int,
    sequence_length: int,
    tokens_per_sequence: int,
    *,
    generator: torch.Generator,
    include_output_token: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens_per_sequence <= 0 or tokens_per_sequence >= sequence_length:
        tokens_per_sequence = sequence_length
    batches, tokens = [], []
    for batch in range(batch_size):
        if tokens_per_sequence == sequence_length:
            selected = torch.arange(sequence_length)
        elif include_output_token:
            random_count = tokens_per_sequence - 1
            selected = torch.randperm(sequence_length - 1, generator=generator)[:random_count]
            selected = torch.cat([selected, torch.tensor([sequence_length - 1])]).sort().values
        else:
            selected = torch.randperm(sequence_length, generator=generator)[:tokens_per_sequence].sort().values
        batches.append(torch.full((len(selected),), batch, dtype=torch.long))
        tokens.append(selected)
    return torch.cat(batches), torch.cat(tokens)


def _source_split_map(source_records: dict[int, dict[str, str]], seed: int) -> dict[int, str]:
    source_ids = sorted(source_records)
    groups = [f"{source_records[source]['environment']}:{source_records[source]['source_group']}" for source in source_ids]
    strata = [source_records[source]["environment"] for source in source_ids]
    splits = stratified_grouped_split(groups, strata, seed=seed)
    return dict(zip(source_ids, splits, strict=True))


@hydra.main(config_path="../../../configs/experiments/circuit_tracing/collect_corpus", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    dataset_config = resolve_dataset_config(root, cfg)
    active_envs = enabled_envs(dataset_config, list(cfg.envs))
    loader = build_loader(root, {env: dataset_config[env] for env in active_envs}, cfg)
    source_caps: dict[int, int] = {}
    source_paths = enable_sample_identity(
        loader,
        max_rows_per_source=int(cfg.get("max_rows_per_source", 0)),
        max_rows_by_source=source_caps,
    )
    source_records: dict[int, dict[str, str]] = {}
    if str(cfg.grouping_mode) == "dataset_source_group":
        source_records = audited_source_records(as_path(root, str(cfg.dataset_manifest_path)), source_paths)
        component_caps = cfg.get("max_rows_per_component")
        for source_id, record in source_records.items():
            cap = (
                int(cfg.get("max_rows_per_source", 0))
                if component_caps is None
                else int(component_caps[record["environment"]][record["component"]])
            )
            if cap <= 0:
                raise ValueError(f"source cap must be positive for {record}")
            source_caps[source_id] = cap
        split_by_source = _source_split_map(source_records, int(cfg.seed))
    else:
        split_by_source = {}

    with marl_gpt_cwd(root):
        model, model_config = load_model(root, cfg)
    checkpoint_path = as_path(root, str(cfg.checkpoint))
    checkpoint_hash = file_sha256(checkpoint_path)
    writer = CLTCorpusWriter(
        output_dir,
        rows_per_shard=int(cfg.rows_per_shard),
        manifest={
            "checkpoint": str(cfg.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "model": {
                "shared_layers": int(model_config.n_layer) - 1,
                "path_layers": int(model_config.n_layer),
                "d_model": int(model_config.n_embd),
                "block_size": int(model_config.block_size),
            },
            "environments": active_envs,
            "token_sampling": {
                "tokens_per_sequence": int(cfg.tokens_per_sequence),
                "include_output_token": bool(cfg.include_output_token),
                "model_has_attention_mask": False,
            },
            "preprocessing": "natural-residual-stream",
            "grouping_mode": str(cfg.grouping_mode),
        },
    )
    generator = torch.Generator().manual_seed(int(cfg.seed))
    rows_per_environment: Counter[str] = Counter()
    sequences_per_environment: Counter[str] = Counter()
    seen_source_rows: set[tuple[int, int]] = set()
    accepted_sequences_per_source: Counter[int] = Counter()
    expected_source_rows = sum(source_caps.values()) if source_records else 0
    global_sample_index = 0
    iterator = iter(loader)
    for batch_index in range(int(cfg.num_batches)):
        with marl_gpt_cwd(root):
            batch_obs, target_actions, _mask_target, _next_obs, batch_info = next(iterator)
        labels = env_labels_for_batch(loader)
        with torch.no_grad(), PathCapture(model, "actor") as actor_capture, PathCapture(
            model, "critic"
        ) as critic_capture, marl_gpt_cwd(root):
            actor_logits, critic_logits, _loss, _loss_info = model(batch_obs)
        actor_path = actor_capture.result()
        critic_path = critic_capture.result()
        for actor_value, critic_value in zip(
            actor_path.residual_inputs[:-1], critic_path.residual_inputs[:-1], strict=True
        ):
            if not torch.equal(actor_value, critic_value):
                raise RuntimeError("actor and critic captures disagree on the shared residual stream")

        batch_size, sequence_length, _width = actor_path.residual_inputs[0].shape
        source_ids = batch_info.get("source_file_id")
        source_rows = batch_info.get("source_row_index")
        if source_records:
            if source_ids is None or source_rows is None:
                raise RuntimeError("audited collection requires source-file and source-row identity")
            eligible_batches = []
            for row_batch in range(batch_size):
                source_id = int(source_ids[row_batch])
                identity = (source_id, int(source_rows[row_batch]))
                if accepted_sequences_per_source[source_id] >= source_caps[source_id]:
                    continue
                if identity not in seen_source_rows:
                    seen_source_rows.add(identity)
                    accepted_sequences_per_source[source_id] += 1
                    eligible_batches.append(row_batch)
            if not eligible_batches:
                continue
        else:
            eligible_batches = list(range(batch_size))
        local_batch_indices, token_indices = sampled_token_coordinates(
            len(eligible_batches),
            sequence_length,
            int(cfg.tokens_per_sequence),
            generator=generator,
            include_output_token=bool(cfg.include_output_token),
        )
        eligible_tensor = torch.tensor(eligible_batches, dtype=torch.long)
        batch_indices = eligible_tensor[local_batch_indices]
        shared_residuals = torch.stack(actor_path.residual_inputs[:-1], dim=2)[batch_indices, token_indices]
        shared_outputs = torch.stack(actor_path.mlp_outputs[:-1], dim=2)[batch_indices, token_indices]
        tensors = {
            "shared_residual_inputs": shared_residuals,
            "shared_mlp_outputs": shared_outputs,
            "actor_residual_input": actor_path.residual_inputs[-1][batch_indices, token_indices],
            "actor_mlp_output": actor_path.mlp_outputs[-1][batch_indices, token_indices],
            "critic_residual_input": critic_path.residual_inputs[-1][batch_indices, token_indices],
            "critic_mlp_output": critic_path.mlp_outputs[-1][batch_indices, token_indices],
        }
        storage_dtype = getattr(torch, str(cfg.storage_dtype))
        tensors = {name: value.to(storage_dtype) for name, value in tensors.items()}

        legal_masks = batch_obs["action_mask"].bool()
        legal_logits = actor_logits.masked_fill(~legal_masks, -torch.inf)
        predicted_actions = legal_logits.argmax(dim=-1)
        predicted_values = model.calculate_all_val_from_logits(critic_logits)
        metadata = []
        for row_batch, token_index in zip(batch_indices.tolist(), token_indices.tolist(), strict=True):
            label = int(labels[row_batch])
            environment = ID_TO_ENV.get(label, str(label))
            source_id = int(source_ids[row_batch]) if source_ids is not None else -1
            if source_id in source_records:
                source_group = source_records[source_id]["source_group"]
                split_group = f"{environment}:{source_group}"
                split = split_by_source[source_id]
            else:
                source_group = f"batch-{batch_index:06d}"
                split_group = f"{environment}:{source_group}"
                split = "train" if batch_index % 10 < 7 else "validation" if batch_index % 10 < 9 else "test"
            predicted_action = int(predicted_actions[row_batch])
            row = {
                "environment": environment,
                "environment_label": label,
                "split_group": split_group,
                "source_group": source_group,
                "source_file_id": source_id,
                "source_file": source_paths.get(source_id),
                "source_row_index": int(source_rows[row_batch]) if source_rows is not None else -1,
                "sample_index": global_sample_index,
                "sequence_batch_index": batch_index,
                "sequence_sample_index": row_batch,
                "token_index": token_index,
                "is_output_token": token_index == sequence_length - 1,
                "split": split,
                "target_action": int(target_actions[row_batch]),
                "predicted_action": predicted_action,
                "predicted_action_value": float(predicted_values[row_batch, predicted_action]),
                "legal_actions": legal_masks[row_batch].nonzero(as_tuple=False).flatten().tolist(),
                "agent_position": int(batch_obs["agent_pos"][row_batch, token_index]),
                "team_position": int(batch_obs["group_pos"][row_batch, token_index]),
                "time_position": int(batch_obs["time_pos"][row_batch, token_index]),
                "attribute_position": int(batch_obs["attr_pos"][row_batch, token_index]),
            }
            metadata.append(row)
            global_sample_index += 1
            rows_per_environment[environment] += 1
        for row_batch in eligible_batches:
            sequences_per_environment[ID_TO_ENV.get(int(labels[row_batch]), str(int(labels[row_batch])))] += 1
        writer.add(tensors, metadata)
        if expected_source_rows and len(seen_source_rows) == expected_source_rows:
            break

    if expected_source_rows and len(seen_source_rows) != expected_source_rows:
        raise RuntimeError(
            f"collection reached {len(seen_source_rows):,} of {expected_source_rows:,} unique capped source rows; "
            "increase num_batches or audit the native loader schedule"
        )

    manifest = writer.close()
    manifest["rows_per_environment"] = dict(rows_per_environment)
    manifest["sequences_per_environment"] = dict(sequences_per_environment)
    manifest["unique_source_rows"] = len(seen_source_rows)
    manifest["expected_source_rows"] = expected_source_rows
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n")
    run_manifest_path = output_dir / "run_manifest.json"
    write_run_manifest(
        run_manifest_path,
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"corpus_manifest": "manifest.json"},
        hashes={"checkpoint": checkpoint_hash, "corpus_manifest": file_sha256(manifest_path)},
        split_manifest={
            split: sum(
                len(rows)
                for _tensors, rows in iter_clt_shards(output_dir, split=split, verify_hashes=False)
            )
            for split in ("train", "validation", "test")
        },
        environment_versions={"corpus_schema": "1", "preprocessing": "natural-residual-stream"},
    )
    provenance_value = cfg.get("provenance_dir")
    if provenance_value is not None:
        provenance_dir = as_path(root, str(provenance_value))
        provenance_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, provenance_dir / "corpus_manifest.json")
        shutil.copy2(run_manifest_path, provenance_dir / "run_manifest.json")
    return manifest


if __name__ == "__main__":
    main()

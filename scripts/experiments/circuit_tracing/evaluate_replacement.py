"""Evaluate actor and critic CLT replacement fidelity on held-out MARL-GPT batches."""

from __future__ import annotations

from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.clt_training import load_trained_clt
from marl_gpt_interp.experiment_io import file_sha256, write_run_manifest
from marl_gpt_interp.marl_gpt_clt import CLTReplacement
from marl_gpt_interp.marl_gpt_tools import (
    ID_TO_ENV,
    as_path,
    build_loader,
    enabled_envs,
    env_labels_for_batch,
    load_model,
    marl_gpt_cwd,
    repo_root,
    resolve_dataset_config,
    to_plain_config,
    write_json,
)


def _policy_kl(reference: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    reference = reference.masked_fill(~mask, -torch.inf)
    candidate = candidate.masked_fill(~mask, -torch.inf)
    reference_log_probs = torch.log_softmax(reference, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate, dim=-1)
    return (
        torch.softmax(reference, dim=-1)
        * (reference_log_probs - candidate_log_probs).masked_fill(~mask, 0.0)
    ).sum(dim=-1)


@hydra.main(config_path="../../../configs/experiments/circuit_tracing/evaluate_replacement", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_clt, actor_spec = load_trained_clt(as_path(root, str(cfg.actor_clt_dir)), device=str(cfg.device))
    critic_clt, critic_spec = load_trained_clt(as_path(root, str(cfg.critic_clt_dir)), device=str(cfg.device))
    if actor_spec["branch"] != "actor" or critic_spec["branch"] != "critic":
        raise ValueError("replacement evaluation needs actor and critic CLTs")
    if actor_spec["source_checkpoint_sha256"] != critic_spec["source_checkpoint_sha256"]:
        raise ValueError("actor and critic CLTs were not trained for the same checkpoint")
    dataset_config = resolve_dataset_config(root, cfg)
    active_envs = enabled_envs(dataset_config, list(cfg.envs))
    loader = build_loader(root, {env: dataset_config[env] for env in active_envs}, cfg)
    with marl_gpt_cwd(root):
        model, _model_config = load_model(root, cfg)
    if file_sha256(as_path(root, str(cfg.checkpoint))) != actor_spec["source_checkpoint_sha256"]:
        raise ValueError("evaluation checkpoint does not match CLT training")

    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    iterator = iter(loader)
    for _batch_index in range(int(cfg.num_batches)):
        with marl_gpt_cwd(root):
            batch_obs, _target, _mask_target, _next_obs, _batch_info = next(iterator)
        labels = env_labels_for_batch(loader)
        with torch.no_grad(), marl_gpt_cwd(root):
            original_actor, original_critic, _loss, _info = model(batch_obs)
            with CLTReplacement(model, actor_clt, "actor"):
                replaced_actor, _unused_critic, _loss, _info = model(batch_obs)
            with CLTReplacement(model, critic_clt, "critic"):
                _unused_actor, replaced_critic, _loss, _info = model(batch_obs)
            mask = batch_obs["action_mask"].bool()
            actor_kl = _policy_kl(original_actor, replaced_actor, mask)
            agreement = (
                original_actor.masked_fill(~mask, -torch.inf).argmax(dim=-1)
                == replaced_actor.masked_fill(~mask, -torch.inf).argmax(dim=-1)
            ).float()
            original_values = model.calculate_all_val_from_logits(original_critic)
            replaced_values = model.calculate_all_val_from_logits(replaced_critic)
            value_mae = (original_values - replaced_values).abs().masked_fill(~mask, 0.0).sum(dim=-1) / mask.sum(dim=-1)
        for index, label in enumerate(labels.tolist()):
            environment = ID_TO_ENV.get(int(label), str(label))
            for group in ("all", environment):
                rows[group]["actor_kl"].append(float(actor_kl[index]))
                rows[group]["action_agreement"].append(float(agreement[index]))
                rows[group]["critic_value_mae"].append(float(value_mae[index]))

    metrics = {
        f"{group}/{name}": sum(values) / len(values)
        for group, group_rows in rows.items()
        for name, values in group_rows.items()
    }
    metrics["examples"] = float(len(rows["all"]["actor_kl"]))
    write_json(output_dir / "replacement_metrics.json", metrics)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"metrics": "replacement_metrics.json"},
        hashes={
            "checkpoint": actor_spec["source_checkpoint_sha256"],
            "actor_clt": file_sha256(as_path(root, str(cfg.actor_clt_dir)) / "model.pt"),
            "critic_clt": file_sha256(as_path(root, str(cfg.critic_clt_dir)) / "model.pt"),
        },
    )
    return metrics


if __name__ == "__main__":
    main()

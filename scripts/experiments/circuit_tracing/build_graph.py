"""Build one actor or critic attribution graph from a held-out MARL-GPT decision."""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.attribution_graph import (
    ActorTarget,
    CriticTarget,
    build_attribution_graph,
    prune_attribution_graph,
)
from marl_gpt_interp.clt_training import load_trained_clt
from marl_gpt_interp.experiment_io import file_sha256, write_run_manifest
from marl_gpt_interp.marl_gpt_tools import (
    as_path,
    build_loader,
    enabled_envs,
    load_model,
    marl_gpt_cwd,
    repo_root,
    resolve_dataset_config,
    slice_batch,
    to_plain_config,
)


@hydra.main(config_path="../../../configs/experiments/circuit_tracing/build_graph", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = as_path(root, str(cfg.eligibility_audit))
    audit = json.loads(audit_path.read_text())
    if audit.get("eligible_for_graph_interpretation") is not True:
        raise ValueError("CLT suite did not pass the graph-interpretation eligibility audit")
    clt, spec = load_trained_clt(as_path(root, str(cfg.clt_dir)), device=str(cfg.device))
    branch = str(cfg.branch)
    if spec["branch"] != branch:
        raise ValueError("selected CLT does not match the attribution branch")
    dataset_config = resolve_dataset_config(root, cfg)
    active_envs = enabled_envs(dataset_config, list(cfg.envs))
    loader = build_loader(root, {env: dataset_config[env] for env in active_envs}, cfg)
    with marl_gpt_cwd(root):
        model, _model_config = load_model(root, cfg)
        batch_obs, _target_actions, _mask, _next_obs, _batch_info = next(iter(loader))
    sample = int(cfg.sample_in_batch)
    if sample < 0 or sample >= next(iter(batch_obs.values())).shape[0]:
        raise ValueError("sample_in_batch is outside the loaded batch")
    one = slice_batch(batch_obs, torch.arange(next(iter(batch_obs.values())).shape[0]) == sample)
    action_value = cfg.target.get("action")
    if branch == "actor":
        comparison = cfg.target.get("comparison_action")
        target = ActorTarget(
            action=None if action_value is None else int(action_value),
            comparison_action=None if comparison is None else int(comparison),
        )
    else:
        target = CriticTarget(action=None if action_value is None else int(action_value))
    graph = build_attribution_graph(
        model,
        one,
        clt,
        branch=branch,
        target=target,
        minimum_edge=float(cfg.minimum_edge),
    )
    graph = prune_attribution_graph(
        graph,
        node_threshold=float(cfg.pruning.node_threshold),
        edge_threshold=float(cfg.pruning.edge_threshold),
    )
    torch.save({key: value.detach().cpu() for key, value in one.items()}, output_dir / "input.pt")
    graph.write_json(output_dir / "graph.json")
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"graph": "graph.json", "input": "input.pt"},
        hashes={
            "checkpoint": file_sha256(as_path(root, str(cfg.checkpoint))),
            "clt": file_sha256(as_path(root, str(cfg.clt_dir)) / "model.pt"),
            "input": file_sha256(output_dir / "input.pt"),
            "eligibility_audit": file_sha256(audit_path),
        },
    )
    return graph.to_dict()


if __name__ == "__main__":
    main()

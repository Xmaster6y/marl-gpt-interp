"""Evaluate one graph-selected CLT feature in the original MARL-GPT model."""

from __future__ import annotations

import json
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.clt_training import load_trained_clt
from marl_gpt_interp.experiment_io import file_sha256, write_run_manifest
from marl_gpt_interp.marl_gpt_clt import CrossLayerFeatureIntervention
from marl_gpt_interp.marl_gpt_tools import (
    as_path,
    load_model,
    marl_gpt_cwd,
    repo_root,
    to_plain_config,
    write_json,
)


def _policy_kl(reference: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> float:
    reference = reference.masked_fill(~mask, -torch.inf)
    candidate = candidate.masked_fill(~mask, -torch.inf)
    reference_log_probs = torch.log_softmax(reference, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate, dim=-1)
    terms = torch.softmax(reference, dim=-1) * (
        reference_log_probs - candidate_log_probs
    ).masked_fill(~mask, 0.0)
    return float(terms.sum(dim=-1).item())


def _expected_action_value(model: Any, critic_logits: torch.Tensor, action: int) -> float:
    return float(model.calculate_all_val_from_logits(critic_logits)[0, action])


def _read_feature(graph: dict[str, Any], feature_id: str | None) -> tuple[int, int, int, str]:
    feature_nodes = [node for node in graph["nodes"] if node["kind"] == "feature"]
    if feature_id is None:
        feature_attribution = {node["id"]: 0.0 for node in feature_nodes}
        for edge in graph["edges"]:
            if edge["source"] in feature_attribution:
                feature_attribution[edge["source"]] += abs(float(edge["attribution"]))
        if not feature_attribution:
            raise ValueError("the graph has no retained feature node")
        feature_id = max(feature_attribution, key=feature_attribution.get)
    matches = [node for node in feature_nodes if node["id"] == feature_id]
    if len(matches) != 1:
        raise ValueError(f"feature_id {feature_id!r} is not a unique retained graph feature")
    node = matches[0]
    return int(node["layer"]), int(node["token"]), int(node["feature"]), str(feature_id)


def _measure(
    model: Any,
    batch_obs: dict[str, torch.Tensor],
    *,
    branch: str,
    action: int,
    comparison_action: int | None,
) -> dict[str, float]:
    actor, critic, _loss, _info = model(batch_obs)
    mask = batch_obs["action_mask"].bool()
    result = {"action": float(action)}
    if branch == "actor":
        if comparison_action is None:
            raise ValueError("actor intervention needs a comparison action")
        result["target"] = float(actor[0, action] - actor[0, comparison_action])
    else:
        result["target"] = _expected_action_value(model, critic, action)
    result["selected_action"] = float(actor.masked_fill(~mask, -torch.inf).argmax(dim=-1).item())
    return result


@hydra.main(config_path="../../../configs/experiments/circuit_tracing/evaluate_intervention", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = as_path(root, str(cfg.graph))
    graph = json.loads(graph_path.read_text())
    branch = str(cfg.branch)
    if graph["branch"] != branch:
        raise ValueError("graph and intervention branch do not match")
    clt, spec = load_trained_clt(as_path(root, str(cfg.clt_dir)), device=str(cfg.device))
    if spec["branch"] != branch:
        raise ValueError("CLT and intervention branch do not match")
    source_layer, token_index, feature, feature_id = _read_feature(
        graph, None if cfg.feature_id is None else str(cfg.feature_id)
    )
    target = graph["target"]
    action = int(target["action"])
    comparison_action = (
        int(target["comparison_action"]) if target["kind"] == "action_logit_contrast" else None
    )

    input_value = cfg.get("input")
    input_path = graph_path.parent / "input.pt" if input_value is None else as_path(root, str(input_value))
    batch_obs = torch.load(input_path, map_location=str(cfg.device), weights_only=True)
    if next(iter(batch_obs.values())).shape[0] != 1:
        raise ValueError("stored graph input must contain exactly one decision")
    with marl_gpt_cwd(root):
        model, _model_config = load_model(root, cfg)
    if file_sha256(as_path(root, str(cfg.checkpoint))) != spec["source_checkpoint_sha256"]:
        raise ValueError("intervention checkpoint does not match CLT training")

    with torch.no_grad(), marl_gpt_cwd(root):
        baseline_actor, _baseline_critic, _loss, _info = model(batch_obs)
        baseline = _measure(
            model,
            batch_obs,
            branch=branch,
            action=action,
            comparison_action=comparison_action,
        )
        mask = batch_obs["action_mask"].bool()
        rows = []
        for factor_value in cfg.factors:
            factor = float(factor_value)
            with CrossLayerFeatureIntervention(
                model,
                clt,
                branch,
                source_layer=source_layer,
                feature=feature,
                token_index=token_index,
                factor=factor,
            ) as intervention:
                actor, critic, _loss, _info = model(batch_obs)
            measured = (
                float(actor[0, action] - actor[0, comparison_action])
                if branch == "actor"
                else _expected_action_value(model, critic, action)
            )
            rows.append(
                {
                    "control": "feature",
                    "factor": factor,
                    "target": measured,
                    "target_change": measured - baseline["target"],
                    "policy_kl": _policy_kl(baseline_actor, actor, mask),
                    "selected_action": int(actor.masked_fill(~mask, -torch.inf).argmax(dim=-1).item()),
                    "activation": float(intervention.activation.item()),
                }
            )
            for seed_value in cfg.random_seeds:
                seed = int(seed_value)
                with CrossLayerFeatureIntervention(
                    model,
                    clt,
                    branch,
                    source_layer=source_layer,
                    feature=feature,
                    token_index=token_index,
                    factor=factor,
                    random_seed=seed,
                ) as intervention:
                    random_actor, random_critic, _loss, _info = model(batch_obs)
                random_target = (
                    float(random_actor[0, action] - random_actor[0, comparison_action])
                    if branch == "actor"
                    else _expected_action_value(model, random_critic, action)
                )
                rows.append(
                    {
                        "control": "norm_matched_random",
                        "random_seed": seed,
                        "factor": factor,
                        "target": random_target,
                        "target_change": random_target - baseline["target"],
                        "policy_kl": _policy_kl(baseline_actor, random_actor, mask),
                        "selected_action": int(
                            random_actor.masked_fill(~mask, -torch.inf).argmax(dim=-1).item()
                        ),
                        "activation": float(intervention.activation.item()),
                    }
                )

    matched_controls = []
    random_rows = [row for row in rows if row["control"] == "norm_matched_random"]
    for feature_row in (row for row in rows if row["control"] == "feature"):
        if random_rows:
            match = min(random_rows, key=lambda row: abs(row["policy_kl"] - feature_row["policy_kl"]))
            matched_controls.append(
                {
                    "feature_factor": feature_row["factor"],
                    "feature_policy_kl": feature_row["policy_kl"],
                    "random_seed": match["random_seed"],
                    "random_factor": match["factor"],
                    "random_policy_kl": match["policy_kl"],
                    "absolute_kl_mismatch": abs(match["policy_kl"] - feature_row["policy_kl"]),
                    "feature_target_change": feature_row["target_change"],
                    "random_target_change": match["target_change"],
                }
            )
    result = {
        "branch": branch,
        "feature_id": feature_id,
        "baseline": baseline,
        "rows": rows,
        "induced_kl_matched_controls": matched_controls,
    }
    write_json(output_dir / "intervention_metrics.json", result)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"metrics": "intervention_metrics.json"},
        hashes={
            "checkpoint": spec["source_checkpoint_sha256"],
            "clt": file_sha256(as_path(root, str(cfg.clt_dir)) / "model.pt"),
            "graph": file_sha256(graph_path),
            "input": file_sha256(input_path),
        },
    )
    return result


if __name__ == "__main__":
    main()

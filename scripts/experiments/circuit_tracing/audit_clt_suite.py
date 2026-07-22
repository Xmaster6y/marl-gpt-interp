"""Apply the frozen CLT and replacement gates before graph interpretation."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from marl_gpt_interp.clt_audit import audit_clt_suite
from marl_gpt_interp.experiment_io import file_sha256, write_run_manifest
from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json


@hydra.main(config_path="../../../configs/experiments/circuit_tracing/audit_clt_suite", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    actor_path = as_path(root, str(cfg.actor_metrics))
    critic_path = as_path(root, str(cfg.critic_metrics))
    replacement_path = as_path(root, str(cfg.replacement_metrics))
    result = audit_clt_suite(
        json.loads(actor_path.read_text()),
        json.loads(critic_path.read_text()),
        json.loads(replacement_path.read_text()),
        environments=[str(value) for value in cfg.environments],
        num_layers=int(cfg.num_layers),
        maximum_normalized_mse=float(cfg.gates.maximum_normalized_mse),
        maximum_dead_feature_fraction=float(cfg.gates.maximum_dead_feature_fraction),
        minimum_l0=float(cfg.gates.minimum_l0),
        maximum_l0=float(cfg.gates.maximum_l0),
        maximum_actor_kl=float(cfg.gates.maximum_actor_kl),
        minimum_action_agreement=float(cfg.gates.minimum_action_agreement),
        maximum_critic_value_mae=float(cfg.gates.maximum_critic_value_mae),
    )
    write_json(output_dir / "clt_suite_audit.json", result)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status=result["status"],
        artifacts={"audit": "clt_suite_audit.json"},
        hashes={
            "actor_metrics": file_sha256(actor_path),
            "critic_metrics": file_sha256(critic_path),
            "replacement_metrics": file_sha256(replacement_path),
        },
    )
    if not result["eligible_for_graph_interpretation"]:
        raise SystemExit(
            f"CLT suite is not eligible for graph interpretation; {len(result['failures'])} gates failed"
        )
    return result


if __name__ == "__main__":
    main()

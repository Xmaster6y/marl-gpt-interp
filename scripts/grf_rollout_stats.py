"""Run MARL-GPT on GRF and write simple rollout statistics."""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.grf_stats import (
    action_entropy,
    average_present,
    count_actions,
    extract_raw_observation_stats,
)


DEFAULT_MODEL_URL = "https://huggingface.co/nortem/marl-gpt-model/resolve/main/marl-gpt-main.pt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_checkpoint(path: Path, url: str, download: bool) -> None:
    if path.exists():
        return
    if not download:
        raise SystemExit(
            f"Checkpoint not found at {path}. Put the MARL-GPT checkpoint there, "
            "set grf_rollout_stats.checkpoint, or rerun with "
            "grf_rollout_stats.download_checkpoint=true."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint to {path}")
    urllib.request.urlretrieve(url, path)


def _load_grf_stack(root: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(root / "marl-gpt"))
    try:
        from envs.grf_env.config import GRFInferenceConfig
        from envs.grf_env.create_env import make_grf_marl_gpt
        from gpt.inference import InferenceConfig, MARLGPTInference
    except ImportError as exc:
        raise SystemExit(
            "GRF/MARL-GPT imports failed. Install the optional GRF environment with "
            "`uv sync --group grf` and make sure the vendored `marl-gpt/` code is present. "
            f"Original error: {exc}"
        ) from exc
    return GRFInferenceConfig, make_grf_marl_gpt, (InferenceConfig, MARLGPTInference)


def _raw_observation(env: Any) -> Any:
    base_env = getattr(env, "unwrapped", env)
    football_env = getattr(base_env, "_env", None)
    football_unwrapped = getattr(football_env, "unwrapped", football_env)
    core_env = getattr(football_unwrapped, "_env", None)
    if core_env is not None and hasattr(core_env, "observation"):
        return core_env.observation()
    return None


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _episode_summary(
    *,
    seed: int,
    episode: int,
    step_rows: list[dict[str, Any]],
    action_size: int,
) -> dict[str, Any]:
    action_counts = [0 for _ in range(action_size)]
    total_reward = 0.0
    for row in step_rows:
        action_counts = [
            left + right
            for left, right in zip(action_counts, row["action_counts"], strict=True)
        ]
        total_reward += float(row["reward_sum"])

    last = step_rows[-1] if step_rows else {}
    return {
        "seed": seed,
        "episode": episode,
        "steps": len(step_rows),
        "terminated": last.get("terminated"),
        "truncated": last.get("truncated"),
        "total_reward": total_reward,
        "left_score": last.get("left_score"),
        "right_score": last.get("right_score"),
        "goal_difference": last.get("goal_difference"),
        "action_counts": action_counts,
        "action_entropy": action_entropy(action_counts),
        "possession_left_fraction": average_present(
            1
            if row.get("ball_owned_team") == 0
            else 0
            if row.get("ball_owned_team") in (-1, 1)
            else None
            for row in step_rows
        ),
        "mean_distance_to_goal": average_present(
            row.get("distance_to_goal") for row in step_rows
        ),
        "mean_nearest_defender_distance": average_present(
            row.get("nearest_defender_distance") for row in step_rows
        ),
        "mean_defensive_compactness": average_present(
            row.get("defensive_compactness") for row in step_rows
        ),
    }


def _aggregate(episode_rows: list[dict[str, Any]], action_size: int) -> dict[str, Any]:
    action_counts = [0 for _ in range(action_size)]
    for row in episode_rows:
        action_counts = [
            left + right
            for left, right in zip(action_counts, row["action_counts"], strict=True)
        ]
    return {
        "episodes": len(episode_rows),
        "mean_steps": average_present(row.get("steps") for row in episode_rows),
        "mean_total_reward": average_present(
            row.get("total_reward") for row in episode_rows
        ),
        "mean_goal_difference": average_present(
            row.get("goal_difference") for row in episode_rows
        ),
        "mean_possession_left_fraction": average_present(
            row.get("possession_left_fraction") for row in episode_rows
        ),
        "mean_distance_to_goal": average_present(
            row.get("mean_distance_to_goal") for row in episode_rows
        ),
        "mean_nearest_defender_distance": average_present(
            row.get("mean_nearest_defender_distance") for row in episode_rows
        ),
        "mean_defensive_compactness": average_present(
            row.get("mean_defensive_compactness") for row in episode_rows
        ),
        "action_counts": action_counts,
        "action_entropy": action_entropy(action_counts),
    }


def main(cfg: DictConfig) -> dict[str, Any]:
    script_cfg = cfg.grf_rollout_stats
    root = _repo_root()
    checkpoint = Path(script_cfg.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    output_dir = Path(script_cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_checkpoint(
        checkpoint,
        str(OmegaConf.select(script_cfg, "model_url", default=DEFAULT_MODEL_URL)),
        bool(script_cfg.download_checkpoint),
    )

    GRFInferenceConfig, make_grf_marl_gpt, inference_types = _load_grf_stack(root)
    InferenceConfig, MARLGPTInference = inference_types

    env_cfg = GRFInferenceConfig(
        device=script_cfg.device,
        history_len=script_cfg.history_len,
        history_step=script_cfg.history_step,
        map_name=script_cfg.map_name,
        obs_per_agent=False,
    )
    policy_cfg = InferenceConfig(
        path_to_weights=str(checkpoint),
        device=script_cfg.device,
        model_type=script_cfg.model_type,
        last_token=script_cfg.last_token,
        sample_actions=script_cfg.sample_actions,
    )
    policy = MARLGPTInference(policy_cfg)
    action_size = int(OmegaConf.select(script_cfg, "action_size", default=policy.action_size))

    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    seeds = list(script_cfg.seeds)

    for seed in seeds:
        for episode_index in range(int(script_cfg.episodes_per_seed)):
            env = make_grf_marl_gpt(env_cfg)
            observations, _ = env.reset(seed=int(seed))
            episode_step_rows: list[dict[str, Any]] = []
            try:
                for step in range(1, int(script_cfg.max_steps) + 1):
                    actions = policy.act(observations)
                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    raw_stats = extract_raw_observation_stats(_raw_observation(env))
                    info_stats = infos[0].get("episode_stats", {}) if infos else {}
                    final_score = info_stats.get("current_score_L_R", [None, None])
                    if raw_stats.get("left_score") is None:
                        raw_stats["left_score"] = final_score[0]
                    if raw_stats.get("right_score") is None:
                        raw_stats["right_score"] = final_score[1]
                    if raw_stats.get("goal_difference") is None:
                        raw_stats["goal_difference"] = info_stats.get("goal_difference")
                    row = {
                        "seed": int(seed),
                        "episode": episode_index,
                        "step": step,
                        "actions": actions,
                        "action_counts": count_actions(actions, action_size),
                        "reward_sum": float(sum(rewards)),
                        "terminated": bool(all(terminated)),
                        "truncated": bool(all(truncated)),
                        "free_kick_count": info_stats.get("free_kick_count"),
                        "goal_kick_count": info_stats.get("goal_kick_count"),
                        "corner_count": info_stats.get("corner_count"),
                        "throw_in_count": info_stats.get("throw_in_count"),
                        "penalty_count": info_stats.get("penalty_count"),
                        **raw_stats,
                    }
                    step_rows.append(row)
                    episode_step_rows.append(row)
                    if bool(all(terminated)) or bool(all(truncated)):
                        break
            finally:
                env.close()

            episode_rows.append(
                _episode_summary(
                    seed=int(seed),
                    episode=episode_index,
                    step_rows=episode_step_rows,
                    action_size=action_size,
                )
            )

    aggregate = _aggregate(episode_rows, action_size)
    config_dump = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    aggregate["config"] = config_dump

    _write_jsonl(output_dir / "steps.jsonl", step_rows)
    _write_jsonl(output_dir / "episodes.jsonl", episode_rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(aggregate, handle, default=_json_default, indent=2, sort_keys=True)
    csv_row = {key: value for key, value in aggregate.items() if key != "config"}
    csv_row["action_counts"] = json.dumps(csv_row["action_counts"])
    _write_csv(output_dir / "summary.csv", csv_row)

    logger.info(f"Wrote GRF rollout statistics to {output_dir}")
    return aggregate

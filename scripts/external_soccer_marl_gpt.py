"""Encode external soccer tracking as GRF and run a bounded MARL-GPT smoke."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.external_soccer import (
    GRFEncodingConfig,
    ModelInputBatch,
    Simple115V2Encoder,
    audit_histories,
    build_model_inputs,
    encode_histories,
    iter_laliga_frames,
    iter_robocup_frames,
)
from marl_gpt_interp.marl_gpt_tools import activation_hooks, marl_gpt_cwd, marl_gpt_path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _encoding_config(cfg: DictConfig) -> GRFEncodingConfig:
    return GRFEncodingConfig(**OmegaConf.to_container(cfg.encoding, resolve=True))


def _select_complete_frames(histories, max_examples: int):
    groups: dict[str, list[Any]] = defaultdict(list)
    order = []
    for history in histories:
        frame_id = history.current.frame_id
        if frame_id not in groups:
            order.append(frame_id)
        groups[frame_id].append(history)
    selected = []
    for frame_id in order:
        group = groups[frame_id]
        if selected and len(selected) + len(group) > max_examples:
            break
        if len(group) > max_examples:
            raise ValueError(f"max_examples_per_source={max_examples} cannot retain all {len(group)} perspectives")
        selected.extend(group)
    return tuple(selected)


def _external_batch(
    root: Path,
    cfg: DictConfig,
    source: str,
    encoding: GRFEncodingConfig,
    *,
    max_examples: int | None = None,
):
    encoder = Simple115V2Encoder(encoding)
    source_cfg = cfg[source]
    input_path = _path(root, str(source_cfg.path))
    if source == "laliga":
        frames = iter_laliga_frames(
            input_path,
            max_sequences=int(source_cfg.max_sequences),
            max_frames=int(source_cfg.max_frames),
        )
    else:
        frames = iter_robocup_frames(input_path, max_frames=int(source_cfg.max_frames))
    histories, diagnostics = encode_histories(frames, encoder)
    selected = _select_complete_frames(
        histories,
        int(cfg.max_examples_per_source) if max_examples is None else max_examples,
    )
    if not selected:
        raise RuntimeError(f"No complete {source} histories were encoded from {input_path}")
    diagnostics["selected_histories"] = len(selected)
    diagnostics["input_path"] = str(input_path)
    audit = audit_histories(selected)
    audit.update(diagnostics)
    return build_model_inputs(selected, encoding), audit


def _metadata_from_external(batch: ModelInputBatch) -> list[dict[str, Any]]:
    return [
        {
            "source": history.current.source,
            "match_id": history.current.match_id,
            "sequence_id": history.current.sequence_id,
            "frame_id": history.current.frame_id,
            "step_index": history.current.step_index,
            "focal_team": history.current.focal_team,
            "focal_player": history.current.focal_player,
            "focal_index": history.current.focal_index,
            "rotated": history.current.rotated,
            "imputed_fields": sorted(history.current.imputed_fields),
        }
        for history in batch.histories
    ]


def _torch_batch(arrays: dict[str, np.ndarray], device: str):
    import torch

    return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}


def _collect_native_grf(
    root: Path,
    cfg: DictConfig,
    checkpoint: Path,
    *,
    max_examples: int | None = None,
):
    import torch

    from scripts.grf_rollout_stats import _load_grf_stack

    GRFInferenceConfig, make_grf_marl_gpt, inference_types = _load_grf_stack(root)
    InferenceConfig, MARLGPTInference = inference_types
    env_cfg = GRFInferenceConfig(
        device=str(cfg.device),
        history_len=int(cfg.encoding.history_len),
        history_step=int(cfg.encoding.history_step),
        map_name=str(cfg.native_grf.map_name),
        obs_per_agent=False,
    )
    policy_cfg = InferenceConfig(
        path_to_weights=str(checkpoint),
        device=str(cfg.device),
        model_type="actor",
        last_token=True,
        env_specific=True,
        env_indx=int(cfg.encoding.environment_id),
        sample_actions=False,
    )
    policy = MARLGPTInference(policy_cfg)
    env = make_grf_marl_gpt(env_cfg)
    observations, _ = env.reset(seed=int(cfg.seed))
    target_examples = int(cfg.max_examples_per_source) if max_examples is None else max_examples
    arrays: dict[str, list[Any]] = defaultdict(list)
    metadata = []
    try:
        for step in range(int(cfg.native_grf.max_steps)):
            actions = policy.act(observations)
            if step >= int(cfg.native_grf.warmup_steps):
                current = policy.last_obs
                remaining = target_examples - len(metadata)
                take = min(int(current["obs"].shape[0]), remaining)
                if take > 0:
                    for key, value in current.items():
                        arrays[key].append(value[:take].detach().cpu())
                    metadata.extend(
                        {
                            "source": "grf",
                            "match_id": f"{cfg.native_grf.map_name}:seed-{cfg.seed}",
                            "sequence_id": "episode-0",
                            "frame_id": f"step-{step}",
                            "step_index": step,
                            "focal_team": "left",
                            "focal_player": str(index),
                            "focal_index": index,
                            "rotated": False,
                            "imputed_fields": [],
                        }
                        for index in range(take)
                    )
            if len(metadata) >= target_examples:
                break
            observations, _, terminated, truncated, _ = env.step(actions)
            if bool(all(terminated)) or bool(all(truncated)):
                break
    finally:
        env.close()
    if not metadata:
        raise RuntimeError("Native GRF collection produced no post-warmup observations")
    native = {key: torch.cat(values, dim=0) for key, values in arrays.items()}
    audit = {
        "histories": len(metadata),
        "finite": bool(torch.isfinite(native["obs"]).all()),
        "shape": list(native["obs"].shape),
        "value_min": float(native["obs"][:, :-1].min()),
        "value_max": float(native["obs"][:, :-1].max()),
        "imputed_history_counts": {},
        "warmup_steps": int(cfg.native_grf.warmup_steps),
        "map_name": str(cfg.native_grf.map_name),
    }
    return native, metadata, audit, policy.net


def _source_input_summary(batch: dict[str, Any], observation_tokens: int) -> dict[str, Any]:
    obs = batch["obs"][:, :observation_tokens].detach().float().cpu()
    return {
        "examples": int(obs.shape[0]),
        "finite": bool(obs.isfinite().all()),
        "minimum": float(obs.min()),
        "maximum": float(obs.max()),
        "mean": float(obs.mean()),
        "standard_deviation": float(obs.std()),
    }


def _evaluate(model, source_batches, source_metadata, cfg: DictConfig):
    import torch

    predictions = []
    activations: dict[str, dict[str, Any]] = {}
    source_summaries = {}
    selected_layers = set(str(value) for value in cfg.activation_layers)
    for source, full_batch in source_batches.items():
        activation_parts: dict[str, list[Any]] = defaultdict(list)
        action_counts: Counter[int] = Counter()
        entropies = []
        critic_values = []
        for start in range(0, len(source_metadata[source]), int(cfg.batch_size)):
            stop = min(start + int(cfg.batch_size), len(source_metadata[source]))
            batch = {key: value[start:stop] for key, value in full_batch.items()}
            captured = {}
            hooks = activation_hooks(model, captured)
            with torch.no_grad():
                act_logits, value_logits, _, _ = model(batch)
                action_probs = torch.softmax(act_logits[:, :19], dim=-1)
                values = model.calculate_all_val_from_logits(value_logits)[:, :19]
            for hook in hooks:
                hook.remove()
            for name, tensor in captured.items():
                if name in selected_layers:
                    activation_parts[name].append(tensor[:, -1, :].detach().float().cpu())
            entropy = -(action_probs * action_probs.clamp_min(1e-12).log()).sum(dim=-1)
            top_actions = action_probs.argmax(dim=-1)
            for local_index, metadata in enumerate(source_metadata[source][start:stop]):
                action = int(top_actions[local_index])
                action_counts[action] += 1
                entropies.append(float(entropy[local_index]))
                critic_values.extend(float(value) for value in values[local_index].tolist())
                predictions.append(
                    {
                        **metadata,
                        "top_grf_action": action,
                        "action_entropy": float(entropy[local_index]),
                        "action_probabilities": action_probs[local_index].tolist(),
                        "critic_values": values[local_index].tolist(),
                    }
                )
        activations[source] = {name: torch.cat(parts) for name, parts in activation_parts.items()}
        source_summaries[source] = {
            "examples": len(source_metadata[source]),
            "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
            "action_entropy_mean": float(np.mean(entropies)),
            "action_entropy_std": float(np.std(entropies)),
            "critic_value_mean": float(np.mean(critic_values)),
            "critic_value_std": float(np.std(critic_values)),
            "finite_activations": all(bool(tensor.isfinite().all()) for tensor in activations[source].values()),
        }
    return predictions, activations, source_summaries


def _activation_distances(activations) -> dict[str, Any]:
    import torch

    reference = activations["grf"]
    distances = {}
    for source in ("laliga", "robocup"):
        distances[source] = {}
        for layer, tensor in activations[source].items():
            source_center = tensor.mean(dim=0)
            reference_center = reference[layer].mean(dim=0)
            distances[source][layer] = {
                "centroid_l2": float((source_center - reference_center).norm()),
                "relative_centroid_l2": float(
                    (source_center - reference_center).norm() / reference_center.norm().clamp_min(1e-12)
                ),
                "centroid_cosine": float(
                    torch.nn.functional.cosine_similarity(source_center, reference_center, dim=0, eps=1e-12)
                ),
            }
    return distances


@hydra.main(config_path="../configs/external_soccer_marl_gpt", version_base=None)
def main(cfg: DictConfig) -> dict[str, Any]:
    import torch

    root = _root()
    output_dir = _path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _path(root, str(cfg.checkpoint))
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    encoding = _encoding_config(cfg)
    external_batches = {}
    audits = {}
    metadata = {}
    for source in ("laliga", "robocup"):
        batch, audit = _external_batch(root, cfg, source, encoding)
        external_batches[source] = _torch_batch(batch.arrays, str(cfg.device))
        audits[source] = audit
        metadata[source] = _metadata_from_external(batch)

    marl_gpt_path(root)
    with marl_gpt_cwd(root):
        native_batch, native_metadata, native_audit, model = _collect_native_grf(root, cfg, checkpoint)
    source_batches = {**external_batches, "grf": {key: value.to(str(cfg.device)) for key, value in native_batch.items()}}
    metadata["grf"] = native_metadata
    audits["grf"] = native_audit

    with marl_gpt_cwd(root):
        predictions, activations, model_summaries = _evaluate(model, source_batches, metadata, cfg)
    observation_tokens = encoding.history_len * 115
    input_summaries = {
        source: _source_input_summary(batch, observation_tokens) for source, batch in source_batches.items()
    }
    distances = _activation_distances(activations)
    summary = {
        "status": "ok",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "input_summaries": input_summaries,
        "model_summaries": model_summaries,
        "activation_distances_to_grf": distances,
        "interpretation_boundary": (
            "This bounded run verifies encoding and model execution only; it is not evidence of human-football "
            "transfer or external action validity."
        ),
    }

    ordered_sources = ("laliga", "robocup", "grf")
    combined = {
        key: torch.cat([source_batches[source][key].detach().cpu() for source in ordered_sources]).numpy()
        for key in source_batches["grf"]
    }
    combined["source"] = np.concatenate(
        [np.full(len(metadata[source]), source) for source in ordered_sources]
    )
    combined["metadata_json"] = np.asarray(
        [json.dumps(row, sort_keys=True) for source in ordered_sources for row in metadata[source]]
    )
    np.savez_compressed(output_dir / "encoded_inputs.npz", **combined)
    torch.save(activations, output_dir / "activations.pt")
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    _write_json(output_dir / "input_audit.json", audits)
    _write_json(output_dir / "summary.json", summary)
    logger.info(f"Wrote external-soccer MARL-GPT smoke outputs to {output_dir}")
    return summary


if __name__ == "__main__":
    main()

"""Run environment identity probes and counterfactual-token analyses for MARL-GPT."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf


ENV_TO_ID = {"smac": 1, "pogema": 2, "grf": 3}


@dataclass(frozen=True)
class ProbeDataset:
    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, default=_json_default, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_torch():
    import torch

    return torch


def _as_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _existing_data_path(root: Path, value: str | PathLike[str]) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)

    candidates = [root / path, root / "marl-gpt" / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _with_absolute_dataset_paths(root: Path, dataset_config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dataset_config)
    for env_cfg in resolved.values():
        env_cfg["folder_paths"] = [
            _existing_data_path(root, folder) for folder in env_cfg.get("folder_paths", [])
        ]
    return resolved


def _missing_dataset_paths(dataset_config: dict[str, Any]) -> list[str]:
    missing = []
    for env_cfg in dataset_config.values():
        for folder in env_cfg.get("folder_paths", []):
            if not Path(folder).exists():
                missing.append(str(folder))
    return missing


@contextmanager
def _marl_gpt_cwd(root: Path):
    old_cwd = Path.cwd()
    os.chdir(root / "marl-gpt")
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _marl_gpt_path(root: Path) -> None:
    path = str(root / "marl-gpt")
    if path not in sys.path:
        sys.path.insert(0, path)


def _to_plain_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[return-value]


def _resolve_dataset_config(root: Path, script_cfg: DictConfig) -> dict[str, Any]:
    if OmegaConf.select(script_cfg, "dataset_config") is not None:
        return _to_plain_config(script_cfg.dataset_config)

    config_path = _as_path(root, str(script_cfg.dataset_config_path))
    with config_path.open() as handle:
        payload = json.load(handle)
    split = str(OmegaConf.select(script_cfg, "dataset_config_split", default="train"))
    if split in payload:
        payload = payload[split]
    return payload


def _enabled_envs(dataset_config: dict[str, Any], requested: Iterable[str]) -> list[str]:
    envs = []
    for env in requested:
        if env not in ENV_TO_ID:
            raise SystemExit(f"Unknown environment {env!r}; expected one of {sorted(ENV_TO_ID)}")
        paths = dataset_config.get(env, {}).get("folder_paths", [])
        if paths:
            envs.append(env)
    if not envs:
        raise SystemExit("No enabled environments have non-empty folder_paths.")
    return envs


def _glob_pt_files(folder: Path, limit: int) -> list[Path]:
    files = sorted(folder.glob("*.pt"))
    if limit > 0:
        files = files[:limit]
    return files


def _tensor_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if hasattr(value, "shape"):
        summary["shape"] = list(value.shape)
    if hasattr(value, "dtype"):
        summary["dtype"] = str(value.dtype)
    if hasattr(value, "numel") and value.numel() > 0:
        try:
            summary["min"] = float(value.min().item())
            summary["max"] = float(value.max().item())
        except Exception:
            pass
    return summary


def inspect_dataset_files(root: Path, dataset_config: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    torch = _load_torch()
    max_files = int(OmegaConf.select(cfg, "max_inspect_files_per_folder", default=2))
    inspection: dict[str, Any] = {}
    dataset_config = _with_absolute_dataset_paths(root, dataset_config)

    for env, env_cfg in dataset_config.items():
        folder_rows = []
        for raw_folder in env_cfg.get("folder_paths", []):
            folder = _as_path(root, raw_folder)
            files = _glob_pt_files(folder, max_files)
            file_rows = []
            for file_path in files:
                try:
                    payload = torch.load(file_path, map_location="cpu")
                except Exception as exc:
                    file_rows.append({"path": str(file_path), "load_error": str(exc)})
                    continue
                keys = sorted(payload.keys()) if isinstance(payload, dict) else []
                file_rows.append(
                    {
                        "path": str(file_path),
                        "keys": keys,
                        "tensors": {key: _tensor_summary(payload[key]) for key in keys},
                    }
                )
            folder_rows.append(
                {
                    "folder": str(folder),
                    "exists": folder.exists(),
                    "num_pt_files_seen": len(files),
                    "files": file_rows,
                }
            )
        inspection[env] = {
            "config": env_cfg.get("config", {}),
            "folder_paths": folder_rows,
            "map_types": env_cfg.get("map_types", []),
        }
    return inspection


def _build_loader(root: Path, dataset_config: dict[str, Any], cfg: DictConfig):
    _marl_gpt_path(root)
    from utils.multi_env_dataset import MultiEnvAggregateDataset

    dataset_config = _with_absolute_dataset_paths(root, dataset_config)
    missing_paths = _missing_dataset_paths(dataset_config)
    if missing_paths:
        preview = "\n".join(f"- {path}" for path in missing_paths[:12])
        suffix = "" if len(missing_paths) <= 12 else f"\n... and {len(missing_paths) - 12} more"
        raise SystemExit(
            "Offline MARL-GPT dataset folders are missing. Install or link the Hugging Face "
            "`nortem/marl-gpt-datasets` tree so these paths exist:\n"
            f"{preview}{suffix}"
        )

    for env_cfg in dataset_config.values():
        env_cfg.setdefault("config", {})
        env_cfg["config"]["last_token"] = bool(OmegaConf.select(cfg, "last_token", default=True))
        env_cfg["config"]["env_specific"] = bool(OmegaConf.select(cfg, "env_specific", default=True))

    with _marl_gpt_cwd(root):
        return MultiEnvAggregateDataset(
            batch_size=int(cfg.batch_size),
            dataset_config=dataset_config,
            device=str(cfg.device),
            max_block_size=int(cfg.max_block_size),
            max_action_size=int(cfg.max_action_size),
        )


def _env_labels_for_batch(loader: Any) -> Any:
    torch = _load_torch()
    labels = []
    for env, batch_size in loader.batch_sizes.items():
        if env in loader.dataloaders and batch_size > 0:
            labels.extend([ENV_TO_ID[env]] * int(batch_size))
    return torch.tensor(labels, dtype=torch.long, device=loader.device)


def _copy_obs(batch_obs: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() for key, value in batch_obs.items()}


def _wrong_prompt_ids(true_env_ids: Any) -> Any:
    torch = _load_torch()
    choices = torch.tensor([1, 2, 3], device=true_env_ids.device)
    wrong = []
    for label in true_env_ids.tolist():
        candidates = choices[choices != int(label)]
        wrong.append(int(candidates[random.randrange(len(candidates))].item()))
    return torch.tensor(wrong, dtype=torch.long, device=true_env_ids.device)


def _flatten_feature(tensor: Any, *, max_columns: int) -> Any:
    torch = _load_torch()
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    if max_columns > 0 and flat.shape[1] > max_columns:
        stride = max(flat.shape[1] // max_columns, 1)
        flat = flat[:, ::stride][:, :max_columns]
    return torch.nan_to_num(flat)


def _feature_groups(batch_obs: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    torch = _load_torch()
    max_columns = int(OmegaConf.select(cfg, "max_feature_columns", default=4096))
    groups: dict[str, Any] = {}

    positional_keys = [key for key in ("group_pos", "agent_pos", "time_pos", "attr_pos") if key in batch_obs]
    for key in ("obs", "action_mask"):
        if key in batch_obs:
            groups[key] = _flatten_feature(batch_obs[key], max_columns=max_columns)
    if positional_keys:
        groups["positions"] = torch.cat(
            [_flatten_feature(batch_obs[key], max_columns=max_columns) for key in positional_keys],
            dim=1,
        )
    available = [groups[key] for key in ("obs", "action_mask", "positions") if key in groups]
    if available:
        groups["full_input"] = torch.cat(available, dim=1)
    groups["final_token"] = torch.stack(
        [
            batch_obs["obs"][:, -1].float(),
            batch_obs["attr_pos"][:, -1].float()
            if "attr_pos" in batch_obs
            else torch.zeros_like(batch_obs["obs"][:, -1]),
        ],
        dim=1,
    )
    return groups


def _standardize(train_x: Any, test_x: Any) -> tuple[Any, Any]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def _split_probe_dataset(x: Any, y: Any, train_fraction: float) -> ProbeDataset:
    torch = _load_torch()
    n = x.shape[0]
    indices = torch.randperm(n, device=x.device)
    train_n = max(1, min(n - 1, int(round(n * train_fraction))))
    train_idx = indices[:train_n]
    test_idx = indices[train_n:]
    train_x, test_x = _standardize(x[train_idx], x[test_idx])
    return ProbeDataset(train_x, y[train_idx], test_x, y[test_idx])


def _macro_f1(y_true: Any, y_pred: Any, labels: list[int]) -> float:
    scores = []
    for label in labels:
        pred_pos = y_pred == label
        true_pos = y_true == label
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def _confusion_matrix(y_true: Any, y_pred: Any, labels: list[int]) -> list[list[int]]:
    matrix = []
    for true_label in labels:
        row = []
        for pred_label in labels:
            row.append(int(((y_true == true_label) & (y_pred == pred_label)).sum().item()))
        matrix.append(row)
    return matrix


def train_linear_probe(x: Any, y: Any, cfg: DictConfig) -> dict[str, Any]:
    torch = _load_torch()
    labels = sorted(int(label) for label in y.unique().tolist())
    if len(labels) < 2 or x.shape[0] < 4:
        return {"status": "skipped", "reason": "need at least two labels and four examples"}

    dataset = _split_probe_dataset(x, y, float(OmegaConf.select(cfg, "train_fraction", default=0.7)))
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_train = torch.tensor([label_to_index[int(label)] for label in dataset.y_train.tolist()], device=x.device)
    y_test = torch.tensor([label_to_index[int(label)] for label in dataset.y_test.tolist()], device=x.device)

    model = torch.nn.Linear(dataset.x_train.shape[1], len(labels), device=x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(OmegaConf.select(cfg, "probe_lr", default=0.05)),
        weight_decay=float(OmegaConf.select(cfg, "probe_weight_decay", default=0.01)),
    )
    steps = int(OmegaConf.select(cfg, "probe_steps", default=200))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(dataset.x_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(dataset.x_test)
        pred_idx = logits.argmax(dim=1)
        pred = torch.tensor([labels[int(index)] for index in pred_idx.tolist()], device=x.device)
        accuracy = float((pred == dataset.y_test).float().mean().item())
        per_label_recall = {}
        for label in labels:
            mask = dataset.y_test == label
            per_label_recall[str(label)] = float((pred[mask] == label).float().mean().item()) if mask.any() else None

    return {
        "status": "ok",
        "n_train": int(dataset.x_train.shape[0]),
        "n_test": int(dataset.x_test.shape[0]),
        "n_features": int(dataset.x_train.shape[1]),
        "labels": labels,
        "accuracy": accuracy,
        "balanced_accuracy": float(
            sum(v for v in per_label_recall.values() if v is not None)
            / max(sum(v is not None for v in per_label_recall.values()), 1)
        ),
        "macro_f1": _macro_f1(dataset.y_test, pred, labels),
        "per_label_recall": per_label_recall,
        "confusion_matrix": _confusion_matrix(dataset.y_test, pred, labels),
    }


def _load_model(root: Path, cfg: DictConfig):
    torch = _load_torch()
    _marl_gpt_path(root)
    from gpt.inference import strip_prefix_from_state_dict
    from gpt.model_ac import CriticGPTConfig, CriticWithLoss

    checkpoint_path = _as_path(root, str(cfg.checkpoint))
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=str(cfg.device))
    model_config = CriticGPTConfig(**checkpoint["model_args"])
    model = CriticWithLoss(model_config)
    model.load_state_dict(strip_prefix_from_state_dict(checkpoint["model"]), strict=False)
    model.to(str(cfg.device))
    model.eval()
    return model, model_config


def _activation_hooks(model: Any, captured: dict[str, Any]):
    hooks = []

    def hook(name: str):
        def save_output(_module, _inputs, output):
            captured[name] = output.detach()

        return save_output

    hooks.append(model.transformer.drop.register_forward_hook(hook("embed")))
    for index, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(hook(f"layer_{index:02d}")))
    hooks.append(model.transformer.act_layers.register_forward_hook(hook("actor_layer")))
    hooks.append(model.transformer.critic_layers.register_forward_hook(hook("critic_layer")))
    return hooks


def _pooled_activations(captured: dict[str, Any]) -> dict[str, Any]:
    pooled = {}
    for name, tensor in captured.items():
        pooled[f"{name}:mean"] = tensor.mean(dim=1)
        pooled[f"{name}:final"] = tensor[:, -1, :]
    return pooled


def _entropy(logits: Any) -> Any:
    torch = _load_torch()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def collect_batches(root: Path, dataset_config: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    torch = _load_torch()
    torch.manual_seed(int(OmegaConf.select(cfg, "seed", default=0)))
    random.seed(int(OmegaConf.select(cfg, "seed", default=0)))

    loader = _build_loader(root, dataset_config, cfg)
    iterator = iter(loader)
    true_feature_tables: dict[str, list[Any]] = defaultdict(list)
    true_labels = []
    activation_tables: dict[str, list[Any]] = defaultdict(list)
    activation_true_labels = []
    activation_prompted_labels = []
    behavior_rows = []

    model = None
    model_config = None
    hooks = []
    if bool(OmegaConf.select(cfg, "run_model", default=False)):
        with _marl_gpt_cwd(root):
            model, model_config = _load_model(root, cfg)
            last_attr_pos = model_config.n_attr - 1
    else:
        last_attr_pos = None

    try:
        for batch_index in range(int(cfg.num_batches)):
            with _marl_gpt_cwd(root):
                batch_obs, _target, _mask_target, _batch_obs_next, _batch_info = next(iterator)
            env_labels = _env_labels_for_batch(loader)
            if env_labels.shape[0] != batch_obs["obs"].shape[0]:
                raise RuntimeError(
                    f"Label count {env_labels.shape[0]} does not match batch size {batch_obs['obs'].shape[0]}"
                )
            feature_groups = _feature_groups(batch_obs, cfg)
            for name, features in feature_groups.items():
                true_feature_tables[name].append(features.detach().cpu())
            true_labels.append(env_labels.detach().cpu())

            if model is None:
                continue

            prompt_sets = {"correct": env_labels, "wrong": _wrong_prompt_ids(env_labels)}
            for env_id in (1, 2, 3):
                prompt_sets[f"prompt_{env_id}"] = torch.full_like(env_labels, env_id)

            correct_logits = None
            for condition, prompt_ids in prompt_sets.items():
                prompted_obs = _copy_obs(batch_obs)
                prompted_obs["obs"][:, -1] = prompt_ids
                if last_attr_pos is not None:
                    prompted_obs["attr_pos"][:, -1] = last_attr_pos

                captured: dict[str, Any] = {}
                hooks = _activation_hooks(model, captured)
                with _marl_gpt_cwd(root), torch.no_grad():
                    act_logits, val_logits, _loss, _info = model(prompted_obs)
                for active_hook in hooks:
                    active_hook.remove()
                hooks = []

                if condition == "correct":
                    correct_logits = act_logits.detach()
                for name, features in _pooled_activations(captured).items():
                    activation_tables[name].append(features.detach().cpu())
                activation_true_labels.append(env_labels.detach().cpu())
                activation_prompted_labels.append(prompt_ids.detach().cpu())

                if correct_logits is not None:
                    delta = (act_logits - correct_logits).detach()
                    behavior_rows.append(
                        {
                            "batch": batch_index,
                            "condition": condition,
                            "mean_abs_action_logit_delta": float(delta.abs().mean().item()),
                            "mean_entropy": float(_entropy(act_logits).mean().item()),
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
        "behavior_rows": behavior_rows,
    }


def run_probes(collected: dict[str, Any], cfg: DictConfig) -> list[dict[str, Any]]:
    rows = []
    for source, feature_key, label_key in [
        ("input", "input_features", "input_true_labels"),
        ("activation_true_env", "activation_features", "activation_true_labels"),
        ("activation_prompted_env", "activation_features", "activation_prompted_labels"),
    ]:
        labels = collected.get(label_key)
        if labels is None:
            continue
        for name, features in collected[feature_key].items():
            result = train_linear_probe(features.to(str(cfg.device)), labels.to(str(cfg.device)), cfg)
            result.update({"source": source, "feature": name})
            rows.append(result)
    return rows


def main(cfg: DictConfig) -> dict[str, Any]:
    script_cfg = cfg.env_mechanism_probes
    root = _repo_root()
    output_dir = _as_path(root, str(script_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = _resolve_dataset_config(root, script_cfg)
    requested_envs = list(OmegaConf.select(script_cfg, "envs", default=["smac", "pogema", "grf"]))
    active_envs = _enabled_envs(dataset_config, requested_envs)
    dataset_config = {env: dataset_config[env] for env in active_envs}

    inspection = inspect_dataset_files(root, dataset_config, script_cfg)
    _write_json(output_dir / "dataset_inspection.json", inspection)

    collected = collect_batches(root, dataset_config, script_cfg)
    probe_rows = run_probes(collected, script_cfg)
    _write_json(output_dir / "probe_results.json", probe_rows)
    _write_csv(output_dir / "probe_results.csv", probe_rows)
    _write_csv(output_dir / "token_swap_behavior.csv", collected["behavior_rows"])
    _write_json(
        output_dir / "summary.json",
        {
            "active_envs": active_envs,
            "num_input_examples": int(collected["input_true_labels"].shape[0])
            if collected["input_true_labels"] is not None
            else 0,
            "num_activation_examples": int(collected["activation_true_labels"].shape[0])
            if collected["activation_true_labels"] is not None
            else 0,
            "probe_rows": len(probe_rows),
            "behavior_rows": len(collected["behavior_rows"]),
            "config": _to_plain_config(cfg),
        },
    )
    logger.info(f"Wrote environment mechanism probe outputs to {output_dir}")
    return {"probe_rows": probe_rows, "behavior_rows": collected["behavior_rows"]}

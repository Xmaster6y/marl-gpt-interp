"""Reusable utilities for MARL-GPT dataset, activation, probe, and gradient analyses."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


ENV_TO_ID = {"smac": 1, "pogema": 2, "grf": 3}
ID_TO_ENV = {value: key for key, value in ENV_TO_ID.items()}


@dataclass(frozen=True)
class ProbeDataset:
    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, default=json_default, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_torch():
    import torch

    return torch


def as_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def to_plain_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[return-value]


def resolve_dataset_config(root: Path, script_cfg: DictConfig) -> dict[str, Any]:
    if OmegaConf.select(script_cfg, "dataset_config") is not None:
        return to_plain_config(script_cfg.dataset_config)

    config_path = as_path(root, str(script_cfg.dataset_config_path))
    with config_path.open() as handle:
        payload = json.load(handle)
    split = str(OmegaConf.select(script_cfg, "dataset_config_split", default="train"))
    if split in payload:
        payload = payload[split]
    return payload


def enabled_envs(dataset_config: dict[str, Any], requested: Iterable[str]) -> list[str]:
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


def _existing_data_path(root: Path, value: str | PathLike[str]) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)

    candidates = [root / path, root / "marl-gpt" / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def with_absolute_dataset_paths(root: Path, dataset_config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dataset_config)
    for env_cfg in resolved.values():
        env_cfg["folder_paths"] = [
            _existing_data_path(root, folder) for folder in env_cfg.get("folder_paths", [])
        ]
    return resolved


def missing_dataset_paths(dataset_config: dict[str, Any]) -> list[str]:
    missing = []
    for env_cfg in dataset_config.values():
        for folder in env_cfg.get("folder_paths", []):
            if not Path(folder).exists():
                missing.append(str(folder))
    return missing


@contextmanager
def marl_gpt_cwd(root: Path):
    old_cwd = Path.cwd()
    os.chdir(root / "marl-gpt")
    try:
        yield
    finally:
        os.chdir(old_cwd)


def marl_gpt_path(root: Path) -> None:
    path = str(root / "marl-gpt")
    if path not in sys.path:
        sys.path.insert(0, path)


def _glob_pt_files(folder: Path, limit: int) -> list[Path]:
    files = sorted(folder.glob("*.pt"))
    if limit > 0:
        files = files[:limit]
    return files


def tensor_summary(value: Any) -> dict[str, Any]:
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
    torch = load_torch()
    max_files = int(OmegaConf.select(cfg, "max_inspect_files_per_folder", default=2))
    inspection: dict[str, Any] = {}
    dataset_config = with_absolute_dataset_paths(root, dataset_config)

    for env, env_cfg in dataset_config.items():
        folder_rows = []
        for raw_folder in env_cfg.get("folder_paths", []):
            folder = as_path(root, raw_folder)
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
                        "tensors": {key: tensor_summary(payload[key]) for key in keys},
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


def build_loader(root: Path, dataset_config: dict[str, Any], cfg: DictConfig):
    marl_gpt_path(root)
    from utils.multi_env_dataset import MultiEnvAggregateDataset

    dataset_config = with_absolute_dataset_paths(root, dataset_config)
    missing_paths = missing_dataset_paths(dataset_config)
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

    with marl_gpt_cwd(root):
        return MultiEnvAggregateDataset(
            batch_size=int(cfg.batch_size),
            dataset_config=dataset_config,
            device=str(cfg.device),
            max_block_size=int(cfg.max_block_size),
            max_action_size=int(cfg.max_action_size),
        )


def env_labels_for_batch(loader: Any) -> Any:
    torch = load_torch()
    labels = []
    for env, batch_size in loader.batch_sizes.items():
        if env in loader.dataloaders and batch_size > 0:
            labels.extend([ENV_TO_ID[env]] * int(batch_size))
    return torch.tensor(labels, dtype=torch.long, device=loader.device)


def copy_obs(batch_obs: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() for key, value in batch_obs.items()}


def flatten_feature(tensor: Any, *, max_columns: int) -> Any:
    torch = load_torch()
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    if max_columns > 0 and flat.shape[1] > max_columns:
        stride = max(flat.shape[1] // max_columns, 1)
        flat = flat[:, ::stride][:, :max_columns]
    return torch.nan_to_num(flat)


def feature_groups(batch_obs: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    torch = load_torch()
    max_columns = int(OmegaConf.select(cfg, "max_feature_columns", default=4096))
    groups: dict[str, Any] = {}

    positional_keys = [key for key in ("group_pos", "agent_pos", "time_pos", "attr_pos") if key in batch_obs]
    for key in ("obs", "action_mask"):
        if key in batch_obs:
            groups[key] = flatten_feature(batch_obs[key], max_columns=max_columns)
    if positional_keys:
        groups["positions"] = torch.cat(
            [flatten_feature(batch_obs[key], max_columns=max_columns) for key in positional_keys],
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


def standardize(train_x: Any, test_x: Any) -> tuple[Any, Any]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def split_probe_dataset(x: Any, y: Any, train_fraction: float) -> ProbeDataset:
    torch = load_torch()
    n = x.shape[0]
    indices = torch.randperm(n, device=x.device)
    train_n = max(1, min(n - 1, int(round(n * train_fraction))))
    train_idx = indices[:train_n]
    test_idx = indices[train_n:]
    train_x, test_x = standardize(x[train_idx], x[test_idx])
    return ProbeDataset(train_x, y[train_idx], test_x, y[test_idx])


def macro_f1(y_true: Any, y_pred: Any, labels: list[int]) -> float:
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


def confusion_matrix(y_true: Any, y_pred: Any, labels: list[int]) -> list[list[int]]:
    matrix = []
    for true_label in labels:
        row = []
        for pred_label in labels:
            row.append(int(((y_true == true_label) & (y_pred == pred_label)).sum().item()))
        matrix.append(row)
    return matrix


def cosine(a: Any, b: Any) -> float:
    torch = load_torch()
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item())


def top_abs_indices(vector: Any, limit: int) -> list[int]:
    if limit <= 0:
        return []
    k = min(limit, int(vector.numel()))
    if k == 0:
        return []
    return [int(index) for index in vector.detach().float().abs().topk(k).indices.tolist()]


def train_linear_probe(x: Any, y: Any, cfg: DictConfig) -> dict[str, Any]:
    torch = load_torch()
    labels = sorted(int(label) for label in y.unique().tolist())
    if len(labels) < 2 or x.shape[0] < 4:
        return {"status": "skipped", "reason": "need at least two labels and four examples"}

    dataset = split_probe_dataset(x, y, float(OmegaConf.select(cfg, "train_fraction", default=0.7)))
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

    with torch.no_grad():
        weights = model.weight.detach().cpu()
        class_weight_norms = {
            str(label): float(weights[index].norm().item()) for index, label in enumerate(labels)
        }
        pairwise_weight_cosine = {}
        for left_index, left_label in enumerate(labels):
            for right_index, right_label in enumerate(labels[left_index + 1 :], start=left_index + 1):
                pairwise_weight_cosine[f"{left_label}_vs_{right_label}"] = cosine(
                    weights[left_index], weights[right_index]
                )

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
        "macro_f1": macro_f1(dataset.y_test, pred, labels),
        "per_label_recall": per_label_recall,
        "confusion_matrix": confusion_matrix(dataset.y_test, pred, labels),
        "class_weight_norms": class_weight_norms,
        "pairwise_weight_cosine": pairwise_weight_cosine,
        "top_weight_features": {
            str(label): top_abs_indices(
                weights[index],
                int(OmegaConf.select(cfg, "top_direction_features", default=8)),
            )
            for index, label in enumerate(labels)
        },
    }


def load_model(root: Path, cfg: DictConfig):
    torch = load_torch()
    marl_gpt_path(root)
    from gpt.inference import strip_prefix_from_state_dict
    from gpt.model_ac import CriticGPTConfig, CriticWithLoss

    checkpoint_path = as_path(root, str(cfg.checkpoint))
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=str(cfg.device))
    model_config = CriticGPTConfig(**checkpoint["model_args"])
    model = CriticWithLoss(model_config)
    model.load_state_dict(strip_prefix_from_state_dict(checkpoint["model"]), strict=False)
    model.to(str(cfg.device))
    model.eval()
    return model, model_config


def activation_hooks(model: Any, captured: dict[str, Any]):
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


def pooled_activations(captured: dict[str, Any]) -> dict[str, Any]:
    pooled = {}
    for name, tensor in captured.items():
        pooled[f"{name}:mean"] = tensor.mean(dim=1)
        pooled[f"{name}:final"] = tensor[:, -1, :]
    return pooled


def slice_batch(batch: Any, mask: Any) -> Any:
    if isinstance(batch, dict):
        return {key: slice_batch(value, mask) for key, value in batch.items()}
    if hasattr(batch, "shape") and batch.shape[:1] == mask.shape[:1]:
        return batch[mask]
    return batch


def parameter_group(name: str) -> str:
    if "obs_encoder" in name or "wte" in name:
        return "token_embeddings"
    if name.startswith("transformer.h."):
        parts = name.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            return f"layer_{int(parts[2]):02d}"
    if "actor" in name:
        return "actor_head"
    if "critic" in name:
        return "critic_head"
    if "pos" in name or "emb" in name:
        return "position_embeddings"
    return "other"


def named_gradient_groups(model: Any) -> dict[str, Any]:
    torch = load_torch()
    groups: dict[str, list[Any]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        groups[parameter_group(name)].append(parameter.grad.detach().float().flatten().cpu())
    return {name: torch.cat(chunks) for name, chunks in groups.items() if chunks}


def collect_parameter_gradients_by_env(
    *,
    root: Path,
    model: Any,
    batch_index: int,
    batch_obs: dict[str, Any],
    target: Any,
    mask_target: Any,
    batch_obs_next: dict[str, Any],
    batch_info: dict[str, Any],
    env_labels: Any,
    parameter_rows: list[dict[str, Any]],
    gradient_sums: dict[str, dict[int, Any]],
    gradient_counts: dict[str, dict[int, int]],
) -> None:
    torch = load_torch()
    for env_id in sorted(int(label) for label in env_labels.unique().tolist()):
        env_mask = env_labels == env_id
        if not bool(env_mask.any()):
            continue
        env_obs = slice_batch(batch_obs, env_mask)
        env_next_obs = slice_batch(batch_obs_next, env_mask)
        env_info = slice_batch(batch_info, env_mask)
        env_target = target[env_mask].long()
        env_mask_target = mask_target[env_mask].long()
        try:
            model.zero_grad(set_to_none=True)
            with marl_gpt_cwd(root), torch.enable_grad():
                target_val = model.calculate_target_val(env_obs, env_next_obs, env_info)
                _act_logits, _val_logits, loss, loss_info = model(env_obs, env_target, env_mask_target, target_val)
                if loss is None:
                    raise RuntimeError("model returned no loss")
                loss.backward()
            for group_name, gradient in named_gradient_groups(model).items():
                parameter_rows.append(
                    {
                        "batch": batch_index,
                        "env": ID_TO_ENV.get(env_id, str(env_id)),
                        "env_id": env_id,
                        "parameter_group": group_name,
                        "gradient_l2": float(gradient.norm().item()),
                        "gradient_l1_mean": float(gradient.abs().mean().item()),
                        "gradient_abs_max": float(gradient.abs().max().item()),
                        "loss": float(loss.detach().item()),
                        "bc_loss": float(loss_info["bc_loss"].detach().item())
                        if loss_info.get("bc_loss") is not None
                        else None,
                        "critic_loss": float(loss_info["critic_loss"].detach().item())
                        if loss_info.get("critic_loss") is not None
                        else None,
                    }
                )
                existing = gradient_sums[group_name].get(env_id)
                gradient_sums[group_name][env_id] = gradient.clone() if existing is None else existing + gradient
                gradient_counts[group_name][env_id] += 1
        except Exception as exc:
            parameter_rows.append(
                {
                    "batch": batch_index,
                    "env": ID_TO_ENV.get(env_id, str(env_id)),
                    "env_id": env_id,
                    "parameter_group": "all",
                    "status": "failed",
                    "error": str(exc),
                }
            )
        finally:
            model.zero_grad(set_to_none=True)


def entropy(logits: Any) -> Any:
    torch = load_torch()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def condition_mask(conditions: list[str], requested: list[str]):
    torch = load_torch()
    if not conditions or not requested:
        return None
    requested_set = set(requested)
    return torch.tensor([condition in requested_set for condition in conditions], dtype=torch.bool)


def activation_direction_rows(
    activation_features: dict[str, Any],
    labels: Any,
    *,
    cfg: DictConfig,
    conditions: list[str] | None = None,
    requested_conditions: list[str] | None = None,
) -> list[dict[str, Any]]:
    torch = load_torch()
    rows = []
    if labels is None:
        return rows

    condition_label = "all"
    base_mask = None
    if conditions and requested_conditions:
        base_mask = condition_mask(conditions, requested_conditions)
        condition_label = ",".join(requested_conditions)

    for feature_name, features in activation_features.items():
        feature_values = features.float() if base_mask is None else features[base_mask].float()
        feature_labels = labels if base_mask is None else labels[base_mask]
        env_ids = sorted(int(label) for label in feature_labels.unique().tolist())
        if len(env_ids) < 2:
            continue
        means = {}
        for env_id in env_ids:
            env_features = feature_values[feature_labels == env_id]
            if env_features.numel() == 0:
                continue
            means[env_id] = env_features.mean(dim=0)
            rows.append(
                {
                    "feature": feature_name,
                    "condition": condition_label,
                    "direction_type": "class_mean",
                    "env": ID_TO_ENV.get(env_id, str(env_id)),
                    "env_id": env_id,
                    "n_examples": int(env_features.shape[0]),
                    "mean_l2": float(means[env_id].norm().item()),
                    "mean_abs": float(means[env_id].abs().mean().item()),
                    "top_abs_features": top_abs_indices(
                        means[env_id],
                        int(OmegaConf.select(cfg, "top_direction_features", default=8)),
                    ),
                }
            )
        for left_index, left_env in enumerate(env_ids):
            for right_env in env_ids[left_index + 1 :]:
                if left_env not in means or right_env not in means:
                    continue
                delta = means[left_env] - means[right_env]
                rows.append(
                    {
                        "feature": feature_name,
                        "condition": condition_label,
                        "direction_type": "mean_difference",
                        "env_pair": f"{ID_TO_ENV.get(left_env, left_env)}_vs_{ID_TO_ENV.get(right_env, right_env)}",
                        "left_env_id": left_env,
                        "right_env_id": right_env,
                        "cosine_between_means": cosine(means[left_env], means[right_env]),
                        "difference_l2": float(delta.norm().item()),
                        "difference_abs_mean": float(delta.abs().mean().item()),
                        "top_abs_features": top_abs_indices(
                            delta,
                            int(OmegaConf.select(cfg, "top_direction_features", default=8)),
                        ),
                    }
                )
    return rows


def parameter_gradient_cosine_rows(
    gradient_sums: dict[str, Any],
    gradient_counts: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for group_name, env_gradients in gradient_sums.items():
        averaged = {}
        for env_id, gradient in env_gradients.items():
            count = int(gradient_counts[group_name].get(env_id, 0))
            if count <= 0:
                continue
            averaged[env_id] = gradient / count
            rows.append(
                {
                    "parameter_group": group_name,
                    "direction_type": "mean_gradient",
                    "env": ID_TO_ENV.get(int(env_id), str(env_id)),
                    "env_id": int(env_id),
                    "n_batches": count,
                    "gradient_l2": float(averaged[env_id].norm().item()),
                    "gradient_abs_mean": float(averaged[env_id].abs().mean().item()),
                    "gradient_abs_max": float(averaged[env_id].abs().max().item()),
                }
            )
        env_ids = sorted(int(env_id) for env_id in averaged)
        for left_index, left_env in enumerate(env_ids):
            for right_env in env_ids[left_index + 1 :]:
                rows.append(
                    {
                        "parameter_group": group_name,
                        "direction_type": "gradient_cosine",
                        "env_pair": f"{ID_TO_ENV.get(left_env, left_env)}_vs_{ID_TO_ENV.get(right_env, right_env)}",
                        "left_env_id": left_env,
                        "right_env_id": right_env,
                        "cosine": cosine(averaged[left_env], averaged[right_env]),
                    }
                )
    return rows


def linear_cka(x: Any, y: Any) -> float:
    x = x.detach().float()
    y = y.detach().float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = (xy * xy).sum()
    denominator = ((xx * xx).sum().sqrt() * (yy * yy).sum().sqrt()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _limited_examples(features: Any, max_examples: int) -> Any:
    if max_examples > 0 and features.shape[0] > max_examples:
        return features[:max_examples]
    return features


def _safe_float(value: Any) -> float:
    return float(value.detach().float().cpu().item()) if hasattr(value, "detach") else float(value)


def _upper_triangle_values(matrix: Any) -> Any:
    torch = load_torch()
    if matrix.shape[0] < 2:
        return torch.empty(0, device=matrix.device, dtype=matrix.dtype)
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1, device=matrix.device)
    return matrix[indices[0], indices[1]]


def _pca_variance_summary(centered: Any, max_rank: int) -> dict[str, Any]:
    torch = load_torch()
    if centered.shape[0] < 2 or centered.numel() == 0:
        return {
            "pca_rank": 0,
            "pca_top1_var": None,
            "pca_top5_var": None,
            "pca_top10_var": None,
            "pca_components_90": None,
            "participation_ratio": None,
            "effective_rank": None,
        }
    singular_values = torch.linalg.svdvals(centered)
    if max_rank > 0:
        singular_values = singular_values[:max_rank]
    variances = singular_values.square()
    total = variances.sum().clamp_min(1e-12)
    ratios = variances / total
    cumulative = ratios.cumsum(dim=0)
    entropy_value = -(ratios * ratios.clamp_min(1e-12).log()).sum()
    components_90 = int((cumulative < 0.9).sum().item()) + 1
    return {
        "pca_rank": int(variances.numel()),
        "pca_top1_var": _safe_float(cumulative[min(0, cumulative.numel() - 1)]),
        "pca_top5_var": _safe_float(cumulative[min(4, cumulative.numel() - 1)]),
        "pca_top10_var": _safe_float(cumulative[min(9, cumulative.numel() - 1)]),
        "pca_components_90": min(components_90, int(cumulative.numel())),
        "participation_ratio": _safe_float(total.square() / variances.square().sum().clamp_min(1e-12)),
        "effective_rank": _safe_float(entropy_value.exp()),
    }


def representation_proximity_rows(
    activation_features: dict[str, Any],
    labels: Any,
    *,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Summarize within-environment compactness for each activation feature."""

    torch = load_torch()
    if labels is None:
        return []
    max_examples = int(OmegaConf.select(cfg, "max_pairwise_examples_per_env", default=256))
    max_rank = int(OmegaConf.select(cfg, "max_pca_rank", default=128))
    rows = []
    for feature_name, features in activation_features.items():
        for env_id in sorted(int(label) for label in labels.unique().tolist()):
            env_features = _limited_examples(features[labels == env_id].float(), max_examples)
            n_examples = int(env_features.shape[0])
            if n_examples < 2:
                continue
            centroid = env_features.mean(dim=0)
            centered = env_features - centroid
            l2_to_centroid = centered.norm(dim=1)
            pairwise_l2 = torch.pdist(env_features, p=2)
            normalized = torch.nn.functional.normalize(env_features, dim=1, eps=1e-12)
            pairwise_cosine_distance = 1 - _upper_triangle_values(normalized @ normalized.T)
            centroid_cosine = torch.nn.functional.cosine_similarity(
                env_features,
                centroid.unsqueeze(0),
                dim=1,
                eps=1e-12,
            )
            row = {
                "feature": feature_name,
                "env": ID_TO_ENV.get(env_id, str(env_id)),
                "env_id": env_id,
                "n_examples": n_examples,
                "n_features": int(env_features.shape[1]),
                "centroid_l2": float(centroid.norm().item()),
                "mean_l2_to_centroid": float(l2_to_centroid.mean().item()),
                "std_l2_to_centroid": float(l2_to_centroid.std(unbiased=False).item()),
                "max_l2_to_centroid": float(l2_to_centroid.max().item()),
                "mean_pairwise_l2": float(pairwise_l2.mean().item()),
                "std_pairwise_l2": float(pairwise_l2.std(unbiased=False).item()),
                "mean_pairwise_cosine_distance": float(pairwise_cosine_distance.mean().item()),
                "std_pairwise_cosine_distance": float(pairwise_cosine_distance.std(unbiased=False).item()),
                "mean_cosine_to_centroid": float(centroid_cosine.mean().item()),
                "std_cosine_to_centroid": float(centroid_cosine.std(unbiased=False).item()),
            }
            row.update(_pca_variance_summary(centered, max_rank))
            rows.append(row)
    return rows


def representation_separation_rows(
    activation_features: dict[str, Any],
    labels: Any,
    *,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Summarize between-environment separation normalized by within-env spread."""

    torch = load_torch()
    if labels is None:
        return []
    max_examples = int(OmegaConf.select(cfg, "max_pairwise_examples_per_env", default=256))
    rows = []
    env_ids = sorted(int(label) for label in labels.unique().tolist())
    for feature_name, features in activation_features.items():
        env_features = {
            env_id: _limited_examples(features[labels == env_id].float(), max_examples) for env_id in env_ids
        }
        for left_index, left_env in enumerate(env_ids):
            left = env_features[left_env]
            if left.shape[0] < 2:
                continue
            left_centroid = left.mean(dim=0)
            left_centered = left - left_centroid
            left_within_sq = left_centered.square().sum(dim=1).mean()
            for right_env in env_ids[left_index + 1 :]:
                right = env_features[right_env]
                if right.shape[0] < 2:
                    continue
                right_centroid = right.mean(dim=0)
                right_centered = right - right_centroid
                right_within_sq = right_centered.square().sum(dim=1).mean()
                centroid_delta = left_centroid - right_centroid
                centroid_l2 = centroid_delta.norm()
                pooled_within_sq = ((left_within_sq + right_within_sq) / 2).clamp_min(1e-12)

                cross_dist = torch.cdist(left, right, p=2)
                left_dist = torch.cdist(left, left, p=2)
                right_dist = torch.cdist(right, right, p=2)
                left_dist.fill_diagonal_(float("inf"))
                right_dist.fill_diagonal_(float("inf"))
                left_same_nearest = left_dist.min(dim=1).values < cross_dist.min(dim=1).values
                right_same_nearest = right_dist.min(dim=1).values < cross_dist.min(dim=0).values
                same_env_nn_fraction = torch.cat([left_same_nearest, right_same_nearest]).float().mean()

                left_a = left_dist.masked_fill(torch.isinf(left_dist), 0).sum(dim=1) / max(left.shape[0] - 1, 1)
                right_a = right_dist.masked_fill(torch.isinf(right_dist), 0).sum(dim=1) / max(right.shape[0] - 1, 1)
                left_b = cross_dist.mean(dim=1)
                right_b = cross_dist.mean(dim=0)
                silhouette_values = torch.cat(
                    [
                        (left_b - left_a) / torch.maximum(left_a, left_b).clamp_min(1e-12),
                        (right_b - right_a) / torch.maximum(right_a, right_b).clamp_min(1e-12),
                    ]
                )
                energy_distance = (
                    2 * cross_dist.mean() - torch.pdist(left, p=2).mean() - torch.pdist(right, p=2).mean()
                )
                rows.append(
                    {
                        "feature": feature_name,
                        "env_pair": f"{ID_TO_ENV.get(left_env, left_env)}_vs_{ID_TO_ENV.get(right_env, right_env)}",
                        "left_env_id": left_env,
                        "right_env_id": right_env,
                        "n_left": int(left.shape[0]),
                        "n_right": int(right.shape[0]),
                        "centroid_l2": float(centroid_l2.item()),
                        "normalized_centroid_l2": float((centroid_l2 / pooled_within_sq.sqrt()).item()),
                        "fisher_ratio": float((centroid_l2.square() / pooled_within_sq).item()),
                        "mean_cross_l2": float(cross_dist.mean().item()),
                        "energy_distance": float(energy_distance.item()),
                        "silhouette_l2": float(silhouette_values.mean().item()),
                        "same_env_nearest_neighbor_fraction": float(same_env_nn_fraction.item()),
                    }
                )
    return rows


def asymmetric_subspace_rows(
    activation_features: dict[str, Any],
    labels: Any,
    *,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Measure directed PCA subspace containment between environment activations."""

    torch = load_torch()
    if labels is None:
        return []
    max_examples = int(OmegaConf.select(cfg, "max_pairwise_examples_per_env", default=256))
    top_ks = [int(value) for value in OmegaConf.select(cfg, "asymmetric_top_ks", default=[1, 2, 4, 8, 16, 32, 64])]
    rows = []
    env_ids = sorted(int(label) for label in labels.unique().tolist())
    for feature_name, features in activation_features.items():
        env_features = {
            env_id: _limited_examples(features[labels == env_id].float(), max_examples) for env_id in env_ids
        }
        bases = {}
        centered_values = {}
        total_variances = {}
        for env_id, values in env_features.items():
            if values.shape[0] < 2:
                continue
            centered = values - values.mean(dim=0, keepdim=True)
            try:
                _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
            except RuntimeError:
                continue
            bases[env_id] = vh
            centered_values[env_id] = centered
            total_variances[env_id] = centered.square().sum().clamp_min(1e-12)

        for source_env in env_ids:
            if source_env not in bases:
                continue
            source_basis = bases[source_env]
            max_k = int(source_basis.shape[0])
            for target_env in env_ids:
                if source_env == target_env or target_env not in centered_values:
                    continue
                target = centered_values[target_env]
                total_target_variance = total_variances[target_env]
                for requested_k in top_ks:
                    k = min(requested_k, max_k)
                    if k <= 0:
                        continue
                    basis = source_basis[:k].T
                    projected = target @ basis @ basis.T
                    explained = projected.square().sum() / total_target_variance
                    rows.append(
                        {
                            "feature": feature_name,
                            "source_env": ID_TO_ENV.get(source_env, str(source_env)),
                            "target_env": ID_TO_ENV.get(target_env, str(target_env)),
                            "source_env_id": source_env,
                            "target_env_id": target_env,
                            "direction": (
                                f"{ID_TO_ENV.get(source_env, source_env)}_to_"
                                f"{ID_TO_ENV.get(target_env, target_env)}"
                            ),
                            "method": "pca_subspace_containment",
                            "requested_rank": requested_k,
                            "rank": k,
                            "n_source": int(env_features[source_env].shape[0]),
                            "n_target": int(env_features[target_env].shape[0]),
                            "target_variance_explained": float(explained.item()),
                        }
                    )
    return rows


def activation_subspace_similarity_rows(activation_features: dict[str, Any], labels: Any) -> list[dict[str, Any]]:
    rows = []
    if labels is None:
        return rows
    env_ids = sorted(int(label) for label in labels.unique().tolist())
    for feature_name, features in activation_features.items():
        for left_index, left_env in enumerate(env_ids):
            left = features[labels == left_env].float()
            for right_env in env_ids[left_index + 1 :]:
                right = features[labels == right_env].float()
                n = min(int(left.shape[0]), int(right.shape[0]))
                if n < 2:
                    continue
                rows.append(
                    {
                        "feature": feature_name,
                        "env_pair": f"{ID_TO_ENV.get(left_env, left_env)}_vs_{ID_TO_ENV.get(right_env, right_env)}",
                        "left_env_id": left_env,
                        "right_env_id": right_env,
                        "n_examples_per_env": n,
                        "linear_cka": linear_cka(left[:n], right[:n]),
                    }
                )
    return rows

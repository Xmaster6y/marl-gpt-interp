"""Minimal MARL-GPT model and dataset integration for circuit tracing."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from os import PathLike
from pathlib import Path
from types import MethodType
from typing import Any

from omegaconf import DictConfig, OmegaConf


ENV_TO_ID = {"smac": 1, "pogema": 2, "grf": 3}
ID_TO_ENV = {value: key for key, value in ENV_TO_ID.items()}


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
    payload = json.loads(config_path.read_text())
    split = str(OmegaConf.select(script_cfg, "dataset_config_split", default="train"))
    return payload[split] if split in payload else payload


def enabled_envs(dataset_config: dict[str, Any], requested: Iterable[str]) -> list[str]:
    envs = []
    for environment in requested:
        if environment not in ENV_TO_ID:
            raise SystemExit(f"Unknown environment {environment!r}; expected one of {sorted(ENV_TO_ID)}")
        if dataset_config.get(environment, {}).get("folder_paths", []):
            envs.append(environment)
    if not envs:
        raise SystemExit("No enabled environments have non-empty folder_paths.")
    return envs


def _existing_data_path(root: Path, value: str | PathLike[str]) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidates = (root / path, root / "marl-gpt" / path)
    return str(next((candidate for candidate in candidates if candidate.exists()), candidates[0]))


def with_absolute_dataset_paths(root: Path, dataset_config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dataset_config)
    for environment in resolved.values():
        environment["folder_paths"] = [
            _existing_data_path(root, folder) for folder in environment.get("folder_paths", [])
        ]
    return resolved


@contextmanager
def marl_gpt_cwd(root: Path):
    previous = Path.cwd()
    os.chdir(root / "marl-gpt")
    try:
        yield
    finally:
        os.chdir(previous)


def marl_gpt_path(root: Path) -> None:
    path = str(root / "marl-gpt")
    if path not in sys.path:
        sys.path.insert(0, path)


def build_loader(root: Path, dataset_config: dict[str, Any], cfg: DictConfig):
    marl_gpt_path(root)
    from utils.multi_env_dataset import MultiEnvAggregateDataset

    dataset_config = with_absolute_dataset_paths(root, dataset_config)
    missing = [
        folder
        for environment in dataset_config.values()
        for folder in environment.get("folder_paths", [])
        if not Path(folder).exists()
    ]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:12])
        suffix = "" if len(missing) <= 12 else f"\n... and {len(missing) - 12} more"
        raise SystemExit(f"Offline MARL-GPT dataset folders are missing:\n{preview}{suffix}")
    for environment in dataset_config.values():
        environment.setdefault("config", {})
        environment["config"]["last_token"] = bool(OmegaConf.select(cfg, "last_token", default=True))
        environment["config"]["env_specific"] = bool(OmegaConf.select(cfg, "env_specific", default=True))
    with marl_gpt_cwd(root):
        return MultiEnvAggregateDataset(
            batch_size=int(cfg.batch_size),
            dataset_config=dataset_config,
            device=str(cfg.device),
            max_block_size=int(cfg.max_block_size),
            max_action_size=int(cfg.max_action_size),
        )


def _stable_source_id(path: str) -> int:
    normalized = Path(path).as_posix()
    return int.from_bytes(hashlib.sha256(normalized.encode()).digest()[:8], "big") % (2**63 - 1)


def enable_sample_identity(
    loader: Any,
    *,
    max_rows_per_source: int = 0,
    max_rows_by_source: Mapping[int, int] | None = None,
) -> dict[int, str]:
    """Instrument the vendored loader with stable source-file and row identities."""

    torch = load_torch()
    source_paths: dict[int, str] = {}
    source_caps = max_rows_by_source if max_rows_by_source is not None else {}
    for aggregate in loader.dataloaders.values():
        for critic_loader in aggregate.datasets:
            data_loader = critic_loader.dataloader
            for path in data_loader.file_paths:
                source_paths[_stable_source_id(path)] = Path(path).as_posix()
            original_load = data_loader.load_and_transfer_data_file

            def load_with_identity(self, filename, *, _original=original_load):
                result = _original(filename)
                self._sample_identity_source_id = _stable_source_id(filename)
                self._sample_identity_row_offset = 0
                if getattr(self, "_load_part_file", False):
                    part_index = int(self._indx_part) - 2
                    self._sample_identity_row_offset = int(self._all_indx_part[part_index])
                row_cap = int(source_caps.get(self._sample_identity_source_id, max_rows_per_source))
                if row_cap > 0:
                    self.indices = self.indices[:row_cap]
                return result

            data_loader.load_and_transfer_data_file = MethodType(load_with_identity, data_loader)
            original_extra = critic_loader.add_extra_information_for_critic

            def extra_with_identity(self, first_index, *, _original=original_extra):
                info = _original(first_index)
                inner = self.dataloader
                indices = inner.indices[first_index : first_index + inner.batch_size]
                info["source_file_id"] = torch.full_like(indices, inner._sample_identity_source_id, dtype=torch.long)
                info["source_row_index"] = indices.to(device=inner.device, dtype=torch.long) + int(
                    inner._sample_identity_row_offset
                )
                return info

            critic_loader.add_extra_information_for_critic = MethodType(extra_with_identity, critic_loader)
    loader.sample_identity_sources = source_paths
    return source_paths


def env_labels_for_batch(loader: Any) -> Any:
    torch = load_torch()
    labels = [
        ENV_TO_ID[environment]
        for environment, batch_size in loader.batch_sizes.items()
        if environment in loader.dataloaders and batch_size > 0
        for _ in range(int(batch_size))
    ]
    return torch.tensor(labels, dtype=torch.long, device=loader.device)


def load_model(root: Path, cfg: DictConfig):
    torch = load_torch()
    marl_gpt_path(root)
    from gpt.inference import strip_prefix_from_state_dict
    from gpt.model_ac import CriticGPTConfig, CriticWithLoss

    checkpoint_path = as_path(root, str(cfg.checkpoint))
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=str(cfg.device), weights_only=False)
    model_config = CriticGPTConfig(**checkpoint["model_args"])
    model = CriticWithLoss(model_config)
    model.load_state_dict(strip_prefix_from_state_dict(checkpoint["model"]), strict=False)
    model.to(str(cfg.device)).eval()
    return model, model_config


def slice_batch(batch: Any, mask: Any) -> Any:
    if isinstance(batch, dict):
        return {key: slice_batch(value, mask) for key, value in batch.items()}
    if hasattr(batch, "shape") and batch.shape[:1] == mask.shape[:1]:
        return batch[mask]
    return batch

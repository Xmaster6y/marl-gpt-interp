"""Production training utilities for cached-activation TopK SAEs."""

from __future__ import annotations

import json
import importlib
import importlib.metadata
import math
import os
import sys
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from marl_gpt_interp.sparse_features import sparse_metrics


def fit_activation_preprocessing(
    x: torch.Tensor,
    labels: torch.Tensor,
    domains: Sequence[str],
    mode: str,
) -> dict[str, Any]:
    """Fit a declared train-only activation transform for SAE diagnostics."""

    if mode == "natural":
        return {"mode": mode, "fit_split": "train", "metric_space": "natural_activation"}
    if mode != "per_domain_center_rms":
        raise ValueError(f"Unknown activation preprocessing mode {mode!r}")
    means, rms_scales = [], []
    for label, domain in enumerate(domains):
        values = x[labels == label]
        if not len(values):
            raise ValueError(f"Cannot fit preprocessing without training rows for {domain}")
        mean = values.mean(dim=0)
        centered = values - mean
        scale = activation_norm_factor(centered)
        means.append(mean.tolist())
        rms_scales.append(scale)
    return {
        "mode": mode,
        "fit_split": "train",
        "metric_space": "per_domain_centered_rms_scaled",
        "domains": list(domains),
        "means": means,
        "rms_scales": rms_scales,
    }


def apply_activation_preprocessing(
    x: torch.Tensor,
    labels: torch.Tensor,
    preprocessing: dict[str, Any],
) -> torch.Tensor:
    """Apply a fitted activation transform without mixing domain statistics."""

    mode = str(preprocessing["mode"])
    if mode == "natural":
        return x
    if mode != "per_domain_center_rms":
        raise ValueError(f"Unknown activation preprocessing mode {mode!r}")
    result = torch.empty_like(x)
    means = torch.tensor(preprocessing["means"], dtype=x.dtype, device=x.device)
    scales = torch.tensor(preprocessing["rms_scales"], dtype=x.dtype, device=x.device)
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= len(means)):
        raise ValueError("Activation preprocessing labels fall outside the fitted domains")
    for label in range(len(means)):
        mask = labels == label
        result[mask] = (x[mask] - means[label]) / scales[label]
    return result


def _dictionary_learning_types():
    try:
        distribution = importlib.metadata.distribution("dictionary-learning")
    except ImportError as error:
        raise ImportError("Install the SAE dependencies with `uv sync --group sae`.") from error

    # dictionary-learning eagerly imports its language-model activation buffer from
    # package __init__ files.  Cached-activation training does not use that stack, so
    # load the upstream TopK module without importing nnsight/datasets/transformers.
    package_root = Path(distribution.locate_file("dictionary_learning"))
    for name, path in (
        ("dictionary_learning", package_root),
        ("dictionary_learning.trainers", package_root / "trainers"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package

    top_k = importlib.import_module("dictionary_learning.trainers.top_k")
    return top_k.AutoEncoderTopK, top_k.TopKTrainer


class DictionaryLearningTopK(torch.nn.Module):
    """MARL-facing adapter around dictionary-learning's reference TopK SAE."""

    def __init__(self, input_dim: int, width: int, k: int) -> None:
        super().__init__()
        autoencoder_type, _ = _dictionary_learning_types()
        self.autoencoder = autoencoder_type(input_dim, width, k)
        self.input_dim = input_dim
        self.width = width
        self.k = k

    def encode(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        del labels
        return self.autoencoder.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(codes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(x, labels)
        return self.decode(codes), codes


def balanced_batch_indices(
    labels: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample the same number of rows from every represented domain."""

    domains = labels.unique(sorted=True)
    if batch_size < len(domains):
        raise ValueError("batch_size must be at least the number of domains")
    base, remainder = divmod(batch_size, len(domains))
    parts = []
    for offset, domain in enumerate(domains):
        candidates = (labels == domain).nonzero(as_tuple=False).flatten()
        count = base + int(offset < remainder)
        chosen = torch.randint(len(candidates), (count,), generator=generator)
        parts.append(candidates[chosen])
    indices = torch.cat(parts)
    return indices[torch.randperm(len(indices), generator=generator)]


def activation_norm_factor(x: torch.Tensor) -> float:
    return float(torch.sqrt(x.float().pow(2).sum(dim=-1).mean()).clamp_min(1e-8))


@torch.no_grad()
def validation_metrics(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    domains: Sequence[str],
    *,
    batch_size: int = 8192,
) -> dict[str, float]:
    reconstructions, all_codes = [], []
    device = next(model.parameters()).device
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device)
        reconstruction, codes = model(batch)
        reconstructions.append(reconstruction.cpu())
        all_codes.append(codes.cpu())
    reconstruction = torch.cat(reconstructions)
    codes = torch.cat(all_codes)
    result = {f"validation/{key}": value for key, value in sparse_metrics(x, reconstruction, codes).items()}
    total_variance = torch.var(x, dim=0, unbiased=False).sum().clamp_min(1e-8)
    residual_variance = torch.var(x - reconstruction, dim=0, unbiased=False).sum()
    result["validation/explained_variance"] = float(1 - residual_variance / total_variance)
    rates = (codes > 0).float().mean(dim=0)
    for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
        result[f"validation/feature_density_q{int(100 * quantile):03d}"] = float(torch.quantile(rates, quantile))
    for label, domain in enumerate(domains):
        mask = labels == label
        if mask.any():
            domain_metrics = sparse_metrics(x[mask], reconstruction[mask], codes[mask])
            result.update({f"validation/{domain}/{key}": value for key, value in domain_metrics.items()})
    return result


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class TrainingResult:
    model: DictionaryLearningTopK
    final_step: int
    norm_factor: float
    metrics: dict[str, float]


def train_topk_sae(
    x: torch.Tensor,
    labels: torch.Tensor,
    domains: Sequence[str],
    validation_x: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    output_dir: Path,
    width: int,
    k: int,
    steps: int,
    batch_size: int,
    learning_rate: float | None,
    warmup_steps: int,
    decay_start: int | None,
    auxk_alpha: float,
    log_every: int,
    checkpoint_every: int,
    device: str,
    seed: int,
    resume_from: Path | None = None,
    wandb_cfg: dict[str, Any] | None = None,
) -> TrainingResult:
    """Train with dictionary-learning's TopK protocol and local lifecycle state."""

    _, trainer_type = _dictionary_learning_types()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    norm_factor = activation_norm_factor(x)
    trainer = trainer_type(
        steps=steps,
        activation_dim=x.shape[1],
        dict_size=width,
        k=k,
        layer=3,
        lm_name="marl-gpt",
        lr=learning_rate,
        auxk_alpha=auxk_alpha,
        warmup_steps=warmup_steps,
        decay_start=decay_start,
        seed=seed,
        device=device,
        wandb_name=output_dir.name,
        submodule_name="layer_03:final",
    )
    start_step = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        trainer.ae.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.scheduler.load_state_dict(state["scheduler"])
        trainer.num_tokens_since_fired.copy_(state["num_tokens_since_fired"].to(device))
        generator.set_state(state["sampler_rng_state"])
        norm_factor = float(state["norm_factor"])
        start_step = int(state["step"]) + 1

    wandb_run = None
    if wandb_cfg and bool(wandb_cfg.get("enabled", False)):
        try:
            import wandb
        except ModuleNotFoundError:
            if bool(wandb_cfg.get("required", False)):
                raise
            warnings.warn(
                "wandb is unavailable; continuing with authoritative local JSONL metrics and checkpoints",
                stacklevel=2,
            )
        else:
            wandb_run = wandb.init(
                project=str(wandb_cfg.get("project", "marl-gpt-sae")),
                name=output_dir.name,
                dir=str(output_dir),
                mode=str(wandb_cfg.get("mode", "offline")),
                config=dict(wandb_cfg.get("config", {})),
                resume="allow",
            )

    metrics_path = output_dir / "training_metrics.jsonl"
    latest_metrics: dict[str, float] = {}
    gradient_state = {"sum_sq": 0.0}

    def accumulate_gradient(gradient: torch.Tensor) -> torch.Tensor:
        gradient_state["sum_sq"] += float(gradient.detach().float().pow(2).sum())
        return gradient

    gradient_handles = [parameter.register_hook(accumulate_gradient) for parameter in trainer.ae.parameters()]
    interval_started = time.perf_counter()
    interval_examples = 0
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with metrics_path.open("a") as metrics_file:
            for step in range(start_step, steps):
                indices = balanced_batch_indices(labels, batch_size, generator)
                batch = (x[indices] / norm_factor).to(device)
                gradient_state["sum_sq"] = 0.0
                loss = float(trainer.update(step, batch))
                gradient_norm = math.sqrt(gradient_state["sum_sq"])
                interval_examples += len(batch)
                should_log = step == start_step or (step + 1) % log_every == 0 or step + 1 == steps
                if should_log:
                    elapsed = max(time.perf_counter() - interval_started, 1e-8)
                    model = DictionaryLearningTopK(x.shape[1], width, k).to(device)
                    model.autoencoder.load_state_dict(trainer.ae.state_dict())
                    normalized_validation = validation_metrics(
                        model,
                        validation_x / norm_factor,
                        validation_labels,
                        domains,
                    )
                    latest_metrics = {
                        "step": step,
                        "train/loss": loss,
                        "train/learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                        "train/dead_features": float(trainer.dead_features),
                        "train/gradient_norm_pre_clip": gradient_norm,
                        "system/examples_per_second": interval_examples / elapsed,
                        **normalized_validation,
                    }
                    if str(device).startswith("cuda"):
                        latest_metrics["system/peak_gpu_memory_bytes"] = float(torch.cuda.max_memory_allocated(device))
                        torch.cuda.reset_peak_memory_stats(device)
                    metrics_file.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
                    metrics_file.flush()
                    if wandb_run is not None:
                        wandb_run.log(latest_metrics, step=step)
                    interval_started = time.perf_counter()
                    interval_examples = 0
                should_checkpoint = (step + 1) % checkpoint_every == 0 or step + 1 == steps
                if should_checkpoint:
                    payload = {
                        "step": step,
                        "model": trainer.ae.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "scheduler": trainer.scheduler.state_dict(),
                        "num_tokens_since_fired": trainer.num_tokens_since_fired,
                        "sampler_rng_state": generator.get_state(),
                        "norm_factor": norm_factor,
                    }
                    checkpoint = checkpoints / f"step-{step + 1:08d}.pt"
                    _atomic_torch_save(payload, checkpoint)
                    _atomic_torch_save(payload, checkpoints / "latest.pt")
    finally:
        for handle in gradient_handles:
            handle.remove()
        if wandb_run is not None:
            wandb_run.finish()

    trainer.ae.scale_biases(norm_factor)
    model = DictionaryLearningTopK(x.shape[1], width, k).to(device)
    model.autoencoder.load_state_dict(trainer.ae.state_dict())
    return TrainingResult(model=model, final_step=steps - 1, norm_factor=norm_factor, metrics=latest_metrics)

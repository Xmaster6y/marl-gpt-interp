"""Training lifecycle for branch-specific MARL-GPT cross-layer transcoders."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from marl_gpt_interp.clt import CLTConfig, CrossLayerTranscoder, clt_health_metrics, clt_loss
from marl_gpt_interp.clt_data import branch_batch, iter_clt_shards


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class CLTTrainingResult:
    model: CrossLayerTranscoder
    final_step: int
    metrics: dict[str, float]


@torch.no_grad()
def evaluate_corpus(
    model: CrossLayerTranscoder,
    corpus_dir: Path,
    *,
    branch: str,
    split: str,
    device: str,
    maximum_rows: int = 100_000,
) -> dict[str, float]:
    residual_parts, output_parts, environments = [], [], []
    rows = 0
    for tensors, _metadata in iter_clt_shards(corpus_dir, split=split):
        residuals, outputs = branch_batch(tensors, branch)
        keep = len(residuals) if maximum_rows <= 0 else min(len(residuals), maximum_rows - rows)
        if keep <= 0:
            break
        residual_parts.append(residuals[:keep])
        output_parts.append(outputs[:keep])
        environments.extend(str(row["environment"]) for row in _metadata[:keep])
        rows += keep
    if not residual_parts:
        raise ValueError(f"CLT corpus has no {split!r} rows")
    model_dtype = next(model.parameters()).dtype
    residuals = torch.cat(residual_parts).to(device=device, dtype=model_dtype)
    outputs = torch.cat(output_parts).to(device=device, dtype=model_dtype)
    metrics = {"examples": float(rows), **clt_health_metrics(model, residuals, outputs)}
    for environment in sorted(set(environments)):
        indices = torch.tensor(
            [index for index, value in enumerate(environments) if value == environment],
            dtype=torch.long,
            device=device,
        )
        local = clt_health_metrics(model, residuals[indices], outputs[indices])
        metrics[f"environment/{environment}/examples"] = float(len(indices))
        metrics.update({f"environment/{environment}/{key}": value for key, value in local.items()})
    return metrics


def train_clt(
    corpus_dir: Path,
    *,
    branch: str,
    config: CLTConfig,
    output_dir: Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    sparsity_coefficient: float,
    sparsity_tanh_scale: float,
    log_every: int,
    checkpoint_every: int,
    validation_rows: int,
    gradient_clip: float,
    device: str,
    seed: int,
    resume_from: Path | None = None,
) -> CLTTrainingResult:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    model = CrossLayerTranscoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1

    metrics_path = output_dir / "training_metrics.jsonl"
    shard_iterator = iter_clt_shards(corpus_dir, split="train")
    current_residuals = current_outputs = None
    cursor = 0
    generator = torch.Generator().manual_seed(seed)
    latest: dict[str, float] = {}
    interval_start = time.perf_counter()
    interval_examples = 0

    def next_batch() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal shard_iterator, current_residuals, current_outputs, cursor
        parts_x, parts_y = [], []
        remaining = batch_size
        while remaining:
            if current_residuals is None or cursor == len(current_residuals):
                try:
                    tensors, _metadata = next(shard_iterator)
                except StopIteration:
                    shard_iterator = iter_clt_shards(corpus_dir, split="train")
                    try:
                        tensors, _metadata = next(shard_iterator)
                    except StopIteration as error:
                        raise ValueError("CLT corpus has no training rows") from error
                current_residuals, current_outputs = branch_batch(tensors, branch)
                order = torch.randperm(len(current_residuals), generator=generator)
                current_residuals = current_residuals[order]
                current_outputs = current_outputs[order]
                cursor = 0
            count = min(remaining, len(current_residuals) - cursor)
            parts_x.append(current_residuals[cursor : cursor + count])
            parts_y.append(current_outputs[cursor : cursor + count])
            cursor += count
            remaining -= count
        model_dtype = next(model.parameters()).dtype
        return (
            torch.cat(parts_x).to(device=device, dtype=model_dtype),
            torch.cat(parts_y).to(device=device, dtype=model_dtype),
        )

    with metrics_path.open("a") as metrics_file:
        for step in range(start_step, steps):
            residuals, outputs = next_batch()
            result = clt_loss(
                model,
                residuals,
                outputs,
                sparsity_coefficient=sparsity_coefficient,
                sparsity_tanh_scale=sparsity_tanh_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            model.normalize_decoder_bundles_()
            interval_examples += len(residuals)

            should_log = step == start_step or (step + 1) % log_every == 0 or step + 1 == steps
            if should_log:
                elapsed = max(time.perf_counter() - interval_start, 1e-8)
                model.eval()
                validation = evaluate_corpus(
                    model,
                    corpus_dir,
                    branch=branch,
                    split="validation",
                    device=device,
                    maximum_rows=validation_rows,
                )
                model.train()
                latest = {
                    "step": float(step),
                    "train/loss": float(result.loss.detach()),
                    "train/reconstruction": float(result.reconstruction.detach()),
                    "train/sparsity": float(result.sparsity.detach()),
                    "train/l0": float(result.l0.detach()),
                    "train/gradient_norm_pre_clip": float(gradient_norm),
                    "system/examples_per_second": interval_examples / elapsed,
                    **{f"validation/{key}": value for key, value in validation.items()},
                }
                metrics_file.write(json.dumps(latest, sort_keys=True) + "\n")
                metrics_file.flush()
                interval_start = time.perf_counter()
                interval_examples = 0

            if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
                state = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "branch": branch,
                }
                _atomic_save(state, checkpoint_dir / f"step-{step + 1:08d}.pt")
                _atomic_save(state, checkpoint_dir / "latest.pt")

    return CLTTrainingResult(model.cpu(), steps - 1, latest)


def load_trained_clt(model_dir: Path, *, device: str = "cpu") -> tuple[CrossLayerTranscoder, dict[str, Any]]:
    spec = json.loads((model_dir / "model_spec.json").read_text())
    config_payload = dict(spec["clt"])
    config_payload["features_per_layer"] = tuple(config_payload["features_per_layer"])
    model = CrossLayerTranscoder(CLTConfig(**config_payload))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    return model, spec

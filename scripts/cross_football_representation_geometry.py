"""Compare MARL-GPT latent geometry across La Liga, RoboCup, and native GRF."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.external_soccer import GRFEncodingConfig
from marl_gpt_interp.marl_gpt_tools import (
    activation_centroid_cosine_similarity_rows,
    activation_direction_rows,
    activation_hooks,
    activation_pairwise_cosine_similarity_rows,
    activation_subspace_similarity_rows,
    asymmetric_subspace_rows,
    marl_gpt_cwd,
    marl_gpt_path,
    pooled_activations,
    representation_proximity_rows,
    representation_separation_rows,
    self_subspace_similarity_rows,
    write_csv,
    write_json,
)
from scripts.external_soccer_marl_gpt import (
    _collect_native_grf,
    _encoding_config,
    _external_batch,
    _metadata_from_external,
    _path,
    _root,
    _torch_batch,
)


SOURCE_TO_ID = {"laliga": 1, "robocup": 2, "grf": 3}
ID_TO_SOURCE = {value: key for key, value in SOURCE_TO_ID.items()}


def _collect_source_latents(model, source_batches, cfg: DictConfig, cache_dir):
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, dict[str, Any]] = {}
    for source, full_batch in source_batches.items():
        cache_path = cache_dir / f"{source}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            expected = int(full_batch["obs"].shape[0])
            if cached and all(int(values.shape[0]) == expected for values in cached.values()):
                by_source[source] = cached
                logger.info(f"Loaded {expected} cached latent examples for {source}")
                continue
        parts: dict[str, list[Any]] = defaultdict(list)
        for start in range(0, int(full_batch["obs"].shape[0]), int(cfg.batch_size)):
            stop = min(start + int(cfg.batch_size), int(full_batch["obs"].shape[0]))
            batch = {key: value[start:stop] for key, value in full_batch.items()}
            captured = {}
            hooks = activation_hooks(model, captured)
            with torch.no_grad():
                model(batch)
            for hook in hooks:
                hook.remove()
            for feature, values in pooled_activations(
                captured,
                exclude_final_token_from_mean=bool(
                    OmegaConf.select(cfg, "exclude_final_token_from_mean", default=False)
                ),
            ).items():
                parts[feature].append(values.detach().float().cpu())
        by_source[source] = {feature: torch.cat(chunks) for feature, chunks in parts.items()}
        torch.save(by_source[source], cache_path)
        logger.info(f"Collected {len(next(iter(by_source[source].values())))} latent examples for {source}")
    return by_source


def _frame_groups(metadata: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    order = []
    for index, row in enumerate(metadata):
        frame_id = str(row["frame_id"])
        if frame_id not in groups:
            order.append(frame_id)
        groups[frame_id].append(index)
    return [groups[frame_id] for frame_id in order]


def select_analysis_frame_groups(
    source_metadata,
    max_frames: int,
    *,
    mode: str = "first",
    seed: int = 0,
    min_step_gap: int = 0,
):
    """Select complete frames deterministically without breaking perspective groups."""

    if mode not in {"first", "random"}:
        raise ValueError(f"Unknown frame sampling mode {mode!r}")
    if min_step_gap < 0:
        raise ValueError("min_step_gap must be non-negative")

    selected_by_source = {}
    audit = {}
    for source, source_id in SOURCE_TO_ID.items():
        candidates = _frame_groups(source_metadata[source])
        if len(candidates) < max_frames:
            raise ValueError(
                f"Not enough complete frames for {source}: requested {max_frames}, found {len(candidates)}"
            )
        if mode == "first":
            selected = candidates[:max_frames]
        else:
            selected = []
            for attempt in range(100):
                shuffled = list(candidates)
                random.Random(seed + source_id * 10_000 + attempt).shuffle(shuffled)
                trial = []
                trial_rows = []
                for group in shuffled:
                    row = source_metadata[source][group[0]]
                    same_sequence_too_close = any(
                        row.get("match_id") == previous.get("match_id")
                        and row.get("sequence_id") == previous.get("sequence_id")
                        and abs(int(row.get("step_index", 0)) - int(previous.get("step_index", 0)))
                        < min_step_gap
                        for previous in trial_rows
                    )
                    if same_sequence_too_close:
                        continue
                    trial.append(group)
                    trial_rows.append(row)
                    if len(trial) == max_frames:
                        selected = trial
                        break
                if selected:
                    break
            if not selected:
                raise ValueError(
                    f"Could not select {max_frames} {source} frames with min_step_gap={min_step_gap}"
                )
        selected_by_source[source] = selected
        selected_rows = [source_metadata[source][group[0]] for group in selected]
        audit[source] = {
            "candidate_frames": len(candidates),
            "selected_frames": len(selected),
            "frame_ids": [str(row["frame_id"]) for row in selected_rows],
            "step_indices": [int(row.get("step_index", 0)) for row in selected_rows],
        }
    audit["policy"] = {
        "mode": mode,
        "seed": seed,
        "min_step_gap": min_step_gap,
        "selection_order": "random" if mode == "random" else "source_order",
    }
    return selected_by_source, audit


def build_analysis_units(source_features, source_metadata, max_frames: int, *, frame_groups=None):
    """Build balanced perspective and frame-mean feature tables."""

    import torch

    if frame_groups is None:
        frame_groups, _audit = select_analysis_frame_groups(source_metadata, max_frames)
    if any(len(groups) < max_frames for groups in frame_groups.values()):
        counts = {source: len(groups) for source, groups in frame_groups.items()}
        raise ValueError(f"Not enough complete frames for balanced analysis: {counts}")

    frame_features: dict[str, list[Any]] = defaultdict(list)
    frame_labels = []
    for source, source_id in SOURCE_TO_ID.items():
        groups = frame_groups[source]
        for feature, values in source_features[source].items():
            frame_features[feature].append(torch.stack([values[indices].mean(dim=0) for indices in groups]))
        frame_labels.extend([source_id] * len(groups))

    perspective_count = min(
        sum(len(group) for group in frame_groups[source]) for source in SOURCE_TO_ID
    )
    perspective_features: dict[str, list[Any]] = defaultdict(list)
    perspective_labels = []
    for source, source_id in SOURCE_TO_ID.items():
        indices = [index for group in frame_groups[source] for index in group][:perspective_count]
        for feature, values in source_features[source].items():
            perspective_features[feature].append(values[indices])
        perspective_labels.extend([source_id] * len(indices))

    return {
        "perspective": (
            {feature: torch.cat(chunks) for feature, chunks in perspective_features.items()},
            torch.tensor(perspective_labels, dtype=torch.long),
        ),
        "frame_mean": (
            {feature: torch.cat(chunks) for feature, chunks in frame_features.items()},
            torch.tensor(frame_labels, dtype=torch.long),
        ),
    }


def _tag(rows: list[dict[str, Any]], sample_unit: str) -> list[dict[str, Any]]:
    for row in rows:
        row["sample_unit"] = sample_unit
    return rows


def _geometry_tables(analysis_units, cfg: DictConfig):
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_unit, (features, labels) in analysis_units.items():
        names = ID_TO_SOURCE
        tables["activation_geometry"].extend(
            _tag(activation_direction_rows(features, labels, cfg=cfg, label_names=names), sample_unit)
        )
        tables["internal_representation_proximity"].extend(
            _tag(representation_proximity_rows(features, labels, cfg=cfg, label_names=names), sample_unit)
        )
        tables["representation_separation"].extend(
            _tag(representation_separation_rows(features, labels, cfg=cfg, label_names=names), sample_unit)
        )
        tables["activation_centroid_cosine_similarity"].extend(
            _tag(
                activation_centroid_cosine_similarity_rows(
                    features,
                    labels,
                    cfg=cfg,
                    label_names=names,
                ),
                sample_unit,
            )
        )
        tables["activation_pairwise_cosine_similarity"].extend(
            _tag(
                activation_pairwise_cosine_similarity_rows(
                    features,
                    labels,
                    cfg=cfg,
                    label_names=names,
                ),
                sample_unit,
            )
        )
        tables["asymmetric_representation_analysis"].extend(
            _tag(asymmetric_subspace_rows(features, labels, cfg=cfg, label_names=names), sample_unit)
        )
        tables["self_subspace_similarity"].extend(
            _tag(self_subspace_similarity_rows(features, labels, label_names=names), sample_unit)
        )
        tables["activation_subspace_similarity"].extend(
            _tag(activation_subspace_similarity_rows(features, labels, label_names=names), sample_unit)
        )
    return tables


@hydra.main(config_path="../configs/cross_football_representation_geometry", version_base=None)
def main(cfg: DictConfig) -> dict[str, Any]:
    import torch

    root = _root()
    output_dir = _path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _path(root, str(cfg.checkpoint))
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    torch.manual_seed(int(cfg.seed))

    encoding: GRFEncodingConfig = _encoding_config(cfg)
    max_frames = int(cfg.max_frames_per_source)
    candidate_frames = int(OmegaConf.select(cfg, "sampling.candidate_frames_per_source", default=max_frames))
    if candidate_frames < max_frames:
        raise ValueError("sampling.candidate_frames_per_source must be at least max_frames_per_source")
    external_target = candidate_frames * 22
    native_target = candidate_frames * 11
    source_batches = {}
    source_metadata = {}
    audits = {}
    for source in ("laliga", "robocup"):
        batch, audit = _external_batch(
            root,
            cfg,
            source,
            encoding,
            max_examples=external_target,
        )
        source_batches[source] = _torch_batch(batch.arrays, str(cfg.device))
        source_metadata[source] = _metadata_from_external(batch)
        audits[source] = audit

    marl_gpt_path(root)
    with marl_gpt_cwd(root):
        native_batch, native_metadata, native_audit, model = _collect_native_grf(
            root,
            cfg,
            checkpoint,
            max_examples=native_target,
        )
    source_batches["grf"] = {key: value.to(str(cfg.device)) for key, value in native_batch.items()}
    source_metadata["grf"] = native_metadata
    audits["grf"] = native_audit

    with marl_gpt_cwd(root):
        source_features = _collect_source_latents(model, source_batches, cfg, output_dir / "latent_cache")
    frame_groups, sampling_audit = select_analysis_frame_groups(
        source_metadata,
        max_frames,
        mode=str(OmegaConf.select(cfg, "sampling.mode", default="first")),
        seed=int(OmegaConf.select(cfg, "sampling.seed", default=cfg.seed)),
        min_step_gap=int(OmegaConf.select(cfg, "sampling.min_step_gap", default=0)),
    )
    analysis_units = build_analysis_units(
        source_features,
        source_metadata,
        max_frames,
        frame_groups=frame_groups,
    )
    tables = _geometry_tables(analysis_units, cfg)

    for name, rows in tables.items():
        write_json(output_dir / f"{name}.json", rows)
        write_csv(output_dir / f"{name}.csv", rows)
    write_json(output_dir / "dataset_inspection.json", audits)
    torch.save(
        {
            "source_features": source_features,
            "source_metadata": source_metadata,
            "analysis_units": analysis_units,
        },
        output_dir / "latent_features.pt",
    )
    summary = {
        "sources": list(SOURCE_TO_ID),
        "source_ids": SOURCE_TO_ID,
        "latent_only": True,
        "pooling": {
            "mean_excludes_final_token": bool(
                OmegaConf.select(cfg, "exclude_final_token_from_mean", default=False)
            ),
            "final_token_retained_as_separate_feature": True,
            "attention_is_causal": False,
        },
        "max_frames_per_source": max_frames,
        "sampling": sampling_audit,
        "raw_examples": {source: len(source_metadata[source]) for source in SOURCE_TO_ID},
        "analysis_examples": {
            unit: {
                source: int((labels == source_id).sum())
                for source, source_id in SOURCE_TO_ID.items()
            }
            for unit, (_features, labels) in analysis_units.items()
        },
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    write_json(output_dir / "summary.json", summary)
    logger.info(f"Wrote cross-football representation geometry to {output_dir}")
    return summary


if __name__ == "__main__":
    main()

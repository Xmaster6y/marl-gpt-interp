"""Summarize held-out SAE features for interpretation and apparent universality."""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import load_activation_cache, write_run_manifest
from marl_gpt_interp.sae_training import apply_activation_preprocessing
from scripts.experiments.sparse_marl_gpt.train_dictionary import build_model


@torch.no_grad()
def feature_rows(
    codes: torch.Tensor,
    labels: torch.Tensor,
    domains: list[str],
    metadata: list[dict],
    *,
    top_n: int,
    minimum_rate: float,
) -> list[dict]:
    rows = []
    for feature in range(codes.shape[1]):
        values = codes[:, feature]
        rates = {}
        means = {}
        for label, domain in enumerate(domains):
            domain_values = values[labels == label]
            active = domain_values > 0
            rates[domain] = float(active.float().mean()) if len(domain_values) else 0.0
            means[domain] = float(domain_values[active].mean()) if active.any() else 0.0
        support = [domain for domain in domains if rates[domain] >= minimum_rate]
        maximum_rate = max(rates.values(), default=0.0)
        universality = min(rates.values(), default=0.0) / maximum_rate if maximum_rate else 0.0
        count = min(top_n, int((values > 0).sum()))
        top_examples = []
        if count:
            top_values, top_indices = torch.topk(values, count)
            for activation, index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                sample = metadata[index]
                top_examples.append(
                    {
                        "activation": activation,
                        "sample_index": sample["sample_index"],
                        "environment": sample["environment"],
                        "source_file_id": sample.get("source_file_id"),
                        "source_row_index": sample.get("source_row_index"),
                        "target_action": sample.get("target_action"),
                        "trajectory_group": sample["trajectory_group"],
                    }
                )
        rows.append(
            {
                "feature": feature,
                "activation_rate": float((values > 0).float().mean()),
                "mean_active_activation": float(values[values > 0].mean()) if (values > 0).any() else 0.0,
                "domain_activation_rate": rates,
                "domain_mean_active_activation": means,
                "apparent_support": support,
                "apparent_universality": universality,
                "top_examples": top_examples,
                "causal_status": "not_evaluated",
            }
        )
    return rows


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/analyze_features", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = as_path(root, str(cfg.model_dir))
    spec = json.loads((model_dir / "model_spec.json").read_text())
    tensors, metadata, cache_manifest = load_activation_cache(as_path(root, str(cfg.cache_dir)))
    split_indices = [index for index, row in enumerate(metadata) if row["split"] == str(cfg.split)]
    if not split_indices:
        raise ValueError(f"No examples in requested split {str(cfg.split)!r}")
    x = tensors[spec["activation_location"]][split_indices].float()
    selected_metadata = [metadata[index] for index in split_indices]
    domains = list(spec["domains"])
    domain_to_label = {domain: index for index, domain in enumerate(domains)}
    labels = torch.tensor([domain_to_label[row["environment"]] for row in selected_metadata])
    preprocessing = dict(spec.get("preprocessing", {"mode": "natural"}))
    x = apply_activation_preprocessing(x, labels, preprocessing)
    model = build_model(DictConfig({"model": spec["model"]}), int(spec["input_dim"]), domains)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    _, codes = model(x, labels)
    rows = feature_rows(
        codes,
        labels,
        domains,
        selected_metadata,
        top_n=int(cfg.top_examples_per_feature),
        minimum_rate=float(cfg.minimum_domain_activation_rate),
    )
    path = output_dir / "feature_summary.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "features": len(rows),
        "dead_features": sum(row["activation_rate"] == 0 for row in rows),
        "apparently_universal_features": sum(len(row["apparent_support"]) == len(domains) for row in rows),
        "warning": "Activation support is descriptive; universality requires per-domain causal interventions.",
    }
    write_json(output_dir / "summary.json", summary)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"features": path.name, "summary": "summary.json"},
        hashes={"source_checkpoint": cache_manifest["checkpoint_sha256"]},
        split_manifest={str(cfg.split): len(split_indices)},
        environment_versions={"cache_schema": str(cache_manifest["format_version"])},
    )
    return summary


if __name__ == "__main__":
    main()

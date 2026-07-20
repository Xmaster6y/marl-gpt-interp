"""Generate cross-run decoder matches and candidate feature-splitting diagnostics."""

from __future__ import annotations

import json

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import load_activation_cache, write_run_manifest
from marl_gpt_interp.sae_training import apply_activation_preprocessing
from scripts.experiments.sparse_marl_gpt.train_dictionary import build_model


def decoder_directions(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "autoencoder"):
        return model.autoencoder.decoder.weight.detach()
    decoder = model.decoder.detach()
    return decoder if decoder.shape[0] == model.width else decoder.T


@torch.no_grad()
def encode_batches(model: torch.nn.Module, x: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks = []
    device = next(model.parameters()).device
    for start in range(0, len(x), batch_size):
        _, codes = model(
            x[start : start + batch_size].to(device),
            labels[start : start + batch_size].to(device),
        )
        chunks.append(codes.cpu())
    return torch.cat(chunks)


def activation_correlations(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return held-out Pearson correlations between every cross-run feature pair."""

    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    left = F.normalize(left, dim=0, eps=1e-12)
    right = F.normalize(right, dim=0, eps=1e-12)
    return left.T @ right


def domain_rate_fingerprints(codes: torch.Tensor, labels: torch.Tensor, domain_count: int) -> torch.Tensor:
    rates = []
    for label in range(domain_count):
        rates.append((codes[labels == label] > 0).float().mean(dim=0))
    return torch.stack(rates, dim=1)


def top_identity_sets(codes: torch.Tensor, metadata: list[dict], top_n: int) -> list[set[tuple]]:
    identities = []
    for feature in range(codes.shape[1]):
        count = min(top_n, int((codes[:, feature] > 0).sum()))
        indices = torch.topk(codes[:, feature], count).indices.tolist() if count else []
        identities.append(
            {
                (
                    metadata[index].get("source_file_id"),
                    metadata[index].get("source_row_index"),
                )
                for index in indices
            }
        )
    return identities


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/compare_features", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors, metadata, cache_manifest = load_activation_cache(as_path(root, str(cfg.cache_dir)))
    split_indices = [index for index, row in enumerate(metadata) if row["split"] == str(cfg.split)]
    if not split_indices:
        raise ValueError(f"No examples in requested split {str(cfg.split)!r}")
    selected_metadata = [metadata[index] for index in split_indices]
    models = []
    specs = []
    for raw_dir in cfg.model_dirs:
        model_dir = as_path(root, str(raw_dir))
        spec = json.loads((model_dir / "model_spec.json").read_text())
        model = build_model(DictConfig({"model": spec["model"]}), int(spec["input_dim"]), list(spec["domains"]))
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        model.to(str(cfg.device)).eval()
        models.append(model)
        specs.append(spec)
    if len(models) != 2:
        raise ValueError("Feature comparison currently requires exactly two model directories")
    if specs[0]["activation_location"] != specs[1]["activation_location"]:
        raise ValueError("Feature comparison requires the same activation location")
    if specs[0]["domains"] != specs[1]["domains"]:
        raise ValueError("Feature comparison requires the same ordered domains")
    if specs[0].get("cache_manifest_sha256") != specs[1].get("cache_manifest_sha256"):
        raise ValueError("Feature comparison requires models trained from the same activation cache")
    if specs[0].get("preprocessing", {"mode": "natural"}) != specs[1].get(
        "preprocessing", {"mode": "natural"}
    ):
        raise ValueError("Feature comparison requires identical fitted preprocessing")
    domains = list(specs[0]["domains"])
    domain_to_label = {domain: index for index, domain in enumerate(domains)}
    labels = torch.tensor([domain_to_label[row["environment"]] for row in selected_metadata])
    x = tensors[specs[0]["activation_location"]][split_indices].float()
    x = apply_activation_preprocessing(x, labels, specs[0].get("preprocessing", {"mode": "natural"}))
    codes = [encode_batches(model, x, labels, int(cfg.batch_size)) for model in models]
    left_decoder = F.normalize(decoder_directions(models[0]).float(), dim=-1)
    right_decoder = F.normalize(decoder_directions(models[1]).float(), dim=-1)
    similarity = left_decoder @ right_decoder.T
    correlations = activation_correlations(codes[0], codes[1])
    fingerprints = [domain_rate_fingerprints(value, labels, len(domains)) for value in codes]
    fingerprint_similarity = F.normalize(fingerprints[0], dim=1, eps=1e-12) @ F.normalize(
        fingerprints[1], dim=1, eps=1e-12
    ).T
    identities = [top_identity_sets(value, selected_metadata, int(cfg.top_examples)) for value in codes]
    threshold = float(cfg.candidate_split_cosine)
    correlation_threshold = float(cfg.minimum_activation_correlation)
    fingerprint_threshold = float(cfg.minimum_fingerprint_cosine)
    overlap_threshold = float(cfg.minimum_top_example_jaccard)
    rows = []
    for left_feature in range(similarity.shape[0]):
        candidates = (similarity[left_feature] >= threshold).nonzero(as_tuple=False).flatten()
        candidate_rows = []
        for index in candidates.tolist():
            overlap = jaccard(identities[0][left_feature], identities[1][index])
            activation_correlation = float(correlations[left_feature, index])
            domain_fingerprint_cosine = float(fingerprint_similarity[left_feature, index])
            validated = (
                activation_correlation >= correlation_threshold
                and domain_fingerprint_cosine >= fingerprint_threshold
                and overlap >= overlap_threshold
            )
            candidate_rows.append(
                {
                    "feature": index,
                    "decoder_cosine": float(similarity[left_feature, index]),
                    "activation_correlation": activation_correlation,
                    "domain_rate_fingerprint_cosine": domain_fingerprint_cosine,
                    "top_example_jaccard": overlap,
                    "activation_validated": validated,
                }
            )
        validated_count = sum(row["activation_validated"] for row in candidate_rows)
        rows.append(
            {
                "left_feature": left_feature,
                "candidate_count": len(candidates),
                "activation_validated_count": validated_count,
                "candidate_split": validated_count > 1,
                "right_candidates": candidate_rows,
                "validation_status": "activation_validated" if validated_count else "decoder_candidate_only",
            }
        )
    matches_path = output_dir / "decoder_matches.jsonl"
    with matches_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "left_features": len(left_decoder),
        "right_features": len(right_decoder),
        "candidate_splits": sum(row["candidate_split"] for row in rows),
        "activation_validated_matches": sum(row["activation_validated_count"] for row in rows),
        "candidate_split_cosine": threshold,
        "minimum_activation_correlation": correlation_threshold,
        "minimum_fingerprint_cosine": fingerprint_threshold,
        "minimum_top_example_jaccard": overlap_threshold,
        "split": str(cfg.split),
        "examples": len(split_indices),
        "warning": "Activation evidence validates stability candidates only; semantic and causal fingerprints remain required.",
    }
    write_json(output_dir / "summary.json", summary)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(cfg.seed),
        status="completed",
        artifacts={"matches": matches_path.name, "summary": "summary.json"},
        hashes={"activation_cache": cache_manifest["shards"][0]["sha256"]},
        split_manifest={str(cfg.split): len(split_indices)},
    )
    return summary


if __name__ == "__main__":
    main()

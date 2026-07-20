"""Generate cross-run decoder matches and candidate feature-splitting diagnostics."""

from __future__ import annotations

import json

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import write_run_manifest
from scripts.experiments.sparse_marl_gpt.train_dictionary import build_model


def decoder_directions(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "autoencoder"):
        return model.autoencoder.decoder.weight.detach()
    decoder = model.decoder.detach()
    return decoder if decoder.shape[0] == model.width else decoder.T


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/compare_features", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    models = []
    specs = []
    for raw_dir in cfg.model_dirs:
        model_dir = as_path(root, str(raw_dir))
        spec = json.loads((model_dir / "model_spec.json").read_text())
        model = build_model(DictConfig({"model": spec["model"]}), int(spec["input_dim"]), list(spec["domains"]))
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        models.append(model)
        specs.append(spec)
    if len(models) != 2:
        raise ValueError("Feature comparison currently requires exactly two model directories")
    left = F.normalize(decoder_directions(models[0]).float(), dim=-1)
    right = F.normalize(decoder_directions(models[1]).float(), dim=-1)
    similarity = left @ right.T
    threshold = float(cfg.candidate_split_cosine)
    rows = []
    for left_feature in range(similarity.shape[0]):
        candidates = (similarity[left_feature].abs() >= threshold).nonzero(as_tuple=False).flatten()
        rows.append(
            {
                "left_feature": left_feature,
                "candidate_count": len(candidates),
                "candidate_split": len(candidates) > 1,
                "right_candidates": [
                    {"feature": int(index), "decoder_cosine": float(similarity[left_feature, index])}
                    for index in candidates
                ],
                "validation_status": "decoder_candidate_only",
            }
        )
    matches_path = output_dir / "decoder_matches.jsonl"
    with matches_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "left_features": len(left),
        "right_features": len(right),
        "candidate_splits": sum(row["candidate_split"] for row in rows),
        "candidate_split_cosine": threshold,
        "warning": "Decoder similarity only proposes split candidates; activation and causal fingerprints must validate them.",
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
        hashes={},
    )
    return summary


if __name__ == "__main__":
    main()

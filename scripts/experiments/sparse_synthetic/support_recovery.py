"""Run a bounded synthetic domain-support recovery comparison."""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root, to_plain_config, write_json
from marl_gpt_interp.sparse_features import (
    DomainLatticeSAE,
    FrozenRandomDictionary,
    IndependentSAEs,
    SparseAutoencoder,
    evaluate_synthetic_gate,
    generate_synthetic_sparse_data,
    greedy_decoder_matches,
    infer_functional_supports,
    sparse_metrics,
    support_macro_f1,
    train_sparse_model,
    write_run_manifest,
)


def _decoder(model):
    if isinstance(model, IndependentSAEs):
        return torch.cat([child.decoder for child in model.models])
    return model.decoder


def _predicted_supports(model, codes, labels, domains, matches, target_width):
    if isinstance(model, DomainLatticeSAE):
        learned = model.latent_supports
    elif isinstance(model, IndependentSAEs):
        learned = tuple(
            frozenset({domain})
            for domain, child in zip(domains, model.models, strict=True)
            for _ in range(child.width)
        )
    else:
        learned = infer_functional_supports(codes, labels, domains)
    target_order = [frozenset() for _ in range(target_width)]
    for learned_index, target_index, _score in matches:
        target_order[target_index] = learned[learned_index]
    return tuple(target_order)


@hydra.main(config_path="../../../configs/experiments/sparse_synthetic/support_recovery", version_base=None)
def main(cfg: DictConfig) -> dict:
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = tuple(str(value) for value in cfg.domains)
    rows = []
    regimes = list(cfg.get("regimes", [cfg.synthetic]))
    for regime in regimes:
        for seed in list(cfg.seeds):
            torch.manual_seed(int(seed))
            data = generate_synthetic_sparse_data(
                domains=domains,
                input_dim=int(cfg.synthetic.input_dim),
                features_per_support=int(cfg.synthetic.features_per_support),
                samples_per_domain=int(cfg.synthetic.samples_per_domain),
                active_features=int(cfg.synthetic.active_features),
                noise_std=float(regime.noise_std),
                correlation=float(regime.correlation),
                anisotropy=float(regime.anisotropy),
                superposition=float(regime.superposition),
                hierarchy=float(regime.get("hierarchy", 0.0)),
                imbalance=list(regime.imbalance),
                seed=int(seed),
            )
            width = int(data.decoder.shape[0])
            per_domain_width = max(1, width // len(domains))
            models = {
                "flat_topk": SparseAutoencoder(data.activations.shape[1], width, int(cfg.model.k)),
                "independent_topk": IndependentSAEs(
                    data.activations.shape[1], domains, per_domain_width, int(cfg.model.k)
                ),
                "domain_lattice": DomainLatticeSAE(
                    data.activations.shape[1],
                    domains,
                    int(cfg.synthetic.features_per_support),
                    int(cfg.model.k),
                ),
                "constrained_random": FrozenRandomDictionary(
                    data.activations.shape[1], width, int(cfg.model.k), seed=int(seed) + 10_000
                ),
            }
            for name, model in models.items():
                losses = (
                    [None]
                    if isinstance(model, FrozenRandomDictionary)
                    else train_sparse_model(
                        model,
                        data.activations,
                        data.labels,
                        steps=int(cfg.training.steps),
                        batch_size=int(cfg.training.batch_size),
                        learning_rate=float(cfg.training.learning_rate),
                        seed=int(seed),
                    )
                )
                model.eval()
                with torch.no_grad():
                    reconstruction, codes = model(data.activations, data.labels)
                matches = greedy_decoder_matches(_decoder(model), data.decoder)
                predicted = _predicted_supports(model, codes, data.labels, domains, matches, len(data.latent_supports))
                metrics = sparse_metrics(data.activations, reconstruction, codes)
                rows.append(
                    {
                        "seed": int(seed),
                        "regime": str(regime.name),
                        "assumption_holding": bool(regime.get("assumption_holding", True)),
                        "method": name,
                        "support_macro_f1": support_macro_f1(predicted, data.latent_supports),
                        "decoder_match_cosine": sum(item[2] for item in matches) / max(len(matches), 1),
                        "final_training_loss": losses[-1],
                        **metrics,
                    }
                )
    write_json(output_dir / "metrics.json", rows)
    gate = evaluate_synthetic_gate(
        rows,
        f1_threshold=float(cfg.gate.support_macro_f1),
        reconstruction_tolerance=float(cfg.gate.reconstruction_tolerance),
        stable_seed_count=int(cfg.gate.stable_seed_count),
        decoder_match_threshold=float(cfg.gate.decoder_match_threshold),
    )
    summary = {"status": "completed", "rows": len(rows), "gate": gate, "metrics": rows}
    write_json(output_dir / "summary.json", summary)
    write_run_manifest(
        output_dir / "run_manifest.json",
        root=root,
        run_id=output_dir.name,
        config=to_plain_config(cfg),
        seed=int(list(cfg.seeds)[0]),
        status="completed",
        artifacts={"metrics": "metrics.json", "summary": "summary.json"},
        split_manifest={"synthetic_examples_per_regime": int(cfg.synthetic.samples_per_domain) * len(domains)},
        environment_versions={"synthetic_generator": "domain-lattice-v1"},
    )
    return summary


if __name__ == "__main__":
    main()

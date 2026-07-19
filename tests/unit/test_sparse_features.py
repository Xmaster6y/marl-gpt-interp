from __future__ import annotations

import json

import pytest
import torch

from marl_gpt_interp.sparse_features import (
    DomainLatticeSAE,
    SparseAutoencoder,
    batch_topk_codes,
    domain_lattice,
    domain_stratified_batch_topk_codes,
    evaluate_synthetic_gate,
    generate_synthetic_sparse_data,
    grouped_split,
    load_activation_cache,
    masked_policy_kl,
    replace_token_activation,
    sparse_replacement_hook,
    sparse_metrics,
    support_macro_f1,
    topk_codes,
    train_sparse_model,
    write_activation_cache,
    write_run_manifest,
)


def test_domain_lattice_has_complete_three_domain_support():
    supports = domain_lattice(("smac", "grf", "pogema"))
    assert len(supports) == 7
    assert supports[0] == frozenset({"smac", "grf", "pogema"})
    assert {len(value) for value in supports} == {1, 2, 3}


def test_topk_and_batch_topk_budgets():
    scores = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    assert torch.count_nonzero(topk_codes(scores, 2), dim=1).tolist() == [2, 2, 2, 2]
    assert int(torch.count_nonzero(batch_topk_codes(scores, 2))) == 8


def test_domain_stratified_batch_topk_is_fair_per_domain():
    scores = torch.tensor([[10.0, 9.0, 8.0], [7.0, 6.0, 5.0], [1.0, 0.9, 0.8], [0.7, 0.6, 0.5]])
    labels = torch.tensor([0, 0, 1, 1])
    codes = domain_stratified_batch_topk_codes(scores, labels, 1)
    assert int(torch.count_nonzero(codes[labels == 0])) == 2
    assert int(torch.count_nonzero(codes[labels == 1])) == 2


def test_lattice_masks_ineligible_features():
    model = DomainLatticeSAE(4, ("smac", "grf", "pogema"), 2, 3)
    _reconstruction, codes = model(torch.randn(3, 4), torch.tensor([0, 1, 2]))
    for row, domain in zip(codes, model.domains, strict=True):
        for active, support in zip(row > 0, model.latent_supports, strict=True):
            if active:
                assert domain in support


def test_synthetic_generator_is_reproducible_and_respects_support():
    left = generate_synthetic_sparse_data(seed=4, samples_per_domain=8)
    right = generate_synthetic_sparse_data(seed=4, samples_per_domain=8)
    assert torch.equal(left.activations, right.activations)
    for codes, label in zip(left.latent_codes, left.labels, strict=True):
        domain = left.domains[int(label)]
        assert all(domain in left.latent_supports[index] for index in (codes > 0).nonzero().flatten().tolist())


def test_tiny_sparse_training_reduces_loss():
    data = generate_synthetic_sparse_data(input_dim=8, samples_per_domain=16, seed=2)
    model = SparseAutoencoder(8, data.decoder.shape[0], 3)
    losses = train_sparse_model(model, data.activations, data.labels, steps=30, batch_size=16, seed=2)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_sparse_metrics_and_support_f1():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metrics = sparse_metrics(x, x.clone(), x.clone())
    assert metrics["normalized_mse"] == 0.0
    assert metrics["l0"] == 1.0
    assert support_macro_f1((frozenset({"a"}),), (frozenset({"a"}),)) == 1.0


def test_grouped_split_never_leaks_a_group():
    groups = ["a", "a", "b", "c", "c", "d", "e", "f"]
    splits = grouped_split(groups, seed=9)
    assert len({split for group, split in zip(groups, splits, strict=True) if group == "a"}) == 1
    assert set(splits) <= {"train", "validation", "test"}


def test_activation_cache_round_trip_and_hash_validation(tmp_path):
    tensors = {"layer_03:final": torch.randn(3, 4)}
    metadata = [
        {
            "environment": "grf",
            "trajectory_group": f"episode-{index}",
            "sample_index": index,
            "activation_location": ["layer_03:final"],
            "token_selector": "final",
            "checkpoint_sha256": "abc",
            "preprocessing_identity": "test-v1",
            "split": "test",
        }
        for index in range(3)
    ]
    manifest = write_activation_cache(tmp_path, tensors, metadata, {"checkpoint_sha256": "abc"})
    loaded, loaded_metadata, loaded_manifest = load_activation_cache(tmp_path)
    assert torch.equal(loaded["layer_03:final"], tensors["layer_03:final"])
    assert loaded_metadata == metadata
    assert loaded_manifest == manifest
    (tmp_path / "metadata.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="metadata hash"):
        load_activation_cache(tmp_path)


def test_activation_cache_rejects_incomplete_metadata(tmp_path):
    with pytest.raises(ValueError, match="missing required fields"):
        write_activation_cache(tmp_path, {"layer": torch.randn(1, 2)}, [{"sample_index": 0}], {})


def test_synthetic_gate_requires_positive_paired_intervals_and_stable_seeds():
    rows = []
    for seed in range(5):
        for method, f1, mse in (
            ("domain_lattice", 0.90, 0.10),
            ("flat_topk", 0.70, 0.10),
            ("independent_topk", 0.65, 0.10),
        ):
            rows.append(
                {
                    "regime": "balanced",
                    "assumption_holding": True,
                    "seed": seed,
                    "method": method,
                    "support_macro_f1": f1,
                    "normalized_mse": mse,
                    "decoder_match_cosine": 0.8,
                }
            )
    result = evaluate_synthetic_gate(rows)
    assert result["passed"]
    assert result["regimes"]["balanced"]["stable_seeds"] == 5


def test_run_manifest_hashes_config(tmp_path):
    payload = write_run_manifest(
        tmp_path / "run.json", root=tmp_path, run_id="smoke", config={"a": 1}, seed=3, status="completed"
    )
    assert payload["wandb_required"] is False
    assert json.loads((tmp_path / "run.json").read_text())["config_sha256"] == payload["config_sha256"]


def test_masked_policy_kl_ignores_invalid_action_and_replacement_is_local():
    reference = torch.tensor([[1.0, 2.0, 100.0]])
    candidate = torch.tensor([[1.0, 2.0, -100.0]])
    mask = torch.tensor([[True, True, False]])
    assert masked_policy_kl(reference, candidate, mask) == pytest.approx(0.0, abs=1e-7)
    hidden = torch.zeros(2, 3, 4)
    replaced = replace_token_activation(hidden, torch.ones(2, 4), token_index=1)
    assert torch.equal(replaced[:, 1], torch.ones(2, 4))
    assert torch.equal(replaced[:, 0], torch.zeros(2, 4))


def test_frozen_checkpoint_activation_replacement_is_deterministic(tmp_path):
    torch.manual_seed(7)
    block = torch.nn.Linear(4, 4, bias=False)
    torch.save(block.state_dict(), tmp_path / "checkpoint.pt")
    restored = torch.nn.Linear(4, 4, bias=False)
    restored.load_state_dict(torch.load(tmp_path / "checkpoint.pt", weights_only=True))
    restored.requires_grad_(False)
    dictionary = SparseAutoencoder(4, 4, 2)
    record = {}
    handle = restored.register_forward_hook(
        sparse_replacement_hook(dictionary, torch.tensor([0, 0]), token_index=-1, record=record)
    )
    output = restored(torch.ones(2, 3, 4))
    handle.remove()
    assert torch.isfinite(output).all()
    assert set(record) == {"original", "reconstruction", "codes"}
    assert torch.equal(output[:, -1], record["reconstruction"])

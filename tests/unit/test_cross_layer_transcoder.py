from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from marl_gpt_interp.attribution_graph import (
    ActorTarget,
    CriticTarget,
    build_attribution_graph,
    prune_attribution_graph,
)
from marl_gpt_interp.clt import CLTConfig, CrossLayerTranscoder, clt_loss, jump_relu
from marl_gpt_interp.clt_audit import audit_clt_suite
from marl_gpt_interp.clt_data import CLTCorpusWriter, branch_batch, iter_clt_shards
from marl_gpt_interp.clt_training import evaluate_corpus
from marl_gpt_interp.marl_gpt_clt import (
    CLTReplacement,
    CrossLayerFeatureIntervention,
    PathCapture,
    local_error_adjustments,
)


def test_jump_relu_threshold_and_threshold_gradient():
    values = torch.tensor([[0.5, 1.5]], requires_grad=True)
    thresholds = torch.tensor([1.0, 1.0], requires_grad=True)
    output = jump_relu(values, thresholds, bandwidth=1.1)
    assert output.tolist() == [[0.0, 1.5]]
    output.sum().backward()
    assert values.grad.tolist() == [[0.0, 1.0]]
    assert thresholds.grad is not None


def test_clt_uses_triangular_cross_layer_writes():
    model = CrossLayerTranscoder(CLTConfig(2, (1, 1), initial_threshold=0.01))
    with torch.no_grad():
        for encoder in model.encoders:
            encoder.weight.fill_(1.0)
            encoder.bias.zero_()
        model.decoder(0, 0).fill_(1.0)
        model.decoder(0, 1).fill_(2.0)
        model.decoder(1, 1).fill_(3.0)
        for bias in model.output_biases:
            bias.zero_()
    result = model((torch.ones(1, 2), torch.full((1, 2), 2.0)))
    assert torch.equal(result.reconstructions[0], torch.full((1, 2), 2.0))
    assert torch.equal(result.reconstructions[1], torch.full((1, 2), 16.0))


def test_clt_training_objective_decreases_on_tiny_problem():
    generator = torch.Generator().manual_seed(4)
    residuals = torch.randn(64, 3, 4, generator=generator)
    outputs = torch.stack(
        [residuals[:, 0], residuals[:, 0] + residuals[:, 1], residuals.sum(dim=1)], dim=1
    )
    model = CrossLayerTranscoder(CLTConfig(4, (8, 8, 8), initial_threshold=1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        result = clt_loss(model, residuals, outputs, sparsity_coefficient=1e-4)
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()
        model.normalize_decoder_bundles_()
        losses.append(float(result.loss.detach()))
    assert sum(losses[-5:]) < sum(losses[:5])


def _tiny_corpus_tensors(rows: int = 5) -> dict[str, torch.Tensor]:
    return {
        "shared_residual_inputs": torch.randn(rows, 2, 4),
        "shared_mlp_outputs": torch.randn(rows, 2, 4),
        "actor_residual_input": torch.randn(rows, 4),
        "actor_mlp_output": torch.randn(rows, 4),
        "critic_residual_input": torch.randn(rows, 4),
        "critic_mlp_output": torch.randn(rows, 4),
    }


def _tiny_metadata(rows: int = 5) -> list[dict]:
    return [
        {
            "environment": "grf",
            "split_group": f"source-{index // 2}",
            "source_file_id": 1,
            "source_row_index": index,
            "sample_index": index,
            "token_index": index,
            "is_output_token": index == rows - 1,
            "split": "train" if index < rows - 1 else "test",
        }
        for index in range(rows)
    ]


def test_clt_corpus_is_sharded_and_branch_complete(tmp_path):
    writer = CLTCorpusWriter(tmp_path, rows_per_shard=3, manifest={"checkpoint_sha256": "abc"})
    tensors = _tiny_corpus_tensors()
    metadata = _tiny_metadata()
    writer.add(tensors, metadata)
    manifest = writer.close()
    assert manifest["rows"] == 5
    assert [shard["rows"] for shard in manifest["shards"]] == [3, 2]
    loaded = list(iter_clt_shards(tmp_path))
    actor_inputs, actor_outputs = branch_batch(loaded[0][0], "actor")
    assert actor_inputs.shape == actor_outputs.shape == (3, 3, 4)


def test_clt_evaluation_casts_float16_corpus_to_model_dtype(tmp_path):
    writer = CLTCorpusWriter(tmp_path, rows_per_shard=5, manifest={"checkpoint_sha256": "abc"})
    tensors = {name: value.half() for name, value in _tiny_corpus_tensors().items()}
    metadata = _tiny_metadata()
    for row in metadata:
        row["split"] = "validation"
    writer.add(tensors, metadata)
    writer.close()
    model = CrossLayerTranscoder(CLTConfig(4, (3, 3, 3), initial_threshold=1e-3))
    metrics = evaluate_corpus(model, tmp_path, branch="actor", split="validation", device="cpu")
    assert metrics["examples"] == 5


def _tiny_marl_gpt():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "marl-gpt"))
    from gpt.model_ac import CriticGPTConfig, CriticWithLoss

    config = CriticGPTConfig(
        block_size=4,
        vocab_size=12,
        n_layer=3,
        n_head=2,
        n_embd=8,
        history_len=2,
        n_team=2,
        n_agents=4,
        n_attr=8,
        action_size=3,
    )
    model = CriticWithLoss(config).eval()
    batch = {
        "obs": torch.randn(1, 4),
        "group_pos": torch.zeros(1, 4, dtype=torch.long),
        "agent_pos": torch.tensor([[0, 1, 2, 3]]),
        "time_pos": torch.tensor([[0, 0, 1, 1]]),
        "attr_pos": torch.tensor([[0, 1, 2, 3]]),
        "action_mask": torch.tensor([[True, True, True]]),
    }
    return model, batch


@pytest.mark.parametrize("branch", ["actor", "critic"])
def test_full_branch_replacement_and_local_error_correction(branch):
    torch.manual_seed(2)
    model, batch = _tiny_marl_gpt()
    clt = CrossLayerTranscoder(CLTConfig(8, (6, 6, 6), initial_threshold=1e-6))
    for encoder in clt.encoders:
        torch.nn.init.constant_(encoder.bias, 0.2)
    with torch.no_grad(), PathCapture(model, branch) as capture:
        original = model(batch)
    errors = local_error_adjustments(capture.result(), clt)
    with torch.no_grad(), CLTReplacement(model, clt, branch, error_adjustments=errors):
        corrected = model(batch)
    index = 0 if branch == "actor" else 1
    assert torch.allclose(original[index], corrected[index], atol=1e-5)


def test_actor_and_critic_graphs_include_attention_conditioned_inputs_and_errors():
    torch.manual_seed(3)
    model, batch = _tiny_marl_gpt()
    actor_clt = CrossLayerTranscoder(CLTConfig(8, (3, 3, 3), initial_threshold=1e-6))
    critic_clt = CrossLayerTranscoder(CLTConfig(8, (3, 3, 3), initial_threshold=1e-6))
    for clt in (actor_clt, critic_clt):
        for encoder in clt.encoders:
            torch.nn.init.constant_(encoder.bias, 0.2)
    actor_graph = build_attribution_graph(
        model, batch, actor_clt, branch="actor", target=ActorTarget(), minimum_edge=0.0
    )
    critic_graph = build_attribution_graph(
        model, batch, critic_clt, branch="critic", target=CriticTarget(), minimum_edge=0.0
    )
    for graph in (actor_graph, critic_graph):
        kinds = {node.kind for node in graph.nodes}
        assert {"input", "feature", "error", "bias", "output"} <= kinds
        assert any(edge.target == "output" and edge.source.startswith("feature:") for edge in graph.edges)
        assert graph.diagnostics["retained_edge_fraction_by_absolute_attribution"] == pytest.approx(1.0)
        assert 0.0 <= graph.diagnostics["output_error_attribution_fraction"] <= 1.0
    pruned = prune_attribution_graph(actor_graph, node_threshold=0.8, edge_threshold=0.98)
    assert len(pruned.nodes) <= len(actor_graph.nodes)
    assert len(pruned.edges) <= len(actor_graph.edges)
    assert any(node.id == "output" for node in pruned.nodes)
    assert pruned.diagnostics["retained_node_influence"] >= 0.8
    assert pruned.diagnostics["retained_edge_influence"] >= 0.98


def test_cross_layer_feature_intervention_changes_original_actor_output():
    torch.manual_seed(5)
    model, batch = _tiny_marl_gpt()
    clt = CrossLayerTranscoder(CLTConfig(8, (3, 3, 3), initial_threshold=1e-6))
    for encoder in clt.encoders:
        torch.nn.init.constant_(encoder.bias, 0.2)
    with torch.no_grad():
        original = model(batch)[0]
        with CrossLayerFeatureIntervention(
            model,
            clt,
            "actor",
            source_layer=0,
            feature=0,
            token_index=-1,
            factor=1.0,
        ) as intervention:
            changed = model(batch)[0]
    assert intervention.activation is not None
    assert not torch.equal(original, changed)


def test_random_feature_intervention_is_reproducible_and_bundle_norm_matched():
    model, _batch = _tiny_marl_gpt()
    clt = CrossLayerTranscoder(CLTConfig(8, (3, 3, 3), initial_threshold=1e-6))
    first = CrossLayerFeatureIntervention(
        model, clt, "actor", source_layer=0, feature=1, token_index=-1, factor=1.0, random_seed=7
    )
    second = CrossLayerFeatureIntervention(
        model, clt, "actor", source_layer=0, feature=1, token_index=-1, factor=1.0, random_seed=7
    )
    for target, random_direction in first.directions.items():
        reference = clt.decoder(0, target)[1]
        assert torch.equal(random_direction, second.directions[target])
        assert float(random_direction.norm()) == pytest.approx(float(reference.detach().norm()))


def test_clt_suite_audit_rejects_a_breached_or_missing_gate():
    health = {}
    for group in ("", "environment/smac/", "environment/pogema/", "environment/grf/"):
        for layer in range(2):
            prefix = f"validation/{group}layer_{layer:02d}"
            health[f"{prefix}/normalized_mse"] = 0.1
            health[f"{prefix}/dead_feature_fraction"] = 0.1
            health[f"{prefix}/l0"] = 32.0
    replacement = {
        f"{group}/{metric}": value
        for group in ("all", "smac", "pogema", "grf")
        for metric, value in (("actor_kl", 0.01), ("action_agreement", 0.95), ("critic_value_mae", 0.5))
    }
    passed = audit_clt_suite(
        health,
        health,
        replacement,
        environments=("smac", "pogema", "grf"),
        num_layers=2,
        maximum_normalized_mse=0.2,
        maximum_dead_feature_fraction=0.5,
        minimum_l0=16,
        maximum_l0=128,
        maximum_actor_kl=0.05,
        minimum_action_agreement=0.9,
        maximum_critic_value_mae=1.0,
    )
    assert passed["eligible_for_graph_interpretation"] is True
    health["validation/environment/grf/layer_01/normalized_mse"] = 0.3
    failed = audit_clt_suite(
        health,
        health,
        replacement,
        environments=("smac", "pogema", "grf"),
        num_layers=2,
        maximum_normalized_mse=0.2,
        maximum_dead_feature_fraction=0.5,
        minimum_l0=16,
        maximum_l0=128,
        maximum_actor_kl=0.05,
        minimum_action_agreement=0.9,
        maximum_critic_value_mae=1.0,
    )
    assert failed["eligible_for_graph_interpretation"] is False

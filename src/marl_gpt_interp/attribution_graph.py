"""Prompt-local attribution graphs for MARL-GPT actor and critic CLTs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from marl_gpt_interp.clt import CrossLayerTranscoder
from marl_gpt_interp.marl_gpt_clt import (
    CLTReplacement,
    FrozenNonlinearities,
    InputResidualLeaf,
    NonlinearityCapture,
    PathCapture,
    action_logit_contrast,
    local_error_adjustments,
)


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    layer: int | None = None
    token: int | None = None
    feature: int | None = None
    activation: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    attribution: float


@dataclass
class AttributionGraph:
    branch: str
    target: dict[str, Any]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    diagnostics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "marl-gpt-attribution-graph",
            "format_version": 1,
            "branch": self.branch,
            "target": self.target,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "diagnostics": self.diagnostics,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def prune_attribution_graph(
    graph: AttributionGraph,
    *,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> AttributionGraph:
    """Prune by indirect influence while retaining input, error, and bias nodes."""

    if not 0 < node_threshold <= 1 or not 0 < edge_threshold <= 1:
        raise ValueError("pruning thresholds must be in (0, 1]")
    incoming_totals: dict[str, float] = {}
    for edge in graph.edges:
        incoming_totals[edge.target] = incoming_totals.get(edge.target, 0.0) + abs(edge.attribution)
    normalized = [
        (edge, abs(edge.attribution) / max(incoming_totals.get(edge.target, 0.0), 1e-12))
        for edge in graph.edges
    ]
    ordered = sorted(
        graph.nodes,
        key=lambda node: (node.layer if node.layer is not None else (-1 if node.kind != "output" else 10**9)),
        reverse=True,
    )
    influence = {node.id: 0.0 for node in graph.nodes}
    influence["output"] = 1.0
    outgoing: dict[str, list[tuple[GraphEdge, float]]] = {}
    for edge, weight in normalized:
        outgoing.setdefault(edge.source, []).append((edge, weight))
    for node in ordered:
        if node.id == "output":
            continue
        influence[node.id] = sum(weight * influence[edge.target] for edge, weight in outgoing.get(node.id, []))

    feature_nodes = [node for node in graph.nodes if node.kind == "feature"]
    ranked_nodes = sorted(feature_nodes, key=lambda node: influence[node.id], reverse=True)
    total_node_score = sum(influence[node.id] for node in ranked_nodes)
    retained_features: set[str] = set()
    cumulative_node_score = 0.0
    for node in ranked_nodes:
        if total_node_score == 0:
            break
        retained_features.add(node.id)
        cumulative_node_score += influence[node.id]
        if cumulative_node_score / total_node_score >= node_threshold:
            break
    retained_nodes = {
        node.id
        for node in graph.nodes
        if node.kind in {"input", "error", "bias", "output"} or node.id in retained_features
    }
    candidate_edges = [
        (edge, weight * influence[edge.target])
        for edge, weight in normalized
        if edge.source in retained_nodes and edge.target in retained_nodes
    ]
    candidate_edges.sort(key=lambda item: item[1], reverse=True)
    total_edge_score = sum(score for _edge, score in candidate_edges)
    kept_edges, cumulative_edge_score = [], 0.0
    for edge, score in candidate_edges:
        kept_edges.append(edge)
        cumulative_edge_score += score
        if total_edge_score and cumulative_edge_score / total_edge_score >= edge_threshold:
            break
    incident = {"output"}
    for edge in kept_edges:
        incident.update((edge.source, edge.target))
    kept_nodes = [node for node in graph.nodes if node.id in incident]
    return AttributionGraph(
        branch=graph.branch,
        target=graph.target,
        nodes=kept_nodes,
        edges=kept_edges,
        diagnostics={
            **graph.diagnostics,
            "unpruned_nodes": float(len(graph.nodes)),
            "unpruned_edges": float(len(graph.edges)),
            "node_influence_threshold": node_threshold,
            "edge_influence_threshold": edge_threshold,
            "retained_node_influence": cumulative_node_score / max(total_node_score, 1e-12),
            "retained_edge_influence": cumulative_edge_score / max(total_edge_score, 1e-12),
        },
    )


@dataclass(frozen=True)
class ActorTarget:
    action: int | None = None
    comparison_action: int | None = None


@dataclass(frozen=True)
class CriticTarget:
    action: int | None = None


def _model_outputs(model: Any, batch_obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(batch_obs)
    if not isinstance(output, tuple) or len(output) < 2:
        raise TypeError("MARL-GPT must return actor and critic logits")
    return output[0], output[1]


def _actor_selector(
    original_actor: torch.Tensor,
    action_mask: torch.Tensor,
    spec: ActorTarget,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dict[str, Any]]:
    _target, action, default_comparison = action_logit_contrast(original_actor, action_mask, spec.action)
    comparison = default_comparison if spec.comparison_action is None else int(spec.comparison_action)
    if comparison == action or not bool(action_mask[0, comparison]):
        raise ValueError("comparison_action must be a different legal action")

    def select(actor_logits: torch.Tensor, _critic_logits: torch.Tensor) -> torch.Tensor:
        return actor_logits[0, action] - actor_logits[0, comparison]

    return select, {"kind": "action_logit_contrast", "action": action, "comparison_action": comparison}


def _critic_selector(
    model: Any,
    original_actor: torch.Tensor,
    original_critic: torch.Tensor,
    action_mask: torch.Tensor,
    spec: CriticTarget,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dict[str, Any]]:
    legal_actor = original_actor.masked_fill(~action_mask, -torch.inf)
    action = int(legal_actor.argmax(dim=-1).item()) if spec.action is None else int(spec.action)
    if not bool(action_mask[0, action]):
        raise ValueError("critic target action must be legal")
    logits = original_critic[0, action]
    probabilities = torch.softmax(logits, dim=-1)
    support = model.critic_loss.support.to(logits)
    centers = (support[:-1] + support[1:]) / 2
    expected_value = (probabilities * centers).sum()
    tangent = (probabilities * (centers - expected_value)).detach()

    def select(_actor_logits: torch.Tensor, critic_logits: torch.Tensor) -> torch.Tensor:
        return (critic_logits[0, action] * tangent).sum()

    return select, {
        "kind": "expected_action_value_local_linearization",
        "action": action,
        "reference_value": float(expected_value),
    }


def _feature_id(layer: int, token: int, feature: int) -> str:
    return f"feature:{layer}:{token}:{feature}"


def _input_id(token: int) -> str:
    return f"input:{token}"


def _error_id(layer: int, token: int) -> str:
    return f"error:{layer}:{token}"


def _bias_id(target: str) -> str:
    return f"bias:{target}"


class _GraphAccumulator:
    def __init__(self, *, minimum_edge: float) -> None:
        self.minimum_edge = minimum_edge
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.total_absolute_attribution = 0.0
        self.retained_absolute_attribution = 0.0

    def node(self, node: GraphNode) -> None:
        self.nodes.setdefault(node.id, node)

    def edge(self, source: GraphNode, target: GraphNode, attribution: torch.Tensor | float) -> float:
        value = float(attribution.detach()) if isinstance(attribution, torch.Tensor) else float(attribution)
        self.total_absolute_attribution += abs(value)
        if abs(value) >= self.minimum_edge:
            self.node(source)
            self.node(target)
            self.edges.append(GraphEdge(source.id, target.id, value))
            self.retained_absolute_attribution += abs(value)
        return value


def _source_contributions(
    accumulator: _GraphAccumulator,
    *,
    target_node: GraphNode,
    target_scalar: torch.Tensor,
    feature_leaves: list[torch.Tensor],
    input_leaf: torch.Tensor,
    error_leaves: list[torch.Tensor],
    maximum_source_layer: int,
) -> float:
    sources: list[torch.Tensor] = [*feature_leaves, input_leaf, *error_leaves]
    gradients = torch.autograd.grad(target_scalar, sources, retain_graph=True, allow_unused=True)
    incoming = 0.0
    feature_gradients = gradients[: len(feature_leaves)]
    input_gradient = gradients[len(feature_leaves)]
    error_gradients = gradients[len(feature_leaves) + 1 :]

    for layer, (leaf, gradient) in enumerate(zip(feature_leaves, feature_gradients, strict=True)):
        if layer >= maximum_source_layer or gradient is None:
            continue
        contributions = leaf * gradient
        active_indices = (leaf != 0).nonzero(as_tuple=False)
        for batch, token, feature in active_indices.tolist():
            if batch != 0:
                raise ValueError("attribution graphs currently support batch size one")
            value = contributions[batch, token, feature]
            incoming += accumulator.edge(
                GraphNode(
                    _feature_id(layer, token, feature),
                    "feature",
                    layer=layer,
                    token=token,
                    feature=feature,
                    activation=float(leaf[batch, token, feature].detach()),
                ),
                target_node,
                value,
            )

    if input_gradient is not None:
        contributions = (input_leaf * input_gradient).sum(dim=-1)
        for token, value in enumerate(contributions[0]):
            incoming += accumulator.edge(
                GraphNode(_input_id(token), "input", token=token), target_node, value
            )

    for layer, (leaf, gradient) in enumerate(zip(error_leaves, error_gradients, strict=True)):
        if layer >= maximum_source_layer or gradient is None:
            continue
        contributions = (leaf * gradient).sum(dim=-1)
        for token, value in enumerate(contributions[0]):
            incoming += accumulator.edge(
                GraphNode(_error_id(layer, token), "error", layer=layer, token=token), target_node, value
            )
    return incoming


def build_attribution_graph(
    model: Any,
    batch_obs: dict[str, torch.Tensor],
    clt: CrossLayerTranscoder,
    *,
    branch: str,
    target: ActorTarget | CriticTarget,
    minimum_edge: float = 1e-6,
) -> AttributionGraph:
    """Build an exact local graph conditional on frozen QK and norm denominators.

    Feature edges include all linear paths through residual connections and
    frozen attention OV circuits. Reconstruction errors are explicit vector
    nodes. Each feature's residual constant is represented by a bias node so
    incoming edges sum to its preactivation before edge thresholding.
    """

    if next(iter(batch_obs.values())).shape[0] != 1:
        raise ValueError("attribution graph construction requires batch size one")
    model.eval()
    with torch.no_grad(), PathCapture(model, branch) as path_capture, NonlinearityCapture(
        model, branch
    ) as nonlinear_capture:
        original_actor, original_critic = _model_outputs(model, batch_obs)
    path = path_capture.result()
    frozen_state = nonlinear_capture.result()
    errors = local_error_adjustments(path, clt)
    action_mask = batch_obs["action_mask"].bool()
    if branch == "actor":
        if not isinstance(target, ActorTarget):
            raise TypeError("actor attribution requires ActorTarget")
        selector, target_metadata = _actor_selector(original_actor, action_mask, target)
    elif branch == "critic":
        if not isinstance(target, CriticTarget):
            raise TypeError("critic attribution requires CriticTarget")
        selector, target_metadata = _critic_selector(
            model, original_actor, original_critic, action_mask, target
        )
    else:
        raise ValueError("branch must be actor or critic")

    with torch.enable_grad(), FrozenNonlinearities(model, branch, frozen_state), InputResidualLeaf(
        model
    ) as input_context, CLTReplacement(
        model, clt, branch, error_adjustments=errors, detach_features=True
    ) as replacement:
        local_actor, local_critic = _model_outputs(model, batch_obs)
        output_scalar = selector(local_actor, local_critic)

    if input_context.value is None:
        raise RuntimeError("input residual was not captured")
    feature_leaves = [value for value in replacement.trace.feature_leaves if value is not None]
    error_leaves = [value for value in replacement.trace.errors if value is not None]
    preactivations = [value for value in replacement.trace.preactivations if value is not None]
    if len(feature_leaves) != clt.config.num_layers or len(error_leaves) != clt.config.num_layers:
        raise RuntimeError("local replacement trace is incomplete")

    accumulator = _GraphAccumulator(minimum_edge=minimum_edge)
    active_target_count = 0
    for layer, (preactivation, activation) in enumerate(zip(preactivations, feature_leaves, strict=True)):
        for batch, token, feature in (activation != 0).nonzero(as_tuple=False).tolist():
            if batch != 0:
                raise ValueError("attribution graphs currently support batch size one")
            active_target_count += 1
            target_node = GraphNode(
                _feature_id(layer, token, feature),
                "feature",
                layer=layer,
                token=token,
                feature=feature,
                activation=float(activation[batch, token, feature].detach()),
                metadata={"preactivation": float(preactivation[batch, token, feature].detach())},
            )
            accumulator.node(target_node)
            scalar = preactivation[batch, token, feature]
            incoming = _source_contributions(
                accumulator,
                target_node=target_node,
                target_scalar=scalar,
                feature_leaves=feature_leaves,
                input_leaf=input_context.value,
                error_leaves=error_leaves,
                maximum_source_layer=layer,
            )
            residual = float(scalar.detach()) - incoming
            accumulator.edge(GraphNode(_bias_id(target_node.id), "bias"), target_node, residual)

    output_node = GraphNode("output", "output", activation=float(output_scalar.detach()), metadata=target_metadata)
    accumulator.node(output_node)
    incoming = _source_contributions(
        accumulator,
        target_node=output_node,
        target_scalar=output_scalar,
        feature_leaves=feature_leaves,
        input_leaf=input_context.value,
        error_leaves=error_leaves,
        maximum_source_layer=clt.config.num_layers,
    )
    accumulator.edge(
        GraphNode(_bias_id(output_node.id), "bias"), output_node, float(output_scalar.detach()) - incoming
    )
    retained_fraction = accumulator.retained_absolute_attribution / max(
        accumulator.total_absolute_attribution, 1e-12
    )
    output_edges = [edge for edge in accumulator.edges if edge.target == "output"]
    output_mass = sum(abs(edge.attribution) for edge in output_edges)
    error_output_mass = sum(
        abs(edge.attribution) for edge in output_edges if edge.source.startswith("error:")
    )
    return AttributionGraph(
        branch=branch,
        target=target_metadata,
        nodes=list(accumulator.nodes.values()),
        edges=accumulator.edges,
        diagnostics={
            "active_feature_nodes": float(active_target_count),
            "retained_edge_fraction_by_absolute_attribution": retained_fraction,
            "output_error_attribution_fraction": error_output_mass / max(output_mass, 1e-12),
            "local_output": float(output_scalar.detach()),
            "original_actor_max_logit": float(original_actor.max()),
        },
    )


def incoming_edges(graph: AttributionGraph, node_id: str) -> Iterable[GraphEdge]:
    return (edge for edge in graph.edges if edge.target == node_id)

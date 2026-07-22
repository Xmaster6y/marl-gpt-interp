"""MARL-GPT instrumentation and replacement hooks for branch-specific CLTs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from marl_gpt_interp.clt import CrossLayerTranscoder


BRANCHES = ("actor", "critic")


def branch_blocks(model: Any, branch: str) -> tuple[Any, ...]:
    if branch not in BRANCHES:
        raise ValueError("branch must be actor or critic")
    shared = tuple(model.transformer.h)
    branch_block = model.transformer.act_layers if branch == "actor" else model.transformer.critic_layers
    return (*shared, branch_block)


def branch_final_norm(model: Any, branch: str) -> Any:
    return model.transformer.ln_actor_f if branch == "actor" else model.transformer.ln_critic_f


def branch_head(model: Any, branch: str) -> Any:
    return model.lm_head_actor if branch == "actor" else model.lm_head_critic


@dataclass(frozen=True)
class PathActivations:
    residual_inputs: tuple[torch.Tensor, ...]
    mlp_outputs: tuple[torch.Tensor, ...]


class PathCapture:
    """Capture the residual input and target output of every MLP on one path."""

    def __init__(self, model: Any, branch: str, *, detach: bool = True) -> None:
        self.blocks = branch_blocks(model, branch)
        self.detach = detach
        self.residual_inputs: list[torch.Tensor | None] = [None] * len(self.blocks)
        self.mlp_outputs: list[torch.Tensor | None] = [None] * len(self.blocks)
        self.handles = []

    def __enter__(self) -> PathCapture:
        for layer, block in enumerate(self.blocks):
            self.handles.append(block.ln_2.register_forward_pre_hook(self._capture_residual(layer)))
            self.handles.append(block.mlp.register_forward_hook(self._capture_mlp(layer)))
        return self

    def _capture_residual(self, layer: int):
        def hook(_module, inputs):
            value = inputs[0]
            self.residual_inputs[layer] = value.detach() if self.detach else value

        return hook

    def _capture_mlp(self, layer: int):
        def hook(_module, _inputs, output):
            self.mlp_outputs[layer] = output.detach() if self.detach else output

        return hook

    def result(self) -> PathActivations:
        if any(value is None for value in (*self.residual_inputs, *self.mlp_outputs)):
            raise RuntimeError("the complete MARL-GPT path has not run")
        return PathActivations(
            tuple(value for value in self.residual_inputs if value is not None),
            tuple(value for value in self.mlp_outputs if value is not None),
        )

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass
class ReplacementTrace:
    preactivations: list[torch.Tensor | None]
    activations: list[torch.Tensor | None]
    reconstructions: list[torch.Tensor | None]
    feature_leaves: list[torch.Tensor | None]
    errors: list[torch.Tensor | None]


class CLTReplacement:
    """Replace every MLP on one actor or critic path with a single CLT."""

    def __init__(
        self,
        model: Any,
        clt: CrossLayerTranscoder,
        branch: str,
        *,
        error_adjustments: Sequence[torch.Tensor] | None = None,
        detach_features: bool = False,
    ) -> None:
        self.blocks = branch_blocks(model, branch)
        if len(self.blocks) != clt.config.num_layers:
            raise ValueError("CLT layer count does not match the MARL-GPT branch path")
        if error_adjustments is not None and len(error_adjustments) != len(self.blocks):
            raise ValueError("one error adjustment is required per path layer")
        self.clt = clt
        self.error_adjustments = tuple(error_adjustments) if error_adjustments is not None else None
        self.detach_features = detach_features
        count = len(self.blocks)
        self.trace = ReplacementTrace([None] * count, [None] * count, [None] * count, [None] * count, [None] * count)
        self.handles = []

    def __enter__(self) -> CLTReplacement:
        for layer, block in enumerate(self.blocks):
            self.handles.append(block.ln_2.register_forward_pre_hook(self._encode(layer)))
            self.handles.append(block.mlp.register_forward_hook(self._replace(layer)))
        return self

    def _encode(self, layer: int):
        def hook(_module, inputs):
            if layer == 0:
                count = len(self.blocks)
                self.trace = ReplacementTrace(
                    [None] * count, [None] * count, [None] * count, [None] * count, [None] * count
                )
            residual = inputs[0]
            preactivation, activation = self.clt.encode_layer(layer, residual)
            self.trace.preactivations[layer] = preactivation
            if self.detach_features:
                leaf = activation.detach().requires_grad_(True)
                activation = leaf
                self.trace.feature_leaves[layer] = leaf
            self.trace.activations[layer] = activation

        return hook

    def _replace(self, layer: int):
        def hook(_module, _inputs, _output):
            activations = self.trace.activations[: layer + 1]
            if any(value is None for value in activations):
                raise RuntimeError("CLT source features are missing")
            reconstruction = self.clt.reconstruct_layer(
                layer, [value for value in activations if value is not None]
            )
            if self.error_adjustments is not None:
                error = self.error_adjustments[layer]
                if self.detach_features:
                    error = error.detach().requires_grad_(True)
                self.trace.errors[layer] = error
                reconstruction = reconstruction + error
            self.trace.reconstructions[layer] = reconstruction
            return reconstruction

        return hook

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class CrossLayerFeatureIntervention:
    """Inject one CLT feature's learned write bundle into the original model."""

    def __init__(
        self,
        model: Any,
        clt: CrossLayerTranscoder,
        branch: str,
        *,
        source_layer: int,
        feature: int,
        token_index: int,
        factor: float,
        target_layers: Sequence[int] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.blocks = branch_blocks(model, branch)
        if not 0 <= source_layer < len(self.blocks):
            raise ValueError("source_layer is outside the branch path")
        if not 0 <= feature < clt.config.features_per_layer[source_layer]:
            raise ValueError("feature is outside the source layer")
        targets = tuple(range(source_layer, len(self.blocks))) if target_layers is None else tuple(target_layers)
        if not targets or any(target < source_layer or target >= len(self.blocks) for target in targets):
            raise ValueError("target layers must be downstream of the feature encoder")
        self.clt = clt
        self.source_layer = source_layer
        self.feature = feature
        self.token_index = token_index
        self.factor = factor
        self.target_layers = targets
        self.directions: dict[int, torch.Tensor] = {}
        if random_seed is not None:
            generator = torch.Generator().manual_seed(random_seed)
            for target in targets:
                reference = clt.decoder(source_layer, target)[feature].detach()
                random = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
                self.directions[target] = random / random.norm().clamp_min(1e-8) * reference.norm()
        self.activation: torch.Tensor | None = None
        self.handles = []

    def __enter__(self) -> CrossLayerFeatureIntervention:
        source_block = self.blocks[self.source_layer]
        self.handles.append(source_block.ln_2.register_forward_pre_hook(self._capture_activation()))
        for layer in self.target_layers:
            self.handles.append(self.blocks[layer].mlp.register_forward_hook(self._inject(layer)))
        return self

    def _capture_activation(self):
        def hook(_module, inputs):
            _preactivation, activation = self.clt.encode_layer(self.source_layer, inputs[0])
            self.activation = activation[:, self.token_index, self.feature]

        return hook

    def _inject(self, target_layer: int):
        def hook(_module, _inputs, output):
            if self.activation is None:
                raise RuntimeError("feature activation was not captured before its intervention")
            result = output.clone()
            direction = self.directions.get(
                target_layer, self.clt.decoder(self.source_layer, target_layer)[self.feature]
            ).to(output)
            result[:, self.token_index, :] += self.factor * self.activation[:, None] * direction
            return result

        return hook

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass(frozen=True)
class FrozenState:
    attention_patterns: Mapping[int, torch.Tensor]
    layer_norm_rstd: Mapping[int, torch.Tensor]


class NonlinearityCapture:
    """Record attention patterns and LayerNorm denominators for one input."""

    def __init__(self, model: Any, branch: str) -> None:
        self.blocks = branch_blocks(model, branch)
        self.norms = [norm for block in self.blocks for norm in (block.ln_1, block.ln_2)]
        self.norms.append(branch_final_norm(model, branch))
        self.attention_patterns: dict[int, torch.Tensor] = {}
        self.layer_norm_rstd: dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self) -> NonlinearityCapture:
        for block in self.blocks:
            self.handles.append(block.attn.register_forward_pre_hook(self._attention_hook(block.attn)))
        for norm in self.norms:
            self.handles.append(norm.register_forward_pre_hook(self._norm_hook(norm)))
        return self

    def _attention_hook(self, attention):
        def hook(_module, inputs):
            x = inputs[0]
            batch, tokens, width = x.shape
            q, k, _v = attention.c_attn(x).split(attention.n_embd, dim=2)
            head_width = width // attention.n_head
            q = q.view(batch, tokens, attention.n_head, head_width).transpose(1, 2)
            k = k.view(batch, tokens, attention.n_head, head_width).transpose(1, 2)
            pattern = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(head_width), dim=-1)
            self.attention_patterns[id(attention)] = pattern.detach()

        return hook

    def _norm_hook(self, norm):
        def hook(_module, inputs):
            x = inputs[0]
            variance = x.var(dim=-1, keepdim=True, unbiased=False)
            self.layer_norm_rstd[id(norm)] = torch.rsqrt(variance + 1e-5).detach()

        return hook

    def result(self) -> FrozenState:
        expected_attention = {id(block.attn) for block in self.blocks}
        expected_norms = {id(norm) for norm in self.norms}
        if self.attention_patterns.keys() != expected_attention or self.layer_norm_rstd.keys() != expected_norms:
            raise RuntimeError("the complete branch path has not run")
        return FrozenState(dict(self.attention_patterns), dict(self.layer_norm_rstd))

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FrozenNonlinearities:
    """Freeze QK attention patterns and normalization denominators via hooks."""

    def __init__(self, model: Any, branch: str, state: FrozenState) -> None:
        self.blocks = branch_blocks(model, branch)
        self.norms = [norm for block in self.blocks for norm in (block.ln_1, block.ln_2)]
        self.norms.append(branch_final_norm(model, branch))
        self.state = state
        self.handles = []

    def __enter__(self) -> FrozenNonlinearities:
        for block in self.blocks:
            self.handles.append(block.attn.register_forward_hook(self._frozen_attention(block.attn)))
        for norm in self.norms:
            self.handles.append(norm.register_forward_hook(self._frozen_norm(norm)))
        return self

    def _frozen_attention(self, attention):
        pattern = self.state.attention_patterns[id(attention)]

        def hook(_module, inputs, _output):
            x = inputs[0]
            batch, tokens, width = x.shape
            _q, _k, value = attention.c_attn(x).split(attention.n_embd, dim=2)
            head_width = width // attention.n_head
            value = value.view(batch, tokens, attention.n_head, head_width).transpose(1, 2)
            mixed = pattern.to(value) @ value
            mixed = mixed.transpose(1, 2).contiguous().view(batch, tokens, width)
            return attention.resid_dropout(attention.c_proj(mixed))

        return hook

    def _frozen_norm(self, norm):
        rstd = self.state.layer_norm_rstd[id(norm)]

        def hook(_module, inputs, _output):
            x = inputs[0]
            centered = x - x.mean(dim=-1, keepdim=True)
            result = centered * rstd.to(x) * norm.weight
            return result if norm.bias is None else result + norm.bias

        return hook

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class InputResidualLeaf:
    """Expose the combined token and positional embedding as graph input nodes."""

    def __init__(self, model: Any) -> None:
        self.module = model.transformer.drop
        self.value: torch.Tensor | None = None
        self.handle = None

    def __enter__(self) -> InputResidualLeaf:
        def hook(_module, _inputs, output):
            self.value = output.detach().requires_grad_(True)
            return self.value

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def local_error_adjustments(path: PathActivations, clt: CrossLayerTranscoder) -> tuple[torch.Tensor, ...]:
    forward = clt(path.residual_inputs)
    return tuple(
        target - reconstruction
        for target, reconstruction in zip(path.mlp_outputs, forward.reconstructions, strict=True)
    )


def action_logit_contrast(logits: torch.Tensor, action_mask: torch.Tensor, action: int | None = None) -> tuple[torch.Tensor, int, int]:
    if logits.shape[0] != 1 or action_mask.shape != logits.shape:
        raise ValueError("attribution currently requires one decision and a matching action mask")
    legal = logits.masked_fill(~action_mask, -torch.inf)
    chosen = int(legal.argmax(dim=-1).item()) if action is None else int(action)
    if not bool(action_mask[0, chosen]):
        raise ValueError("the attributed action must be legal")
    alternatives = legal.clone()
    alternatives[0, chosen] = -torch.inf
    comparison = int(alternatives.argmax(dim=-1).item())
    if not torch.isfinite(alternatives[0, comparison]):
        raise ValueError("an action contrast needs at least two legal actions")
    return logits[0, chosen] - logits[0, comparison], chosen, comparison

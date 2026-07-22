"""Cross-layer transcoders for actor and critic computation paths."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn


class _JumpReLU(torch.autograd.Function):
    """JumpReLU with a rectangular-kernel surrogate gradient for the threshold."""

    @staticmethod
    def forward(ctx, preactivations: torch.Tensor, thresholds: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(preactivations, thresholds)
        ctx.bandwidth = bandwidth
        return preactivations * (preactivations > thresholds)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        preactivations, thresholds = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        active = preactivations > thresholds
        distance = (preactivations - thresholds).abs()
        kernel = (distance < bandwidth / 2).to(preactivations.dtype) / bandwidth
        grad_preactivations = grad_output * active
        grad_thresholds = -(grad_output * preactivations * kernel).sum_to_size(thresholds.shape)
        return grad_preactivations, grad_thresholds, None


def jump_relu(
    preactivations: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    bandwidth: float = 1e-3,
) -> torch.Tensor:
    if bandwidth <= 0:
        raise ValueError("JumpReLU bandwidth must be positive")
    return _JumpReLU.apply(preactivations, thresholds, bandwidth)


@dataclass(frozen=True)
class CLTConfig:
    """Architecture shared by one branch-specific cross-layer transcoder."""

    input_dim: int
    features_per_layer: tuple[int, ...]
    initial_threshold: float = 1e-2
    jump_relu_bandwidth: float = 1e-3

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.features_per_layer or any(width <= 0 for width in self.features_per_layer):
            raise ValueError("features_per_layer must contain positive widths")
        if self.initial_threshold <= 0:
            raise ValueError("initial_threshold must be positive")

    @property
    def num_layers(self) -> int:
        return len(self.features_per_layer)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["features_per_layer"] = list(self.features_per_layer)
        return payload


@dataclass(frozen=True)
class CLTForward:
    preactivations: tuple[torch.Tensor, ...]
    activations: tuple[torch.Tensor, ...]
    reconstructions: tuple[torch.Tensor, ...]


class CrossLayerTranscoder(nn.Module):
    """Jointly reconstruct every MLP output from current and earlier features.

    A feature encoded at layer ``source`` owns one decoder vector for every
    ``target >= source``. Decoder bundles are indexed explicitly so replacement
    and attribution code use the same learned computation.
    """

    def __init__(self, config: CLTConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleList(
            nn.Linear(config.input_dim, width, bias=True) for width in config.features_per_layer
        )
        initial_log_threshold = math.log(config.initial_threshold)
        self.log_thresholds = nn.ParameterList(
            nn.Parameter(torch.full((width,), initial_log_threshold)) for width in config.features_per_layer
        )
        self.output_biases = nn.ParameterList(
            nn.Parameter(torch.zeros(config.input_dim)) for _ in range(config.num_layers)
        )
        self.decoders = nn.ParameterDict()
        for source, width in enumerate(config.features_per_layer):
            for target in range(source, config.num_layers):
                parameter = nn.Parameter(torch.empty(width, config.input_dim))
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                self.decoders[self.decoder_key(source, target)] = parameter
        self.normalize_decoder_bundles_()

    @staticmethod
    def decoder_key(source: int, target: int) -> str:
        return f"{source}_to_{target}"

    def decoder(self, source: int, target: int) -> torch.Tensor:
        if target < source:
            raise ValueError("CLT features cannot write to earlier layers")
        return self.decoders[self.decoder_key(source, target)]

    def thresholds(self, layer: int) -> torch.Tensor:
        return self.log_thresholds[layer].exp()

    def encode_layer(self, layer: int, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preactivations = self.encoders[layer](residual)
        activations = jump_relu(
            preactivations,
            self.thresholds(layer),
            bandwidth=self.config.jump_relu_bandwidth,
        )
        return preactivations, activations

    def reconstruct_layer(self, target: int, activations: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(activations) <= target:
            raise ValueError("reconstruct_layer needs activations through its target layer")
        reconstruction = self.output_biases[target]
        for source in range(target + 1):
            reconstruction = reconstruction + activations[source] @ self.decoder(source, target)
        return reconstruction

    def forward(self, residual_inputs: Sequence[torch.Tensor]) -> CLTForward:
        if len(residual_inputs) != self.config.num_layers:
            raise ValueError(
                f"expected {self.config.num_layers} residual tensors, received {len(residual_inputs)}"
            )
        preactivations, activations, reconstructions = [], [], []
        for layer, residual in enumerate(residual_inputs):
            preactivation, activation = self.encode_layer(layer, residual)
            preactivations.append(preactivation)
            activations.append(activation)
            reconstructions.append(self.reconstruct_layer(layer, activations))
        return CLTForward(tuple(preactivations), tuple(activations), tuple(reconstructions))

    def decoder_bundle_norms(self, source: int) -> torch.Tensor:
        bundle = torch.cat(
            [self.decoder(source, target) for target in range(source, self.config.num_layers)], dim=-1
        )
        return bundle.norm(dim=-1)

    @torch.no_grad()
    def normalize_decoder_bundles_(self) -> None:
        for source in range(self.config.num_layers):
            norms = self.decoder_bundle_norms(source).clamp_min(1e-8)
            for target in range(source, self.config.num_layers):
                self.decoder(source, target).div_(norms[:, None])


@dataclass(frozen=True)
class CLTLoss:
    loss: torch.Tensor
    reconstruction: torch.Tensor
    sparsity: torch.Tensor
    layer_normalized_mse: tuple[torch.Tensor, ...]
    l0: torch.Tensor


def clt_loss(
    model: CrossLayerTranscoder,
    residual_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    sparsity_coefficient: float,
    sparsity_tanh_scale: float = 1.0,
) -> CLTLoss:
    """Compute layer-balanced reconstruction and decoder-aware sparsity losses.

    Inputs have shape ``[..., layers, d_model]``. The leading dimensions are
    treated as independent token examples. Natural activations are used; the
    method never branches on environment labels.
    """

    if residual_inputs.shape != mlp_outputs.shape:
        raise ValueError("residual inputs and MLP outputs must have identical shapes")
    if residual_inputs.shape[-2:] != (model.config.num_layers, model.config.input_dim):
        raise ValueError("activation tensors do not match the CLT configuration")
    if sparsity_coefficient < 0 or sparsity_tanh_scale <= 0:
        raise ValueError("invalid sparsity hyperparameters")
    leading_shape = residual_inputs.shape[:-2]
    if valid_mask is None:
        valid_mask = torch.ones(leading_shape, dtype=torch.bool, device=residual_inputs.device)
    if valid_mask.shape != leading_shape:
        raise ValueError("valid_mask must match the leading token dimensions")
    if not valid_mask.any():
        raise ValueError("CLT loss requires at least one valid token")

    forward = model(tuple(residual_inputs[..., layer, :] for layer in range(model.config.num_layers)))
    layer_losses = []
    for layer, reconstruction in enumerate(forward.reconstructions):
        target = mlp_outputs[..., layer, :]
        error = (reconstruction - target)[valid_mask]
        variance = target[valid_mask].float().var(unbiased=False).clamp_min(1e-8)
        layer_losses.append(error.float().pow(2).mean() / variance)
    reconstruction_loss = torch.stack(layer_losses).mean()

    sparsity_terms, active_counts = [], []
    for layer, activation in enumerate(forward.activations):
        selected = activation[valid_mask]
        norms = model.decoder_bundle_norms(layer)
        sparsity_terms.append(torch.tanh(sparsity_tanh_scale * selected * norms).sum(dim=-1).mean())
        active_counts.append((selected > 0).float().sum(dim=-1).mean())
    sparsity_loss = torch.stack(sparsity_terms).mean()
    l0 = torch.stack(active_counts).mean()
    return CLTLoss(
        loss=reconstruction_loss + sparsity_coefficient * sparsity_loss,
        reconstruction=reconstruction_loss,
        sparsity=sparsity_loss,
        layer_normalized_mse=tuple(layer_losses),
        l0=l0,
    )


@torch.no_grad()
def clt_health_metrics(
    model: CrossLayerTranscoder,
    residual_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    result = clt_loss(
        model,
        residual_inputs,
        mlp_outputs,
        valid_mask=valid_mask,
        sparsity_coefficient=0.0,
    )
    forward = model(tuple(residual_inputs[..., layer, :] for layer in range(model.config.num_layers)))
    if valid_mask is None:
        valid_mask = torch.ones(residual_inputs.shape[:-2], dtype=torch.bool, device=residual_inputs.device)
    rates = []
    metrics = {
        "normalized_mse": float(result.reconstruction),
        "l0": float(result.l0),
    }
    for layer, (loss, activation) in enumerate(zip(result.layer_normalized_mse, forward.activations, strict=True)):
        layer_rates = (activation[valid_mask] > 0).float().mean(dim=0)
        rates.append(layer_rates)
        metrics[f"layer_{layer:02d}/normalized_mse"] = float(loss)
        metrics[f"layer_{layer:02d}/l0"] = float((activation[valid_mask] > 0).float().sum(dim=-1).mean())
        metrics[f"layer_{layer:02d}/dead_feature_fraction"] = float((layer_rates == 0).float().mean())
    metrics["dead_feature_fraction"] = float((torch.cat(rates) == 0).float().mean())
    return metrics

"""Sparse feature models, synthetic data, caches, and evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def support_name(support: Iterable[str]) -> str:
    return "+".join(sorted(support))


def domain_lattice(domains: Sequence[str]) -> tuple[frozenset[str], ...]:
    """Return every non-empty support subset, largest and then lexical first."""

    from itertools import combinations

    ordered = tuple(sorted(set(domains)))
    return tuple(frozenset(group) for size in range(len(ordered), 0, -1) for group in combinations(ordered, size))


def topk_codes(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k > scores.shape[-1]:
        raise ValueError(f"k must be in [1, {scores.shape[-1]}], got {k}")
    values, indices = torch.topk(scores, k=k, dim=-1)
    values = F.relu(values)
    output = torch.zeros_like(scores)
    return output.scatter(-1, indices, values)


def batch_topk_codes(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Keep batch_size * k positive scores across the whole batch."""

    if scores.ndim != 2:
        raise ValueError("BatchTopK expects a two-dimensional score matrix")
    if k <= 0 or k > scores.shape[1]:
        raise ValueError(f"k must be in [1, {scores.shape[1]}], got {k}")
    flat = scores.flatten()
    count = min(scores.shape[0] * k, flat.numel())
    values, indices = torch.topk(flat, k=count)
    output = torch.zeros_like(flat)
    output.scatter_(0, indices, F.relu(values))
    return output.view_as(scores)


def domain_stratified_batch_topk_codes(scores: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """Apply an independent BatchTopK budget to every represented domain."""

    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("labels must have one entry per score row")
    output = torch.zeros_like(scores)
    for label in labels.unique(sorted=True):
        mask = labels == label
        output[mask] = batch_topk_codes(scores[mask], k)
    return output


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, width: int, k: int, activation: str = "topk") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.k = k
        self.activation = activation
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, width, bias=True)
        self.decoder = nn.Parameter(torch.empty(width, input_dim))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))
        self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.data = F.normalize(self.decoder.data, dim=-1)

    def encode(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.encoder(x - self.pre_bias)
        if self.activation == "topk":
            return topk_codes(scores, self.k)
        if self.activation == "batch_topk":
            return batch_topk_codes(scores, self.k)
        if self.activation == "domain_batch_topk":
            if labels is None:
                raise ValueError("domain_batch_topk requires labels")
            return domain_stratified_batch_topk_codes(scores, labels, self.k)
        raise ValueError(f"Unknown activation {self.activation!r}")

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.decoder + self.pre_bias

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(x, labels)
        return self.decode(codes), codes


class DomainLatticeSAE(SparseAutoencoder):
    """One dictionary whose latent eligibility follows declared domain supports."""

    def __init__(
        self,
        input_dim: int,
        domains: Sequence[str],
        width_per_support: int,
        k: int,
        activation: str = "topk",
    ) -> None:
        self.domains = tuple(domains)
        self.supports = domain_lattice(self.domains)
        self.width_per_support = width_per_support
        super().__init__(input_dim, len(self.supports) * width_per_support, k, activation)
        self.register_buffer(
            "latent_domain_mask",
            torch.tensor(
                [
                    [domain in support for support in self.supports for _ in range(width_per_support)]
                    for domain in self.domains
                ],
                dtype=torch.bool,
            ),
        )

    @property
    def latent_supports(self) -> tuple[frozenset[str], ...]:
        return tuple(support for support in self.supports for _ in range(self.width_per_support))

    def encode(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            raise ValueError("DomainLatticeSAE requires integer domain labels")
        scores = self.encoder(x - self.pre_bias)
        eligibility = self.latent_domain_mask[labels]
        scores = scores.masked_fill(~eligibility, -torch.inf)
        if self.activation == "topk":
            return topk_codes(scores, self.k)
        if self.activation == "domain_batch_topk":
            return domain_stratified_batch_topk_codes(scores, labels, self.k)
        raise ValueError("Lattice activation must be topk or domain_batch_topk")


class IndependentSAEs(nn.Module):
    def __init__(self, input_dim: int, domains: Sequence[str], width: int, k: int) -> None:
        super().__init__()
        self.domains = tuple(domains)
        self.models = nn.ModuleList(SparseAutoencoder(input_dim, width, k) for _ in self.domains)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction = torch.zeros_like(x)
        codes = torch.zeros(x.shape[0], len(self.domains) * self.models[0].width, device=x.device, dtype=x.dtype)
        for label, model in enumerate(self.models):
            mask = labels == label
            if not mask.any():
                continue
            reconstructed, local_codes = model(x[mask])
            reconstruction[mask] = reconstructed
            start = label * model.width
            codes[mask, start : start + model.width] = local_codes
        return reconstruction, codes

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        for model in self.models:
            model.normalize_decoder_()


class FrozenRandomDictionary(nn.Module):
    """A constrained-random control with the same width and L0 budget as an SAE."""

    def __init__(self, input_dim: int, width: int, k: int, *, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        decoder = F.normalize(torch.randn(width, input_dim, generator=generator), dim=-1)
        self.register_buffer("decoder", decoder)
        self.k = k

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del labels
        codes = topk_codes(x @ self.decoder.T, self.k)
        return codes @ self.decoder, codes


@dataclass(frozen=True)
class SyntheticSparseData:
    activations: torch.Tensor
    labels: torch.Tensor
    decoder: torch.Tensor
    latent_supports: tuple[frozenset[str], ...]
    latent_codes: torch.Tensor
    domains: tuple[str, ...]


def generate_synthetic_sparse_data(
    *,
    domains: Sequence[str] = ("smac", "grf", "pogema"),
    input_dim: int = 32,
    features_per_support: int = 2,
    samples_per_domain: int = 256,
    active_features: int = 3,
    noise_std: float = 0.02,
    correlation: float = 0.0,
    anisotropy: float = 0.0,
    superposition: float = 0.0,
    hierarchy: float = 0.0,
    imbalance: Sequence[float] | None = None,
    seed: int = 0,
) -> SyntheticSparseData:
    """Generate activations with known support over the complete domain lattice."""

    generator = torch.Generator().manual_seed(seed)
    domain_tuple = tuple(domains)
    supports = domain_lattice(domain_tuple)
    latent_supports = tuple(s for s in supports for _ in range(features_per_support))
    width = len(latent_supports)
    decoder = F.normalize(torch.randn(width, input_dim, generator=generator), dim=-1)
    if hierarchy:
        universal = F.normalize(torch.randn(input_dim, generator=generator), dim=0)
        decoder = F.normalize(
            torch.stack(
                [
                    row + hierarchy * (len(support) / len(domain_tuple)) * universal
                    for row, support in zip(decoder, latent_supports, strict=True)
                ]
            ),
            dim=-1,
        )
    if superposition:
        shared = F.normalize(torch.randn(input_dim, generator=generator), dim=0)
        decoder = F.normalize(decoder + superposition * shared, dim=-1)
    scale = torch.linspace(1.0, 1.0 + anisotropy, input_dim)
    decoder = F.normalize(decoder * scale, dim=-1)
    counts = [samples_per_domain] * len(domain_tuple)
    if imbalance is not None:
        if len(imbalance) != len(domain_tuple):
            raise ValueError("imbalance must have one multiplier per domain")
        counts = [max(1, round(samples_per_domain * value)) for value in imbalance]

    all_x, all_labels, all_codes = [], [], []
    previous = torch.zeros(width)
    for label, domain in enumerate(domain_tuple):
        eligible = torch.tensor([domain in support for support in latent_supports])
        eligible_indices = eligible.nonzero(as_tuple=False).flatten()
        for _ in range(counts[label]):
            chosen = eligible_indices[torch.randperm(len(eligible_indices), generator=generator)[:active_features]]
            codes = torch.zeros(width)
            values = 0.5 + torch.rand(len(chosen), generator=generator)
            codes[chosen] = values
            if correlation:
                codes = (1 - correlation) * codes + correlation * previous
                codes *= eligible
            previous = codes
            noise = noise_std * torch.randn(input_dim, generator=generator)
            all_x.append(codes @ decoder + noise)
            all_codes.append(codes)
            all_labels.append(label)
    return SyntheticSparseData(
        activations=torch.stack(all_x),
        labels=torch.tensor(all_labels, dtype=torch.long),
        decoder=decoder,
        latent_supports=latent_supports,
        latent_codes=torch.stack(all_codes),
        domains=domain_tuple,
    )


def train_sparse_model(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 500,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(steps):
        indices = torch.randint(x.shape[0], (min(batch_size, x.shape[0]),), generator=generator, device=x.device)
        reconstruction, _codes = model(x[indices], labels[indices])
        loss = F.mse_loss(reconstruction, x[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        normalize = getattr(model, "normalize_decoder_", None)
        if normalize is not None:
            normalize()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def sparse_metrics(x: torch.Tensor, reconstruction: torch.Tensor, codes: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(reconstruction, x)
    baseline = x.var(unbiased=False).clamp_min(torch.finfo(x.dtype).eps)
    active = codes > 0
    rates = active.float().mean(dim=0)
    nonzero_rates = rates[rates > 0]
    entropy = -(
        nonzero_rates * nonzero_rates.log2() + (1 - nonzero_rates) * (1 - nonzero_rates).clamp_min(1e-8).log2()
    )
    return {
        "mse": float(mse),
        "normalized_mse": float(mse / baseline),
        "l0": float(active.sum(dim=-1).float().mean()),
        "activation_code_bits": float(entropy.sum()) if entropy.numel() else 0.0,
        "dead_feature_fraction": float((rates == 0).float().mean()),
    }


@torch.no_grad()
def infer_functional_supports(
    codes: torch.Tensor,
    labels: torch.Tensor,
    domains: Sequence[str],
    *,
    minimum_rate: float = 0.01,
) -> tuple[frozenset[str], ...]:
    supports = []
    for feature in range(codes.shape[1]):
        active_domains = []
        for label, domain in enumerate(domains):
            mask = labels == label
            rate = (codes[mask, feature] > 0).float().mean() if mask.any() else torch.tensor(0.0)
            if float(rate) >= minimum_rate:
                active_domains.append(domain)
        supports.append(frozenset(active_domains))
    return tuple(supports)


def support_macro_f1(predicted: Sequence[frozenset[str]], target: Sequence[frozenset[str]]) -> float:
    if len(predicted) != len(target):
        raise ValueError("predicted and target support lists must have equal length")
    scores = []
    for left, right in zip(predicted, target, strict=True):
        if not left and not right:
            scores.append(1.0)
            continue
        precision = len(left & right) / max(len(left), 1)
        recall = len(left & right) / max(len(right), 1)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def greedy_decoder_matches(learned: torch.Tensor, target: torch.Tensor) -> list[tuple[int, int, float]]:
    similarity = F.normalize(learned, dim=-1) @ F.normalize(target, dim=-1).T
    candidates = [
        (float(similarity[i, j].abs()), i, j) for i in range(similarity.shape[0]) for j in range(similarity.shape[1])
    ]
    matches, used_left, used_right = [], set(), set()
    for score, left, right in sorted(candidates, reverse=True):
        if left in used_left or right in used_right:
            continue
        matches.append((left, right, score))
        used_left.add(left)
        used_right.add(right)
    return matches


def grouped_split(
    groups: Sequence[str], *, seed: int = 0, fractions: Sequence[float] = (0.7, 0.15, 0.15)
) -> list[str]:
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain train/validation/test values summing to one")
    unique = sorted(set(groups), key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    train_end = round(len(unique) * fractions[0])
    validation_end = train_end + round(len(unique) * fractions[1])
    mapping = {
        group: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, group in enumerate(unique)
    }
    return [mapping[group] for group in groups]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_activation_cache(
    directory: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    if not tensors:
        raise ValueError("activation cache requires at least one tensor")
    row_counts = {int(value.shape[0]) for value in tensors.values()}
    if len(row_counts) != 1 or row_counts.pop() != len(metadata):
        raise ValueError("all tensors and metadata must have the same row count")
    required = {
        "trajectory_group",
        "sample_index",
        "activation_location",
        "token_selector",
        "checkpoint_sha256",
        "preprocessing_identity",
        "split",
    }
    for index, row in enumerate(metadata):
        missing = required - row.keys()
        if "environment" not in row and "source" not in row:
            missing.add("environment_or_source")
        if missing:
            raise ValueError(f"metadata row {index} is missing required fields: {sorted(missing)}")
    shard = directory / "shard-00000.pt"
    torch.save(dict(tensors), shard)
    metadata_path = directory / "metadata.jsonl"
    with metadata_path.open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    payload = {
        **dict(manifest),
        "format_version": 1,
        "tensor_only": True,
        "shards": [{"path": shard.name, "sha256": file_sha256(shard)}],
        "metadata": {"path": metadata_path.name, "sha256": file_sha256(metadata_path)},
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()},
        "rows": len(metadata),
    }
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_activation_cache(directory: Path) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    shard_info = manifest["shards"][0]
    shard = directory / shard_info["path"]
    if file_sha256(shard) != shard_info["sha256"]:
        raise ValueError("activation shard hash mismatch")
    tensors = torch.load(shard, map_location="cpu", weights_only=True)
    metadata_path = directory / manifest["metadata"]["path"]
    if file_sha256(metadata_path) != manifest["metadata"]["sha256"]:
        raise ValueError("activation metadata hash mismatch")
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line]
    return tensors, metadata, manifest


def git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_run_manifest(
    path: Path,
    *,
    root: Path,
    run_id: str,
    config: Mapping[str, Any],
    seed: int,
    status: str,
    artifacts: Mapping[str, str] | None = None,
    hashes: Mapping[str, str] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    environment_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config_json = json.dumps(dict(config), sort_keys=True, default=str)
    payload = {
        "format_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "config": dict(config),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "seed": seed,
        "status": status,
        "artifacts": dict(artifacts or {}),
        "hashes": dict(hashes or {}),
        "split_manifest": dict(split_manifest or {}),
        "environment_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            **dict(environment_versions or {}),
        },
        "wandb_required": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload


@torch.no_grad()
def masked_policy_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, mask: torch.Tensor) -> float:
    reference = reference_logits.masked_fill(~mask, -torch.inf)
    candidate = candidate_logits.masked_fill(~mask, -torch.inf)
    reference_log_probs = F.log_softmax(reference, dim=-1)
    candidate_log_probs = F.log_softmax(candidate, dim=-1)
    terms = F.softmax(reference, dim=-1) * (reference_log_probs - candidate_log_probs)
    return float(terms.masked_fill(~mask, 0.0).sum(dim=-1).mean())


@torch.no_grad()
def behavior_fidelity_metrics(
    reference_actor_logits: torch.Tensor,
    candidate_actor_logits: torch.Tensor,
    action_mask: torch.Tensor,
    reference_critic: torch.Tensor,
    candidate_critic: torch.Tensor,
) -> dict[str, float]:
    """Measure native masked actor fidelity and critic-output preservation."""

    reference_actor = reference_actor_logits.masked_fill(~action_mask, -torch.inf)
    candidate_actor = candidate_actor_logits.masked_fill(~action_mask, -torch.inf)
    actor_agreement = (reference_actor.argmax(-1) == candidate_actor.argmax(-1)).float().mean()
    if reference_critic.shape[-1] > 1:
        critic_kl = F.kl_div(
            F.log_softmax(candidate_critic, dim=-1),
            F.softmax(reference_critic, dim=-1),
            reduction="batchmean",
        )
    else:
        critic_kl = torch.tensor(0.0, device=reference_critic.device)
    return {
        "actor_kl": masked_policy_kl(reference_actor_logits, candidate_actor_logits, action_mask),
        "selected_action_agreement": float(actor_agreement),
        "critic_kl": float(critic_kl),
        "critic_mean_absolute_deviation": float((reference_critic - candidate_critic).abs().mean()),
    }


def paired_interval_lower(differences: Sequence[float], *, z: float = 1.96) -> float:
    """Normal-approximation lower bound for a paired mean difference."""

    values = torch.tensor(list(differences), dtype=torch.float64)
    if values.numel() < 2:
        return float("-inf")
    standard_error = values.std(unbiased=True) / math.sqrt(values.numel())
    return float(values.mean() - z * standard_error)


def evaluate_synthetic_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    f1_threshold: float = 0.80,
    reconstruction_tolerance: float = 0.05,
    stable_seed_count: int = 4,
    decoder_match_threshold: float = 0.70,
) -> dict[str, Any]:
    """Evaluate the prespecified lattice gate per assumption-holding regime."""

    by_regime: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("assumption_holding", True):
            by_regime[str(row["regime"])].append(row)
    results: dict[str, Any] = {}
    for regime, regime_rows in by_regime.items():
        by_seed = defaultdict(dict)
        for row in regime_rows:
            by_seed[int(row["seed"])][str(row["method"])] = row
        complete = [methods for methods in by_seed.values() if "domain_lattice" in methods]
        baselines = ("flat_topk", "independent_topk")
        lattice_f1 = [float(methods["domain_lattice"]["support_macro_f1"]) for methods in complete]
        interval_bounds = {
            baseline: paired_interval_lower(
                [
                    float(methods["domain_lattice"]["support_macro_f1"]) - float(methods[baseline]["support_macro_f1"])
                    for methods in complete
                    if baseline in methods
                ]
            )
            for baseline in baselines
        }
        stable = 0
        for methods in complete:
            lattice = methods["domain_lattice"]
            best_mse = min(float(method["normalized_mse"]) for method in methods.values())
            reconstruction_ok = float(lattice["normalized_mse"]) <= (1 + reconstruction_tolerance) * best_mse
            decoder_ok = float(lattice["decoder_match_cosine"]) >= decoder_match_threshold
            stable += int(float(lattice["support_macro_f1"]) >= f1_threshold and reconstruction_ok and decoder_ok)
        passed = (
            len(complete) >= stable_seed_count
            and min(lattice_f1, default=0.0) >= f1_threshold
            and all(bound > 0 for bound in interval_bounds.values())
            and stable >= stable_seed_count
        )
        results[regime] = {
            "passed": passed,
            "seed_count": len(complete),
            "minimum_lattice_support_macro_f1": min(lattice_f1, default=None),
            "paired_95pct_lower_bounds": interval_bounds,
            "stable_seeds": stable,
        }
    return {"passed": bool(results) and all(item["passed"] for item in results.values()), "regimes": results}


def replace_token_activation(
    hidden: torch.Tensor, reconstruction: torch.Tensor, token_index: int = -1
) -> torch.Tensor:
    if hidden.shape[0] != reconstruction.shape[0] or hidden.shape[-1] != reconstruction.shape[-1]:
        raise ValueError("reconstruction must match hidden batch and feature dimensions")
    output = hidden.clone()
    output[:, token_index, :] = reconstruction
    return output


def sparse_replacement_hook(
    dictionary: nn.Module,
    labels: torch.Tensor,
    *,
    token_index: int = -1,
    record: dict[str, torch.Tensor] | None = None,
):
    """Create a forward hook that replaces one sequence position with its sparse reconstruction."""

    def hook(_module, _inputs, module_output):
        hidden = module_output[0] if isinstance(module_output, tuple) else module_output
        selected = hidden[:, token_index, :]
        reconstruction, codes = dictionary(selected, labels.to(selected.device))
        if record is not None:
            record.update(
                {
                    "original": selected.detach(),
                    "reconstruction": reconstruction.detach(),
                    "codes": codes.detach(),
                }
            )
        replaced = replace_token_activation(hidden, reconstruction, token_index)
        if isinstance(module_output, tuple):
            return (replaced, *module_output[1:])
        return replaced

    return hook

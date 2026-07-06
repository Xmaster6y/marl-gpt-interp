# Cross-Environment Compute Sharing In MARL-GPT

## Status

Active.

## Question

How much of MARL-GPT's effective computation, parameters, and learned knowledge is shared across SMAC, POGEMA,
and GRF, and how much is environment-specific?

## Motivation

The model was trained across three environments with different observation schemas, action semantics, objectives, and
entity structures. It is not scientifically interesting by itself that a probe can classify the source environment. The
useful question is whether the trained model reuses common multi-agent computation across environments or whether each
environment is handled by mostly separate effective circuits.

This matters for generalization. If MARL-GPT learns shared abstractions such as local crowding, action availability,
progress, value, or multi-agent interaction load, then GRF coordination probes may connect to broader cross-environment
capabilities. If the model mostly uses environment-specific computation, then football-specific probes should be
interpreted as local to GRF and should not be treated as evidence of general multi-agent knowledge.

## Scope

The primary condition is natural trained-model inference:

- correct environment token;
- normal observation stream;
- normal action mask;
- normal positional/index channels;
- no wrong-token or counterfactual environment-token condition.

Wrong-token experiments remain useful as diagnostics for whether environment identity can be recovered without the
explicit token, but they are not the main evidence for compute sharing or cross-environment capability.

## Levels Of Evidence

### 1. Input-Format Controls

Some environment separation is trivial because the raw inputs differ. Controls should measure how much environment
information is available from:

- observation values only;
- action masks only;
- positional/index channels only;
- final environment token only;
- the full model input before learned transformer computation.

These controls define the nuisance baseline. They should not be mistaken for learned shared knowledge.

### 2. Representation Sharing

Measure whether hidden states occupy shared, partially shared, or environment-separated subspaces.

Candidate analyses:

- layerwise CKA, SVCCA, RSA, or principal-angle similarity between environment activation sets;
- class-mean and pairwise mean-difference directions by layer, branch, and token group;
- PCA or low-rank subspace overlap across environments;
- probe directions used only as diagnostic axes, not as the headline result.

Primary question:

Do SMAC, POGEMA, and GRF share activation subspaces after controlling for trivial input-format differences?

### 3. Effective Computation Sharing

Measure whether the same model components matter for decisions across environments.

Candidate analyses:

- per-environment gradient vectors under the model's own action/critic loss;
- pairwise gradient cosine by layer, embedding group, actor head, critic head, attention, and MLP group;
- overlap of top-k gradient, attribution, or ablation-sensitive components;
- activation attribution to token groups such as observation tokens, final token, action-mask-related tokens, and
  positional channels;
- targeted ablations or activation patching for components ranked as shared or environment-specific.

Primary question:

Do the same layers, heads, MLP channels, embeddings, and output heads support behavior across environments, or do
different environments rely on distinct effective computation?

### 4. Cross-Environment Knowledge Transfer

The strongest evidence would show that abstract directions or readouts learned in one environment transfer to another.

Candidate abstract variables:

- action availability or action-mask entropy;
- value or return bin;
- local entity density or crowding;
- number or fraction of active/visible agents;
- proximity or progress toward an objective, where an environment-specific proxy exists;
- interaction load, such as many nearby entities or contested states;
- time/history position;
- controllability or role-like proxies.

Experiment sketch:

1. Define an abstract variable in each environment with a comparable interpretation.
2. Train a linear readout or direction in one environment, or in two environments.
3. Evaluate on held-out examples from the same environment and from the other environments.
4. Compare cross-environment transfer to same-environment held-out performance.
5. Test whether the transferring direction uses the same layers or parameter groups as the environment-specific
   direction.

Primary question:

Does MARL-GPT represent reusable multi-agent abstractions, or only environment-local patterns?

## Metrics

- Pairwise activation-subspace similarity by layer and token group.
- Pairwise gradient cosine by parameter group.
- Top-k overlap of important heads, MLP channels, embeddings, or token groups.
- Env-pair distance summaries for activations and gradients.
- Cross-environment concept-transfer accuracy, AUROC, balanced accuracy, or correlation.
- Same-env versus cross-env transfer gap for each abstract concept.
- Behavioral degradation from ablating shared-ranked versus environment-specific-ranked components.
- Residual GRF tactical-probe performance after controlling for environment identity or projecting out environment
  directions.

## Expected Outcomes

### Mostly Shared Compute

Evidence:

- high activation-subspace similarity across environments;
- high gradient cosine and high top-k attribution overlap;
- abstract concept directions transfer across environments;
- ablating shared-ranked components hurts multiple environments.

Interpretation:

The model likely reuses general multi-agent computation across environments.

### Mostly Environment-Specific Compute

Evidence:

- low activation-subspace overlap;
- low gradient cosine across most parameter groups;
- environment-specific top-k components;
- poor cross-environment concept transfer;
- ablating environment-ranked components selectively hurts that environment.

Interpretation:

The model may solve each environment through mostly separate effective computation, limiting claims about general
multi-agent knowledge.

### Mixed Shared Abstractions

Evidence:

- raw environment identity is strongly separable;
- some abstract concept directions transfer across two or three environments;
- some parameter groups are shared while others are environment-specific.

Interpretation:

This is likely the most interesting result. The model may have environment-specific surface handling but shared
abstractions for spatial, value, or interaction structure.

## Reviewer Objection

A reviewer may argue that cross-environment differences only reflect schema, action-space, or dataset-format
differences. A strong result must therefore go beyond environment classification. It should show whether abstract
concepts transfer, whether the same effective computation supports behavior across environments, and whether causal
ablations distinguish shared from environment-specific components.

## Decision Rule

If activation subspaces, parameter gradients, attribution overlap, and abstract concept transfer all point to shared
structure, use cross-environment compute sharing as a central interpretability claim. If only environment identity is
decodable but abstract concepts do not transfer and important components are disjoint, treat environment identity as a
nuisance control and keep claims environment-local. If shared abstractions appear for only some concepts or environment
pairs, focus the paper around partial generalization and clearly state which capabilities are shared.

## Links

- [Environment representations in MARL-GPT](2026-07-06-environment-representations-in-marl-gpt.md)
- [Environment mechanism probes](../experiments/2026-07-06-environment-mechanism-probes.md)
- [Coordination representations in MARL-GPT](2026-06-30-coordination-representations-in-marl-gpt.md)

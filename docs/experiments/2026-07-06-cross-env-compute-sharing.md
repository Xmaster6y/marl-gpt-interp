# Cross-Environment Compute Sharing

## Status

Completed JZ small run. Job `1387279` completed successfully on 2026-07-06 with exit code `0:0`.

## Question

How much of MARL-GPT's effective computation is shared across SMAC, POGEMA, and GRF under natural trained-model
inference?

## Hypothesis

MARL-GPT may contain a mixture of shared and environment-specific computation. Low-level input handling may be
environment-specific because observation schemas and action semantics differ, while some intermediate or late
representations may share abstract multi-agent structure such as action availability, local density, value, progress,
or interaction load.

## Primary Condition

Use only natural inference:

- correct environment token;
- normal observation stream;
- normal action mask;
- normal positional/index channels;
- no wrong-token condition;
- no environment-token sweep.

The earlier wrong-token runs are diagnostic context only. They should not be used as the main evidence for
cross-environment generalization.

## Data

Start with the same small balanced JZ subset used by the environment-mechanism probes:

- SMAC: `dataset/zerg_5_vs_5`;
- POGEMA: `dataset/dataset_pogema_ll/random`;
- GRF: `dataset/dataset_grf/trajectories/academy_corner`.

Scale only after the cache schema, layer grouping, and summary metrics are stable.

## Implementation

- Entrypoint: [`../../scripts/cross_env_compute_sharing.py`](../../scripts/cross_env_compute_sharing.py)
- Shared helpers: [`../../src/marl_gpt_interp/marl_gpt_tools.py`](../../src/marl_gpt_interp/marl_gpt_tools.py)
- Config: [2026-07-06-jz-small.yaml][cross-env-config]
- Launch artifact:
  [`archived/2026-07-06-cross-env-compute-sharing-v100.sh`](archived/2026-07-06-cross-env-compute-sharing-v100.sh)
- Results: [`../../results/experiments/2026-07-06-cross-env-compute-sharing/`](../../results/experiments/2026-07-06-cross-env-compute-sharing/)
- Slurm logs: `../../results/slurm/cross-env-share-1387279.out` and
  `../../results/slurm/cross-env-share-1387279.err`

## Measures

### Representation Geometry

For each layer, branch, and token group:

- class means per environment;
- pairwise mean-difference L2 and cosine;
- CKA or centered linear kernel similarity across environment activation matrices;
- principal-angle or PCA subspace overlap;
- RSA-style distance correlation if useful.

Outputs:

- `activation_geometry.csv`;
- `activation_subspace_similarity.csv`;
- `env_distance_summary.json`.

### Effective Computation

For each parameter group:

- token embeddings;
- positional embeddings;
- transformer layer groups;
- attention and MLP subgroups if stable to expose;
- actor head;
- critic head.

Measure:

- per-environment gradient norm under the model's own action/critic loss;
- pairwise gradient cosine;
- top-k gradient-coordinate overlap;
- loss-normalized gradient norm to reduce scale artifacts.

Outputs:

- `parameter_gradients.csv`;
- `parameter_gradient_overlap.csv`;
- `parameter_group_summary.json`.

### Abstract Knowledge Transfer

Define abstract variables with comparable semantics across environments where possible:

- action availability or action-mask entropy;
- value or return bin;
- local entity density or crowding;
- number or fraction of active or visible agents;
- time/history position;
- interaction load;
- objective progress or proximity, when a defensible proxy exists.

For each variable:

1. train a simple readout within one environment;
2. evaluate on held-out examples from the same environment;
3. evaluate on other environments when the label is comparable;
4. train on two environments and evaluate on the third;
5. compare transfer to same-env performance and to input-channel controls.

Outputs:

- `concept_transfer.csv`;
- `concept_direction_overlap.csv`;
- `concept_transfer_summary.json`.

## Baselines And Controls

- Full raw input channel probe.
- Observation-only probe.
- Action-mask-only probe.
- Position/index-only probe.
- Final-token-only probe.
- Same-env held-out concept transfer.
- Random-label or permuted-label concept controls.
- Environment-label-controlled concept probes when testing whether GRF tactical concepts survive environment confound
  control.

## Metrics

- Activation CKA/SVCCA/principal-angle similarity by env pair and layer.
- Pairwise gradient cosine by env pair and parameter group.
- Top-k overlap of gradient, attribution, or ablation-ranked components.
- Same-env versus cross-env concept-transfer gap.
- Shared compute score: normalized overlap of important components across all three environments.
- Partially shared score: normalized overlap for two-env pairs.
- Environment-specific score: importance concentrated in one environment and low elsewhere.

## Decision Rule

If activation subspaces, parameter gradients, and concept directions transfer across environments, treat this as
evidence for shared multi-agent computation. If environment identity is separable but concept transfer fails and
important components do not overlap, treat the model as mostly environment-specific. If only some concepts or
environment pairs transfer, focus on partial generalization and name which abstractions appear shared.

## Result

The small natural-inference run gives evidence for pairwise POGEMA-GRF effective-computation overlap, but not for
broad compute sharing across all three environments.

The run used 480 activation/input examples from eight natural batches. The Slurm job completed in 5m31s. It wrote the
expected activation geometry, activation CKA, parameter gradient, gradient overlap, input-probe, dataset inspection, and
natural behavior artifacts.

Environment identity is an easy confound in this setup. Linear probes classify the environment with perfect held-out
accuracy from observation features, action masks, positions, full input, and even the two-dimensional final-token input
summary. This means later cross-environment claims cannot rest on environment separability or raw input controls.

Activation subspace similarity is low in all environment pairs. Mean linear CKA is 0.0247 for SMAC-POGEMA, 0.0158 for
SMAC-GRF, and 0.0368 for POGEMA-GRF. The largest CKA value is only 0.0910, for `pogema_vs_grf` at `layer_01:mean`.
Mean activation differences are also smallest for POGEMA-GRF: mean difference L2 is 15.69 for POGEMA-GRF, 30.82 for
SMAC-POGEMA, and 34.83 for SMAC-GRF.

Gradient alignment shows the strongest positive result. Mean gradient cosine is 0.7792 for POGEMA-GRF, but only 0.0692
for SMAC-POGEMA and 0.0647 for SMAC-GRF. The POGEMA-GRF alignment is high across transformer layers: 0.7159 at
`layer_00`, 0.9295 at `layer_01`, 0.9603 at `layer_02`, and 0.8262-0.8686 across `layer_03` through `layer_06`.
Token embeddings are less aligned but still positive for POGEMA-GRF at 0.4148. SMAC gradients are near orthogonal or
slightly negative against both other environments in most transformer layers.

The current run does not yet answer the abstract knowledge transfer part of the plan. No `concept_transfer.csv` or
concept direction outputs were produced, so the strongest publishable form of the claim is narrower than the original
question.

Conclusion: treat SMAC as environment-specific relative to the POGEMA-GRF pair in this small run. Treat POGEMA-GRF
gradient alignment as a promising partial-sharing signal that needs concept-transfer validation and robustness checks
before it can support a claim about shared multi-agent abstractions.

## Expected Reviewer Objection

The strongest objection is that SMAC, POGEMA, and GRF are too different, so any environment separation may only reflect
dataset format. The experiment must therefore emphasize sharing and transfer of abstract variables, not environment
classification.

## Links

- [Cross-environment compute sharing in MARL-GPT](../questions/2026-07-06-cross-env-compute-sharing.md)
- [Environment mechanism probes](2026-07-06-environment-mechanism-probes.md)

[cross-env-config]: ../../configs/cross_env_compute_sharing/2026-07-06-jz-small.yaml

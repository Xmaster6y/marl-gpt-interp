# Internal Representation Geometry

## Status

Planned JZ small run.

## Question

Are SMAC, POGEMA, and GRF internally coherent in MARL-GPT representation space, and do cross-environment relationships
look different under asymmetric analyses than under symmetric CKA?

## Motivation

The cross-environment compute-sharing run found low activation CKA for all environment pairs but high POGEMA-GRF
gradient alignment. Low CKA is ambiguous without internal geometry: environments might occupy tight but separate
manifolds, or each environment might already be internally diffuse. The next experiment measures within-environment
compactness, between-environment separation normalized by internal spread, and directed subspace containment.

## Hypothesis

SMAC will be more isolated than POGEMA and GRF. POGEMA and GRF may still have low symmetric CKA while showing stronger
directed subspace containment or lower normalized separation than pairs involving SMAC. This would support the
interpretation that POGEMA and GRF use similar effective computation over partially different coordinates.

## Data

Use the same small balanced JZ subset as the completed cross-environment compute-sharing run:

- SMAC: `dataset/zerg_5_vs_5`;
- POGEMA: `dataset/dataset_pogema_ll/random`;
- GRF: `dataset/dataset_grf/trajectories/academy_corner`.

Use natural trained-model inference only: correct environment token, normal observation stream, normal action mask, and
normal positional/index channels.

## Implementation

- Entrypoint: [`../../scripts/internal_representation_geometry.py`](../../scripts/internal_representation_geometry.py)
- Shared helpers: [`../../src/marl_gpt_interp/marl_gpt_tools.py`](../../src/marl_gpt_interp/marl_gpt_tools.py)
- Config: [2026-07-06-jz-small.yaml][config]
- Launch artifact:
  [`to-launch/2026-07-06-internal-representation-geometry-v100.sh`](to-launch/2026-07-06-internal-representation-geometry-v100.sh)

## Measures

### Within-Environment Proximity

For each environment, layer, and pooled token group:

- centroid norm;
- mean, standard deviation, and maximum L2 distance to the environment centroid;
- mean and standard deviation of pairwise L2 distance;
- mean and standard deviation of pairwise cosine distance;
- mean cosine to centroid;
- PCA variance concentration, components required for 90% variance, participation ratio, and effective rank.

Output: `internal_representation_proximity.csv`.

### Normalized Between-Environment Separation

For each environment pair, layer, and pooled token group:

- centroid L2 distance;
- centroid distance normalized by pooled within-environment spread;
- Fisher-style between/within ratio;
- mean cross-environment L2 distance;
- energy distance;
- silhouette score using L2 distances;
- fraction of examples whose nearest neighbor is from the same environment.

Output: `representation_separation.csv`.

### Asymmetric Representation Analysis

For each directed source-target environment pair:

1. fit a PCA basis on centered source-environment activations;
2. project centered target-environment activations onto the source basis;
3. record the fraction of target variance explained at ranks 1, 2, 4, 8, 16, 32, 64, and 128;
4. compare `A -> B` against `B -> A`.

Output: `asymmetric_representation_analysis.csv`.

This implements a simple asymmetric subspace-containment baseline. If `tdhook` is available later, this experiment is
the right place to add its asymmetric representation method and compare it against the PCA-containment baseline.

### Symmetric Comparison

Also write the existing linear CKA table for direct comparison with the previous run.

Output: `activation_subspace_similarity.csv`.

## Decision Rule

If within-environment spread is low and between-environment separation is high, treat low CKA as evidence of
environment-specific representation geometry. If all environments are internally diffuse, downgrade low CKA as a weak
separation signal. If POGEMA-GRF has stronger directed containment or lower normalized separation than SMAC pairs,
interpret the earlier POGEMA-GRF gradient alignment as partial sharing despite coordinate mismatch. If asymmetric scores
are directionally imbalanced, identify which environment's representation basis better contains the other.

## Expected Reviewer Objection

A reviewer may argue that this still measures dataset format rather than abstract multi-agent knowledge. This run is
therefore not sufficient for a concept-sharing claim. Its purpose is to decide whether representation geometry is
stable enough to support later concept-transfer and intervention experiments.

## Links

- [Cross-environment compute sharing](2026-07-06-cross-env-compute-sharing.md)
- [Cross-environment compute sharing in MARL-GPT](../questions/2026-07-06-cross-env-compute-sharing.md)

[config]: ../../configs/internal_representation_geometry/2026-07-06-jz-small.yaml

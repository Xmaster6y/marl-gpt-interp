# Internal Representation Geometry

## Status

Completed JZ small run. Job `1398530` completed successfully on 2026-07-06 with exit code `0:0`.
Rerun job `1462029` completed successfully on 2026-07-07 with exit code `0:0` after installing `tdhook`
in the JZ environment.

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
  [`archived/2026-07-06-internal-representation-geometry-v100.sh`](archived/2026-07-06-internal-representation-geometry-v100.sh)
- Results: [`../../results/experiments/2026-07-06-internal-representation-geometry/`](../../results/experiments/2026-07-06-internal-representation-geometry/)
- Slurm logs: `../../results/slurm/repr-geometry-1398530.out`,
  `../../results/slurm/repr-geometry-1398530.err`,
  `../../results/slurm/repr-geometry-1462029.out`, and
  `../../results/slurm/repr-geometry-1462029.err`

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

Future reruns should also write `self_subspace_similarity.csv`, a split-half within-environment CKA table. This is the
baseline for deciding whether low cross-environment CKA is low relative to the model's own within-environment
activation reliability.

Implementation update: the local script now writes `self_subspace_similarity.csv` using an even/odd split within each
environment. The completed JZ run documented below predates that output, so any paper claim should wait for a rerun
before comparing cross-environment CKA against within-environment self-CKA.

The improved homogeneous readout should report an environment-by-environment CKA matrix where diagonal entries are
within-environment self-CKA and off-diagonal entries are cross-environment CKA. This avoids mixing CKA against
distance-based compactness as the main baseline.

Local rerun update, 2026-07-07: the CPU rerun completed and wrote `self_subspace_similarity.csv` with 60 rows. Across
layer and actor/critic final-state features, mean even/odd self-CKA is `0.0360` for SMAC, `0.0860` for POGEMA, and
`0.0354` for GRF. Comparable cross-environment CKA means are `0.0260` for SMAC-POGEMA, `0.0175` for SMAC-GRF, and
`0.0241` for POGEMA-GRF. This makes the CKA evidence more homogeneous, but it also weakens any overly broad statement
that all cross-environment CKA is far below internal reliability: POGEMA has the clearest self-vs-cross gap, while
SMAC and GRF self-CKA are only modestly above or near cross-CKA.

## Decision Rule

If within-environment spread is low and between-environment separation is high, treat low CKA as evidence of
environment-specific representation geometry. If all environments are internally diffuse, downgrade low CKA as a weak
separation signal. If POGEMA-GRF has stronger directed containment or lower normalized separation than SMAC pairs,
interpret the earlier POGEMA-GRF gradient alignment as partial sharing despite coordinate mismatch. If asymmetric scores
are directionally imbalanced, identify which environment's representation basis better contains the other.

## Result

The rerun used 480 natural activation examples from eight batches and wrote the expected proximity, separation,
asymmetric containment, CKA, natural-behavior, dataset-inspection, and summary artifacts. `tdhook` is now available in
the JZ environment and recorded as `available: true` in `summary.json`. The current script still records `used: false`
because it uses the built-in PCA subspace-containment baseline rather than a `tdhook` method.

The strongest result is that environments are internally coherent enough for low CKA to be meaningful. Every
environment pair has same-environment nearest-neighbor fraction `1.0` across all analyzed features. Median
within-environment pairwise L2 is smallest for GRF (`1.54`), larger for POGEMA (`2.37`), and largest for SMAC (`3.22`).
Median pairwise cosine distance follows the same order: GRF `0.0004`, POGEMA `0.0041`, SMAC `0.0754`. SMAC therefore
appears more internally diffuse, especially in final-token transformer states, but not so diffuse that cross-env
separation becomes uninterpretable.

CKA remains low, consistent with the earlier compute-sharing run. Mean linear CKA is `0.0231` for SMAC-POGEMA,
`0.0162` for SMAC-GRF, and `0.0217` for POGEMA-GRF, with no pair exceeding `0.0512`.

Normalized separation is high for all pairs, but it depends strongly on token pooling. After excluding degenerate
near-zero-spread rows, median normalized centroid L2 is `5.38` for SMAC-POGEMA, `13.40` for SMAC-GRF, and `6.35` for
POGEMA-GRF. Median silhouette scores are `0.798`, `0.905`, and `0.798`, respectively. This means all environments are
well separated, and SMAC-GRF is the most separated pair overall. POGEMA-GRF is not uniformly closest: mean-pooled
transformer states often separate POGEMA-GRF more than SMAC-POGEMA, while final-token and actor/critic branch states
make POGEMA-GRF the closest pair.

Asymmetric PCA containment gives a partial-sharing signal but not a clean POGEMA-GRF-only story. At rank 16, median
target-variance explained is highest for `pogema_to_grf` (`0.5865`) and `grf_to_smac` (`0.5787`), followed by
`pogema_to_smac` (`0.5451`), `smac_to_grf` (`0.3575`), `grf_to_pogema` (`0.3237`), and `smac_to_pogema` (`0.2685`).
The strongest interpretable direction is that POGEMA's low-rank basis contains GRF better than GRF contains POGEMA.
SMAC's basis is weakest at containing POGEMA.

Conclusion: the low CKA result should not be dismissed as all environments being internally scattered. The
representations are environment-separated with perfect nearest-neighbor identity, and SMAC is the most internally
diffuse. The previous POGEMA-GRF gradient alignment is compatible with a coordinate-mismatch story: final/branch
representations and POGEMA-to-GRF low-rank containment show some closeness, but mean-pooled hidden states remain clearly
separated. The next useful step is concept-level transfer or targeted representation patching, because geometry alone
still cannot distinguish shared abstract knowledge from dataset-format structure.

## Expected Reviewer Objection

A reviewer may argue that this still measures dataset format rather than abstract multi-agent knowledge. This run is
therefore not sufficient for a concept-sharing claim. Its purpose is to decide whether representation geometry is
stable enough to support later concept-transfer and intervention experiments.

## Links

- [Cross-environment compute sharing](2026-07-06-cross-env-compute-sharing.md)
- [Cross-environment compute sharing in MARL-GPT](../questions/2026-07-06-cross-env-compute-sharing.md)

[config]: ../../configs/internal_representation_geometry/2026-07-06-jz-small.yaml

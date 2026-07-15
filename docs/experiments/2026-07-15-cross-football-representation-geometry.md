# Cross-Football Representation Geometry

## Status

Completed locally on 2026-07-15, but its contiguous-frame CKA interpretation is superseded by the
[random frame sampling control](2026-07-15-random-frame-sampling-control.md). The original analysis compares latent
representations only for La Liga, raw RoboCup STP, and native GRF states encoded through the same MARL-GPT GRF input
path.

## Question

After exact `simple115v2` encoding, does the pretrained MARL-GPT GRF branch organize La Liga and RoboCup tracking
states with geometry resembling native GRF, and are the source distributions nevertheless distinguishable?

## Hypothesis

The three football sources may share some transformer-layer geometry because their inputs use the same soccer schema,
but source-specific dynamics, sampling, and imputation should leave separable latent distributions. A bounded contiguous
sample can validate the comparison workflow but cannot establish human-football transfer.

## Data Or Command

```bash
uv run -m scripts.cross_football_representation_geometry --config-name 2026-07-15-local-small
```

The run uses one La Liga sequence, one contiguous raw STP interval, and a native `11vs11a11` GRF rollout. It captures
mean-pooled and final-token representations at the embedding, seven transformer blocks, actor layer, and critic layer.
The primary analysis averages all player perspectives within each frame, giving 12 balanced frame-level observations
per source. A supplementary balanced analysis retains 132 individual perspectives per source. Per-source latent caches
make interrupted local runs resumable.

Outputs are untracked under
`results/experiments/2026-07-15-cross-football-representation-geometry/`. Their table names and columns match the
existing cross-environment representation analyses:

- `activation_geometry`
- `internal_representation_proximity`
- `representation_separation`
- `activation_centroid_cosine_similarity`
- `activation_pairwise_cosine_similarity`
- `activation_subspace_similarity`
- `self_subspace_similarity`
- `asymmetric_representation_analysis`

## Metrics And Decision Rule

The primary evidence is transformer-block linear CKA, normalized centroid distance, silhouette score, and same-source
nearest-neighbor fraction on frame means. Centroid and pairwise cosine similarity, PCA/effective-rank summaries,
self-CKA, and directional PCA containment are retained for schema parity and diagnostics.

Evidence for shared latent geometry requires consistently non-trivial cross-source CKA beyond the shared embedding
token. Evidence for distributional alignment would additionally require weak source separability. Neither condition is
treated as transfer evidence without broader held-out matches and non-contiguous sampling.

## Result

All 102 stored latent tensors were finite. The workflow collected 264 La Liga and 264 RoboCup player perspectives and
132 native-GRF perspectives before constructing balanced analysis units.

Across the seven transformer blocks, frame-level CKA was high:

| Pair | Mean-pooled CKA | Final-token CKA |
| --- | ---: | ---: |
| La Liga vs GRF | 0.824 (0.763-0.963) | 0.869 (0.792-0.963) |
| RoboCup vs GRF | 0.859 (0.757-0.990) | 0.948 (0.922-0.973) |
| La Liga vs RoboCup | 0.731 (0.595-0.955) | 0.844 (0.694-0.934) |

This geometric similarity did not imply overlapping distributions. On mean-pooled transformer representations, every
pair had a same-source nearest-neighbor fraction of `1.0`. Mean silhouette scores were `0.766` for La Liga-GRF,
`0.777` for RoboCup-GRF, and `0.836` for La Liga-RoboCup; corresponding mean normalized centroid distances were
`4.967`, `5.806`, and `8.063`.

The player-perspective analysis gives a materially different answer because the 22 views derived from a frame are not
independent. Transformer CKA was very high between the two external encodings (mean-pooled `0.951`, final-token
`0.979`) but low against native GRF (La Liga `0.111`/`0.045`; RoboCup `0.172`/`0.133`). It is therefore supplementary,
not the primary unit of inference.

Raw centroid and pairwise cosine similarities were near one even when nearest-neighbor separation was perfect. The
latents are strongly anisotropic and the frame samples have low effective rank, so uncentered cosine similarity is not
a useful standalone alignment measure here. Embedding CKA is exactly zero because the shared tokenized input produces
a degenerate centered embedding comparison; it is excluded from the transformer summaries above.

## Conclusion

The latent-comparison infrastructure now has the same representation-table schema as the prior SMAC/POGEMA/GRF work.
The initial claim that MARL-GPT preserves similar centered variation across football sources is not retained: later
random, temporally spaced sampling showed that the high CKA depended strongly on contiguous index ordering. Strong
source identity in absolute latent space remains, while near-unit raw cosine is an anisotropy diagnostic rather than
alignment evidence.

The strongest reviewer objection is sampling: each external source comes from one short contiguous sequence, native GRF
has only 12 frames, CKA examples are index-aligned rather than event-matched, and the frame means discard player-level
structure. The next claim-bearing run should use multiple held-out matches, temporally separated frames, repeated GRF
seeds, uncertainty intervals, and event- or phase-matched comparisons. Action logits, critic values, input probes,
parameter gradients, fine-tuning, and action mapping are deliberately out of scope for this analysis.

## Links

- [Random frame sampling control](2026-07-15-random-frame-sampling-control.md)
- [Final-token-excluded pooling control](2026-07-15-final-token-excluded-pooling-control.md)
- [External soccer GRF encoding](2026-07-15-external-soccer-grf-encoding.md)
- [Cross-environment compute sharing](2026-07-06-cross-env-compute-sharing.md)
- [Internal representation geometry](2026-07-06-internal-representation-geometry.md)
- [Simulation-human modelling gap](../questions/2026-06-30-simulation-human-modelling-gap.md)

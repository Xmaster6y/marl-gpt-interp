# Environment-Separated Representations With Partial POGEMA-GRF Sharing

## Status

Tentative.

## Claim

MARL-GPT's natural-inference activations are strongly environment-separated across SMAC, POGEMA, and GRF, but its
effective computation shows a partial POGEMA-GRF sharing signal. This is not yet evidence of shared abstract
multi-agent knowledge; it is evidence that similar parameter directions can be used despite low representational CKA
and environment-specific activation geometry.

## Presentation Figures

Editable SVGs for presenting this claim are in
[`../../results/figures/2026-07-06-environment-separated-partial-sharing/`](../../results/figures/2026-07-06-environment-separated-partial-sharing/).

- [`methodology-schema.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/methodology-schema.svg): high-level
  pipeline from datasets to activation capture, representation analyses, gradient analyses, and claims.
- [`env-identity-corruption-schema.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/env-identity-corruption-schema.svg):
  counterfactual-token methodology: keep the true observation stream and corrupt or sweep the final environment token.
- [`representation-analysis-schema.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/representation-analysis-schema.svg):
  definitions of cross-env CKA, self-CKA, normalized separation, asymmetric containment, and the plotted aggregation.
- [`gradient-cosine-heatmap.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/gradient-cosine-heatmap.svg): main
  effective-computation result; POGEMA-GRF gradients align while SMAC pairs are near orthogonal.
- [`cka-layer-mean.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/cka-layer-mean.svg) and
  [`cka-layer-final.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/cka-layer-final.svg): low symmetric CKA across
  transformer layers.
- [`internal-compactness.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/internal-compactness.svg): within-env
  compactness; SMAC is most internally diffuse, GRF most compact.
- [`normalized-separation.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/normalized-separation.svg): all env
  pairs are separated after normalizing by within-env spread.
- [`final-branch-separation.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/final-branch-separation.svg): POGEMA-GRF
  is closest in final-token and actor/critic branch states.
- [`asymmetric-containment-r16.svg`](../../results/figures/2026-07-06-environment-separated-partial-sharing/asymmetric-containment-r16.svg):
  POGEMA's rank-16 basis contains GRF better than GRF contains POGEMA.

## Evidence

The cross-environment compute-sharing run shows low activation CKA but high POGEMA-GRF gradient alignment. Mean
gradient cosine is `0.7792` for POGEMA-GRF, compared with `0.0692` for SMAC-POGEMA and `0.0647` for SMAC-GRF. Across
transformer layers, POGEMA-GRF gradient cosine ranges from `0.7159` to `0.9603`, while SMAC pairs are near zero or
slightly negative in many groups.

The internal representation geometry run shows that low CKA should not be dismissed as pure within-env scatter. Same-env
nearest-neighbor fraction is `1.0` for every analyzed feature and environment pair. Median within-env pairwise cosine
distance is `0.0004` for GRF, `0.0041` for POGEMA, and `0.0754` for SMAC. SMAC is more diffuse, but the environments
remain clearly separable.

Symmetric cross-env CKA remains low in the internal geometry run: `0.0231` for SMAC-POGEMA, `0.0162` for SMAC-GRF, and
`0.0217` for POGEMA-GRF. The layer-curve figures plot exact centered linear CKA values for `layer_k:mean` or
`layer_k:final`; the scalar values above are means across all feature rows. This suggests that POGEMA-GRF gradient
sharing is not explained by globally aligned activation subspaces.

Asymmetric containment gives a weak but useful bridge between the two results. At rank 16, POGEMA's PCA basis explains
median `0.5865` of GRF variance, while GRF's basis explains median `0.3237` of POGEMA variance. Final-token and
actor/critic branch representations also make POGEMA-GRF the closest pair by normalized centroid distance.

## Limitations

- The evidence is from a small JZ subset: eight batches and 480 activation examples.
- Environment identity remains a major confound because inputs, action spaces, and dataset formats differ.
- `tdhook` was not available on JZ, so asymmetric analysis used a simple PCA-containment baseline.
- Self-CKA was not produced by the retrieved run. It is now implemented as split-half within-env CKA for the next
  internal-geometry rerun, and should be used as the reliability baseline for cross-env CKA.
- The experiments do not yet show transfer of abstract concepts such as action availability, crowding, value, progress,
  or interaction load.
- The experiments do not include causal interventions or activation patching, so the claim should stay diagnostic.

## Next Evidence Needed

The next result should test whether a direction or readout for an abstract variable transfers across environments,
especially between POGEMA and GRF. If concept directions transfer and targeted patching changes behavior in the
predicted direction, the claim can be strengthened from partial effective-computation sharing to shared multi-agent
abstraction.

## Links

- [Cross-environment compute sharing](../experiments/2026-07-06-cross-env-compute-sharing.md)
- [Internal representation geometry](../experiments/2026-07-06-internal-representation-geometry.md)
- [Cross-environment compute sharing in MARL-GPT](../questions/2026-07-06-cross-env-compute-sharing.md)

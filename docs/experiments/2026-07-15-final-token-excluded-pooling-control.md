# Final-Token-Excluded Pooling Control

## Status

Completed locally on 2026-07-15 for both the SMAC/POGEMA/GRF and La Liga/RoboCup/GRF representation workflows.

## Question

Are the high layerwise pairwise cosine similarities an artifact of including MARL-GPT's final environment-token
position in mean-pooled activation states?

## Hypothesis

Removing the final position from mean pooling should reduce direct environment-token contamination. The effect may be
small because the final position is only one of 700 tokens and MARL-GPT uses non-causal self-attention, allowing every
retained token to attend to the environment token before pooling.

## Implementation And Commands

`pooled_activations` now accepts `exclude_final_token_from_mean=True`. The `:mean` feature then averages positions
`0:699`, while `:final` remains available as a separate diagnostic. Run summaries record the pooling policy and that
attention is non-causal.

```bash
uv run -m scripts.internal_representation_geometry \
  --config-name 2026-07-15-local-no-env-mean

uv run -m scripts.cross_football_representation_geometry \
  --config-name 2026-07-15-local-no-env-mean
```

Outputs are untracked under:

- `results/experiments/2026-07-15-internal-representation-geometry-no-env-mean/`
- `results/experiments/2026-07-15-cross-football-representation-geometry-no-env-mean/`

The cross-environment rerun retained the original 480-example, eight-batch condition. The primary football comparison
retained 12 balanced frame means per source. Both write pairwise cosine for the embedding, every transformer block,
the actor layer, and the critic layer.

## Attention Audit

The checkpoint model uses `NonCausalSelfAttention`. Its flash-attention path calls
`scaled_dot_product_attention(..., attn_mask=None, is_causal=False)`, and the manual path's triangular-mask operation is
commented out. Therefore, excluding the last state from pooling does not remove environment-token influence from
post-attention states. It only removes that state's direct weight in the average.

## Result

Mean cross-source pairwise cosine over transformer blocks changed as follows:

| Pair | Original mean | Excluding final position | Change |
| --- | ---: | ---: | ---: |
| POGEMA vs GRF | 0.5516 | 0.5427 | -0.0089 |
| SMAC vs GRF | 0.8722 | 0.8725 | +0.0003 |
| SMAC vs POGEMA | 0.6861 | 0.6777 | -0.0084 |
| La Liga vs GRF | 0.996847 | 0.996842 | -0.000006 |
| RoboCup vs GRF | 0.996482 | 0.996480 | -0.000002 |
| La Liga vs RoboCup | 0.998060 | 0.998057 | -0.000003 |

The largest cross-environment reductions occur from layers 2 through 4: approximately `-0.015` for POGEMA-GRF and
`-0.014` for SMAC-POGEMA. SMAC-GRF is effectively unchanged at every layer. All cross-football layerwise changes are
smaller than `0.00006`, and layers 2 through 6 change by only about `0.000002`.

As a sanity check, every final-token cosine value is exactly unchanged by the pooling option. All generated numeric
tables are finite.

## Figures

The config-driven Matplotlib workflow
[`scripts.make_cross_domain_representation_figures`](../../scripts/make_cross_domain_representation_figures.py) reads
the result CSVs and writes PNG, PDF, and SVG versions of:

- layerwise cross-environment and cross-football pairwise cosine;
- cross-environment CKA and within-environment self-CKA;
- cross-football CKA and within-source self-CKA.

The dated config is
[`2026-07-15-no-env-mean.yaml`](../../configs/make_cross_domain_representation_figures/2026-07-15-no-env-mean.yaml),
and generated files are untracked under `results/figures/2026-07-15-cross-domain-no-env-mean/`.

Both CKA figures use the same fixed vertical range, `0` to `1.02`. This shared scale is intentional: the earlier
cross-environment-only range of `0` to `0.15` visually magnified small SMAC/POGEMA/GRF values and made them difficult
to compare directly with the random-frame cross-football control. The common scale changes presentation only, not any
reported metric.

## Conclusion

The near-unit cross-football cosine is not caused by directly averaging the environment-token state. Removing that
position has negligible effect. The result instead reflects dominant shared activation directions and possibly
environment information already mixed into retained states by non-causal attention. Raw cosine remains insufficient
as an alignment metric; centered CKA, normalized separation, and same-source nearest neighbors remain necessary.

This control also does not deconfound the cross-environment comparison. A stronger intervention must rerun the forward
pass with the environment token ablated, replaced by a common token, or prevented from serving as an attention key and
value, while preserving a separate readout/query mechanism if the actor and critic still require the final position.

## Links

- [Cross-football representation geometry](2026-07-15-cross-football-representation-geometry.md)
- [Internal representation geometry](2026-07-06-internal-representation-geometry.md)
- [Cross-environment compute sharing](2026-07-06-cross-env-compute-sharing.md)

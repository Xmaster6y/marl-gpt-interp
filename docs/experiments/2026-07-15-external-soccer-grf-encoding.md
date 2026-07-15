# External Soccer GRF Encoding

## Status

Completed locally on 2026-07-15. La Liga and raw RoboCup STP states can be encoded as GRF `simple115v2`, grouped into
six-frame histories, and passed through the MARL-GPT checkpoint alongside native GRF observations.

## Question

Can external human and simulator tracking states be converted into the exact structured GRF input contract expected by
MARL-GPT without claiming that their actions or dynamics are equivalent to GRF?

## Hypothesis

Complete 22-player tracking frames should support exact 115-field layout compatibility after explicit coordinate,
perspective, possession, game-mode, and missing-height policies. The checkpoint should produce finite activations and
outputs, although a tiny schema smoke should not establish cross-domain representation or behavioral transfer.

## Data Or Command

```bash
uv run --group grf -m scripts.external_soccer_marl_gpt --config-name 2026-07-15-local-smoke
```

The config uses one bounded La Liga Arrow sequence, 12 raw STP cycles, a native `11vs11a11` GRF rollout, six-frame
histories, all 22 mirrored external player perspectives, and at most 44 complete examples per source.

Outputs are untracked under `results/experiments/2026-07-15-external-soccer-marl-gpt-smoke/`:

- `encoded_inputs.npz`
- `activations.pt`
- `predictions.jsonl`
- `input_audit.json`
- `summary.json`

## Validation And Metrics

- Exact synthetic-vector parity with Google Football's official `Simple115StateWrapper.convert_observation`.
- Exact positional-template parity with MARL-GPT's GRF tokenizer.
- Complete player-perspective grouping, newest-first history construction, finite-value checks, and imputation counts.
- Input ranges, GRF actor entropy and top actions, critic values, and final-token activation-centroid distances to native
  GRF.

## Result

All three sources produced 44 finite model inputs. External histories had shape `(44, 6, 115)` before padding; model
inputs had shape `(44, 700)`. La Liga values ranged from `-0.7632` to `1.0`, raw STP values from `-0.9333` to `1.0`,
and native GRF values from `-1.0239` to `1.0110`.

The La Liga adapter recorded ball height, vertical ball direction, game mode, and attacking-team possession as imputed
for all 44 selected histories. The STP adapter recorded ball height, vertical direction, and unknown possession as
imputed; its sampled play modes mapped without fallback.

Actor logits, critic values, and selected activations were finite for every example. Relative final-token centroid
distances to native GRF were approximately `0.016` to `0.074` for La Liga and `0.021` to `0.078` for RoboCup across the
selected transformer layers. These similarities are not transfer evidence: the final embedding is identical by design
because every source uses the same GRF environment token, the sample contains only a few adjacent frames, and all three
sources selected action `1` for every example with near-zero actor entropy. The action collapse also occurs on the
native GRF control, so it is not evidence against either external adapter.

The reusable code streams La Liga Arrow/JSONL and named raw STP CSV fields, rejects the known-invalid derived RoboCup
`.npy` arrays, generates all 22 focal-player perspectives, and preserves per-history imputation provenance.

## Conclusion

The input-compatibility gate passes: both external sources can be encoded and fed through the pretrained GRF branch
without shape, positional-token, range, or finite-value failures. This is infrastructure evidence only. A representation
claim requires broader match-level samples, non-adjacent train/test grouping, stronger native-GRF controls, and metrics
that do not rely only on the shared final environment-token position. External action mapping and fine-tuning remain out
of scope.

## Links

- [Cross-football representation geometry](2026-07-15-cross-football-representation-geometry.md)
- [Fuji soccer schema inspection](2026-07-14-fuji-soccer-data-schema-and-sample.md)
- [Cross-dataset soccer statistics](2026-06-30-cross-dataset-soccer-statistics.md)
- [GRF-human gap analysis](2026-06-30-grf-human-gap-analysis.md)
- [Simulation-human modelling gap](../questions/2026-06-30-simulation-human-modelling-gap.md)

# Random Frame Sampling Control

## Status

Completed locally on 2026-07-15. This control supersedes the contiguous-frame cross-football CKA interpretation.

## Question

Do near-unit cross-football cosine and high CKA persist when the analysis uses randomly selected, temporally spaced,
complete frames instead of the first 12 adjacent frames?

## Sampling

The workflow collected 24 history-valid candidate frames per source, then selected 12 complete frames with seed `41`.
Selection preserved every perspective belonging to a frame and required selected frames from the same sequence to be
at least two steps apart. The selected frames remained in independent random order for each source.

This is better than contiguous sampling but remains a bounded control: La Liga and RoboCup each come from one sequence,
and native GRF comes from one rollout.

## Command

```bash
uv run -m scripts.cross_football_representation_geometry \
  --config-name 2026-07-15-local-random-frames
```

Outputs are untracked under
`results/experiments/2026-07-15-cross-football-representation-geometry-random-frames/`. The run summary records all
candidate counts, selected frame IDs, step indices, seed, selection order, and temporal-gap policy.

## Result

Random sampling had almost no effect on raw cross-source pairwise cosine averaged over transformer blocks:

| Pair | Contiguous mean pooling | Random mean pooling | Contiguous final | Random final |
| --- | ---: | ---: | ---: | ---: |
| La Liga vs GRF | 0.996842 | 0.996631 | 0.999440 | 0.999526 |
| RoboCup vs GRF | 0.996480 | 0.995978 | 0.999281 | 0.999230 |
| La Liga vs RoboCup | 0.998057 | 0.998016 | 0.999202 | 0.999244 |

The stability of these near-unit values shows that temporal adjacency was not the main cause of cosine saturation. Raw
activations share dominant directions or offsets, so uncentered cosine is insensitive to the source separation visible
under distance-based metrics.

CKA changed sharply after independent random ordering:

| Pair | Contiguous mean CKA | Random mean CKA | Contiguous final CKA | Random final CKA |
| --- | ---: | ---: | ---: | ---: |
| La Liga vs GRF | 0.824 | 0.195 | 0.869 | 0.185 |
| RoboCup vs GRF | 0.859 | 0.412 | 0.948 | 0.329 |
| La Liga vs RoboCup | 0.731 | 0.132 | 0.844 | 0.122 |

Mean-pooled self-CKA also fell to `0.141` for La Liga, `0.197` for RoboCup, and `0.201` for GRF. Final-token self-CKA
fell to approximately `0.09-0.10`. These random-half values are not reliability estimates because their rows are not
matched; the drop confirms that the previous adjacent even/odd split primarily measured temporal ordering.

## Conclusion

The original cross-football CKA result was overstated by contiguous, index-aligned sampling and should not support a
shared-representation claim. Random complete-frame sampling is the correct default for distributional metrics.

Raw pairwise cosine is also overstated as an alignment measure, but random sampling does not repair it. Its near-unit
value is driven primarily by anisotropic activation geometry rather than temporal duplication. Future cosine analysis
should use a common held-out centering/whitening reference or remove dominant principal components and should report
the correction alongside uncentered cosine.

Cross-source CKA still requires semantically matched rows; independently sampled frames do not provide that. When
event- or phase-level matching is unavailable, the primary comparison should use distributional metrics such as energy
distance or MMD with match-level bootstrap uncertainty.

## Links

- [Cross-football representation geometry](2026-07-15-cross-football-representation-geometry.md)
- [Final-token-excluded pooling control](2026-07-15-final-token-excluded-pooling-control.md)
- [Internal representation geometry](2026-07-06-internal-representation-geometry.md)

# Cross-Environment Circuit Tracing

## Status

Active.

## Question

Does the frozen MARL-GPT policy reuse causal computations across SMACv2, POGEMA, and GRF, and can those computations be used for selective training-free or training-light steering?

## Falsifiable Alternatives

1. **Shared circuits:** matched situations recruit corresponding feature paths with consistent actor or critic effects across environments.
2. **Partial reuse:** some motifs recur, while task-specific paths dominate other decisions.
3. **Parameter sharing without mechanism sharing:** each environment uses substantially disjoint paths despite common weights.
4. **Method failure:** the CLT replacement or local graphs are not faithful enough to distinguish the alternatives.

## Required Evidence

- Separate actor and critic CLTs pass held-out reconstruction and global replacement gates in every environment.
- Local graphs expose rather than hide reconstruction-error contributions.
- Candidate shared motifs agree in examples, token roles, signed paths, and original-model intervention fingerprints.
- Cross-environment conclusions are evaluated on source-group-held-out matched situations; native episode provenance is not assumed where the dataset does not expose it.
- Steering claims use fresh rollouts and norm- and induced-KL-matched random writes.

Activation overlap, environment decodability, decoder cosine, or a visually plausible graph is insufficient alone.

## Resolution

The [actor/critic CLT experiment](../experiments/2026-07-22-actor-critic-clt-attribution-graphs.md) resolves the question. A later TacSIm study is permitted only after the official benchmark and one published baseline are reproduced.

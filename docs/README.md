# MARL-GPT Circuit Tracing

## Current State

- **Direction:** two independent full-path cross-layer transcoders (CLTs), one for the actor and one for the critic.
- **Primary object:** input-specific attribution graphs from structured tokens through sparse features and frozen-attention OV paths to action contrasts or action value.
- **Implementation:** corpus, CLT training, replacement, local graph, pruning, and graph-bound original-model intervention workflows are implemented and covered by local tests.
- **Evidence:** no claim-bearing CLT has been trained; replacement, graph, intervention, and rollout results are pending.
- **Paper:** reorganized around circuit tracing; football and TacSIm are downstream steering endpoints rather than the organizing contribution.

## Canonical Records

- [Research question](questions/2026-07-22-cross-environment-circuit-tracing.md)
- [Method decision and prior-history summary](decisions/2026-07-22-adopt-actor-critic-clts.md)
- [Primary experiment](experiments/2026-07-22-actor-critic-clt-attribution-graphs.md)
- [CLT and attribution-graph literature](literature/2026-07-22-cross-layer-transcoders-and-attribution-graphs.md)
- [Tentative claim](claims/2026-07-22-recurring-causal-circuits.md)
- [Paper](../latex/main.tex)

## Next Gate

Collect and audit the token-level corpus, then train the actor and critic CLTs. Do not interpret graphs until both branch-specific replacement gates pass on held-out data in every environment.

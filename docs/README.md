# MARL-GPT Circuit Tracing

## Current State

- **Direction:** two independent full-path cross-layer transcoders (CLTs), one for the actor and one for the critic.
- **Primary object:** input-specific attribution graphs from structured tokens through sparse features and frozen-attention OV paths to action contrasts or action value.
- **Implementation:** corpus, CLT training, replacement, local graph, pruning, and graph-bound original-model intervention workflows are implemented and covered by local tests.
- **Launch:** the SCRATCH-backed Jean Zay suite is submitted as jobs `53452`, `53453`, `53454_[0-1]`, and `53455`; the first job is pending for priority and the others are dependency-held. The original zero-runtime chain was replaced because Jean Zay derived an unintended 240 GB request from eight preprocessing CPUs.
- **Evidence:** no claim-bearing CLT has been trained; submission and pending scheduler state are operational evidence only, while replacement, graph, intervention, and rollout results remain unavailable.
- **Paper:** reorganized around circuit tracing; football and TacSIm are downstream steering endpoints rather than the organizing contribution.

## Canonical Records

- [Research question](questions/2026-07-22-cross-environment-circuit-tracing.md)
- [Method decision and prior-history summary](decisions/2026-07-22-adopt-actor-critic-clts.md)
- [Primary experiment](experiments/2026-07-22-actor-critic-clt-attribution-graphs.md)
- [Jean Zay runtime and SCRATCH contract](2026-07-02-grf-jz-setup.md)
- [CLT and attribution-graph literature](literature/2026-07-22-cross-layer-transcoders-and-attribution-graphs.md)
- [Tentative claim](claims/2026-07-22-recurring-causal-circuits.md)
- [Paper](../latex/main.tex)

## Next Gate

Wait for the dataset audit and corpus collection, then inspect both branch-specific CLT health and replacement gates. Do not interpret graphs merely because downstream jobs ran; graph interpretation requires the hard audit to pass on held-out data in every environment.

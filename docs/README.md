# MARL-GPT Circuit Tracing

## Current State

- **Direction:** two independent full-path cross-layer transcoders (CLTs), one for the actor and one for the critic.
- **Primary object:** input-specific attribution graphs from structured tokens through sparse features and frozen-attention OV paths to action contrasts or action value.
- **Implementation:** corpus, CLT training, replacement, local graph, pruning, and graph-bound original-model intervention workflows are implemented and covered by local tests.
- **Launch:** all `nwq@v100` jobs were cancelled at zero runtime on 2026-07-23. The canonical Jean Zay target is now `jhr@a100`; runtime setup job `83591` is priority-pending on `prepost`, and the claim-bearing suite remains gated on setup plus the A100 CUDA/model smoke.
- **Evidence:** no claim-bearing CLT has been trained; submission and pending scheduler state are operational evidence only, while replacement, graph, intervention, and rollout results remain unavailable.
- **Paper:** reorganized around circuit tracing; football and TacSIm are downstream steering endpoints rather than the organizing contribution.

## Canonical Records

- [Research question](questions/2026-07-22-cross-environment-circuit-tracing.md)
- [Method decision and prior-history summary](decisions/2026-07-22-adopt-actor-critic-clts.md)
- [Primary experiment](experiments/2026-07-22-actor-critic-clt-attribution-graphs.md)
- [Jean Zay runtime and WORK/SCRATCH contract](2026-07-02-grf-jz-setup.md)
- [CLT and attribution-graph literature](literature/2026-07-22-cross-layer-transcoders-and-attribution-graphs.md)
- [Tentative claim](claims/2026-07-22-recurring-causal-circuits.md)
- [Paper](../latex/main.tex)

## Next Gate

Pass the `jhr@a100` runtime preflight, submit the dependency-linked suite, then inspect the dataset audit, both branch-specific CLT health reports, and replacement gates. Do not interpret graphs merely because downstream jobs ran; graph interpretation requires the hard audit to pass on held-out data in every environment.

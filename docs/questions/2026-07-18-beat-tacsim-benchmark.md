# Beat the TacSIm Benchmark

## Status

Deferred endpoint question, blocked on readiness gates, official artifacts, and baseline reproduction.

## Question

Can a MARL-GPT-derived multi-agent imitation model outperform every reported method on the official TacSIm football
tactical-style imitation benchmark?

This asks about conditional continuation quality, not trajectory-failure diagnosis. Sparse graph verification or
candidate reranking is only a possible later method, not the current hypothesis.

## Target

Use the official TacSIm data, split, observation protocol, rollout horizons, preprocessing, and scoring implementation.
The reported comparison set is:

- behavior cloning (BC);
- coordinated multi-agent imitation learning (CMIL);
- inverse reinforcement learning (IRL);
- decentralized adversarial imitation learning with correlated policies (CoDAIL); and
- diffusion-reward adversarial imitation learning (DRAIL).

The target is a new best benchmark score.

## Required Evidence

1. Reproduce the official evaluation and at least the strongest reported baseline within expected variance.
2. Evaluate the proposed method on the unchanged official test split.
3. Report every official grid-resolution and rollout-horizon setting, not only favorable slices.
4. Beat the strongest reported method on the predeclared primary aggregate score.
5. Run matched random-initialization and MARL-GPT-initialization variants so any pretraining claim is identifiable.

## Decision Rule

- **Success:** the proposed method exceeds the strongest TacSIm baseline on the official primary score and does not hide
  material regressions in the per-setting table.
- **Method success without foundation-model success:** the new architecture wins, but MARL-GPT initialization does not
  improve over matched random initialization.
- **No result:** the official artifact or score cannot be reproduced closely enough for a valid comparison.
- **Failure:** the method does not beat the strongest reproduced baseline.

## Links

- [TacSIm literature note](../literature/2026-07-18-tacsim-benchmark.md)
- [Benchmark decision](../decisions/2026-07-18-target-tacsim-benchmark.md)
- [Benchmark experiment](../experiments/2026-07-18-tacsim-benchmark.md)
- [Staged direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)

# Target the TacSIm Benchmark

## Status

Accepted as the football objective.

## Choice

Treat TacSIm as the benchmark, not as a competing model. The immediate objective is to train a method using the official
TacSIm protocol and beat all reported baselines on its primary tactical-style imitation score.

## Rationale

TacSIm already supplies the real-video-derived data, virtual imitation task, split, rollout protocol, baseline family,
and evaluation metrics. This gives MARL-GPT a direct and falsifiable use: test whether its structured multi-agent
representation or pretraining improves tactical imitation against BC, CMIL, IRL, CoDAIL, and DRAIL.

## Consequences

- First acquire and reproduce the official TacSIm evaluation artifact.
- Preserve the official input, output, split, horizons, and score for the claim-bearing comparison.
- Use the strongest reported or reproduced TacSIm method as the baseline to beat.
- Compare MARL-GPT initialization with the same model trained from scratch.
- Do not launch a full run until the exact artifact, metric implementation, compute budget, and reproduction command are
  recorded.

## Revisit Condition

Revisit the decision if the benchmark cannot be reproduced from the released artifacts.

## Links

- [Benchmark question](../questions/2026-07-18-beat-tacsim-benchmark.md)
- [TacSIm literature note](../literature/2026-07-18-tacsim-benchmark.md)
- [Benchmark experiment](../experiments/2026-07-18-tacsim-benchmark.md)

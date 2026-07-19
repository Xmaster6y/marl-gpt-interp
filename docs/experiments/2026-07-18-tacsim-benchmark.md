# TacSIm Benchmark

## Status

Deferred endpoint. Readiness stages, official artifact audit, and baseline reproduction are required before method
training. No run has been launched.

## Question

Can a MARL-GPT-derived multi-agent model achieve a new best result on the official TacSIm benchmark?

## Hypothesis

MARL-GPT's structured agent, group, attribute, and temporal representations may improve coordinated continuation from
TacSIm states. The architecture and the pretrained initialization are separate hypotheses and require a matched
from-scratch control.

## Locked Benchmark

- Official TacSIm dataset and match-level train/validation/test split.
- Official first-frame test context and rollout protocol.
- Official 3-, 5-, and 10-second evaluation horizons.
- Official grid-resolution settings.
- Official spatial occupancy score, movement-vector score, and combined score.
- Complete comparison with BC, CMIL, IRL, CoDAIL, and DRAIL.

The paper says that official scoring uses the ball trajectory. The claim-bearing run must follow the released evaluator
exactly, even if additional all-player metrics are reported separately.

## Stages

This benchmark evaluates conditional tactical-style trajectory imitation, not trajectory-failure diagnosis. A
TacSIm-shaped local proxy may be built only after cross-football sparse-feature robustness passes; proxy results are not
TacSIm transfer evidence.

### 0. Artifact audit

Record the official dataset URL, license, code revision, environment, preprocessing command, split files, metric code,
and expected table. Resolve the paper's duration/segment-count and combined-score formula inconsistencies from the
released implementation.

### 1. Reproduction

Run the official evaluator and reproduce at least the strongest feasible baseline. Do not proceed if the score or split
cannot be matched closely enough for a trustworthy comparison.

### 2. Matched model comparison

Train and evaluate:

1. strongest reproduced TacSIm baseline;
2. structured MARL-GPT-compatible model from random initialization;
3. the identical model initialized from MARL-GPT;
4. ablations required to identify which multi-agent structural components matter.

Match the proposed variants in data, input history, output target, parameter budget, optimizer search budget, and number
of seeds wherever possible.

Sparse attribution-graph verification or candidate reranking may be added only if fixed-layer decomposition, per-layer
transcoders, and graph faithfulness have already passed their own gates.

## Primary Metric

The primary metric is the official combined TacSIm score aggregated exactly as implemented by the benchmark. Report the
full grid-by-horizon table and both component scores. The exact aggregation is pending the artifact audit.

## Decision Rule

The experiment succeeds only if the proposed method beats the strongest reported/reproduced TacSIm method on the locked
primary score. MARL-GPT pretraining is supported only if the pretrained variant also beats the identical randomly
initialized variant under the matched protocol.

## Launch Boundary

No full local or cluster run is authorized until Stage 0 records the artifact and Stage 1 has a concrete config, command,
expected score, runtime, and resource estimate. Minimal evaluator and batch-shape smoke tests are allowed.

## Links

- [Benchmark question](../questions/2026-07-18-beat-tacsim-benchmark.md)
- [Benchmark decision](../decisions/2026-07-18-target-tacsim-benchmark.md)
- [TacSIm literature note](../literature/2026-07-18-tacsim-benchmark.md)
- [Staged direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)

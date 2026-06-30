# Simulation-Human Modelling Gap

## Status

Active.

## Question

Where do GRF and MARL-GPT trajectories differ from human football tracking data, and can the gap be measured at trajectory, tactical-concept, and representation levels?

## Motivation

Sports analytics needs models that explain or evaluate human behavior, not only agents that maximize simulator reward. Human tracking data is continuous and simulator actions are discrete, so direct action matching is fragile. A concept-level gap may be more robust and more interpretable.

## Assumptions

- Human tracking data from the host lab will be available.
- Event labels may be sparse or incomplete, so tracking-derived concepts should be useful even without full action labels.
- GRF and human data can be aligned at least approximately through state features, phases of play, or trajectory matching.

## Expected Evidence

- Distributional comparisons between human and simulator states: spacing, speeds, possession phases, pressure, and attacking shape.
- Concept frequency comparisons: pass opportunities, shot opportunities, defensive pressure, support, and compactness.
- Representation comparisons: whether human-like states activate the same internal directions or layers as simulator states.
- Examples of simulator-specific shortcuts or human-like patterns.

## Metrics

- Trajectory statistics and distances.
- DTW or phase-aligned trajectory distances.
- Concept distribution divergence.
- Activation-space distances or classifier domain accuracy between human and simulator states.

## Decision Rule

If the largest gaps are visible at concept or representation level, the project can motivate interpretability-guided alignment. If the largest gaps are low-level state normalization or observation mismatch, the project must first solve data mapping before claiming model-level insight.

## Links

- [Adaptive Action Supervision](../literature/2026-06-30-adaptive-action-supervision.md)
- [GRF-human gap analysis](../experiments/2026-06-30-grf-human-gap-analysis.md)

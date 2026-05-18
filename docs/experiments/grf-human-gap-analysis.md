# GRF-Human Gap Analysis

## Status

Planned. Blocked until human tracking data access and a data-use agreement are available.

## Question

How do GRF and MARL-GPT trajectories differ from human football tracking data at the levels of trajectories, tactical concepts, and internal model representations?

## Hypothesis

The largest useful gaps will appear in concept distributions and phase-specific behavior rather than raw positions alone. Examples may include pressure response, support positioning, pass timing, and shot selection.

## Data Or Command

Use human tracking and event data from the host lab, plus GRF trajectories from MARL-GPT or simulator rollouts. No launch command is fixed yet.

## Metrics

- Position, velocity, spacing, and formation statistics.
- Possession and attacking-phase statistics.
- Concept distribution divergence.
- DTW or phase-aligned trajectory distances.
- Activation-space similarity between human-derived states and simulator states.
- Domain classifier accuracy over representations as a measure of human-simulator separability.

## Baseline Or Comparison

- Human versus raw GRF simulator trajectories.
- Human versus MARL-GPT rollouts.
- Human versus simple scripted or expert simulator policies if available.
- State-space gap versus representation-space gap.

## Expected Result

Human and simulator data will not align cleanly at the discrete action level. A concept-level comparison should still identify meaningful similarities and differences that can guide later alignment.

## Decision Rule

If human and simulator states can be mapped into shared concepts, proceed to concept-level alignment. If mapping is unreliable, restrict claims to qualitative or exploratory gap analysis and avoid action-supervision claims.

## Links

- [Simulation-human modelling gap](../questions/simulation-human-modelling-gap.md)
- [Adaptive Action Supervision](../literature/adaptive-action-supervision.md)

# Interpretability-Guided Alignment

## Status

Active but downstream of initial probes and gap analysis.

## Question

Can interpretability identify representations that should be preserved, shifted, or regularized when adapting MARL-GPT-style agents from simulator trajectories toward human football behavior?

## Motivation

The project should aim beyond passive visualization. If interpretable concepts reveal why simulator and human behavior differ, they may guide alignment through data selection, concept regularization, representation steering, or adaptive action supervision.

## Assumptions

- Initial probes identify stable concept representations.
- Human and simulator states can be compared through shared concepts or aligned phases of play.
- Alignment can be evaluated without requiring perfect action labels.

## Candidate Mechanisms

- Select simulator trajectories whose concept activations are closest to human trajectories.
- Regularize online or offline training to preserve human-like concept profiles.
- Use representation directions for steering or counterfactual tests.
- Use DTW-style alignment to compare or supervise at matched phases of play.

## Expected Evidence

- A representation-level intervention changes behavior in a predicted direction.
- Concept-level similarity to human data improves without catastrophic loss of simulator performance.
- Alignment failures become diagnosable in terms of specific concepts or tokens.

## Decision Rule

If interventions predictably affect behavior and improve human-like metrics, interpretability-guided alignment becomes a main contribution. If not, alignment should remain a future-work motivation and the main contribution should be diagnostic interpretability.

## Links

- [MARL-GPT](../literature/2026-06-30-marl-gpt.md)
- [Adaptive Action Supervision](../literature/2026-06-30-adaptive-action-supervision.md)
- [Simulation-human modelling gap](2026-06-30-simulation-human-modelling-gap.md)

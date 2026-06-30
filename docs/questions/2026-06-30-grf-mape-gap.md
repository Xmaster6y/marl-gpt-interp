# GRF-MAPE Gap

## Status

Secondary active question.

## Question

Can a simpler multi-agent environment such as MAPE help separate general multi-agent representation issues from football-specific GRF modelling issues?

## Motivation

GRF is the primary environment because it is football-specific and included in MARL-GPT training. A simpler environment may still be useful for controlled tests of role permutation, pursuit, spacing, and coordination without the complexity of football action semantics.

## Assumptions

- MAPE or a related particle environment can provide controlled multi-agent trajectories with known latent variables.
- The simplified environment will not replace GRF as the main sports-facing setting.
- Any MAPE result should be used to clarify mechanisms, not to make the main sports claim.

## Expected Evidence

- Controlled concept probes transfer or fail in predictable ways.
- Representation sensitivity to agent identity, team identity, and spatial relations can be isolated.
- Results explain some failure modes that are harder to diagnose in GRF.

## Decision Rule

Use MAPE only if GRF analyses are blocked or if a specific mechanism needs a controlled setting. Do not let MAPE displace the human-football modelling-gap story for the workshop abstract.

## Links

- [Project brief](../2026-06-30-project-brief.md)

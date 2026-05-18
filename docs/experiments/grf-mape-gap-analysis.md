# GRF-MAPE Gap Analysis

## Status

Optional planned analysis.

## Question

Can a controlled MAPE-style environment isolate multi-agent representation issues that are obscured in GRF?

## Hypothesis

A simpler environment will make role, team, spatial relation, and coordination variables easier to label and manipulate, but it will not answer the main football-human modelling question by itself.

## Data Or Command

No command is fixed. This analysis should only be started after the GRF path is either working or explicitly blocked.

## Metrics

- Probe accuracy for controlled latent variables.
- Sensitivity to agent permutation, team identity, and spatial perturbations.
- Transfer or mismatch between simple-environment concepts and GRF concepts.

## Baseline Or Comparison

- GRF concept probes.
- Randomized or permuted-agent controls.

## Expected Result

MAPE may help debug methods and clarify mechanisms, but it should remain secondary to GRF and human football data.

## Decision Rule

Run this only if it answers a specific blocked or ambiguous GRF question. Do not use it as the main workshop story.

## Links

- [GRF-MAPE gap](../questions/grf-mape-gap.md)

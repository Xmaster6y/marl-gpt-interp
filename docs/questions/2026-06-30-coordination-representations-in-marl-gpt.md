# Coordination Representations In MARL-GPT

## Status

Active.

## Question

Does MARL-GPT internally represent football-relevant coordination concepts in GRF, such as possession, pressure, support, spacing, pass opportunity, and shot opportunity?

## Motivation

MARL-GPT is trained to act in GRF, but task performance alone does not show whether it learns concepts that are useful for sports analysis or human behavior modelling. If these concepts are represented, they can become the bridge between simulator agents and human trajectories.

## Assumptions

- Pretrained MARL-GPT weights and GRF-compatible trajectories will become available.
- Simulator state or observations can be converted into concept labels or weak labels.
- Linear or sparse probes can reveal candidate representations, but causal tests are needed before interpreting them as functional.

## Expected Evidence

- Probe accuracy above baselines for tactical concepts across model layers.
- Layer or token locations where concepts become more linearly available.
- Causal token or representation interventions that change relevant action logits or Q-values.
- Failure cases where probes work but interventions do not affect behavior.

## Metrics

- Probe accuracy, AUROC, or regression error depending on concept type.
- Intervention effect on action logits, Q-values, entropy, and selected action.
- Token attribution or ablation effect sizes for ball, teammate, opponent, history, and global tokens.

## Decision Rule

If football concepts are predictable and causally affect action selection, the project can claim that MARL-GPT contains interpretable coordination representations. If concepts are predictable but not causal, the claim should be limited to diagnostic representations. If neither is true, the project should pivot to modelling-gap measurement rather than concept interpretation.

## Links

- [MARL-GPT](../literature/2026-06-30-marl-gpt.md)
- [GRF representation probes](../experiments/2026-06-30-grf-representation-probes.md)

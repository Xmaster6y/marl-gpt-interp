# GRF Representation Probes

## Status

Planned.

## Question

Which football concepts are represented in MARL-GPT activations during GRF inference?

## Hypothesis

Concepts that are directly useful for action selection, such as possession, pressure, ball distance, and shot opportunity, will be more linearly decodable in later transformer layers and actor or critic representations than in raw token embeddings.

## Data Or Command

Use GRF trajectories from MARL-GPT evaluation, dataset batches, or small rollouts after pretrained weights are available. No launch command is fixed yet.

## Metrics

- Probe performance for each concept across layers.
- Layerwise and tokenwise localization of concept information.
- Generalization of probes across GRF scenarios.
- Causal ablation effect on action logits, Q-values, and selected action.

## Baseline Or Comparison

- Raw observation features or simple handcrafted features.
- Randomly initialized model if feasible.
- Earlier versus later layers.
- Pretrained versus online fine-tuned checkpoints if available.

## Expected Result

Some football concepts should be decodable and behaviorally relevant. Concepts tied to simulator-specific action semantics may be less stable across scenarios or weaker under human-data comparison.

## Decision Rule

If probes reveal stable and causal concepts, use them as the basis for alignment analyses. If concepts are decodable but not causal, report them as diagnostic rather than mechanistic. If probes fail, shift emphasis to trajectory-level modelling gap and tokenization limitations.

## Links

- [Coordination representations](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [MARL-GPT](../literature/2026-06-30-marl-gpt.md)

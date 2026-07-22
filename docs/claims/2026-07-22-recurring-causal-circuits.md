# Recurring Causal Circuits

## Strength

Tentative.

## Claim

Some of MARL-GPT's cross-environment generalization is implemented by recurring causal circuits that transform structured multi-agent observations into action preferences and value estimates across SMACv2, POGEMA, and GRF.

## Evidence Status

No direct evidence yet. Existing decodability and representation-geometry results are motivation only. The claim requires faithful actor and critic CLTs, recurring held-out graph motifs, original-model interventions with predicted effects, and fresh behavioral rollouts.

## Refutation

The claim is contradicted if faithful branch replacements yield environment-disjoint causal paths in matched situations, or if apparently recurring paths fail intervention tests. It remains unresolved if CLT or graph fidelity fails.

## Links

- [Question](../questions/2026-07-22-cross-environment-circuit-tracing.md)
- [Experiment](../experiments/2026-07-22-actor-critic-clt-attribution-graphs.md)
- [Paper](../../latex/main.tex)

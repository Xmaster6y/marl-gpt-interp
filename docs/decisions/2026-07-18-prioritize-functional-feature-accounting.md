# Prioritize Functional Feature Accounting

## Status

Accepted.

## Choice

Make functional sparse-feature accounting in the fixed MARL-GPT checkpoint the primary research direction. Develop and
evaluate a domain-lattice sparse autoencoder that composes universal, pairwise-shared, and environment-private feature
blocks across SMACv2, GRF, and POGEMA.

Do not retrain MARL-GPT into single-environment policies for the primary study. The object is the internal organization
of the existing jointly trained generalist model.

The two primary naive baselines are mandatory:

1. one balanced mixture SAE trained across all three environments, treated as the naive universal dictionary; and
2. three independent environment SAEs, treated as the naive private dictionaries.

Compare the three-independent-SAE baseline both at matched total capacity and as a `3x`-capacity per-domain oracle.

## Rationale

MARL-GPT's shared transformer and multi-environment performance do not establish that it reuses internal features.
Existing sparse-feature methods also make a clean answer difficult: mixture SAEs can suppress domain-specific factors,
while independent SAEs are non-identifiable across seeds and datasets. A composed dictionary tied to explicit domain
support and evaluated through policy fidelity can distinguish a compact shared core, pairwise sharing, and a union of
private mechanisms.

This direction produces a useful result under several outcomes. The lattice may reveal functionally shared capacity; a
single mixture SAE may prove sufficient; three private dictionaries may dominate; or apparent activation sharing may
fail causal tests. Each outcome directly constrains what can be inferred from a generalist MARL architecture.

## Alternatives Considered

- **Continue cross-football concept transfer:** superseded as the primary direction because handcrafted concept probes do
  not answer whether the generalist model internally reuses features.
- **Mechanistic offline-to-online adaptation:** deferred; it requires a checkpoint trajectory and stable repeated online
  fine-tuning, while the current fixed checkpoint already supports a cleaner feature-accounting study.
- **Unsupervised football strategy discovery:** retained as possible later feature interpretation, but not needed to
  validate the shared/private decomposition method.
- **Only compare separate SAEs geometrically:** rejected because independent dictionaries can permute, split, merge, and
  vary across seeds.

## Consequences

- Keep the MARL-GPT checkpoint frozen.
- Start with synthetic support recovery before interpreting real MARL-GPT features.
- Use balanced, trajectory-grouped activation samples from all three environments.
- Compare methods along capacity-sparsity-functional-fidelity frontiers rather than one width or one reconstruction score.
- Evaluate actor and critic fidelity using native environment masks and outputs; do not align action indices across
  environments.
- Require multiple seeds, constrained-random baselines, ablation fingerprints, and feature substitution.
- Treat domain-general causal use as weaker than shared semantics unless a separate semantic test justifies the latter.
- Do not launch cluster work until a local synthetic and activation-schema smoke validates the objective and metrics.

## Revisit Condition

Revisit the method if synthetic experiments show that the lattice cannot recover known domain support, if functional
loss causes trivial output-preserving reconstructions, or if the balanced single-mixture SAE matches the lattice across
the complete fair-comparison frontier. In the last case, prefer the simpler mixture SAE and report the negative result
rather than preserving unnecessary method complexity.

## Links

- [Functional feature accounting question](../questions/2026-07-18-functional-feature-accounting.md)
- [Sparse feature accounting literature](../literature/2026-07-18-sparse-feature-accounting.md)
- [Domain-lattice method validation experiment](../experiments/2026-07-18-domain-lattice-sae-method-validation.md)

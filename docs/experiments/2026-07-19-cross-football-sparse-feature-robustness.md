# Cross-Football Sparse-Feature Robustness

## Status

Blocked on the synthetic support-recovery gate. It may start after that gate and then run alongside the internal
SMAC–GRF–POGEMA study. No run has been launched.

## Question

Do fixed MARL-GPT features and their functional effects remain stable across native GRF, La Liga tracking converted
with recorded imputation, and raw named-column RoboCup STP data?

## Design

Use the common cache, training, logging, and evaluation interfaces but train a separate seven-block lattice: one
universal, three pairwise, and three source-private blocks. Do not combine these sources with the internal environments
into a five-domain power set.

Required corpus controls are match/episode-grouped splits, balanced sources, explicit imputation provenance, random
temporal spacing, and a native-GRF subset whose masked actor outputs are non-collapsed. Reject obsolete derived RoboCup
arrays; raw named columns are authoritative.

Compare flat per-example TopK, capacity-matched independent TopK, the source lattice, domain-stratified BatchTopK
sensitivity variants, PCA/ICA, and constrained-random controls. Report normalized reconstruction, code length, L0,
dead features, cross-seed feature stability, distributional/reconstructive/functional source support, native-GRF output
preservation, and bounded intervention effects.

## Decision Rule

Proceed to a TacSIm-shaped trajectory proxy only if feature matches and bounded effects reproduce across seeds and
native-GRF actor behavior remains informative and preserved. Collapsed native actions fail the functional gate even if
reconstruction is good.

## Non-Claims

Shared features do not establish shared tactical semantics or tactical transfer. This milestone contains no continuous
trajectory prediction and is not TacSIm evidence.

## Links

- [Staged direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)
- [External soccer encoding](2026-07-15-external-soccer-grf-encoding.md)
- [Domain-lattice validation](2026-07-18-domain-lattice-sae-method-validation.md)

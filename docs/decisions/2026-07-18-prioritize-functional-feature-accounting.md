# Prioritize Functional Feature Accounting

## Status

Accepted as the research question and expanded on July 19. Domain-lattice method comparisons were paused on July 21
after the prespecified synthetic gate and a longer-optimization diagnosis both failed. The balanced pooled SAE is now
the primary decomposition baseline. This supersedes only the timing in the TacSIm decision; TacSIm remains the
conditional external endpoint.

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

## Gated Sequence

1. Synthetic support recovery gates all real interpretation.
2. Fixed-layer SMAC–GRF–POGEMA validation begins at `layer_03:final`; `layer_06:final` remains a schema check until the
   first location is stable.
3. After the synthetic gate, a separate GRF–La Liga–RoboCup robustness branch may proceed alongside the internal study,
   without tactical-transfer claims.
4. Per-layer MLP transcoders and bounded attribution graphs follow fixed-layer fidelity and stability. Cross-layer
   transcoders require a separate faithfulness and cost comparison.
5. A 3/5/10-second TacSIm-shaped continuation proxy follows trustworthy football robustness.
6. Official TacSIm work begins only after artifact auditing and reproduction of one published baseline.

Local smoke runs may debug schemas, losses, hooks, and metrics. Claim-bearing evidence requires frozen manifests, all
declared controls, five seeds where prescribed, paired intervals, and complete per-domain reporting. Failed gates are
diagnosed or simplified rather than weakened after inspection. No five-domain lattice, continuous trajectory head,
cross-layer transcoder, or cluster launch belongs to the first milestone.

## July 21 Gate Outcome

Pause all domain-lattice method comparisons and real-feature interpretation. The seven-regime, five-seed synthetic gate
failed every regime. A bounded balanced-regime diagnosis increased training from 500 to 5,000 steps but again produced
zero stable seeds: support F1 passed in two of five seeds, decoder recovery in one, and matched reconstruction in none.
Longer optimization is therefore not a sufficient repair.

Keep the queued full-mixture flat SAE only as a corpus and training-health diagnostic. Do not treat its apparent feature
support as interpretation, and do not proceed to lattice-versus-flat-versus-independent comparisons. The next method
artifact must either correct a diagnosed objective failure or simplify the structured model, then pass the unchanged
balanced convergence diagnostic before the full synthetic gate is rerun. Thresholds remain fixed.

The immediate specification gap is that the implemented model uses hard support-eligibility masks with a single
full-reconstruction loss; it does not implement the proposed hierarchical or residual block objective. The next design
must explain how support is learned rather than merely permitted, while avoiding an objective that forces the desired
universal or pairwise answer by construction.

Continue the broader functional-accounting question with one pooled SAE before designing another structured objective.
Natural pooled training collapsed at widths 512 and 2,048, while per-domain centering/RMS scaling repaired feature
usage. Test global train-only centering/RMS next so the primary preprocessing choice does not depend on domain labels
unless necessary. Select sparsity on `{8,16,32,64}` only after that coordinate-space gate, then inspect top and
random-active examples across all three environments and infer candidate empirical support. The failed lattice gate
blocks the lattice contribution, not this simpler baseline study; functional-sharing claims still require native-policy
substitution or intervention.

## Revisit Condition

Revisit the method if synthetic experiments show that the lattice cannot recover known domain support, if functional
loss causes trivial output-preserving reconstructions, or if the balanced single-mixture SAE matches the lattice across
the complete fair-comparison frontier. In the last case, prefer the simpler mixture SAE and report the negative result
rather than preserving unnecessary method complexity.

## Links

- [Functional feature accounting question](../questions/2026-07-18-functional-feature-accounting.md)
- [Sparse feature accounting literature](../literature/2026-07-18-sparse-feature-accounting.md)
- [Domain-lattice method validation experiment](../experiments/2026-07-18-domain-lattice-sae-method-validation.md)
- [Cross-football robustness](../experiments/2026-07-19-cross-football-sparse-feature-robustness.md)
- [TacSIm endpoint](2026-07-18-target-tacsim-benchmark.md)

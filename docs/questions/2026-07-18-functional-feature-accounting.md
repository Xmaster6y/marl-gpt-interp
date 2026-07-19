# Functional Feature Accounting In MARL-GPT

## Status

Active. Synthetic method validity is the current gate; real MARL-GPT claims remain blocked until it passes.

## Question

How does the fixed, jointly trained MARL-GPT policy allocate its internal sparse feature capacity across SMACv2,
GRF, and POGEMA: through features used universally, features shared by environment pairs, or features private to one
environment?

More specifically, can a composed sparse dictionary preserve MARL-GPT's activations and actor/critic behavior more
at a better functional rate–distortion point than either of the two naive alternatives:

1. one sparse autoencoder trained on a balanced mixture of all three environments; or
2. three independent sparse autoencoders, one per environment?

The MARL-GPT checkpoint remains frozen. This question concerns the organization of representations already present in
the generalist policy, not the effect of retraining it.

## Why This Is Worth Testing

A single shared transformer performing several tasks does not establish internal feature reuse. The model could learn a
compact shared feature basis, a union of environment-specific mechanisms, or a mixture in which sharing is mostly
pairwise. Current mixture-level geometry and environment classification do not distinguish these possibilities.

The question also exposes a methodological gap. A flat mixture SAE can hide domain-specific features, while independent
SAEs are not directly comparable because features can permute, split, merge, or vary across seeds. A domain-structured
dictionary, evaluated by its ability to preserve policy behavior, could provide a more defensible accounting of shared
and private feature capacity.

## Proposed Object: Domain-Lattice SAE

For domain `d` in `{smac, grf, pogema}`, decompose an activation `h_d` into blocks whose allowed support follows the
non-empty subsets of the three domains:

```text
universal
|- smac-grf
|- smac-pogema
|- grf-pogema
|- smac-private
|- grf-private
`- pogema-private
```

Operationally,

```text
h_hat_d = b_d
          + D_universal z_universal(h_d)
          + sum_{pair A containing d} D_A z_A(h_d)
          + D_d z_d(h_d).
```

Domain masks prevent a sample from using an incompatible private or pairwise block. A hierarchical or residual training
objective should require the universal block to explain a compact cross-domain core, pairwise blocks to explain
two-domain residuals, and private blocks to explain the remaining source-specific residuals. Capacity sweeps are required
because one fixed block allocation would make the conclusion depend on an arbitrary design choice.

## Meanings Of Sharedness

Keep three targets separate:

- **Distributional sharing:** a feature activates in multiple environments.
- **Reconstructive sharing:** removing the feature degrades held-out activation reconstruction in multiple environments.
- **Functional sharing:** removing the feature changes native actor or critic outputs in multiple environments.

Only functional sharing supports a claim that MARL-GPT reuses a computation direction. Even then, common causal use does
not prove identical semantics across environments because observation layouts and action meanings are not aligned.

## Expected Evidence

- Held-out reconstruction and sparsity frontiers for the composed dictionary and both naive baselines.
- Native actor and critic fidelity after replacing an activation with its sparse reconstruction.
- Minimum feature capacity required to meet precommitted per-domain fidelity thresholds.
- Universal, pairwise, and private capacity fractions by layer and actor/critic branch.
- Per-feature activation support, ablation fingerprints, and direct substitution tests.
- Stability across random seeds, data resamples, sparsity levels, and dictionary widths.
- Synthetic support-recovery results where the true universal, pairwise, and private factors are known.

## Primary Baselines

### One Mixture SAE: Naive Universal Baseline

Train one balanced per-example TopK SAE on pooled SMACv2, GRF, and POGEMA activations. Classify its features post hoc using
domain-conditional activation and causal-effect statistics. This is the strongest simple baseline for the proposition
that a flat shared dictionary is sufficient.

Calling it `universal` describes its training corpus, not a validated property of its features.

### Three Independent SAEs: Naive Private Baseline

Train one per-example TopK SAE per environment. Compare it under two capacity conventions:

- **Total-capacity matched:** the three dictionary widths sum to the width of the mixture or lattice model.
- **Per-domain-capacity matched:** each private SAE receives the same width as the mixture SAE, producing a `3x`-capacity
  oracle that estimates the best reconstruction available without any pressure to share.

Independent features must not be equated by decoder cosine alone. Candidate matches should be checked with held-out
ablation fingerprints and feature substitution.

### Additional Controls

- PCA, ICA, and random or constrained-random dictionaries.
- Private-only domain-gated dictionary with no shared blocks.
- Universal-plus-private model without pairwise blocks.
- Standard Matryoshka or hierarchical SAE without domain structure.
- Unequal-mixture and activation-norm controls.
- Domain-stratified BatchTopK sensitivity variants; ordinary mixed-domain BatchTopK is not a primary condition.

## Metrics

- Per-domain normalized MSE and explained variance.
- Mean active latents and dead-feature rate.
- Actor KL after native action masking.
- Critic distribution KL and scalar-value deviation.
- Expert-action or dataset-action negative log-likelihood change where valid.
- Domain-conditional activation rate and feature-domain mutual information.
- Per-domain feature-ablation effect and functional support set.
- Feature-substitution quality across seeds and dictionary types.
- Functional rate–distortion: activation code length and dictionary capacity at fixed per-domain reconstruction and
  actor/critic fidelity.

Raw feature count is not the primary outcome. At a fixed distortion threshold, summarize reuse efficiency as:

```text
reuse_efficiency = 1 - K_lattice / (K_smac + K_grf + K_pogema).
```

This metric is only meaningful when sparsity, reconstruction, and actor/critic fidelity thresholds are matched.

## Assumptions And Threats

- Sparse linear dictionaries are an adequate local description of the selected MARL-GPT activations.
- Environment identity, padding, sequence length, positional embeddings, activation norm, and action-mask structure can
  create apparently domain-specific features without representing meaningful policy computation.
- A feature active and causally relevant in two environments may still serve unrelated functions.
- Hierarchical training can force generic high-variance directions into the universal block; functional loss and random
  controls are required to detect this.
- Recent SAE evaluations show that reconstruction and common interpretability scores can be matched by weak or random
  baselines. The project must not rely on reconstruction, top-activating examples, or automatic feature naming alone.

## Decision Rule

Treat the composed dictionary as a useful method only if it:

1. recovers known support structure better than flat-mixture and independently matched dictionaries on synthetic data;
2. improves the held-out behavior-preserving capacity frontier on MARL-GPT;
3. yields functionally supported feature assignments that reproduce across seeds; and
4. beats constrained-random and simpler hierarchical controls.

The claim-bearing location is `layer_03:final`. `layer_06:final` is cached for schema validation and expands only after
the first location is stable. Per-layer transcoders and attribution graphs are a later research stage, not substitutes
for this fixed-layer gate.

If a single mixture SAE matches the lattice at every fair capacity and fidelity point, prefer the simpler method and
report that explicit domain structure is unnecessary. If three independent SAEs dominate and no stable functionally
shared core appears, conclude that MARL-GPT behaves as a sparse union of environment specialists. If activation-level
sharing disappears under functional tests, report superficial rather than mechanistic sharing.

## Links

- [Sparse feature accounting literature](../literature/2026-07-18-sparse-feature-accounting.md)
- [Domain-lattice method validation experiment](../experiments/2026-07-18-domain-lattice-sae-method-validation.md)
- [Domain-lattice direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)
- [Staged direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)

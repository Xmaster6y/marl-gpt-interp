# Sparse Feature Accounting Across Data Domains

## Scope

This note maps the work most relevant to decomposing a fixed MARL-GPT activation space into universal, pairwise-shared,
and environment-private sparse features. The intended contribution is not a generic application of SAEs to MARL. It is
a domain-structured and functionally evaluated rate–distortion analysis for one generalist policy operating on
heterogeneous data sources. The architectural lattice alone has weak novelty.

## Sparse Autoencoder Foundations

### Scaling And Evaluating Sparse Autoencoders

- Source: [Gao et al., 2024](https://arxiv.org/abs/2406.04093)
- Contribution: k-sparse autoencoders provide direct sparsity control, reduce dead latents, and expose reconstruction and
  feature-quality scaling frontiers.
- Project relevance: establishes TopK-style sparse dictionary learning and the need to compare models along matched
  reconstruction-sparsity frontiers rather than at one arbitrary width.
- Limitation for this project: learns one flat dictionary from one activation distribution and does not assign features
  to overlapping data-domain support sets.

### BatchTopK Sparse Autoencoders

- Source: [BatchTopK Sparse Autoencoders](https://openreview.net/pdf?id=d4dpOCqybL)
- Contribution: applies the sparsity budget across a batch rather than forcing every example to activate exactly the same
  number of latents.
- Project relevance: a sensitivity condition because SMACv2, GRF, and POGEMA activations may have different local
  reconstruction complexity. Per-example TopK is primary.
- Risk: mixed-domain batches could allocate disproportionate feature activity to high-norm or high-complexity domains;
  any use must stratify the BatchTopK budget by domain and report per-domain diagnostics.

### Routing And Multi-Layer Alternatives

- [RouteSAE](https://aclanthology.org/2025.emnlp-main.346/) is a flexible learned-routing alternative to hard support
  masks and becomes important if fixed lattice eligibility loses to routing.
- [Cross-layer transcoders and attribution graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
  provide a more direct computational-graph substrate but compound approximation error and cost across layers. They are
  reserved until fixed-layer reconstruction and native-output substitution are reliable.
- [Concept Relevance Vectors](https://arxiv.org/abs/2510.09312) reinforce evaluating concepts by downstream relevance,
  not merely decoder geometry or top activations.

### Matryoshka Sparse Autoencoders

- Source: [Bussmann et al., ICML 2025](https://openreview.net/forum?id=m25T5rAy43)
- Contribution: nested dictionaries preserve broad features while allowing larger dictionaries to add specificity,
  reducing feature absorption and splitting.
- Project relevance: motivates a hierarchical loss in which a compact universal core is supplemented by pairwise and
  private residual blocks.
- Difference: Matryoshka nesting is organized by dictionary size or abstraction level, not by known domain subsets.

## Cross-Model And Cross-Domain Feature Work

### Universal Sparse Autoencoders

- Source: [Thasarathan et al., ICML 2025](https://arxiv.org/abs/2502.03714)
- Contribution: learns a universal concept space that reconstructs activations across multiple pretrained vision models.
- Project relevance: strongest precedent for a shared sparse feature space spanning heterogeneous sources.
- Difference: aligns different models and activation spaces. MARL-GPT supplies one fixed activation space under multiple
  data distributions, so cross-model decoders are unnecessary and the principal problem is domain support accounting.

### Dedicated Feature Crosscoders

- Source: [Cross-Architecture Model Diffing With Crosscoders](https://openreview.net/forum?id=YXB8uigyOg)
- Contribution: partitions a crosscoder into model-A-exclusive, model-B-exclusive, and shared feature blocks.
- Project relevance: demonstrates that architectural masks can encode an explicit shared/private decomposition.
- Difference and warning: model-exclusive partitions do not automatically imply clean capability isolation. A domain
  lattice needs held-out functional validation rather than trusting the assigned block label.

### Language-Specific SAE Features

- Source: [Deng et al., 2025](https://arxiv.org/abs/2505.05111)
- Contribution: measures SAE-feature monolinguality and validates selected language-specific features through targeted
  ablation and steering.
- Project relevance: closest example of post-hoc source specificity inside one multilingual model.
- Gap: it starts from a flat multilingual dictionary and classifies support afterward; it does not learn or compare a
  universal/pairwise/private composed dictionary or quantify behavior-preserving capacity savings.

## Reliability And Evaluation

### Cross-Seed Feature Correspondence

- Source: [Benchmarking Cross-Seed Feature Correspondence](https://openreview.net/forum?id=5cy6WtSC8f)
- Contribution: evaluates feature matching through ablation fingerprints and direct substitution, finding a
  quality-coverage tradeoff and stronger tail matching from Sinkhorn optimal transport.
- Project consequence: decoder cosine can propose matches between independent domain SAEs, but correspondence claims
  require functional fingerprints and substitution on held-out activations.

### SAEBench And Its Audit

- Sources: [SAEBench](https://arxiv.org/abs/2503.09532) and
  [Are Sparse Autoencoder Benchmarks Reliable?](https://arxiv.org/abs/2605.18229)
- Contribution: SAEBench broadens evaluation beyond reconstruction, while its later audit finds several metrics noisy or
  insufficiently discriminative.
- Project consequence: report direct policy-specific actor/critic fidelity and causal effects rather than importing one
  aggregate SAE score as validation.

### Random Baseline Sanity Checks

- Source: [Sanity Checks for Sparse Autoencoders](https://openreview.net/forum?id=bEYHoD7fCj)
- Finding: several trained SAE architectures recover few synthetic ground-truth features despite good explained
  variance, and constrained-random baselines can match common interpretability, probing, and editing metrics.
- Project consequence: include constrained-random feature directions and activation-pattern controls. Reconstruction and
  named top-activating examples cannot establish meaningful feature recovery.

### Cue Capture Instead Of Mechanism

- Source: [Do Sparse Autoencoders Identify Reasoning Features?](https://openreview.net/forum?id=TCFtA9CI3U)
- Finding: contrastively selected features can be explained by simple distributional cues rather than the claimed
  computation.
- Project consequence: environment-specific padding, position indices, final environment tokens, action masks, sequence
  lengths, and activation norms are likely cue features. Sharedness tests must explicitly falsify these alternatives.

### Synthetic Ground-Truth Evaluation

- Source: [SynthSAEBench](https://openreview.net/forum?id=kALCcAhJa1)
- Contribution: evaluates SAE recovery with known correlated, hierarchical, and superposed features and demonstrates a
  disconnect between reconstruction and latent recovery.
- Project relevance: motivates a controlled domain-support generator with known universal, pairwise, and private factors
  before applying the proposed method to MARL-GPT.

## Research Gap

The literature supplies flat SAEs, hierarchical SAEs, cross-model universal dictionaries, model-exclusive crosscoder
partitions, and post-hoc domain-specificity analyses. It does not yet supply a validated method for decomposing one fixed
generalist policy's activation space over the full lattice of data-domain supports while measuring the minimal sparse
capacity needed to preserve native policy behavior.

The project-specific gap is therefore:

> How can universal, partially shared, and private features be identified inside one multi-environment policy without
> confusing data-distribution cues, arbitrary dictionary correspondence, or reconstructive overlap with functional
> reuse, and what activation code length and dictionary capacity buy a fixed level of native-policy fidelity?

## Adversarial Novelty Verdict

The lattice architecture alone has weak novelty. Dedicated Feature Crosscoders already encode shared and exclusive
partitions, Universal SAEs span heterogeneous sources, Matryoshka SAEs impose nested structure, and RouteSAE supplies a
learned-routing alternative. The defensible contribution is the empirical object: functional domain-support
rate–distortion in one frozen generalist policy, validated through known-support recovery, native-output interventions,
cue controls, and random baselines.

The principal reviewer objections are that the mask imposes the answer, reconstruction is mistaken for function,
domain blocks capture source cues, decoder matches are seed artifacts, and football source sharing is overinterpreted as
tactical sharing. The required answers are flat, independent, simpler-hierarchy, routing, and constrained-random
controls; masked actor and critic replacement; explicit token/padding/length/norm/action-mask probes; five-seed
functional substitution; and a strict no-tactical-transfer claim boundary.

## Proposed Contribution Relative To Prior Work

1. Functional domain-support rate–distortion curves using native actor and critic outputs.
2. A domain-lattice SAE as one tested allocation mechanism, not the novelty claim by itself.
3. Explicit separation of distributional, reconstructive, and functional support.
4. Functional support sets based on per-domain ablation effects.
5. Cross-seed correspondence verified through ablation fingerprints and feature substitution.
6. Synthetic support recovery plus constrained-random baselines before MARL-GPT claims.

The work is a methods contribution only if the lattice decomposition beats a balanced single-mixture SAE and three
independent domain SAEs under matched capacity, sparsity, and functional fidelity. Otherwise it should be reported as a
negative analysis of whether explicit domain structure is needed.

## Links

- [Functional feature accounting question](../questions/2026-07-18-functional-feature-accounting.md)
- [Domain-lattice method validation experiment](../experiments/2026-07-18-domain-lattice-sae-method-validation.md)
- [Domain-lattice direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)

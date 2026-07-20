# Domain-Lattice SAE Method Validation

## Status

Infrastructure implemented and the JZ end-to-end schema smoke completed. No claim-bearing real-data experiment has been
launched. Local configs are available for support recovery, activation collection, dictionary training, and evaluation.

The first real-data pilot is now specified at `layer_03:final`: a balanced pooled TopK SAE with width 2,048, `k=16`,
and seed 0. It uses the `dictionary-learning` TopK implementation and training recipe, local resumable checkpoints,
offline-first W&B, per-domain validation, and held-out feature summaries. The pilot is an infrastructure and training-
health gate, not evidence for the lattice method.

## Question

Can a domain-lattice sparse autoencoder recover known universal, pairwise, and private factors and then provide a more
compact behavior-preserving decomposition of fixed MARL-GPT activations than one balanced mixture SAE or three
independent domain SAEs?

## Hypotheses

### Method hypothesis

A hierarchical dictionary with universal, pairwise, and private blocks will recover factor support more accurately than
post-hoc feature classification from a flat mixture SAE or matching among independent SAEs.

### MARL-GPT hypothesis

MARL-GPT will contain a mixture of environment-private and partially shared functional features. The experiment does not
assume that a universal semantic core exists; a private-only or superficially shared result is equally admissible.

## Stage 0: Synthetic Support Recovery

Generate activations from known sparse factors:

```text
x_d = A_universal s
      + sum_{pair S containing d} A_S p_S
      + A_d u_d
      + noise.
```

Vary:

- factor frequency and magnitude by domain;
- correlated universal and private factors;
- pairwise factors;
- feature hierarchy and splitting;
- activation anisotropy;
- unequal domain sample counts;
- additive superposition and nonlinear or manifold-like deviations.

Primary synthetic metrics:

- precision and recall of recovered support sets;
- decoder-direction recovery up to permutation and sign;
- firing-pattern recovery;
- reconstruction-sparsity frontier;
- stability across seeds;
- robustness to unequal frequencies, scales, and sample counts.

The MARL-GPT stage is blocked until every declared assumption-holding regime satisfies all of these gates over five
seeds: lattice support macro-F1 `>= 0.80`; paired 95% intervals above both flat and independent baselines are strictly
positive; normalized reconstruction error is no more than 5% worse than the best matched-L0 baseline; and support plus
decoder recovery are stable in at least four of five seeds. Failure triggers diagnosis or method simplification, not a
post-hoc threshold change. Stable decoder recovery is operationalized as matched decoder cosine `>= 0.70` for the
initial gate.

## Stage 1: Activation Corpus

Collect balanced activations from the frozen MARL-GPT checkpoint:

- equal numbers of independent trajectories or source files per environment;
- SMACv2, GRF, and POGEMA;
- splits grouped by trajectory or source file;
- common sampling budget and declared token-selection rule;
- activation-norm and sequence-length audits before training.

Candidate locations:

- scalar-token embedding output;
- early, middle, and late shared-transformer residual streams;
- final shared-trunk decision-token representation;
- actor-branch representation;
- critic-branch representation.

Claim-bearing training starts at the middle shared block's final decision position, `layer_03:final`. Cache
`layer_06:final` for schema validation only; expand to it after the first location is stable.

Activation shards are tensor-only `.pt` files loaded with `weights_only=True`. Their JSON manifest and sample metadata
record source, trajectory group, sample index, location, token selector, checkpoint hash, preprocessing identity, and
grouped split. Local resolved configs, hashes, environment versions, statuses, metrics, checkpoints, and artifact paths
are authoritative; W&B is optional.

### Leakage boundary in the native datasets

The native loader reads one source file at a time, randomly permutes its flattened row indices, and constructs an
observation from the selected row plus up to five preceding rows when `history_len=6`. It returns rewards and terminal
flags but historically discarded the source-file path and original row index. A split assigned after collection at the
activation-row or collector-batch level can therefore place overlapping history windows in different splits. In the
multi-agent datasets it can also separate different agents or nearby timesteps from the same underlying episode.

The collector now instruments the native loader without changing the vendored MARL-GPT submodule. Every example records
a stable source-file identifier, original source-row index, target action, and grouping field. The initial safe contract
groups by source file and stratifies the grouped split within each environment. This is deliberately conservative: all
trajectories in one file remain in one split, avoiding reliance on ambiguous terminal-marker reconstruction in flattened
multi-agent arrays. A collection is rejected if an environment has fewer than six source groups or any environment/split
cell would be empty. Finer trajectory grouping is allowed later only when the dataset exports an authoritative episode
identity.

Because the native iterator exhausts one file before advancing, the pilot caps accepted loader rows at 8,192 per source.
Without that cap, a fixed example budget can be dominated by the first large file and never reach enough independent
groups. The cap is applied after the loader's row permutation, and the retained examples preserve their original row
indices.

Batch grouping remains schema-smoke-only. It proves tensor and manifest compatibility but has no scientific train/test
meaning because a later batch can revisit the same file, episode, row, or overlapping history.

Controls for the explicit environment channel:

- natural activations;
- per-domain-centered diagnostic, reported separately rather than substituted for the main condition;
- final-environment-token counterfactual where technically valid;
- padding, position-index, sequence-length, action-mask, and activation-norm probes.

## Stage 2: Methods And Baselines

### Baseline A: Single Balanced Mixture SAE

- One per-example TopK SAE trained on pooled, environment-balanced activations.
- Sweep dictionary width and target sparsity.
- Assign feature support post hoc from domain-conditional activation and ablation statistics.
- This is the naive universal baseline.

### Baseline B: Three Independent Domain SAEs

- One per-example TopK SAE each for SMACv2, GRF, and POGEMA.
- **Total-capacity-matched condition:** widths sum to the competing joint model width.
- **Per-domain-capacity-matched condition:** each SAE receives the full competing width, yielding a `3x`-capacity oracle.
- Candidate matches use greedy cosine and Sinkhorn, then require held-out ablation-fingerprint and substitution validation.
- This is the naive private baseline.

### Proposed: Domain-Lattice SAE

- Universal, three pairwise, and three private blocks.
- Hard domain-eligibility masks.
- Hierarchical reconstruction at universal, universal-plus-pair, and full levels.
- Per-example TopK as the primary sparsity budget.
- Sweep total capacity and block allocations.
- Compare reconstruction-only training with a functionally regularized variant.

### Additional Baselines

- Private-only domain-gated SAE.
- Universal-plus-private SAE without pairwise blocks.
- Standard Matryoshka SAE without domain masks.
- PCA, ICA, random decoder, random encoder, and random activation-pattern controls where applicable.
- Domain-stratified BatchTopK sensitivity variants. Do not use unstratified mixed-domain BatchTopK for the primary claim.

## Functional Fidelity

For an original activation `h` and reconstruction `h_hat`, pass both through the frozen remainder of MARL-GPT and
measure:

- actor KL after applying the native environment action mask;
- selected-action agreement;
- critic distribution KL;
- scalar-value deviation;
- dataset-action negative log-likelihood change where valid.

Do not compare the semantics of action index `i` across environments. Functional fidelity is evaluated within each
environment and aggregated only after normalization.

Train two proposed variants:

1. activation-reconstruction objective only;
2. reconstruction plus differentiable actor/critic fidelity regularization.

This comparison tests whether a functional loss improves meaningful accounting or merely optimizes directly for the
evaluation output.

## Feature-Level Functional Support

For feature `k` in domain `d`, remove its decoded contribution and measure the expected within-domain policy change:

```text
F_kd = E_d[KL(policy(h) || policy(h - D_k z_k))].
```

Define a feature's functional support set using precommitted effect and uncertainty thresholds. Compare targeted
ablation against:

- active random SAE features matched for activation magnitude and frequency;
- inactive random SAE features;
- norm-matched random directions;
- within-domain state shuffles;
- out-of-domain state shuffles.

For cross-seed correspondence, use decoder similarity only for candidate generation. Validate matched features by their
per-domain ablation fingerprints and direct contribution substitution.

## Primary Comparison

Construct functional rate–distortion curves. At each activation code length, total capacity, and sparsity level, report
per-domain:

- normalized reconstruction error;
- actor KL;
- critic KL;
- selected-action agreement;
- dead-feature fraction.
- activation code length.

For each method, find the minimum capacity satisfying all precommitted per-domain fidelity thresholds. Compute reuse
efficiency only at matched thresholds:

```text
reuse_efficiency = 1 - K_lattice / (K_smac + K_grf + K_pogema).
```

## Decision Rules

### Support the proposed method

Only if the lattice:

- recovers synthetic support more accurately than both naive baselines;
- improves the MARL-GPT behavior-preserving capacity frontier;
- produces reproducible functional support assignments;
- survives random, cue, and simpler-hierarchy controls.

### Prefer the single mixture SAE

If it matches the lattice at all fair capacity, sparsity, and fidelity points. Conclude that explicit domain structure is
unnecessary for this model and corpus.

### Prefer the independent SAEs

If they dominate and the lattice finds no stable functionally shared core. Conclude that MARL-GPT's sparse functional
capacity is primarily environment-private.

### Restrict the claim to superficial sharing

If features activate or reconstruct across environments but their causal effects do not reproduce across domains or
seeds.

## Reviewer Objections To Preempt

- The lattice support is imposed rather than discovered.
- Universal blocks capture only environment tokens, padding, scale, or other low-dimensional cues.
- Better policy fidelity follows trivially from training on the policy-fidelity objective.
- Independent SAEs receive unfairly more or less capacity.
- Feature matches are artifacts of decoder cosine or seed choice.
- Linear sparse factors are the wrong ontology for these activations.
- The same causal direction can have unrelated semantics across environments.

The synthetic benchmark, two capacity conventions, reconstruction-only variant, cue controls, multiple seeds, and
functional substitution tests are required responses to these objections.

## JZ Schema-Smoke Result

JZ job `2107050` completed the four-stage workflow on one V100 in 4 minutes 25 seconds at commit `41662ce`. It collected
360 `layer_03:final` activations, split as 240 train, 60 validation, and 60 test examples using 12 synthetic batch groups
per environment; trained a width-512, `k=8` upstream `dictionary-learning` TopK SAE for 50 steps; wrote resumable steps
25 and 50; evaluated; and produced 512 held-out feature summaries. Raw artifacts are under
`results/experiments/2026-07-20-layer03-sae-jz-smoke/` on WORK, with Slurm output under `results/slurm/`.

The aggregate held-out normalized MSE was `0.0657`, explained variance at the final validation checkpoint was `0.795`,
and L0 was exactly 8. The per-domain normalized MSEs were GRF `0.0287`, POGEMA `0.0480`, and SMAC `2.936`; 501 of 512
features were dead on the 60-example held-out split. These numbers establish only that collection, training,
checkpointing, evaluation, and analysis execute. Fifty steps and 60 held-out examples are insufficient for feature
quality, and the poor SMAC fidelity and 97.9% held-out dead-feature fraction explicitly fail any training-health claim.
Seven features fired in all three domains, but they are only apparent-support outputs, not universal features.

The run also confirmed the current leakage blocker: each available environment dataset contains exactly one source file.
A file-grouped train/validation/test split is therefore impossible without more independently generated source files or
an authoritative episode identity. Batch grouping was used only for the non-claim schema smoke. The configured pilot
continues to reject fewer than six source groups per environment and must not be launched on the current three-file
corpus.

Two preceding smoke attempts are retained as diagnostic failures: job `2106780` found that compute nodes lack `git`, so
the launcher now injects the login-node commit into manifests; job `2106813` found the schema-smoke grouping precedence
bug that initially collapsed each environment to its single source-file group. Neither attempt produced scientific
evidence.

## Implementation Boundary

The schema-only cluster smoke is complete; the claim-bearing pilot remains blocked by the corpus-group gate. The reusable
core, six Hydra entrypoints, smoke configs, full gate config, cache schema, manifests, and unit tests now exist. Per-layer
MLP transcoders and bounded attribution graphs begin only
after fixed-layer fidelity and feature stability pass; cross-layer transcoders require a later faithfulness/cost gate.
Continuous trajectory prediction and a five-domain lattice are out of scope.

The native files do not expose a uniform authoritative trajectory identity. The supplied smoke remains a
`batch_schema_smoke`, records `claim_bearing: false`, and cannot satisfy the corpus gate. The pilot instead uses the
conservative source-file grouping described above. Source files must be audited to ensure they are independently
generated units; if a preprocessing pipeline shards one trajectory across files, those shard families require a shared
upstream group identifier before a claim-bearing run.

The real-data pilot workflow is:

- collect: `2026-07-20-layer03-pilot`;
- train: `2026-07-20-layer03-topk-pilot` with `uv run --group sae`;
- post-pilot sweep: `2026-07-20-layer03-topk-sweep`, varying widths `{2048, 4096}`, `k` `{8, 16, 32}`, and seeds
  `{0, 1, 2}` only after the pilot's health checks pass;
- analyze: `2026-07-20-layer03-topk-pilot`;
- compare widths or seeds: `2026-07-20-example` after replacing its second model directory.

The completed JZ end-to-end smoke used the four `2026-07-20-jz-smoke` configs and
`archived/2026-07-20-layer03-sae-smoke-v100.sh`. It collected 12 schema-only batches, trained a width-512 TopK SAE for 50
steps, evaluated the held-out schema split, and wrote feature summaries. It is infrastructure evidence only. The job
read staged datasets and reusable caches from SCRATCH, used JOBSCRATCH for temporary files, and retained results in WORK.

Feature summaries include per-domain firing rates, active magnitudes, stable source-row references, and top examples.
They call cross-domain firing **apparent support**. Universality requires later per-domain policy ablation or substitution;
decoder-cosine one-to-many matches similarly identify candidate splitting only, pending activation and causal validation.

Training logs loss, learning rate, pre-clipping gradient norm, throughput, peak GPU memory, normalized reconstruction error,
explained variance, L0, dead-feature fraction, feature-density quantiles, and per-domain validation metrics. Checkpoints
contain model, optimizer, scheduler, dead-feature counters, balanced-sampler RNG state, normalization, and step, and are
written atomically. Local JSONL and manifests remain authoritative when W&B is offline or unavailable.

Entrypoints and configs are grouped by experiment domain:

- synthetic recovery: `scripts.experiments.sparse_synthetic.support_recovery` with
  `configs/experiments/sparse_synthetic/support_recovery/`;
- MARL-GPT collection, training, evaluation, and feature diagnostics:
  `scripts.experiments.sparse_marl_gpt.{collect_activations,train_dictionary,evaluate_dictionary,analyze_features,compare_features}`
  with matching folders under `configs/experiments/sparse_marl_gpt/`.

## Links

- [Functional feature accounting question](../questions/2026-07-18-functional-feature-accounting.md)
- [Sparse feature accounting literature](../literature/2026-07-18-sparse-feature-accounting.md)
- [Domain-lattice direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)
- [Staged direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)

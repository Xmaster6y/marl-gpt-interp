# Domain-Lattice SAE Method Validation

## Status

Infrastructure implemented, the JZ end-to-end schema smoke completed, and the corrected six-group dataset passed its
structural audit. The six-group core diagnostic completed and failed its dead-feature health check. The 12-group
full-training-mixture view is materializing, with its GPU suite submitted behind the acquisition audit. Neither run is
claim-bearing because upstream episode provenance remains unresolved.

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
a stable source-file identifier, original source-row index, target action, and grouping field. Balanced views use the
audited manifest's `source_group`: multipart `chunk_N_part_M` representatives inherit group `chunk_N`, and the grouped
split is stratified within each environment. This is deliberately conservative: all retained data from one candidate
upstream family remain in one split, avoiding reliance on ambiguous terminal-marker reconstruction in flattened
multi-agent arrays. A collection is rejected if an environment has fewer than the configured source-group count or any
environment/split cell would be empty. Finer trajectory grouping is allowed only when the dataset exports an
authoritative episode identity.

Because the native iterator exhausts one file before advancing, each suite caps accepted loader rows by source and
component. Without those caps, a fixed example budget can be dominated by the first large file and never reach enough
independent groups. The cap is applied after the loader's row permutation, and the retained examples preserve their
original row indices. Dataset manifests now supply the grouping identity: every `chunk_N_part_M` representative maps
back to `chunk_N`, and the collector rejects a missing, ambiguous, unaudited, or incomplete manifest mapping.

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

The schema-only cluster smoke and structural corpus gate are complete. The reusable core, eight Hydra entrypoints, smoke
configs, full gate config, cache schema, manifests, and unit tests now exist. Per-layer MLP transcoders and bounded
attribution graphs begin only
after fixed-layer fidelity and feature stability pass; cross-layer transcoders require a later faithfulness/cost gate.
Continuous trajectory prediction and a five-domain lattice are out of scope.

The native files do not expose a uniform authoritative trajectory identity. The supplied smoke remains a
`batch_schema_smoke`, records `claim_bearing: false`, and cannot satisfy the corpus gate. The new suites consume the
audited dataset source-group mapping described above. This closes known multipart leakage but does not prove different
`chunk_N` groups are independently generated episodes, so the runs remain exploratory pending upstream provenance.

The current real-data workflow is:

- acquisition: `2026-07-20-training-small`, then `2026-07-20-training-pilot` only after the smaller suite passes;
- immediate diagnostic: all five stages use `2026-07-20-layer03-balanced-core-small`;
- first full-mixture run: all five stages use `2026-07-20-layer03-balanced-training-small`;
- later 18-group pilot train: `2026-07-20-layer03-topk-pilot` with `uv run --group sae`;
- post-pilot sweep: `2026-07-20-layer03-topk-sweep`, varying widths `{2048, 4096}`, `k` `{8, 16, 32}`, and seeds
  `{0, 1, 2}` only after the pilot's health checks pass;
- analyze: `2026-07-20-layer03-topk-pilot`;
- compare widths or seeds: `2026-07-20-example` after replacing its second model directory.

The core diagnostic collects exactly 6,144 rows per environment from six audited groups and trains width 1,024,
`k=16` for 2,000 steps. The full-mixture run collects exactly 48,840 rows per environment from 12 audited groups and
trains width 2,048, `k=16` for 10,000 steps. Its 814 batches match one complete pass through every native component:
SMAC/GRF sources contribute 4,070 rows, POGEMA maze sources 3,996, and the POGEMA random source 4,884. This compensates
for the native loader's actual 60-row-per-environment batch and 9:1 POGEMA allocation rather than assuming the requested
192-row global batch survives folder-level integer division. After whole-group split assignment, deterministic
round-robin subsampling equalizes each environment within train, validation, and test separately without dropping a
source group.

The final `audit_sae_suite` stage verifies group-disjoint splits, equal environment rows, complete split coverage,
upstream completion, held-out L0, aggregate and per-domain normalized MSE, and dead-feature fraction. The precommitted
full-mixture health gates are aggregate normalized MSE `<=0.20`, every domain `<=0.30`, held-out dead-feature fraction
`<=0.50`, and L0 within `0.10` of 16. Failure blocks the width/sparsity/seed sweep. The core records the same diagnostics
without treating its single-scenario mixture as a strict gate.

Launch artifact: `to-launch/2026-07-20-layer03-balanced-sae-v100.sh`. It refuses unaudited dataset manifests and runs
collection, training, evaluation, feature analysis, and the suite audit in order with local manifests authoritative and
W&B offline.

JZ job `2111294` was submitted at commit `eff3b2b` for the immediate six-group core diagnostic. The 12-group acquisition
job is `2111291`; its audit must pass before the full-mixture GPU suite is submitted.

Job `2111294` completed the collection gate before failing at training because the offline JZ environment lacked the
optional `wandb` package. The cache contains 18,432 rows: exactly 6,144 per environment, six dataset source groups per
environment, and identical per-environment split cells of 4,096 train, 1,024 validation, and 1,024 test rows. It records
`dataset_source_group` and `claim_bearing: false`. No cache recollection is needed. The launcher now accepts a resume
stage, and W&B absence falls back explicitly to authoritative local JSONL/checkpoint observability unless a config marks
W&B required.

Resume job `2112970` completed training, held-out evaluation, feature analysis, and the suite audit in 1m52s. All split
integrity checks passed. Held-out aggregate normalized MSE was `0.000555`, with GRF `0.0000425`, POGEMA `0.0000672`, and
SMAC `0.0682`; L0 was `15.994`. Feature stability failed: held-out dead-feature fraction was `0.939` (962/1,024 feature
summary rows dead), above the precommitted `0.50` ceiling. Validation showed the same `0.940` dead fraction despite
`0.998` explained variance, so this is not merely a test-sample anomaly.

Only feature 683 met the descriptive three-domain firing threshold. It fired on all SMAC test examples but roughly one
third of GRF and POGEMA examples; mean active magnitude was `24.68` for SMAC, `0.324` for POGEMA, and `0.127` for GRF,
and all top-20 examples were SMAC. It is therefore an environment/scale-cue candidate, not evidence of semantic
universality. No causal intervention has been run.

Authoritative core artifacts on JZ WORK are:

- activation cache: `results/experiments/2026-07-20-layer03-balanced-core-small-cache/`;
- SAE model and checkpoints: `results/experiments/2026-07-20-layer03-balanced-core-small-topk-w1024-k16-seed0/`;
- held-out metrics: `results/experiments/2026-07-20-layer03-balanced-core-small-topk-w1024-k16-seed0-evaluation/`;
- feature rows: `results/experiments/2026-07-20-layer03-balanced-core-small-topk-w1024-k16-seed0-features/`;
- final gate: `results/experiments/2026-07-20-layer03-balanced-core-small-suite-audit/`.

Full-mixture job `2113434` is submitted with `afterok:2111291`. It will start only if the 36-file acquisition and
8,192-row-per-source audit pass. Its strict suite audit will test whether task/scenario diversity repairs held-out feature
collapse; failure blocks the width, sparsity, and seed sweep.

## Next Steps And Gates

1. **Finish acquisition without changing the mixture.** Let job `2111291` complete its size/hash and row audit. If any
   source is below 8,192 rows, replace only that representative within the same task and source group; do not relax the
   common accepted-row budget or remove the task.
2. **Run the already-submitted full-mixture flat SAE.** Job `2113434` must retain 12 groups and equal examples in every
   environment/split cell. Record reconstruction, L0, dead features, apparent support, and top-example provenance. This
   is the decisive training-health result, not a method comparison.
3. **Branch on held-out feature usage.** If dead-feature fraction is `<=0.50` and every domain meets normalized-MSE gates,
   proceed to the width/k/seed stability sweep. If collapse persists, freeze the broad sweep and run a bounded diagnostic
   comparing widths `{512, 2048}` and natural pooled scaling versus a separately reported per-domain-centered/RMS-scaled
   condition. Do not substitute the scaled diagnostic for the natural-activation result.
4. **Add semantic rehydration before naming features.** Reconstruct each selected top example from `source_file`,
   `source_row_index`, scenario, action, reward, and terminal metadata; attach environment-specific state summaries. A
   feature receives a semantic label only when its top and random-active examples distinguish a reproducible concept.
5. **Test splitting and stability.** Generate cross-seed decoder matches only as candidates, then require held-out
   activation fingerprints and matched top-example concepts. One-to-many matches remain candidate feature splits until
   causal fingerprints agree.
6. **Test functional universality.** For high-activity, stable candidates first, run batched per-domain ablation and direct
   contribution substitution through the frozen model remainder. Apparent cross-domain firing alone is never universal
   support.
7. **Close claim gates before lattice comparisons.** Obtain authoritative upstream episode/chunk provenance and complete
   the five-seed synthetic support-recovery gate. Only then launch flat-versus-independent-versus-lattice
   rate--distortion comparisons or make universality/reuse claims.

### Queued Post-Health Suite

The next suite is a mutually exclusive conditional launch rather than an unconditional sweep. Launch artifact
`to-launch/2026-07-20-layer03-post-health-v100.sh` consumes the full-mixture cache and checks the authoritative primary
suite audit before doing any GPU work:

- if the primary suite passes, a two-task array trains the identical natural-activation width-2,048, `k=16` condition
  at seeds 1 and 2; together with primary seed 0 this is the first feature-usage stability check;
- if the primary suite fails only its health gates while every structural check passes, a three-task diagnostic array
  adds natural width 512 and per-domain-centered/RMS-scaled widths 512 and 2,048. The primary natural width-2,048 result
  is the fourth cell, so it is not recomputed;
- if acquisition, collection, or another structural prerequisite fails, neither scientific branch is valid. The
  diagnostic launcher fails closed rather than treating an infrastructure failure as dead-feature evidence.

The per-domain transform is fitted on training rows only and replayed unchanged on validation and test rows. Scaled
reconstruction errors live in transformed coordinates and are not compared numerically with natural-space
reconstruction. The diagnostic comparison is whether centering/scaling and/or lower width materially repair held-out
feature usage at matched `k`; the scaled condition cannot replace the natural corpus in later claims.

The primary job is still pending, so both arrays are submitted behind mutually exclusive Slurm dependencies: stability
uses `afterok:2113434`, while the diagnostic branch uses `afternotok:2113434` and additionally requires a written audit
with all structural checks passing and `health.passed=false`.

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
  `scripts.experiments.sparse_marl_gpt.{collect_activations,train_dictionary,evaluate_dictionary,analyze_features,compare_features,audit_sae_suite}`
  with matching folders under `configs/experiments/sparse_marl_gpt/`.

## Links

- [Functional feature accounting question](../questions/2026-07-18-functional-feature-accounting.md)
- [Sparse feature accounting literature](../literature/2026-07-18-sparse-feature-accounting.md)
- [Domain-lattice direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)
- [Staged direction decision](../decisions/2026-07-18-prioritize-functional-feature-accounting.md)

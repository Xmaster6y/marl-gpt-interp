# Actor and Critic CLT Attribution Graphs

## Status

Implementation complete locally. Claim-bearing corpus collection, training, graph analysis, and interventions are not launched.

## Question and Hypothesis

Can separate full-path actor and critic CLTs faithfully replace MARL-GPT's MLP computations and reveal recurring causal paths across SMACv2, POGEMA, and GRF?

If MARL-GPT reuses computation, matched situations should recruit corresponding sparse paths with similar signed output effects and intervention fingerprints. If parameter sharing hides separate policies, faithful graphs should instead be environment-specific.

## Frozen Method

- One actor CLT and one critic CLT with independent weights.
- Eight encoder layers per CLT: seven shared MLP sites plus the relevant branch MLP.
- Width 1,024 per layer for the initial run; learned JumpReLU thresholds and a decoder-bundle-aware tanh sparsity penalty.
- Natural residual-stream inputs; no environment-conditioned preprocessing or weights.
- Layer-balanced normalized reconstruction loss.
- One combined sharded corpus storing shared tensors once and both branch targets.
- Audited source-group-disjoint train, validation, and test splits. These groups are file or chunk-family provenance units, not reconstructed episode identities.
- Uniform sample of 64 model-visible token positions per input window, always including the final output position.
- Exactly one pass over 259,200 capped source rows (86,400 per environment), yielding 16,588,800 token rows; repeated loader epochs are deduplicated by source-file and source-row identity.
- The float16 tensor payload is approximately 152.9 GB (142.4 GiB) before JSONL metadata and filesystem overhead. The corpus, checkpoints, metrics, graphs, Hydra state, Slurm logs, caches, and per-job temporary files are all written under `$SCRATCH/marl-gpt-interp/`. The Git checkout and software environment remain on `$WORK`; no generated claim-bearing artifact is written there during the suite.

The actor graph targets the selected legal-action logit minus the strongest other legal-action logit. The critic graph targets the local linearization of expected value for the selected action. Attention patterns and normalization denominators are frozen for local attribution; attention OV transport is included and QK pattern formation is not.

## Canonical Workflow

```text
scripts.experiments.circuit_tracing.build_balanced_dataset
scripts.experiments.circuit_tracing.audit_balanced_dataset
scripts.experiments.circuit_tracing.collect_corpus
scripts.experiments.circuit_tracing.train_clt        # actor
scripts.experiments.circuit_tracing.train_clt        # critic
scripts.experiments.circuit_tracing.evaluate_replacement
scripts.experiments.circuit_tracing.audit_clt_suite  # hard eligibility gate
scripts.experiments.circuit_tracing.build_graph      # target-specific
scripts.experiments.circuit_tracing.evaluate_intervention
```

Matching configs live under `configs/experiments/circuit_tracing/`.

## Data Gate

Before training:

- checkpoint and every shard hash verify;
- 18 audited source groups are available per environment;
- no audited source group crosses splits;
- all tensor rows have source, token-role, action-mask, action, and value metadata;
- environment and split counts are reported;
- token-position and role coverage are reported;
- full sequence positions are never called padding-invalid because the model has no attention padding mask.

## CLT and Replacement Gates

Both branches must pass, overall and in each environment:

- every layer normalized MSE at most `0.20`;
- dead-feature fraction at most `0.50`;
- mean per-layer `L0` between `16` and `128`;
- actor masked policy KL at most `0.05` and top-legal-action agreement at least `0.90`;
- critic expected-value MAE at most `1.0` on its `[-20, 20]` support.

Failure blocks graph interpretation. Width or sparsity may be revised once from the initial health report, with the revision recorded before test analysis; this is engineering calibration, not a baseline study.

## Graph and Causal Gates

- Reconstruction-error nodes contribute at most `20%` of absolute target attribution in the median analyzed graph.
- Pruned graphs retain at least `80%` node influence and `98%` edge influence under the declared procedure.
- Every graph stores the exact tensor input used to build it; intervention runs consume that artifact rather than re-sampling a nominal dataset index.
- Across selected source--target feature pairs, predicted and observed intervention magnitudes have Spearman correlation at least `0.50`, and signs agree in at least `75%` of cases.
- A cross-environment motif requires corresponding examples, token roles, signed paths, and intervention fingerprints on held-out source groups.
- Steering is evaluated in fresh paired rollouts against norm- and induced-KL-matched random writes. A positive claim requires the lower paired 95% confidence bound on the task metric to exceed the random control without more than a 5% return regression.

## Decision Rule

- **Shared-mechanism evidence:** both CLTs and causal gates pass, and recurring motifs validate across environments.
- **Partial reuse:** faithful graphs show a mixture of recurring and environment-specific validated motifs.
- **No shared mechanism:** faithful graphs are consistently environment-disjoint in matched situations.
- **Method failure:** replacement or graph faithfulness fails; make no statement about MARL-GPT sharing.

## Artifacts

- [Corpus config](../../configs/experiments/circuit_tracing/collect_corpus/2026-07-22-training-pilot.yaml)
- [Actor CLT config](../../configs/experiments/circuit_tracing/train_clt/2026-07-22-actor-pilot.yaml)
- [Critic CLT config](../../configs/experiments/circuit_tracing/train_clt/2026-07-22-critic-pilot.yaml)
- [Replacement config](../../configs/experiments/circuit_tracing/evaluate_replacement/2026-07-22-pilot.yaml)
- [CLT suite audit](../../configs/experiments/circuit_tracing/audit_clt_suite/2026-07-22-pilot.yaml)
- [Actor graph config](../../configs/experiments/circuit_tracing/build_graph/2026-07-22-actor-example.yaml)
- [Critic graph config](../../configs/experiments/circuit_tracing/build_graph/2026-07-22-critic-example.yaml)
- [Actor intervention config](../../configs/experiments/circuit_tracing/evaluate_intervention/2026-07-22-actor-example.yaml)
- [Critic intervention config](../../configs/experiments/circuit_tracing/evaluate_intervention/2026-07-22-critic-example.yaml)
- [Paper](../../latex/main.tex)

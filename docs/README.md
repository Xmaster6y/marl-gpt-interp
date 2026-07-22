# Project Status

## Snapshot

- Project: `marl-gpt-interp`
- Phase: pooled feature-decomposition normalization diagnosis
- Status: natural pooled SAEs collapsed at widths 512 and 2,048, while per-domain centering/RMS scaling passed health;
  a global-normalization control is the next gate before any sparsity sweep or feature interpretation;
  no sparse feature-sharing, tactical-transfer, TacSIm-transfer, or trajectory-failure claim is currently supported
- Program: [Prioritize functional feature accounting](decisions/2026-07-18-prioritize-functional-feature-accounting.md)
- Paper: [`../latex/`](../latex/)
- Project brief: [`2026-06-30-project-brief.md`](2026-06-30-project-brief.md)
- Setup notes: [`2026-07-02-grf-jz-setup.md`](2026-07-02-grf-jz-setup.md)
- Historical target: [NU Sports ML Workshop 2026](literature/2026-06-30-nu-sports-ml-workshop-2026.md), abstract deadline June 1, 2026

## Index

- Questions: [`questions/`](questions/)
- Claims: [`claims/`](claims/)
- Decisions: [`decisions/`](decisions/)
- Reviews: [`reviews/`](reviews/)
- Literature: [`literature/`](literature/)
- Experiments: [`experiments/`](experiments/)
- Presentations: [`presentations/`](presentations/)
- Raw outputs: [`../results/`](../results/) untracked

## Current Claims

- No feature-sharing claim is currently evidenced. The planned study will distinguish a compact functionally shared core, pairwise sharing, a union of environment-private features, and superficial activation overlap.

## Current Decisions

- [Target the TacSIm benchmark](decisions/2026-07-18-target-tacsim-benchmark.md): retained as the eventual endpoint, conditional on artifacts and baseline reproduction.
- [Prioritize functional feature accounting](decisions/2026-07-18-prioritize-functional-feature-accounting.md): current staged program from synthetic validity through internal and football robustness to TacSIm.
- [Causal cross-football concept transfer](decisions/2026-07-15-prioritize-causal-cross-football-concept-transfer.md) is superseded as the primary direction and retained as possible later feature interpretation.
- [Start with GRF rollout statistics](decisions/2026-06-30-start-with-grf-rollout-statistics.md): use simple GRF statistics as the first reproducible gate before probes, steering, or flank-pass comparison.
- [Stage work from GRF to human gap](decisions/2026-06-30-stage-work-from-grf-to-human-gap.md): run and instrument GRF before human-football modelling-gap claims.
- Treat R2DRL as a later online-bridge candidate, not the first environment target.
- Treat human tracking data first as trajectory and tactical-concept evidence, not as clean discrete action labels.
- Use interpretability as the main contribution; online transfer and alignment are downstream questions.

## Current Infrastructure

- [GRF on JZ setup](2026-07-02-grf-jz-setup.md): login-node preparation path is ready; JZ rollout statistics run now completes.
- [External soccer GRF encoding](experiments/2026-07-15-external-soccer-grf-encoding.md): reusable La Liga and raw STP adapters produce audited GRF `simple115v2` histories and finite MARL-GPT outputs.
- Sparse foundation: tensor-only hashed caches, local run manifests, synthetic recovery, TopK/BatchTopK, flat/independent/lattice/random dictionaries, and functional-fidelity evaluation are implemented.

## Active Questions

- [Beat the TacSIm benchmark](questions/2026-07-18-beat-tacsim-benchmark.md), deferred endpoint
- [Functional feature accounting in MARL-GPT](questions/2026-07-18-functional-feature-accounting.md)
- [Environment representations in MARL-GPT](questions/2026-07-06-environment-representations-in-marl-gpt.md)
- [Cross-environment compute sharing in MARL-GPT](questions/2026-07-06-cross-env-compute-sharing.md)
- [Coordination representations in MARL-GPT](questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [Simulation-human modelling gap](questions/2026-06-30-simulation-human-modelling-gap.md)
- [Interpretability-guided alignment](questions/2026-06-30-interpretability-guided-alignment.md)
- [GRF-MAPE gap](questions/2026-06-30-grf-mape-gap.md)

## Recent Experiments

- [Random frame sampling control](experiments/2026-07-15-random-frame-sampling-control.md): random spaced frames reduce cross-football CKA sharply, while cosine remains near one and is not usable as alignment evidence.
- [Final-token-excluded pooling control](experiments/2026-07-15-final-token-excluded-pooling-control.md): removing the last token directly from mean pooling changes cross-football cosine by less than `0.00006`; non-causal attention means this is not an environment-token ablation.
- [Cross-football representation geometry](experiments/2026-07-15-cross-football-representation-geometry.md): superseded by the random-frame control; its contiguous CKA result was an ordering artifact.
- [External soccer GRF encoding](experiments/2026-07-15-external-soccer-grf-encoding.md): 44 examples per source passed the input and checkpoint smoke; the result is infrastructure evidence only.
- [Fuji soccer data schema inspection and tiny sample](experiments/2026-07-14-fuji-soccer-data-schema-and-sample.md): La Liga and raw STP samples are ready for adapter development; the Fuji RoboCup extractor bug is identified exactly.
- [Pretrained weights smoke test](experiments/2026-06-30-pretrained-weights-smoke-test.md): checkpoint loads and runs a short GRF rollout locally; activation capture pending.

## Planned Analyses

- [Domain-lattice SAE method validation](experiments/2026-07-18-domain-lattice-sae-method-validation.md), paused after the synthetic gate and convergence diagnosis failed
- [Cross-football sparse-feature robustness](experiments/2026-07-19-cross-football-sparse-feature-robustness.md), blocked on the synthetic gate
- [TacSIm benchmark](experiments/2026-07-18-tacsim-benchmark.md), deferred endpoint
- [Causal cross-football pressure transfer](experiments/2026-07-15-causal-cross-football-pressure-transfer.md) is superseded as the primary track.
- [Environment mechanism probes](experiments/2026-07-06-environment-mechanism-probes.md)
- [Cross-environment compute sharing](experiments/2026-07-06-cross-env-compute-sharing.md)
- [GRF rollout statistics](experiments/2026-06-30-grf-rollout-statistics.md)
- [Soccer analytics statistics](experiments/2026-06-30-soccer-analytics-statistics.md)
- [GRF representation probes](experiments/2026-06-30-grf-representation-probes.md)
- [GRF-human gap analysis](experiments/2026-06-30-grf-human-gap-analysis.md)
- [GRF-MAPE gap analysis](experiments/2026-06-30-grf-mape-gap-analysis.md)

# Experiments

- [Templates](templates/)
- [To Launch](to-launch/)
- [Archived](archived/)

## Planned Analyses

- [Balanced offline MARL-GPT corpus](2026-07-20-balanced-offline-corpus.md): Six-group-per-environment JZ view passed the structural 6,144-row balance gate; episode provenance and the 18-group full-training-mixture view gate the SAE pilot.
- [Domain-lattice SAE method validation](2026-07-18-domain-lattice-sae-method-validation.md): JZ schema smoke completed end to end; the claim-bearing pilot is blocked because the current corpus has only one source file per environment.
- [Cross-football sparse-feature robustness](2026-07-19-cross-football-sparse-feature-robustness.md): Blocked on the synthetic gate; separate GRF–La Liga–RoboCup robustness branch with no tactical-transfer claim.
- [TacSIm benchmark](2026-07-18-tacsim-benchmark.md): Deferred endpoint; reproduce official artifacts and a baseline after readiness stages.
- [Causal cross-football pressure transfer](2026-07-15-causal-cross-football-pressure-transfer.md): Superseded as the primary planned track; retained as historical context and possible later feature interpretation.
- [Random frame sampling control](2026-07-15-random-frame-sampling-control.md): Completed; random spaced frames collapse the contiguous CKA result while raw cosine remains near one, identifying temporal ordering and anisotropy as separate confounds.
- [Final-token-excluded pooling control](2026-07-15-final-token-excluded-pooling-control.md): Completed; removing the environment-token position from mean pooling barely changes football cosine and does not deconfound non-causal transformer states.
- [Cross-football representation geometry](2026-07-15-cross-football-representation-geometry.md): Superseded; its high CKA depended on contiguous index ordering, while strong source separation remains.
- [External soccer GRF encoding](2026-07-15-external-soccer-grf-encoding.md): Completed; La Liga and raw STP states encode into exact GRF `simple115v2` histories and run through MARL-GPT, but the bounded result is an infrastructure gate rather than transfer evidence.
- [Fuji soccer data schema inspection and tiny sample](2026-07-14-fuji-soccer-data-schema-and-sample.md): Completed; La Liga and raw STP samples are valid for adapter work, while the Fuji RoboCup arrays exactly reproduce an obsolete eight-field positional stride and must be regenerated from named columns.
- [Environment mechanism probes](2026-07-06-environment-mechanism-probes.md): JZ subset runs completed; wrong-token activations decode true environment after the first transformer block, and parameter gradients separate SMAC from the more aligned POGEMA/GRF pair.
- [Cross-environment compute sharing](2026-07-06-cross-env-compute-sharing.md): JZ small run completed; POGEMA-GRF gradients align strongly, while SMAC is near orthogonal to both and activation CKA is low across all pairs.
- [Internal representation geometry](2026-07-06-internal-representation-geometry.md): JZ small run completed; environments are internally coherent and perfectly nearest-neighbor separated, with partial POGEMA-to-GRF low-rank containment but low CKA across all pairs.
- [GRF rollout statistics](2026-06-30-grf-rollout-statistics.md): JZ small run completed; self-contained GRF rollout path writes expected statistics artifacts.
- [Pretrained weights smoke test](2026-06-30-pretrained-weights-smoke-test.md): Checkpoint loads and runs a 20-step GRF rollout; activation capture pending.
- [GRF representation probes](2026-06-30-grf-representation-probes.md): Planned.
- [GRF-human gap analysis](2026-06-30-grf-human-gap-analysis.md): Planned; blocked until human data access.
- [GRF-MAPE gap analysis](2026-06-30-grf-mape-gap-analysis.md): Optional planned analysis.
- [Cross-dataset soccer statistics](2026-06-30-cross-dataset-soccer-statistics.md): Scaffolded normalization and comparison scripts for human, GRF, and RoboCup 2D data.
- [Soccer analytics statistics](2026-06-30-soccer-analytics-statistics.md): Concept and metric catalogue for GRF probes and simulator-human comparison.

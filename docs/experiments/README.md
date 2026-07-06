# Experiments

- [Templates](templates/)
- [To Launch](to-launch/)
- [Archived](archived/)

## Planned Analyses

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

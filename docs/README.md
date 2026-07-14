# Project Status

## Snapshot

- Project: `marl-gpt-interp`
- Phase: fresh-environment readiness and first GRF statistics analysis
- Status: MARL-GPT GRF smoke path exists locally; next gate is a self-contained config-driven statistics run before probing or steering
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

- Planned claim, not yet evidenced: MARL-GPT may contain interpretable coordination representations that can diagnose the gap between GRF simulator behavior and human football trajectories.

## Current Decisions

- [Start with GRF rollout statistics](decisions/2026-06-30-start-with-grf-rollout-statistics.md): use simple GRF statistics as the first reproducible gate before probes, steering, or flank-pass comparison.
- [Stage work from GRF to human gap](decisions/2026-06-30-stage-work-from-grf-to-human-gap.md): run and instrument GRF before human-football modelling-gap claims.
- Treat R2DRL as a later online-bridge candidate, not the first environment target.
- Treat human tracking data first as trajectory and tactical-concept evidence, not as clean discrete action labels.
- Use interpretability as the main contribution; online transfer and alignment are downstream questions.

## Current Infrastructure

- [GRF on JZ setup](2026-07-02-grf-jz-setup.md): login-node preparation path is ready; JZ rollout statistics run now completes.

## Active Questions

- [Environment representations in MARL-GPT](questions/2026-07-06-environment-representations-in-marl-gpt.md)
- [Cross-environment compute sharing in MARL-GPT](questions/2026-07-06-cross-env-compute-sharing.md)
- [Coordination representations in MARL-GPT](questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [Simulation-human modelling gap](questions/2026-06-30-simulation-human-modelling-gap.md)
- [Interpretability-guided alignment](questions/2026-06-30-interpretability-guided-alignment.md)
- [GRF-MAPE gap](questions/2026-06-30-grf-mape-gap.md)

## Recent Experiments

- [Fuji soccer data schema inspection and tiny sample](experiments/2026-07-14-fuji-soccer-data-schema-and-sample.md): La Liga and raw STP samples are ready for adapter development; the Fuji RoboCup extractor bug is identified exactly.
- [Pretrained weights smoke test](experiments/2026-06-30-pretrained-weights-smoke-test.md): checkpoint loads and runs a short GRF rollout locally; activation capture pending.

## Planned Analyses

- [Environment mechanism probes](experiments/2026-07-06-environment-mechanism-probes.md)
- [Cross-environment compute sharing](experiments/2026-07-06-cross-env-compute-sharing.md)
- [GRF rollout statistics](experiments/2026-06-30-grf-rollout-statistics.md)
- [Soccer analytics statistics](experiments/2026-06-30-soccer-analytics-statistics.md)
- [GRF representation probes](experiments/2026-06-30-grf-representation-probes.md)
- [GRF-human gap analysis](experiments/2026-06-30-grf-human-gap-analysis.md)
- [GRF-MAPE gap analysis](experiments/2026-06-30-grf-mape-gap-analysis.md)

# Project Status

## Snapshot

- Project: `marl-gpt-interp`
- Phase: fresh-environment readiness and first GRF statistics analysis
- Status: MARL-GPT GRF smoke path exists locally; next gate is a self-contained config-driven statistics run before probing or steering
- Paper: [`../latex/`](../latex/)
- Project brief: [`2026-06-30-project-brief.md`](2026-06-30-project-brief.md)
- Roadmap: [`2026-06-30-roadmap.md`](2026-06-30-roadmap.md)
- MARL-GPT architecture visual: [`marl-gpt-architecture.html`](marl-gpt-architecture.html)
- Soccer analytics statistics: [`2026-06-30-soccer-analytics-statistics.md`](2026-06-30-soccer-analytics-statistics.md)
- Workshop abstract draft: [`2026-06-30-workshop-abstract.md`](2026-06-30-workshop-abstract.md)
- Historical target: [NU Sports ML Workshop 2026](literature/2026-06-30-nu-sports-ml-workshop-2026.md), abstract deadline June 1, 2026

## Index

- Questions: [`questions/`](questions/)
- Claims: [`claims/`](claims/)
- Decisions: [`decisions/`](decisions/)
- Reviews: [`reviews/`](reviews/)
- Literature: [`literature/`](literature/)
- Experiments: [`experiments/`](experiments/)
- Raw outputs: [`../results/`](../results/) untracked

## Current Claims

- Planned claim, not yet evidenced: MARL-GPT may contain interpretable coordination representations that can diagnose the gap between GRF simulator behavior and human football trajectories.

## Current Decisions

- [Start with GRF rollout statistics](decisions/2026-06-30-start-with-grf-rollout-statistics.md): use simple GRF statistics as the first reproducible gate before probes, steering, or flank-pass comparison.
- Treat R2DRL as a later online-bridge candidate, not the first environment target.
- Treat human tracking data first as trajectory and tactical-concept evidence, not as clean discrete action labels.
- Use interpretability as the main contribution; online transfer and alignment are downstream questions.

## Active Questions

- [Coordination representations in MARL-GPT](questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [Simulation-human modelling gap](questions/2026-06-30-simulation-human-modelling-gap.md)
- [Interpretability-guided alignment](questions/2026-06-30-interpretability-guided-alignment.md)
- [GRF-MAPE gap](questions/2026-06-30-grf-mape-gap.md)

## Recent Experiments

- [Pretrained weights smoke test](experiments/2026-06-30-pretrained-weights-smoke-test.md): checkpoint loads and runs a short GRF rollout locally; activation capture pending.

## Planned Analyses

- [GRF rollout statistics](experiments/2026-06-30-grf-rollout-statistics.md)
- [GRF representation probes](experiments/2026-06-30-grf-representation-probes.md)
- [GRF-human gap analysis](experiments/2026-06-30-grf-human-gap-analysis.md)
- [GRF-MAPE gap analysis](experiments/2026-06-30-grf-mape-gap-analysis.md)

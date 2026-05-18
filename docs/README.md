# Project Status

## Snapshot

- Project: `marl-gpt-interp`
- Phase: project definition and workshop abstract planning
- Status: research framing established; no code or experiments yet
- Paper: [`../latex/`](../latex/)
- Project brief: [`project-brief.md`](project-brief.md)
- Roadmap: [`roadmap.md`](roadmap.md)
- Workshop abstract draft: [`workshop-abstract.md`](workshop-abstract.md)
- Target: [NU Sports ML Workshop 2026](literature/nu-sports-ml-workshop-2026.md), abstract deadline June 1, 2026

## Index

- Questions: [`questions/`](questions/)
- Literature: [`literature/`](literature/)
- Experiments: [`experiments/`](experiments/)
- Raw outputs: [`../results/`](../results/) untracked

## Current Claims

- Planned claim, not yet evidenced: MARL-GPT may contain interpretable coordination representations that can diagnose the gap between GRF simulator behavior and human football trajectories.

## Current Decisions

- Focus the initial project on GRF because MARL-GPT is trained on it and it is football-specific.
- Treat R2DRL as a later online-bridge candidate, not the first environment target.
- Treat human tracking data first as trajectory and tactical-concept evidence, not as clean discrete action labels.
- Use interpretability as the main contribution; online transfer and alignment are downstream questions.

## Active Questions

- [Coordination representations in MARL-GPT](questions/coordination-representations-in-marl-gpt.md)
- [Simulation-human modelling gap](questions/simulation-human-modelling-gap.md)
- [Interpretability-guided alignment](questions/interpretability-guided-alignment.md)
- [GRF-MAPE gap](questions/grf-mape-gap.md)

## Recent Experiments

- None yet.

## Planned Analyses

- [Pretrained weights smoke test](experiments/pretrained-weights-smoke-test.md)
- [GRF representation probes](experiments/grf-representation-probes.md)
- [GRF-human gap analysis](experiments/grf-human-gap-analysis.md)
- [GRF-MAPE gap analysis](experiments/grf-mape-gap-analysis.md)

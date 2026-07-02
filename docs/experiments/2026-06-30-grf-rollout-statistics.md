# GRF Rollout Statistics

## Status

JZ setup path is ready; rollout completion is still the active gate as of 2026-07-02.

## Question

Can a fresh clone run MARL-GPT on GRF and produce simple, inspectable football statistics before representation probing or steering?

## Hypothesis

A short config-driven rollout should produce stable infrastructure evidence: checkpoint loading, environment stepping, action distributions, reward and score summaries, and basic position-derived statistics when raw GRF observations expose positions.

## Config Or Command

Local smoke:

```bash
just grf-install
uv run -m scripts.run_experiment grf_rollout_stats=2026-06-30-smoke
```

JZ small run:

```bash
just run grf_rollout_stats=2026-06-30-v100-small
```

Launch artifact:

- [`to-launch/2026-06-30-grf-rollout-statistics-v100.sh`](to-launch/2026-06-30-grf-rollout-statistics-v100.sh)

## Metrics

- Episode length, termination, truncation, reward, score, and goal difference.
- Action counts and action entropy.
- Possession fraction when raw observations expose `ball_owned_team`.
- Distance to goal, nearest defender distance, attacking team width and depth, and defensive compactness when raw positions are available.
- Set-piece counters from the GRF wrapper info when present.

## Baseline Or Comparison

No scientific baseline. This is an infrastructure and descriptive-statistics gate.

## Expected Result

The run writes per-step JSONL, per-episode JSONL, aggregate JSON, and aggregate CSV under `results/experiments/2026-06-30-grf-rollout-statistics/`. The output should be enough to decide whether GRF is ready for probes.

## Setup Notes

The durable GRF/JZ setup path is now documented in [GRF on JZ setup](../2026-07-02-grf-jz-setup.md). Raw Slurm logs remain under `results/slurm/`.

## Decision Rule

If the fresh-env run works on local and JZ, move to activation capture and first concept probes. If GRF runtime or checkpoint access fails, document the blocker and use offline checkpoint inspection or dataset batches before claiming any probing result.

## Links

- [Start with GRF rollout statistics](../decisions/2026-06-30-start-with-grf-rollout-statistics.md)
- [Pretrained weights smoke test](2026-06-30-pretrained-weights-smoke-test.md)
- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [GRF on JZ setup](../2026-07-02-grf-jz-setup.md)

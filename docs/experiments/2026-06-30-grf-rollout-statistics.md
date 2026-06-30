# GRF Rollout Statistics

## Status

Planned for 2026-06-30.

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

## Verification Notes

Base unit tests and checks do not require GRF. On the local macOS machine, plain `uv sync --group grf` with Python 3.12 fails while building `gfootball` because CMake finds a different Python/Boost.Python stack. `uv sync --python 3.13 --group grf` installs with uv-managed CPython 3.13.5, but the GRF runtime smoke fails because the compiled extension links against the system Python 3.13 framework. The working local path is `just grf-install`, which uses system Python 3.13 and rebuilds `gfootball`; a direct `gfootball.env.create_environment` reset and one-step smoke test passes. The JZ run should be validated after `secret-env.sh` or the cluster module setup provides matching Python 3.13, CMake, and Boost.Python build dependencies.

## Decision Rule

If the fresh-env run works on local and JZ, move to activation capture and first concept probes. If GRF runtime or checkpoint access fails, document the blocker and use offline checkpoint inspection or dataset batches before claiming any probing result.

## Links

- [Start with GRF rollout statistics](../decisions/2026-06-30-start-with-grf-rollout-statistics.md)
- [Pretrained weights smoke test](2026-06-30-pretrained-weights-smoke-test.md)
- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)

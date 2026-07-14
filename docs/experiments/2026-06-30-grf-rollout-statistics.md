# GRF Rollout Statistics

## Status

JZ small run completed.

## Question

Can a fresh clone run MARL-GPT on GRF and produce simple, inspectable football statistics before representation
probing or steering?

## Hypothesis

A short config-driven rollout should produce stable infrastructure evidence: checkpoint loading, environment stepping,
action distributions, reward and score summaries, and basic position-derived statistics when raw GRF observations
expose positions.

## Config Or Command

Local smoke:

```bash
just grf-install
uv run -m scripts.grf_rollout_stats --config-name 2026-06-30-smoke
```

JZ small run:

```bash
just run grf_rollout_stats 2026-06-30-v100-small
```

Launch artifact:

- [`archived/2026-06-30-grf-rollout-statistics-v100.sh`](archived/2026-06-30-grf-rollout-statistics-v100.sh)

## Metrics

- Episode length, termination, truncation, reward, score, and goal difference.
- Action counts and action entropy.
- Possession fraction when raw observations expose `ball_owned_team`.
- Distance to goal, nearest defender distance, attacking team width and depth, and defensive compactness when raw
  positions are available.
- Set-piece counters from the GRF wrapper info when present.

## Baseline Or Comparison

No scientific baseline. This is an infrastructure and descriptive-statistics gate.

## Expected Result

The run writes per-step JSONL, per-episode JSONL, aggregate JSON, and aggregate CSV under
`results/experiments/2026-06-30-grf-rollout-statistics/`. The output should be enough to decide whether GRF is ready
for probes.

## Setup Notes

The durable GRF/JZ setup path is now documented in [GRF on JZ setup](../2026-07-02-grf-jz-setup.md). Raw Slurm logs
remain under `results/slurm/`.

## Decision Rule

If the fresh-env run works on local and JZ, move to activation capture and first concept probes. If GRF runtime or
checkpoint access fails, document the blocker and use offline checkpoint inspection or dataset batches before claiming
any probing result.

## Result: 2026-07-06 JZ Small Run

Slurm job `1382246` completed with exit code `0:0` in 1 minute 56 seconds on JZ.

Result location: `results/experiments/2026-06-30-grf-rollout-statistics/`.

Run scope:

- Map: `academy_pass_and_shoot_with_keeper`.
- Seeds: `0`, `1`, `2`.
- Episodes per seed: `3`.
- Total episodes: `9`.
- Mean steps: `29.33`.
- Mean total reward: `2.0`.
- Mean possession-left fraction: `0.079`.
- Mean distance to goal: `0.306`.
- Mean nearest-defender distance: `0.093`.
- Action entropy: `3.19`.

Conclusion:

The self-contained JZ GRF rollout path works and writes the expected step, episode, summary CSV, and summary JSON
artifacts. This clears the infrastructure gate for GRF-side activation capture and representation probes.

Limitations:

- Slurm stdout reports GRF episode score lines as `[1, 0]`, but the episode JSONL score fields are all `0-0`. Treat
  the current score fields as a wrapper logging issue until score extraction is audited.
- This is a short sanity run, not a behavioral evaluation.

## Links

- [Start with GRF rollout statistics](../decisions/2026-06-30-start-with-grf-rollout-statistics.md)
- [Pretrained weights smoke test](2026-06-30-pretrained-weights-smoke-test.md)
- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [GRF on JZ setup](../2026-07-02-grf-jz-setup.md)

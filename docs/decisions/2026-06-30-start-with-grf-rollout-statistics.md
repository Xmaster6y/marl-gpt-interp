# Start With GRF Rollout Statistics

## Status

Accepted on 2026-06-30.

## Choice

Use a simple MARL-GPT-on-GRF rollout statistics experiment as the first reproducible experiment gate before representation probes, steering, human-data alignment, or flank-pass comparisons.

## Rationale

The project needs a fresh-environment path that can be cloned to Jean Zay and run after dependencies and the checkpoint are installed. GRF is the primary environment because MARL-GPT is trained on it and it is football-specific. Simple trajectory and action statistics are enough to validate the environment, checkpoint, config system, result writing, and launch templates before higher-risk interpretability work.

## Alternatives

- Start with activation probes: deferred until the rollout path is self-contained.
- Start with flank-pass comparison: deferred because it depends on `../interp-gfootball` and `light_malib`.
- Start with MAPE: deferred because it does not answer the football-specific fresh-env question.

## Consequence

The next experiment should produce per-step, per-episode, and aggregate GRF statistics under `results/experiments/2026-06-30-grf-rollout-statistics/`. Probe and steering claims remain out of scope until this gate works.

## Revisit Condition

Revisit if GRF cannot be installed or run on the target environment after the documented dependency setup, or if the checkpoint cannot be legally or practically accessed on JZ.

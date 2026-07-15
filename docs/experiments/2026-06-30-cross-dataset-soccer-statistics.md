# Cross-Dataset Soccer Statistics

## Status

Provider-neutral statistics remain scaffolded. The external-to-GRF encoding gate completed on 2026-07-15 using bounded
La Liga, raw STP, and native GRF samples.

## Question

Can human football data, GRF rollouts, and RoboCup 2D logs be converted into a shared schema that supports comparable tactical statistics?

## Hypothesis

The first useful comparison should avoid brittle provider-specific action semantics and start with robust counts and geometry: pass counts, shot counts, pass completion, pass length, tracking frames, and an approximate pitch-control distribution.

## Data Or Command

Normalize raw provider exports:

```bash
just run normalize_soccer_data template
```

Compare normalized datasets:

```bash
just run compare_soccer_stats template
```

Default config templates:

- `configs/normalize_soccer_data/template.yaml`
- `configs/compare_soccer_stats/template.yaml`

Outputs:

- `results/normalized/<dataset>/*.jsonl`
- `results/analysis/soccer-comparison/comparison.json`
- `results/analysis/soccer-comparison/comparison.csv`

## Metrics

- Event volume, pass count, completed pass count, pass completion rate, mean pass length, shot count, goal count, carry count.
- Tracking frame count.
- Approximate pitch control: nearest-player grid-cell control fraction by team and contested-cell fraction.

## Baseline Or Comparison

- Human versus GRF.
- Human versus RoboCup 2D.
- GRF versus RoboCup 2D.

## Expected Result

The scripts should expose obvious distribution shifts before any representation work: event frequency differences, pass-length differences, and pitch-control imbalance or field-coverage artifacts caused by simulator dynamics.

## Decision Rule

If the same schema supports at least one small sample from each source, extend the metric set to pressure, support, progression, and phase-of-play labels. If a provider cannot be normalized without lossy assumptions, document the mismatch and keep that provider out of claims requiring direct action-level comparison.

## Links

- [External soccer GRF encoding](2026-07-15-external-soccer-grf-encoding.md)
- [GRF-human gap analysis](2026-06-30-grf-human-gap-analysis.md)
- [Soccer analytics statistics](2026-06-30-soccer-analytics-statistics.md)

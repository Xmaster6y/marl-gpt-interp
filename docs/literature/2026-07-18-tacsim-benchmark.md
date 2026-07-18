# TacSIm Benchmark

## Citation

Peng Wen, Yuting Wang, and Qiurui Wang. [TacSIm: A Dataset and Benchmark for Football Tactical Style
Imitation](https://openaccess.thecvf.com/content/CVPR2026/papers/Wen_TacSIm_A_Dataset_and_Benchmark_for_Football_Tactical_Style_Imitation_CVPR_2026_paper.pdf),
CVPR 2026.

## Benchmark Object

TacSIm reconstructs player and ball trajectories from 2024--2025 Premier League broadcast footage, maps them into a
standard pitch coordinate system compatible with a virtual football environment, and asks models to continue tactical
behavior from observed context.

At test time the paper supplies the first-frame player and ball state and evaluates generated continuations at multiple
spatial resolutions and 3-, 5-, and 10-second horizons. The official benchmark text states that scoring is performed on
the ball trajectory.

## Data

- 140 Premier League matches represented in the source collection;
- 180 hours of source broadcast footage;
- 194,565 retained seconds in the dataset table;
- 38,913 possession segments in the dataset table;
- 70/15/15 match-level train/validation/test split;
- attack, defense, and transition phase annotations;
- broadcast detection, tracking, camera calibration, pitch projection, and off-camera trajectory imputation.

The paper's prose reverses the duration and segment-count quantities relative to its table. Use the released artifact as
the source of truth before fixing experiment sizes.

## Reported Methods

- behavior cloning (BC);
- coordinated multi-agent imitation learning (CMIL);
- inverse reinforcement learning (IRL);
- decentralized adversarial imitation learning with correlated policies (CoDAIL); and
- diffusion-reward adversarial imitation learning (DRAIL).

No single method dominates every reported horizon/resolution cell. The proposed work must compare against the best
reported value under the benchmark's official aggregation rather than selecting one convenient baseline.

## Metrics

TacSIm reports spatial occupancy similarity, movement-vector similarity, and a combined score over several grid sizes
and rollout horizons. The paper calls the combined score a harmonic mean but prints an arithmetic-mean equation. The
released evaluator must determine the claim-bearing definition.

## Project Use

TacSIm is the task and benchmark for the football track. The research goal is a new benchmark state of the art.
MARL-GPT is a candidate method or initialization.

## Links

- [Benchmark question](../questions/2026-07-18-beat-tacsim-benchmark.md)
- [Benchmark decision](../decisions/2026-07-18-target-tacsim-benchmark.md)
- [Benchmark experiment](../experiments/2026-07-18-tacsim-benchmark.md)

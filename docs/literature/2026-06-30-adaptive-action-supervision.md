# Adaptive Action Supervision

## Source

- Paper: [Adaptive action supervision in reinforcement learning from real-world multi-agent demonstrations](https://arxiv.org/abs/2305.13030)
- Authors include Keisuke Fujii and collaborators from the lab connected to the project host.

## Takeaway

The paper frames the real-to-sim problem for multi-agent behavior: human or biological trajectories come from unknown real-world dynamics, while RL happens in a simulator with different dynamics. Direct timestep-by-timestep imitation can therefore be wrong. The proposed solution uses dynamic time warping to align demonstration actions with simulated trajectories before applying action supervision.

## Method Facts

- The method combines Q-learning from demonstrations with supervised action loss.
- Demonstration data is kept in replay and used during pretraining and online training.
- The key adaptation is DTW-based action assignment: a simulated timestep can be supervised by an aligned demonstration timestep rather than the same timestamp.
- The total loss combines Double DQN loss, adaptive action supervision, and regularization.
- Experiments include chase-and-escape and football-like settings.
- The football experiments use professional football tracking and event data from J1 League games.

## Evidence Claimed

- Action-supervised methods improve reward and trajectory similarity compared with pure DQN-style baselines in the reported tasks.
- The football setting demonstrates that real tracking data can guide simulated multi-agent policies despite a source-target dynamics gap.
- The paper explicitly studies the tradeoff between imitation reproducibility and reward-maximizing generalization.

## Limitations For This Project

- The method is built around Q-learning-style models rather than transformer foundation policies.
- The football environments are simplified relative to full football simulators.
- DTW handles timing mismatch but does not fully solve action-semantics mismatch or multi-agent role permutation.
- In some settings, adaptive and non-adaptive action supervision have similar reported outcomes.
- The method does not focus on internal representations or interpretability.

## Project Relevance

This paper provides the strongest framing for human-data alignment. It supports treating human tracking as real-to-sim evidence rather than as clean behavior-cloning labels. The project can reuse the idea of trajectory alignment, but apply it at representation and tactical-concept levels before attempting discrete action supervision.

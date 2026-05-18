# Workshop Abstract

## Target

- Workshop: [NU Sports ML Workshop 2026](literature/nu-sports-ml-workshop-2026.md)
- Deadline: June 1, 2026
- Limit: 500 words
- Status: Draft for planned work

## Candidate Title

Interpreting the Modelling Gap Between MARL Foundation Models and Human Football Play

## Alternate Titles

- Interpreting Football Concepts in MARL-GPT
- Towards Interpretable Alignment of MARL-GPT with Human Football Trajectories
- Probing Coordination Representations in Foundation Models for Multi-Agent Football
- From Simulated Agents to Human Play: Interpreting MARL-GPT in Football Environments

## Draft Abstract

Recent work on MARL-GPT suggests that transformer-based policies trained offline on large-scale multi-agent trajectories can generalize across diverse environments, including Google Research Football. However, high simulator performance does not imply that such models learn coordination structures that transfer to human football play or support policy improvement in realistic tactical settings. This is especially important for sports analytics, where the goal is not only to optimize reward in a simulator, but also to model, evaluate, improve, and transfer decision-making behavior grounded in human play.

We propose to study the modelling and transfer gap between MARL-GPT-style agents, football simulators, and human tracking data through interpretability. Our central question is: what football-relevant coordination concepts are represented inside a pretrained multi-agent transformer policy, and how do these representations affect policy behavior across simulated and human-like phases of play? We focus on concepts such as possession, pressure, spacing, support, passing opportunity, and shot opportunity. We plan to probe internal activations of MARL-GPT for these tactical variables, analyze token-level contributions from teammates, opponents, ball-related features, and temporal context, and perform causal ablations to test whether identified representations affect action selection.

A second goal is to connect these model-level analyses to policy improvement and transfer. Human tracking data provides rich continuous trajectories, while simulator environments provide controllable rollouts, rewards, and environment-specific control interfaces. The practical challenge is therefore not simply missing supervision, but mismatch between human and simulated timing, dynamics, action spaces, and tactical context. We propose to compare human and simulated behavior at the level of trajectory statistics, interpretable tactical concepts, and, where available, aligned actions, controls, or events. Inspired by adaptive action supervision, we will investigate trajectory-alignment methods that match human and simulated phases of play despite timing and dynamics mismatch. This allows us to ask whether simulator-trained agents activate similar tactical representations to human players, where they rely on simulator-specific shortcuts, and which representations are most relevant for transfer.

Ultimately, we aim to use interpretability to identify mechanisms that can support better adaptation, not only post-hoc explanation. By isolating representations associated with human-like spacing, pressure response, support creation, and passing decisions, we can study how concept-level regularization, data selection, representation steering, or offline-to-online adaptation affect both policy behavior and transfer to human football data. This project therefore contributes a framework for interpreting, improving, and transferring foundation-style MARL policies in sports settings, with the long-term goal of bridging reinforcement learning agents and human football behavior.

## Abstract Claims To Keep Conservative

- The work is planned and exploratory.
- The analyses assume access to pretrained MARL-GPT policies and human football data.
- The human-data bridge should use trajectory, concept, and available event/control alignment rather than assume exact simulator-human action correspondence.
- Interpretability is used to understand representations that may support policy improvement and transfer, not only to diagnose model behavior.
- Policy improvement and transfer are central research goals, but specific gains should remain empirical questions until experiments run.

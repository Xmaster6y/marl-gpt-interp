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

Recent work on MARL-GPT suggests that transformer-based policies trained offline on large-scale multi-agent trajectories can generalize across diverse environments, including Google Research Football. However, high simulator performance does not imply that such models learn coordination structures comparable to human play. This is especially important for sports analytics, where the goal is often not only to optimize reward in a simulator, but also to model, evaluate, and understand human tactical behavior.

We propose to study the modelling gap between MARL-GPT-style agents, football simulators, and human tracking data through interpretability. Our central question is: what football-relevant coordination concepts are represented inside a pretrained multi-agent transformer policy, and how do these representations differ between simulated and human trajectories? We focus on concepts such as possession, pressure, spacing, support, passing opportunity, and shot opportunity. Using trajectories from Google Research Football, we plan to probe internal activations of MARL-GPT for these tactical variables, analyze token-level contributions from teammates, opponents, ball-related features, and temporal context, and perform causal ablations to test whether identified representations affect action selection.

A second goal is to connect these model-level analyses to human data. Human tracking data provides rich continuous trajectories but only partially resolves the discrete action labels used by reinforcement learning environments. Rather than assuming direct action-level correspondence, we propose to first compare human and simulated behavior at the level of trajectory statistics and interpretable tactical concepts. Inspired by adaptive action supervision, we will investigate trajectory-alignment methods that match human and simulated phases of play despite timing and dynamics mismatch. This allows us to ask whether simulator-trained agents activate similar tactical representations to human players, and where they rely on simulator-specific shortcuts.

Ultimately, we aim to use interpretability not only as a diagnostic tool, but also as a route toward alignment. By identifying representations associated with human-like spacing, pressure response, and passing decisions, we can explore concept-level regularization, data selection, or representation steering during offline-to-online adaptation. This project therefore contributes a framework for interpreting and aligning foundation-style MARL policies in sports settings, with the long-term goal of bridging reinforcement learning agents and human football behavior.

## Abstract Claims To Keep Conservative

- The work is planned and exploratory.
- MARL-GPT weights and human data access are expected but not assumed complete at submission time.
- The human-data bridge begins with trajectory and concept alignment, not full discrete action supervision.
- Interpretability is the primary contribution; policy improvement is a possible downstream result.

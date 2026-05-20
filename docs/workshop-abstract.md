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

We propose to study this gap by comparing MARL-GPT-style agents, football simulators, and human tracking data through football-relevant tactical concepts. Rather than assuming that simulator performance directly transfers to human play, we will examine whether simulator-trained policies show representations or behaviours related to concepts such as possession, pressure, spacing, support, passing opportunity, and shot opportunity. The project will focus on an initial diagnostic analysis of where simulator-trained agents appear to align with human football behaviour and where they may rely on simulator-specific patterns. The expected outcome is not a complete transfer method, but an initial framework for interpreting the modelling gap between MARL foundation models and human football play. This can provide a basis for later work on policy adaptation, concept-level evaluation, and sports analytics applications.

## Abstract Claims To Keep Conservative

- The work is planned and exploratory.
- Model-access contingencies should stay out of the public abstract unless required for submission framing.
- The human-data bridge should use trajectory, concept, and available event/control alignment rather than assume exact simulator-human action correspondence.
- Diagnostic analysis is the initial deliverable; policy adaptation and concept-level evaluation are later directions.
- Avoid naming specific analyses such as probing, token-level attribution, causal ablations, representation steering, or policy-improvement methods in the public abstract until experiments justify them.

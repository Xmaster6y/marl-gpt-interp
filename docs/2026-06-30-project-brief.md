# Project Brief

## Working Title

Interpreting the Modelling Gap Between MARL Foundation Models and Human Football Play

## Goal

The project studies what MARL-GPT learns about multi-agent coordination in football-like environments and whether interpretability can diagnose or reduce the gap between simulator-trained agents and human tracking data.

## Workshop Target

- Venue: [NU Sports ML Workshop 2026](https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/nu-ml-sports-workshop-2026/home?authuser=0)
- Abstract deadline: June 1, 2026, now historical as of June 30, 2026
- Abstract format: title, authors, and max 500 words
- Fit: agent-based modelling and reinforcement learning, play evaluation, player dynamics, and practical sports analytics

## Core Question

What football-relevant coordination concepts are represented inside a pretrained multi-agent transformer policy, and how do those representations differ between simulated GRF trajectories and human football trajectories?

## Proposed Contribution

The contribution is interpretability-first. The project should not claim to build a new foundation model unless later evidence justifies that claim.

Current contribution target:

- Interpret MARL-GPT football representations using probes, token attribution, and causal ablations.
- Measure the modelling gap between simulator rollouts and human tracking data at trajectory, concept, and representation levels.
- Explore whether concept-level interpretability can guide alignment between offline pretraining, human data, and online adaptation.

## Why MARL-GPT

MARL-GPT is trained on GRF and exposes structured tokens with attribute, agent, team, and time embeddings. This makes it a useful target for interpretability because many football concepts can be tied back to input structure: ball features, teammates, opponents, history, action masks, and predicted actions or Q-values.

## Human Data Position

Human tracking data is central, but direct action supervision is risky because tracking trajectories do not always provide clean discrete actions such as pass, shot, or dribble. The first planned use of human data is therefore trajectory-level and concept-level comparison. Action-level alignment is a later goal when event labels or robust pseudo-actions are available.

## Environment Scope

GRF is the primary environment because MARL-GPT is trained on it and it is football-specific. MAPE or another simple multi-agent environment may be used as a controlled secondary setting for studying simulator gap and transfer in a simpler domain. R2DRL is a possible later online bridge, but it has higher engineering risk and should be gated by a smoke test.

## Candidate Concepts

- Ball possession
- Distance and angle to goal
- Nearest defender pressure
- Passing lane openness
- Teammate support
- Defensive compactness
- Role or phase of play
- Action affordances such as pass, shot, dribble, and movement

## Key Risks

- The MARL-GPT checkpoint loads locally, but this repository still needs a self-contained fresh-environment setup for GRF and JZ.
- Human tracking data may lack direct action labels or may only provide event labels for sparse moments.
- GRF actions and human football events are not semantically identical.
- Attention maps alone are not sufficient evidence for interpretability; causal tests and representation probes are needed.
- Flank-pass comparison currently depends on `../interp-gfootball` and `light_malib`, so it is deferred until the GRF-only path is reproducible.
- R2DRL integration may consume engineering time before it produces research evidence.

## Near-Term Milestones

- First: run a config-driven GRF rollout statistics experiment locally and on JZ.
- Then: add activation capture to the MARL-GPT inference path.
- After activation capture: define simulator-derived concept labels and run first probes.
- After human data access: compare human and simulator trajectories at the concept level before attempting action-level supervision.

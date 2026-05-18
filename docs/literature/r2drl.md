# R2DRL

## Source

- Repository: [open-starlab/R2DRL](https://github.com/open-starlab/R2DRL)

## Takeaway

R2DRL is a reinforcement-learning interface for RoboCup 2D Soccer Simulation. It is a possible future online bridge for soccer-like MARL, but it is not the first environment target because MARL-GPT is already trained on GRF and R2DRL has substantial integration cost.

## Method And Interface Facts

- The repository wraps or modifies the RoboCup 2D stack: `rcssserver`, `rcssmonitor`, `librcsc`, and `helios-base`.
- It exposes a Python environment through `Robocup2dEnv`.
- The example interface includes `reset`, `get_obs`, `get_state`, `get_avail_actions`, and `step`.
- The documented action example samples from 17 discrete actions.
- The environment supports configurable team sizes, including reduced-agent settings such as 3v3.

## Limitations For This Project

- Setup requires multiple C++ dependencies and modified external components.
- The fetched documentation does not provide a complete benchmark pipeline or pretrained policies.
- It is unclear whether its observations map cleanly to MARL-GPT tokenization without custom work.
- It is not yet necessary for the June workshop abstract.

## Project Relevance

R2DRL can become a stronger online soccer simulator after the initial GRF interpretation work. Before committing, the project needs a smoke test that verifies reset, rollout, observation schema, action semantics, action masks, and reduced-team configuration.

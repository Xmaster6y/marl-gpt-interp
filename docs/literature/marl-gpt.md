# MARL-GPT

## Source

- Paper: [MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.05943)
- Code and local copy: [`../../marl-gpt/`](../../marl-gpt/)
- Weights and datasets are announced through the MARL-GPT repository and Hugging Face links in its README.

## Takeaway

MARL-GPT argues that one structured transformer policy can be trained offline on large multi-agent expert trajectories and perform across SMACv2, GRF, and POGEMA. For this project, its most important property is not only performance, but the structured tokenization that creates interpretable handles for agent, team, attribute, and time.

## Method Facts

- Offline data comes from expert policies or planners: IPPO experts for SMACv2 and GRF, RHCR for POGEMA.
- Reported data scale is large: hundreds of millions to billions of samples across domains.
- Observations are represented as scalar tokens enriched with learned embeddings for attribute, agent index, team or group, and timestep.
- The model uses a non-causal transformer observation encoder with actor and critic heads.
- The critic is distributional and trained with TD and conservative regularization losses.
- The actor combines behavior cloning and advantage-weighted actor losses.
- Different action spaces are handled with a shared output layer and environment-specific action masks.
- Online adaptation is handled by fine-tuning in environments such as GRF with PPO-style training.

## Evidence Claimed

- The model is evaluated on SMACv2, GRF, and POGEMA.
- It performs competitively against single-domain offline baselines.
- It shows limited but useful within-domain generalization to unseen tasks when related training data are available.
- Online fine-tuning improves performance and can adapt faster than training from scratch in the evaluated setup.

## Limitations For This Project

- The paper does not explain which coordination structures are learned internally.
- Human demonstrations are not the main data source.
- Environment-specific positional encodings and action masks remain necessary.
- Action semantics are not aligned across environments.
- Generalization to genuinely new environments remains limited.
- Full-scale training is expensive and not needed for the first project phase.

## Project Relevance

MARL-GPT is the main model substrate. The project will inspect whether GRF-trained MARL-GPT representations encode football concepts such as possession, pressure, spacing, support, and pass or shot affordances. These analyses connect to active questions on [coordination representations](../questions/coordination-representations-in-marl-gpt.md), the [simulation-human modelling gap](../questions/simulation-human-modelling-gap.md), and [interpretability-guided alignment](../questions/interpretability-guided-alignment.md).

# Cross-Layer Transcoders and Attribution Graphs

## Primary Sources

- Ameisen et al., [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), 2025.
- Lindsey et al., [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html), 2024.
- Hanna et al., [circuit-tracer](https://github.com/decoderesearch/circuit-tracer), open-source attribution and intervention tooling.
- Nesterova et al., [MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.05943), 2026.

## Takeaways

A CLT divides sparse features across model layers. A feature reads the residual stream at its encoder layer and has separate decoder writes to every subsequent MLP output. All layers are trained jointly. Replacing MLPs with these writes exposes direct feature interactions and shortens paths relative to repeated per-layer representations.

For one input, circuit tracing adds the observed MLP reconstruction error back at every token and layer, then freezes attention patterns and normalization denominators. The resulting local model matches the reference output exactly and is linear between sparse feature nonlinearities. Attribution edges include residual and attention-OV paths. They do not explain QK attention-pattern formation.

The local graph remains an indirect explanation of the original model. Reconstruction errors, off-distribution global replacement, feature splitting/absorption, and changed responses under intervention are central limitations. Mechanistic claims require perturbations in the original model.

## MARL-GPT Adaptation

The public circuit-tracer implementation targets supported language-model backends. MARL-GPT needs a native implementation because it has structured scalar tokens, non-causal attention, no padding attention mask, a legal-action mask, an actor--critic fork, and a categorical action-value head.

The actor output should be a legal-action contrast. The critic's expected value is nonlinear in categorical logits, so its graph target is a local expected-value linearization. Two independent branch-complete CLTs avoid forcing a shared actor/critic feature basis.

## Project Boundary

Sparse autoencoders remain relevant related work but are not the primary method. Cross-environment sharing is established by recurring validated graph paths, not by feature activation support alone. TacSIm is a later external steering endpoint and does not share MARL-GPT's discrete-action output contract.

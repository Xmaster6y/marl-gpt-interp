# Environment Representations In MARL-GPT

## Status

Active.

## Question

Does a MARL-GPT model trained across GRF, SMAC, and POGEMA contain identifiable environment-specific representations, directions, or parameters that explain which environment it is acting in?

## Motivation

Before interpreting football-specific coordination structure, the project needs to know whether the model separates environments at all. If the model has a natural GRF direction, SMAC direction, or POGEMA direction, then football probes may partly measure environment identity rather than tactical concepts. If environment-specific parameters or subnetworks exist, they may explain where later concept probes, attribution analyses, or interventions should be applied.

## Subquestions

- Can hidden states, token activations, or summary embeddings predict the source environment: GRF, SMAC, or POGEMA?
- If the explicit environment token is randomly changed, do activations still recover the true source environment, the prompted environment token, or both?
- Are there natural per-environment directions in representation space, such as linear classifier normals, mean-difference vectors, sparse directions, or principal components aligned with environment identity?
- Are there environment-localized parameters, attention heads, MLP channels, or token positions whose attribution is concentrated on one environment?
- How much of the model's effective computation is shared across environments versus specialized to one environment?
- Does removing, projecting out, or steering along an environment direction change behavior in the expected environment-specific way?
- Do environment directions explain away apparent tactical or coordination probes, or are concept directions separable from environment identity?

## Assumptions

- Comparable trajectories, observations, model activations, and action outputs can be collected for GRF, SMAC, and POGEMA.
- The model architecture exposes enough shared representation to compare activations across environments.
- Environment labels are known, but the scientifically useful result is not just classification; it is localization, separability, and behavioral relevance.
- Debug experiments should start with counterfactual environment-token swaps and non-causal probes, then move to attribution, ablation, or activation patching only if the discovered directions are stable.

## Expected Evidence

- Environment labels are decodable above simple baselines from particular layers, tokens, or pooled activations.
- Under wrong-token or token-sweep conditions, activations reveal whether the model tracks true environment, prompted environment, or both.
- Decoding performance localizes to interpretable parts of the model rather than being uniformly available everywhere.
- Candidate directions are stable across seeds, rollout batches, and train/test splits.
- Attribution identifies heads, MLP channels, token positions, or parameters with environment-concentrated influence.
- Ablation or patching separates shared components from environment-specific components by effect on action logits, values, entropy, and selected actions.
- Interventions on candidate directions change environment-relevant logits, values, or action choices without globally destroying behavior.
- Football-specific concept probes remain meaningful after controlling for environment identity, or are explicitly downgraded if they collapse.

## Metrics

- Environment classification accuracy, balanced accuracy, macro-F1, and AUROC for one-vs-rest probes.
- Probe sparsity, layerwise localization, and cross-batch stability of direction vectors.
- Attribution or ablation effect sizes by layer, head, MLP channel, token type, and environment.
- True-environment versus prompted-environment probe accuracy under environment-token swaps.
- Overlap of important heads, channels, token groups, and activation subspaces across environments.
- Change in action logits, value estimates, entropy, selected action, rollout return, or task-specific behavior after direction removal or steering.
- Residual concept-probe performance after adding environment controls or projecting out environment directions.

## Debug Experiment Sketches

- Collect matched activation caches from GRF, SMAC, and POGEMA examples using the same checkpoint and logging interface.
- Run correct-token, wrong-token, and all-token-sweep passes to test true-environment recovery under counterfactual environment tokens.
- Train simple probes for both true environment and prompted environment on layerwise hidden states, pooled sequence representations, and candidate token groups.
- Compare natural directions from probe weights, class-mean differences, PCA, and sparse probes.
- Run activation clustering, attribution, patching, and ablation over heads, MLP activations, token types, and parameters to estimate shared versus environment-specific effective computation.
- Test whether projecting out or steering along candidate directions changes environment-specific behavior while preserving basic action validity.
- Re-run tactical or coordination probes with environment-label controls to distinguish concept evidence from environment identification.

## Reviewer Objection

A reviewer may argue that environment identity is a trivial nuisance variable: GRF, SMAC, and POGEMA have different observations, actions, maps, and reward structures, so any strong environment probe may only recover dataset format. The useful contribution requires showing where environment identity is represented, whether it has behavioral relevance, and whether it confounds or clarifies later coordination claims.

## Decision Rule

If environment identity is decodable, localized, stable, and behaviorally relevant, use environment directions and attribution as the first debug layer before football-specific probes. If identity is decodable but diffuse or purely format-driven, treat it as a control variable and avoid claiming meaningful environment structure. If identity is weakly decodable, prioritize concept probes and rollout statistics, but still report environment controls when comparing GRF to other settings.

## Links

- [Environment mechanism probes](../experiments/2026-07-06-environment-mechanism-probes.md)
- [Coordination representations in MARL-GPT](2026-06-30-coordination-representations-in-marl-gpt.md)
- [GRF representation probes](../experiments/2026-06-30-grf-representation-probes.md)

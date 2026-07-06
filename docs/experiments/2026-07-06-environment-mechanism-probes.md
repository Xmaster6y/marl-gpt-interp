# Environment Mechanism Probes

## Status

Initial JZ subset runs completed.

## Question

How does MARL-GPT represent and use environment identity across GRF, SMAC, and POGEMA?

Can the model recover the true environment from observations and activations when the explicit environment token is counterfactually changed?

Is model computation shared across environments, or are there identifiable environment-specific activations, heads, MLP channels, or parameters?

## Hypothesis

The explicit `env_specific` token is a control signal, not the whole story. If the observation stream contains enough environment-specific structure, then hidden activations should still encode true environment identity even when the final environment token is randomly changed. If the model has environment-specialized computation, that specialization should appear as reproducible differences in activation usage, attribution, routing, or ablation sensitivity across GRF, SMAC, and POGEMA.

## Data

Use a small balanced subset of [`nortem/marl-gpt-datasets`](https://huggingface.co/datasets/nortem/marl-gpt-datasets), sampled through the MARL-GPT dataset loaders where possible.

Initial split:

- Environments: GRF, SMAC, POGEMA.
- Train/validation/test: split by source file or trajectory, not by individual timestep, to avoid near-duplicate leakage.
- Balance: equal number of examples per environment for probe training and evaluation.
- Size: start small enough for local CPU/GPU iteration, then scale only after the schema and caching path are verified.

The first step is a dataset-inspection pass that records actual file names, tensor keys, tensor shapes, action-space sizes, available scenario labels, and whether each source exposes trajectory or file identifiers needed for non-leaky splits.

## Environment Information Channels

MARL-GPT does not receive only a flat observation. In the local code path, each token is built from several additive components:

- Observation value: `obs`, embedded by the scalar token embedding.
- Agent position: `agent_pos`, embedded by an agent-position table.
- Team or group position: `group_pos`, embedded by a team/group-position table.
- History/time position: `time_pos`, embedded by a time-position table.
- Attribute position: `attr_pos`, embedded by an attribute-position table.
- Action availability: `action_mask`, used by the actor and also available as an input feature in some environment loaders.
- Final token: when `last_token` is enabled, the loader sets the last token's `attr_pos` to a special final-token attribute; when `env_specific` is enabled, it also sets that token's `obs` value to an environment indicator.

The current dataset-loader convention assigns environment indicators as SMAC = `1`, POGEMA = `2`, and GRF = `3` when `env_specific` is enabled. This is not a bug for the actual model path; it is one of the possible mechanisms by which the model can condition behavior on the environment.

The experiment should therefore measure which channel carries environment information instead of removing the final token from the main condition.

## Experiment 1: Counterfactual Environment Token

For each example, run the model under multiple final-token conditions:

- Correct token: final `env_specific` token matches the true environment.
- Random wrong token: final token is sampled from one of the other environments.
- All-token sweep: the same observation is evaluated once with each environment token.

Probe hidden activations to predict two labels:

- True environment: where the trajectory actually came from.
- Prompted environment: which final token the model was given.

This is the cleanest first mechanism test. If early or middle layers recover true environment while the final-token position encodes prompted environment, then the model carries both observation-derived and token-derived environment information. If all layers only recover the prompted token, then the model may be mostly using explicit conditioning. If all layers recover true environment despite the wrong token, then the observation stream dominates environment identification.

Primary outputs:

- Layerwise probe accuracy for true environment and prompted environment.
- Confusion matrices under correct-token and wrong-token conditions.
- Representation shift from changing only the environment token.
- Action-logit, value, entropy, and selected-action changes under token swaps.

## Experiment 2: Activation And Parameter Partitioning

Measure whether environments use different parts of the model.

Candidate analyses:

- Activation clustering by environment at each layer and token group.
- Per-head and per-MLP-channel activation statistics by environment.
- Linear probe weights or sparse probe features mapped back to layers, heads, channels, and token positions.
- Gradient attribution or activation patching from the environment label or action logits back to token groups and layers.
- Unit/head ablations ranked by environment-specific effect on logits or values.

This should start with activations and ablations rather than parameter attribution alone. Parameter-level claims are harder because the same weights can implement shared computation with environment-specific activation patterns. A parameter is only meaningfully environment-specific if changing, ablating, or patching its associated activation path affects one environment much more than the others.

Primary outputs:

- Environment specialization score per layer, head, MLP channel, and token group.
- Shared-versus-specific ranking: units that matter for all environments versus one environment.
- Stability of specialized components across random samples and held-out scenario families.
- Behavioral effect of ablating top environment-specific components.

## Experiment 3: Shared Compute Estimate

The right question is not whether a parameter belongs exclusively to one environment, but how much of the model's effective computation is shared.

Operational definitions:

- Shared component: high activation, attribution, or ablation effect across all three environments.
- Environment-specific component: high effect for one environment and low effect for the others.
- Partially shared component: high effect for two environments or for a common structural role such as spatial coordination.

Possible metrics:

- Activation overlap: cosine similarity or CKA between environment-conditioned activation subspaces.
- Attribution overlap: overlap of top-k heads, channels, layers, or token positions by effect size.
- Ablation overlap: correlation of ablation effects across environments.
- Probe subspace overlap: principal angles between environment probe directions.
- Compute-share summary: fraction of total positive attribution or ablation effect assigned to shared, partially shared, and environment-specific groups.

This is exploratory; it should not claim literal FLOP routing unless the implementation exposes sparse routing or conditional execution. For a dense transformer, every token nominally uses every layer, so "shared compute" means shared effective computation measured through activation and causal effect, not fewer operations.

## Baseline Or Comparison

Controls:

- Probe trained on raw observation features before MARL-GPT encoding.
- Probe trained only on final-token features.
- Probe trained only on positional channels.
- Probe trained only on action masks.

Raw-observation variants:

- Flattened padded observation tensors from the loader: `obs`, and where present `obs_own`, `obs_ally`, `obs_enemy`, `action_mask`, and positional fields.
- Observation-value-only probes.
- Positional-channel-only probes using `group_pos`, `agent_pos`, `time_pos`, and `attr_pos`.
- Action-mask-only probes, because action-space size and legal-action patterns may identify the environment.
- Combined input-channel probes that use the same full input dictionary the model receives.
- Raw observations with a small MLP baseline, to distinguish linear separability from generic nonlinear separability.

Activation comparisons:

- Token embedding output.
- Layerwise residual stream or hidden states.
- Pooled sequence representation.
- Candidate token groups: observation tokens, history/time tokens, and final token if present.
- Optional head or MLP-channel features after the first layerwise pass works.

## Metrics

- Environment classification balanced accuracy, macro-F1, and one-vs-rest AUROC.
- Per-environment confusion matrix.
- Gap between activation-probe performance and each input-channel probe.
- Layerwise localization: best layer, earliest layer above baseline, and performance curve across layers.
- Channel attribution: how much performance is explained by observation values, positional channels, action masks, final token, and learned hidden states.
- Token-counterfactual sensitivity: change in true-env and prompted-env probe accuracy when the final token is swapped.
- Behavioral token sensitivity: change in action logits, values, entropy, and selected action after token swaps.
- Shared-compute score: overlap of important heads, channels, token groups, and activation subspaces across environments.
- Environment-specific effect size: change in outputs after ablating or patching components selected for one environment.
- Stability of probe directions across random seeds, sampled files, and held-out scenarios.
- Probe simplicity: linear probe first; sparse or low-rank probe only after the basic baseline is established.

## Expected Result

The full-input baseline will probably classify environments well because GRF, SMAC, and POGEMA use different observation formats, positional layouts, action masks, final-token settings, and task geometry. The useful result is not high absolute environment-probe accuracy. The useful result is the decomposition: whether activations track the true environment, the prompted environment token, or both; whether token swaps change behavior; and whether the same model components carry computation for all environments or specialize by environment.

## Minimal Procedure

1. Inspect the Hugging Face dataset subset and record actual files, tensor keys, shapes, and loader compatibility.
2. Sample matched examples from GRF, SMAC, and POGEMA with trajectory/file-level train/validation/test splits.
3. Build input-channel feature tables for the same examples: observation values, positional indices, action masks, final-token features, and full model input.
4. Cache activations for correct-token, random-wrong-token, and all-token-sweep conditions.
5. Train regularized multinomial logistic-regression probes for true environment and prompted environment.
6. Compare true-env and prompted-env decodability across layers, token groups, and token-swap conditions.
7. Measure action-logit, value, entropy, and selected-action changes under token swaps.
8. If token-swap results are stable, run activation clustering, sparse probes, attribution, and targeted ablations to estimate shared versus environment-specific computation.

## Launch

Initial JZ run:

```bash
uv run -m scripts.run_experiment env_mechanism_probes=2026-07-06-jz-small
```

Cluster launch artifact:

- Config: [`../../configs/env_mechanism_probes/2026-07-06-jz-small.yaml`](../../configs/env_mechanism_probes/2026-07-06-jz-small.yaml)
- Slurm script: [`archived/2026-07-06-environment-mechanism-probes-v100.sh`](archived/2026-07-06-environment-mechanism-probes-v100.sh)
- Expected results: `results/experiments/2026-07-06-environment-mechanism-probes/`

This initial run records dataset/file schemas, trains input-channel probes, caches layerwise pooled activations under correct, wrong, and all-token-sweep environment prompts, trains true-environment and prompted-environment linear probes on those activations, and writes token-swap behavior summaries.

## Result: 2026-07-06 JZ Small Subset

Slurm job `1377379` completed with exit code `0:0` in 1 minute 58 seconds on JZ.

Result location: `results/experiments/2026-07-06-environment-mechanism-probes/`.

Run scope:

- Environments: SMAC zerg 5v5, POGEMA random, GRF academy corner.
- Input examples: 480.
- Activation examples: 2,400, from correct-token, wrong-token, and all-token prompt sweeps.
- Probe rows: 45.

Main findings:

- Environment identity is trivially available in the input. Linear probes reached about 0.99 to 1.00 accuracy from observation values, action masks, positional channels, the final token, and full input features.
- True environment is perfectly decodable from mean pooled embedding and all later layer activations in this subset. The raw embedded final-token position alone is near chance for true environment, but becomes perfectly true-environment decodable after the first transformer block.
- Prompted environment is perfectly decodable from the final-token representation at every layer, including the embedding. Mean-pooled representations only partially encode prompted environment and decline from about 0.81 at layer 0 to about 0.48 by layer 6.
- Counterfactual environment tokens have a behavioral effect. Random wrong-token prompts changed the selected action for about 28.5% of examples, with mean absolute action-logit shift about 3.61. Prompting all examples as POGEMA or GRF produced similar selected-action change rates; prompting all examples as SMAC shifted logits less and did not change selected actions in this run.

Interpretation:

This supports the "both channels" mechanism for the subset: hidden states retain true environment information from observation/action/position structure while the final token retains the prompted environment. Token swaps are behaviorally relevant, but the current result should not be treated as a general environment-specialization claim because environment identity is already linearly separable from shallow input channels.

Limitations:

- This is a small file-level subset chosen for JZ feasibility, not the full MARL-GPT dataset.
- The split is random over sampled examples, not a held-out file or trajectory split.
- The run demonstrates environment identity and token sensitivity, not stable head/channel specialization or shared-compute estimates.

## Result: 2026-07-06 Wrong-Token True-Label Probes

Slurm job `1379511` completed with exit code `0:0` in 2 minutes 3 seconds on JZ.

Result location: `results/experiments/2026-07-06-environment-mechanism-probes/`.

This run trained activation probes only on activations from the random-wrong-token condition while keeping the target label as the true source environment. It kept the same 480 sampled input examples and produced 480 wrong-token activation examples for probe training and evaluation.

Main findings:

- True environment remains linearly decodable from wrong-token activations.
- Mean-pooled embedding activations reached 1.00 accuracy, showing that the observation/action/position stream already identifies the source environment before transformer blocks mix information into the final-token position.
- The corrupted final-token embedding itself was near chance for the true source environment, with 0.333 accuracy.
- After the first transformer block, both mean-pooled and final-token activations reached 1.00 accuracy for true environment across nearly every layer and head-specific branch summary. The only non-perfect row was `layer_06:mean` at 0.993 accuracy.

Wrong-token true-label probe accuracy:

| Feature | Accuracy |
| --- | ---: |
| `embed:mean` | 1.000 |
| `embed:final` | 0.333 |
| `layer_00:mean` | 1.000 |
| `layer_00:final` | 1.000 |
| `layer_01:mean` | 1.000 |
| `layer_01:final` | 1.000 |
| `layer_02:mean` | 1.000 |
| `layer_02:final` | 1.000 |
| `layer_03:mean` | 1.000 |
| `layer_03:final` | 1.000 |
| `layer_04:mean` | 1.000 |
| `layer_04:final` | 1.000 |
| `layer_05:mean` | 1.000 |
| `layer_05:final` | 1.000 |
| `layer_06:mean` | 0.993 |
| `layer_06:final` | 1.000 |
| `critic_layer:mean` | 1.000 |
| `critic_layer:final` | 1.000 |
| `actor_layer:mean` | 1.000 |
| `actor_layer:final` | 1.000 |

Interpretation:

This strengthens the observation-derived environment channel result. Corrupting the explicit final environment token does not prevent recovery of the true environment from hidden states. The final-token position does not know the true environment at the embedding stage when the token is wrong, but it recovers the true label after one transformer block, presumably by attending to or mixing with environment-identifying observation, action-mask, and positional tokens.

This still should not be treated as a mechanistic localization claim. Because input-channel probes are already near perfect, this run establishes robustness of true-environment decodability under token corruption, not the existence of dedicated environment-specific heads, channels, or parameters.

## Planned Run: 2026-07-06 Direction And Parameter Probes

Question:

Which activation directions and parameter groups differ most between GRF, SMAC, and POGEMA observations under the wrong-token condition?

Hypothesis:

If true-environment identity is observation-derived rather than only copied from the explicit environment token, then class-mean activation directions and linear-probe directions should remain separated under wrong-token prompts. If model computation is partly environment-specialized, gradient sensitivity under the model's action/critic loss should show environment-dependent magnitudes or low gradient cosine in some parameter groups. If gradient directions are highly aligned across environments, the safer conclusion is shared effective computation with environment-identifying activations, not parameter specialization.

Config and launch:

- Config: [`../../configs/env_mechanism_probes/2026-07-06-jz-directions.yaml`](../../configs/env_mechanism_probes/2026-07-06-jz-directions.yaml)
- Slurm script: [`archived/2026-07-06-environment-direction-parameter-probes-v100.sh`](archived/2026-07-06-environment-direction-parameter-probes-v100.sh)
- Expected results: `results/experiments/2026-07-06-environment-direction-parameter-probes/`

Run scope:

- Environments: SMAC zerg 5v5, POGEMA random, GRF academy corner.
- Examples: 8 balanced batches of 60 examples.
- Activation condition for direction analysis: random wrong environment token.
- Parameter sensitivity: first 2 batches, split by true environment, using the checkpoint's existing action/critic loss.

Primary outputs:

- `activation_directions.csv`: class-mean vectors and pairwise mean-difference directions per activation feature.
- `probe_results.csv`: linear probe metrics plus compact probe-weight direction summaries.
- `parameter_sensitivity.csv`: per-environment gradient norms by parameter group.
- `parameter_gradient_cosines.csv`: average gradient cosine between environment pairs by parameter group.
- `token_swap_behavior.csv`: action-logit, entropy, value, and selected-action changes under token swaps.

Decision rule:

Treat large activation separation plus high parameter-gradient overlap as evidence for environment-identifying representations implemented by mostly shared computation. Treat stable, low-cosine or high-magnitude differences in specific parameter groups only as localization candidates until a follow-up ablation or patching run shows environment-specific behavioral effects.

## Result: 2026-07-06 Direction And Parameter Probes

Slurm job `1379980` completed with exit code `0:0`, but the parameter-sensitivity rows failed because the action targets reached the model with a non-index tensor dtype. The script now casts per-environment action targets to `long` before the loss forward pass.

Fixed Slurm job `1380926` completed with exit code `0:0` in 3 minutes 0 seconds on JZ.

Result location: `results/experiments/2026-07-06-environment-direction-parameter-probes/`.

Run scope:

- Environments: SMAC zerg 5v5, POGEMA random, GRF academy corner.
- Input examples: 480.
- Activation examples: 2,400, from correct-token, wrong-token, and all-token prompt sweeps.
- Activation-direction rows: 120.
- Parameter-sensitivity rows: 66.
- Parameter-gradient cosine rows: 66.

Main activation findings:

- True environment again remains perfectly decodable from wrong-token activations after the first transformer block. The wrong final-token embedding itself is near chance for true environment, with accuracy 0.326.
- Mean-pooled activation directions separate POGEMA versus GRF most strongly. The largest mean-difference directions are `actor_layer:mean` for POGEMA versus GRF with L2 37.36 and cosine 0.475, followed by `critic_layer:mean` with L2 34.12 and cosine 0.698.
- POGEMA versus GRF separation grows across the transformer: `layer_02:mean` through `layer_06:mean` all have L2 around 32-33. SMAC versus GRF also separates in later mean-pooled layers, but the means are much more aligned, with cosines around 0.97-0.98 in layers 4-6.
- Token swaps remain behaviorally relevant. Random wrong-token prompts changed the selected action for 28.1% of examples, with mean absolute action-logit delta 3.60. Prompting all examples as POGEMA or GRF produced the same selected-action change rate; prompting all examples as SMAC shifted logits less and did not change selected actions in this subset.

Main parameter-gradient findings:

- Gradient magnitudes differ by environment and layer. SMAC has much larger sensitivity in token embeddings, layer 0, and miscellaneous non-layer parameters; POGEMA and GRF have larger sensitivity in layer 2.
- SMAC gradients are almost orthogonal to both POGEMA and GRF across most parameter groups. Transformer-layer gradient cosines for SMAC versus POGEMA range from -0.003 to 0.099, and SMAC versus GRF from -0.008 to 0.068 across layers 1-6.
- POGEMA and GRF gradients are much more aligned in transformer layers, with cosines 0.724 at layer 0, 0.930 at layer 1, 0.961 at layer 2, and 0.827-0.866 in layers 3-6. Actor and critic heads show the same pattern: POGEMA versus GRF cosines are 0.838 and 0.706, while SMAC-pair cosines are much lower.

Interpretation:

The result strengthens the claim that environment identity is not only copied from the explicit final token. Wrong-token activations recover the true environment almost everywhere after the first block, and mean-difference directions are large in later pooled activations and actor/critic branches.

The parameter-gradient result is a localization signal, not yet a causal parameter-specialization claim. The cleanest pattern is that SMAC has a distinct loss-gradient direction from both POGEMA and GRF, while POGEMA and GRF share much more aligned gradient directions through most transformer layers. This suggests that the model's effective computation may be more similar for POGEMA and GRF than for SMAC in this subset, or that the current SMAC sample/loss scale induces a different gradient regime.

Limitations:

- Parameter sensitivity used only the first two balanced batches, so the gradient-cosine pattern needs a seed and sample-size rerun before being treated as stable.
- Gradient directions are measured under the training-style action/critic loss, not under a causal ablation or patching intervention.
- The environment identity signal remains trivially available in raw input channels, so these results localize separability and sensitivity but do not prove semantically meaningful environment concepts.

## Implementation Notes

The local MARL-GPT loader already emits tokenized observation dictionaries and targets from `MultiEnvAggregateDataset`. The design should reuse that path so padding, history length, and positional fields match MARL-GPT inference. The experiment should also preserve input-channel feature tables before the model forward pass, because those controls explain how much environment identity is available without learned transformer computation.

The loader and inference code can add a final token. With `last_token`, the last token receives a special `attr_pos`; with `env_specific`, the last token's `obs` value is set to the environment index. This should be kept for the primary condition if it matches the trained checkpoint. The ablation is still useful because it tells us whether hidden environment directions are simply copied from this explicit token or also recoverable from observations, positions, masks, and learned task structure.

## Decision Rule

If wrong-token runs still recover true environment from hidden activations and behavior remains closer to the true environment than the prompted token, then the model differentiates environments from observation-derived structure. If wrong-token runs shift representations and behavior toward the prompted token, then explicit environment conditioning is functionally important. If attribution and ablation effects overlap heavily across environments, claim mostly shared effective computation. If stable components have large environment-specific effects, proceed to localization and intervention experiments for environment-specialized computation. If the apparent partition is unstable across samples or disappears under held-out scenarios, treat it as a weak diagnostic rather than a mechanistic claim.

## Links

- [Environment representations in MARL-GPT](../questions/2026-07-06-environment-representations-in-marl-gpt.md)
- [MARL-GPT](../literature/2026-06-30-marl-gpt.md)

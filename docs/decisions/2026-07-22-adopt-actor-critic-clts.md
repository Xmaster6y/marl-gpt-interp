# Adopt Separate Actor and Critic Cross-Layer Transcoders

## Status

Accepted on 2026-07-22.

## Choice

Use two independent full-path CLTs as the primary interpretability method:

- actor: seven shared MLPs plus the actor-branch MLP;
- critic: seven shared MLPs plus the critic-branch MLP.

Train both on one pooled natural-activation corpus without environment-conditioned normalization, routing, or weights. Construct input-specific attribution graphs to legal-action logit contrasts and local expected-value targets. Include residual and frozen-attention OV paths, retain reconstruction errors as graph nodes, and validate hypotheses by intervening in the original model.

Do not retain the SAE/lattice implementation as a compatibility path. Sparse autoencoders remain related work, not the project method. TacSIm remains a downstream football endpoint and requires a separate continuous-trajectory model or head.

## Rationale

A single-layer SAE describes a representation but does not identify the multi-layer computation producing actions or values. MARL-GPT has seven shared blocks followed by distinct actor and critic blocks, so a feature at the former `layer_03:final` site is not a decision bottleneck. CLTs approximate the MLP transformations along complete paths and support direct feature-to-feature attribution through the residual stream and realized attention routing.

Separate branch dictionaries avoid assuming that action selection and value estimation use the same sparse basis. Correspondence between them must emerge through examples and causal effects.

## Minimal Prior History

- The frozen checkpoint loads and has 7,232,256 parameters, width 256, eight heads, seven shared blocks, and separate actor/critic blocks.
- Preliminary probes found environment identity easily decodable, but decodability and representation geometry did not establish shared computation.
- Natural pooled TopK SAEs at widths 512 and 2,048 reconstructed well but had approximately 95% and 98% dead features. Per-domain normalization improved health only by adding domain-conditioned preprocessing, which is not an accepted universal representation.
- A hard domain-lattice SAE failed its synthetic recovery gate and was cancelled.
- A balanced 18-source-group-per-environment data view was planned and partially materialized. Its raw data and audit machinery remain useful, but the old final-token activation caches are not CLT corpora.
- No trained CLT, attribution graph on the frozen checkpoint, mechanistic-sharing result, steering result, or TacSIm result currently exists.

## Consequences

- Collect token-level residual inputs and MLP outputs for every shared layer and both branches.
- Treat all sequence positions as model-visible because MARL-GPT applies no attention padding mask; sample positions for storage rather than declaring padding causally absent.
- Use the original MARL-GPT checkpoint as the fidelity reference. No per-layer transcoder or additional SAE baseline is required.
- Require matched random interventions only as a causal damage/scale control.
- Keep claims prospective until replacement, error contribution, intervention, and fresh-rollout gates pass.

## Revisit Conditions

Revisit the method if neither branch CLT can meet replacement gates without collapse, or if attribution is consistently dominated by reconstruction errors or missing QK effects. In that case, diagnose corpus coverage and CLT capacity first; add QK analysis only if OV-conditioned graphs are faithful but incomplete.

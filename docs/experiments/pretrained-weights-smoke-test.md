# Pretrained Weights Smoke Test

## Status

Planned. Blocked until pretrained MARL-GPT weights are available locally.

## Question

Can the project load MARL-GPT weights, run GRF-compatible inference, and capture the activations needed for interpretability analysis?

## Hypothesis

The released checkpoint will contain `model_args` sufficient to reconstruct the model and expose activations from the token embeddings, transformer layers, actor head, and critic head.

## Data Or Command

No command is fixed yet. The first run should use the smallest available GRF trajectory or a minimal evaluation rollout rather than a full benchmark.

## Metrics

- Checkpoint loads without architecture mismatch.
- One batch or rollout produces action logits, action masks, Q-values, and token metadata.
- Activation hooks capture tensors from embeddings, residual streams, and output heads.
- Runtime is practical on local or available compute.

## Baseline Or Comparison

No scientific baseline. This is an infrastructure gate for later analyses.

## Expected Result

The model should run in evaluation mode and expose enough internals for probes and ablations.

## Decision Rule

If loading or inference fails because of missing dependencies or unavailable GRF runtime, use offline trajectory batches and CPU checkpoint inspection first. If token metadata cannot be recovered, prioritize reconstructing the tokenizer mapping before any probe claims.

## Links

- [MARL-GPT](../literature/marl-gpt.md)
- [Coordination representations](../questions/coordination-representations-in-marl-gpt.md)

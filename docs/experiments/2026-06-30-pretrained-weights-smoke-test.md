# Pretrained Weights Smoke Test

## Status

Partially completed on 2026-06-29. Checkpoint loading and short GRF rollout work locally; activation capture is still pending.

## Question

Can the project load MARL-GPT weights, run GRF-compatible inference, and capture the activations needed for interpretability analysis?

## Hypothesis

The released checkpoint will contain `model_args` sufficient to reconstruct the model and expose activations from the token embeddings, transformer layers, actor head, and critic head.

## Data Or Command

Checkpoint:

- `results/marl-gpt-main.pt`, downloaded from `https://huggingface.co/nortem/marl-gpt-model/resolve/main/marl-gpt-main.pt`.

Smoke command:

```bash
../interp-gfootball/.venv/bin/python scripts/smoke_marl_gpt_grf.py \
  --checkpoint results/marl-gpt-main.pt \
  --steps 20 \
  --device cpu
```

## Metrics

- Checkpoint loads without architecture mismatch.
- One batch or rollout produces action logits, action masks, Q-values, and token metadata.
- Activation hooks capture tensors from embeddings, residual streams, and output heads.
- Runtime is practical on local or available compute.

## Baseline Or Comparison

No scientific baseline. This is an infrastructure gate for later analyses.

## Expected Result

The model should run in evaluation mode and expose enough internals for probes and ablations.

## Actual Result

The checkpoint loads with no missing or unexpected keys after stripping the `_orig_mod.` prefix used by compiled PyTorch checkpoints. The loaded config has `block_size=700`, `action_size=20`, `history_len=10`, and 7,232,256 parameters.

The smoke script ran 20 steps on `academy_pass_and_shoot_with_keeper` with two controlled left-team agents. Latest verification output:

```text
{'map_name': 'academy_pass_and_shoot_with_keeper', 'steps': 20, 'last_actions': [0, 2], 'total_rewards': [0.0, 0.0], 'terminated': [False, False], 'truncated': [False, False], 'metrics': {}}
```

The current project `.venv` is not yet sufficient for this run because it lacks Torch, NumPy, Gymnasium, and GRF. The smoke run used the sibling `../interp-gfootball/.venv`, which already has Torch and `gfootball`. The vendored GRF adapter also needed compatibility fixes for Gymnasium wrapper metadata and four-value versus five-value GRF step returns.

Activation hooks were not tested in this run.

## Decision Rule

If loading or inference fails because of missing dependencies or unavailable GRF runtime, use offline trajectory batches and CPU checkpoint inspection first. If token metadata cannot be recovered, prioritize reconstructing the tokenizer mapping before any probe claims.

## Conclusion

The Hugging Face MARL-GPT weights are usable for local CPU inference and can drive a short GRF rollout. The next gate is making this repository's own `uv` environment self-contained and adding an activation-capture smoke test.

## Links

- [MARL-GPT](../literature/2026-06-30-marl-gpt.md)
- [Coordination representations](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)

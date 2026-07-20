# Balanced Offline MARL-GPT Corpus

## Status

Builder and scalable configs implemented; the six-source-per-environment core view is ready to materialize on JZ
SCRATCH through the pre/post-processing partition. No view is claim-bearing until its episode and shard-family audit
passes.

## Actual Problem

The complete public MARL-GPT repository is an acquisition pool, not a balanced experimental dataset. At pinned revision
`b70333a`, it contains approximately 275 GB of SMACv2, 633 GB of LMAPF/POGEMA, and 465 GB of GRF data. Mirroring all
1.37 TB would preserve this imbalance. Equal file counts are also insufficient because file sizes and transition counts
vary sharply and `chunk_*_part_*` files may share an upstream shard family.

The target variable for balance is the number of accepted activation examples per environment after leakage-safe group
splitting. Secondary controls are equal selected source-file counts per environment, a declared within-environment task
mixture, and a common maximum contribution per source. Raw byte counts are not a balance metric because observation
formats differ.

## Builder Contract

`scripts.experiments.sparse_marl_gpt.build_balanced_dataset` queries the pinned Hugging Face tree from a JZ pre/post node,
selects files deterministically from config weights, downloads them resumably into one revision-addressed SCRATCH cache,
and exposes config-specific views through symlinks. It never overwrites a conflicting view path. Each view manifest
records expected sizes, available hashes, environment/component identities, row caps, revision, seed, and status.

The core launch then runs `scripts.experiments.sparse_marl_gpt.audit_balanced_dataset`, which reads each materialized file
sequentially, records raw and capped row counts, terminal counts, files below the configured cap, and candidate multipart
`chunk_*` families. Candidate families remain warnings rather than silently becoming ground-truth episode identities.

Materialization status is `materialized_pending_audit`, with `claim_bearing: false`. A separate audit must establish:

- whether trajectories cross file or `part_*` boundaries;
- how per-agent `done` segments combine into one multi-agent episode;
- usable transition and episode counts by environment and scenario;
- absence of source groups shared across train, validation, and test;
- equal accepted row budgets after loader batch-size truncation.

## Scale And Mixture Configs

| Config | Source files per environment | Mixture | Purpose |
| --- | ---: | --- | --- |
| `2026-07-20-core-small` | 6 | zerg 5v5 / random / academy corner | Cheap end-to-end builder and audit development; not representative |
| `2026-07-20-training-small` | 12 | MARL-GPT training tasks | First full-mixture materialization; sparse minority-task coverage |
| `2026-07-20-training-pilot` | 30 | MARL-GPT training tasks | Intended fixed-layer activation pilot after audit |
| `2026-07-20-training-large` | 36 | MARL-GPT training tasks | Largest equal-component view supported by every configured GRF task |

The training mixture uses the six SMAC 5v5/5v6 race-task folders equally, POGEMA mazes/random at the native loader's
9:1 mixture, and the six GRF training folders equally. Domain batches remain equal. The 30-source activation config caps
each source at 3,330 accepted rows and runs 1,665 balanced batches, targeting 99,900 activation rows per environment.

## JZ Layout And Launch

- Immutable shared cache: `SCRATCH/marl-gpt-interp/hf-cache/nortem--marl-gpt-datasets/<revision>/`
- Balanced views: `SCRATCH/marl-gpt-interp/balanced-datasets/<config>/`
- Manifests: `balanced_dataset_manifest.json` inside each view
- Launch artifact: `to-launch/2026-07-20-balanced-offline-corpus-prepost.sh`

The launch runs on `prepost`, consumes no GPU allocation, and inherits the JZ login proxy. The first launch uses
`2026-07-20-core-small`; larger views reuse any already cached files.

## Decision Rule

Proceed from core-small to training-pilot only if the builder is resumable, every selected file matches its expected
size/hash, and the audit produces authoritative non-overlapping groups. Proceed to SAE training only when the activation
cache contains equal train, validation, and test example counts per environment with all configured scenarios represented
in the declared splits. Otherwise revise grouping or mixture before spending GPU time.

## Links

- [SAE method validation](2026-07-18-domain-lattice-sae-method-validation.md)
- [JZ setup](../2026-07-02-grf-jz-setup.md)
- [Balanced dataset configs](../../configs/experiments/sparse_marl_gpt/build_balanced_dataset/)

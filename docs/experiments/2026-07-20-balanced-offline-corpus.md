# Balanced Offline MARL-GPT Corpus

## Status

Builder and scalable configs implemented. The first six-file-per-environment view was materialized on JZ SCRATCH but
rejected by its audit: five of six GRF files fell below the common row cap and represented only four distinct
`chunk_*` families. A corrected six-group view is ready to materialize through the pre/post-processing partition. No
view is claim-bearing until its episode and shard-family audit passes.

## Actual Problem

The complete public MARL-GPT repository is an acquisition pool, not a balanced experimental dataset. At pinned revision
`b70333a`, it contains approximately 275 GB of SMACv2, 633 GB of LMAPF/POGEMA, and 465 GB of GRF data. Mirroring all
1.37 TB would preserve this imbalance. Equal file counts are also insufficient because file sizes and transition counts
vary sharply and `chunk_*_part_*` files may share an upstream shard family.

The target variable for balance is the number of accepted activation examples per environment after leakage-safe group
splitting. Secondary controls are equal selected source-group counts per environment, a declared within-environment task
mixture, and a common maximum contribution per source. Raw byte counts are not a balance metric because observation
formats differ.

## Builder Contract

`scripts.experiments.sparse_marl_gpt.build_balanced_dataset` queries the pinned Hugging Face tree from a JZ pre/post node,
selects groups deterministically from config weights, downloads one largest physical part per selected group resumably
into one revision-addressed SCRATCH cache, and exposes config-specific views through symlinks. Multipart `chunk_N_part_M`
files share group `chunk_N`; selecting only one part prevents a group from being counted twice while keeping this first
corpus small. The builder never overwrites a conflicting view path. Each view manifest records group identity, expected
size, available hash, environment/component identity, row cap, revision, seed, and status.

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

| Config | Source groups per environment | Mixture | Purpose |
| --- | ---: | --- | --- |
| `2026-07-20-core-small` | 6 files, 4 GRF groups observed | zerg 5v5 / random / academy corner | Rejected acquisition/audit evidence; never use for training |
| `2026-07-20-core-balanced-small` | 6 | zerg 5v5 / random / academy corner | Corrected end-to-end builder and audit smoke; not representative |
| `2026-07-20-training-small` | 12 | MARL-GPT training tasks | First full-mixture materialization; sparse minority-task coverage |
| `2026-07-20-training-pilot` | 18 | MARL-GPT training tasks | Intended fixed-layer activation pilot after audit |
| `2026-07-20-training-large` | 20 | MARL-GPT training tasks | Larger feasible group-balanced view; weakest GRF task remains at three groups |

The training mixture uses the six SMAC 5v5/5v6 race-task folders equally, POGEMA mazes/random at the native loader's
9:1 mixture, and the six GRF training folders equally. Domain batches remain equal. The 18-group activation config caps
each source at 5,120 accepted rows and runs 1,440 balanced batches, targeting 92,160 activation rows per environment.

## Materialization Evidence

JZ pre/post jobs `2107826` and `2107896` materialized and then idempotently re-audited the original core view. The shared
cache contains 8,743,842,683 downloaded bytes, so corrected and larger views can reuse matching objects. The audit found:

| Environment | Physical files | Candidate groups | Raw rows | Accepted rows at 8,192 cap | Files below cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRF | 6 | 4 | 393,730 | 26,019 | 5 |
| POGEMA | 6 | 6 | 12,582,912 | 49,152 | 0 |
| SMAC | 6 | 6 | 4,329,134 | 49,152 | 0 |

This view is rejected because its accepted environment budgets differ and its GRF group count is below six. The
corrected core uses six distinct groups per environment, a 1,024-row cap per group, and exactly 96 equal-domain batches,
targeting 6,144 accepted activation rows per environment. A successful file audit is necessary but not sufficient:
authoritative episode provenance and group-disjoint activation splits remain required before a scientific claim.

## JZ Layout And Launch

- Immutable shared cache: `SCRATCH/marl-gpt-interp/hf-cache/nortem--marl-gpt-datasets/<revision>/`
- Balanced views: `SCRATCH/marl-gpt-interp/balanced-datasets/<config>/`
- Manifests: `balanced_dataset_manifest.json` inside each view
- Launch artifact: `to-launch/2026-07-20-balanced-offline-corpus-prepost.sh`

The launch runs on `prepost`, consumes no GPU allocation, and inherits the JZ login proxy. Its default is the corrected
`2026-07-20-core-balanced-small`; larger views reuse any already cached files.

## Decision Rule

Proceed from core-balanced-small to training-pilot only if the builder is resumable, every selected file matches its
expected size/hash, and every environment reaches six distinct groups and 6,144 accepted rows. The audit exits nonzero
on any duplicate group, undersized file, or unequal environment budget. Proceed to SAE training only when the activation
cache contains equal train, validation, and test example counts per environment with all configured scenarios represented
in the declared splits. Otherwise revise grouping or mixture before spending GPU time.

## Links

- [SAE method validation](2026-07-18-domain-lattice-sae-method-validation.md)
- [JZ setup](../2026-07-02-grf-jz-setup.md)
- [Balanced dataset configs](../../configs/experiments/sparse_marl_gpt/build_balanced_dataset/)

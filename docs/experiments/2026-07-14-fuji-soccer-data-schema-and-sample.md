# Fuji Soccer Data Schema Inspection And Tiny Sample

## Status

Schema inspection and bounded sample transfer completed on 2026-07-14. The La Liga mini sample is structurally valid
for adapter development. The public STP raw tracking schema is also valid, and a 300-cycle sample was downloaded. The
Fuji-derived RoboCup arrays must not be used: comparison against their source CSV identifies an obsolete player-field
stride in their extractor.

## Question

What are the on-disk schemas, coordinate conventions, temporal structure, and sampling units of the authorized La Liga
human-tracking and RoboCup 2D state datasets on Fuji, and can a tiny structure-preserving sample support the existing
cross-dataset normalization pipeline without copying either full dataset?

## Why This Is Worth Doing

The current soccer normalization code assumes flat CSV, JSON, or JSONL rows and the comparison code assumes a
105-by-68 pitch. Those assumptions have not been checked against either source dataset. A bounded schema-and-sample
gate prevents a full transfer or experiment from silently mixing coordinate systems, incomplete frames, incompatible
identifiers, or provider-specific semantics.

The output is infrastructure evidence, not evidence of a human-simulator modelling gap. Its purpose is to determine
whether a defensible small comparison can be designed.

## Authorized Source Data

- Human football data: `/mnt/datapool00/laliga_23_EDMS`
- RoboCup 2D particle/state data: `/mnt/datapool00/robocup2d_extracted_states_92d`
- Public RoboCup raw data: the RoboCup 2024 round-robin archives linked by `open-starlab/STP-challenge-2025`
- Source host: Fuji, configured locally as SSH host `fuji`
- Authorization: the project owner confirmed authorization to use the data on 2026-07-14.

The source directories must be inspected in place. Do not recursively copy or synchronize them.

## Hypothesis

Each dataset will expose a match or game identifier, a frame or cycle identifier, team and player identities, ball
state, and planar positions. A complete contiguous interval from one match or game should be sufficient to validate
schema mapping and basic geometry. Provider coordinate frames, sampling frequencies, velocity conventions, and missing
entity behavior are expected to differ and must be measured rather than assumed.

## Inspection Procedure

Run read-only commands on Fuji and record:

1. Total directory size and file count without traversing file contents.
2. Directory structure to the first two or three levels.
3. File extensions, representative file sizes, compression, and container formats.
4. For one representative file per distinct format, metadata and only the minimum header/schema information needed to
   identify fields and array shapes.
5. Candidate match/game, period, frame/cycle, team, player, and ball identifiers.
6. Coordinate ranges and orientation, pitch dimensions, units, origin, and whether teams switch direction.
7. Timestamp or cycle units, observed sampling interval, and whether velocities are stored or must be derived.
8. Entities per complete frame, missing-player conventions, substitutions, ball representation, and duplicated rows.
9. Event-data availability and any keys that join events to tracking or state frames.
10. Provider documentation, manifests, or schema files stored alongside the data.

Do not print player names or other unnecessary identifying metadata into logs or committed documentation. Prefer field
names, types, shapes, ranges, and pseudonymous identifiers.

## Tiny Sample Design

Choose samples only after the schema inspection identifies the natural grouping keys.

### Human Sample

- One match and one period.
- One contiguous 10-to-30-second interval.
- Every player and ball record for every selected frame.
- Include the minimum match/frame metadata needed to interpret coordinates and playing direction.
- If events are available, include only events overlapping the interval plus the identifiers needed to align them.

### RoboCup 2D Sample

- One game.
- One contiguous interval with approximately the same duration or number of complete states as the human sample.
- Every player and ball state for every selected cycle.
- Include the minimum game/cycle metadata needed to interpret coordinates and sides.

Do not sample with `head` over row-oriented tracking data unless the file is first grouped by complete frame. Arbitrary
row truncation can produce incomplete teams and invalid geometry.

## Sample Storage And Provenance

Derived samples should remain untracked under:

- `results/raw/human/fuji-schema-smoke/`
- `results/raw/robocup2d/fuji-schema-smoke/`

For each sample, write an untracked manifest containing:

- source host and absolute source path;
- source relative file path;
- match/game and period identifiers;
- frame/cycle or timestamp bounds;
- selection rule and requested row/frame count;
- actual row, frame, player, team, and ball counts;
- source and sample checksums where practical;
- extraction timestamp and script/config version.

Commit the extraction script and config, but not the derived data, unless the project owner explicitly decides that the
sample is distributable and useful as a test fixture.

## Normalization Target

Map complete state rows into the existing tracking schema in
[`../../src/marl_gpt_interp/soccer_schema.py`](../../src/marl_gpt_interp/soccer_schema.py):

- `dataset`
- `match_id`
- `frame_id`
- `timestamp`
- `team_id`
- `player_id`
- `x`, `y`
- `vx`, `vy`
- `is_ball`

Before using [`../../scripts/normalize_soccer_data.py`](../../scripts/normalize_soccer_data.py), verify that the source
format is supported and small enough for its current whole-file read. Add streaming or provider-specific extraction
only if the inspected formats require it.

Coordinates must be converted into one documented canonical frame before running pitch-control or distance metrics.
The conversion must record source units, target units, origin, axis directions, and team-direction handling.

## Validation Metrics

- Files, matches/games, periods, frames/cycles, rows, teams, players, and ball records in the sample.
- Rows per frame and fraction of frames with the expected entity set.
- Duplicate and missing identifier counts.
- Timestamp monotonicity and empirical sampling interval.
- Coordinate and velocity minima, maxima, and non-finite counts.
- Fraction of entities inside documented pitch bounds after canonicalization.
- Team-side and playing-direction consistency.
- Ball-presence rate.
- Successful round trip into normalized JSONL.
- Pitch-control output only after complete-frame and coordinate validation pass.

## Baseline Or Comparison

- Compare observed schemas with the repository's provider-neutral tracking fields.
- Compare coordinate and timing conventions between human and RoboCup sources.
- Use the existing fixture-backed normalization example only as an infrastructure baseline; it is not evidence that the
  real providers use the same fields or semantics.

## Expected Result

The inspection should produce a factual schema table for both datasets and a reproducible extraction rule for two small,
complete temporal samples. The samples should reveal whether the existing normalization code can be configured directly
or needs streaming, nested-array support, coordinate conversion, or provider-specific adapters.

## Decision Rule

- Proceed to a tiny cross-dataset statistics run only if both samples preserve complete frames, have documented
  coordinate transforms, and normalize without silently dropping required identifiers.
- Implement a minimal provider adapter if a source is nested, columnar, compressed, or otherwise incompatible with the
  current flat-file reader.
- Keep processing on Fuji and retrieve only normalized samples or aggregate diagnostics if even the bounded raw sample
  is too large or operationally awkward to transfer.
- Defer pitch-control comparisons if coordinate orientation, team direction, or entity completeness cannot be verified.
- Do not make human-versus-simulator claims from this smoke sample; use it only to approve or revise the full experiment.

## Expected Reviewer Objection

A reviewer could argue that one short interval is unrepresentative. That objection is correct for scientific claims but
does not invalidate this experiment's infrastructure purpose. Representativeness, match-level splitting, and sample-size
selection belong in the subsequent full experiment after schema compatibility is established.

## Attempt: 2026-07-14

The local SSH configuration resolves `fuji` to `192.168.1.65` for user `y_poupart`. A read-only directory inventory
initially failed with `No route to host`. After local-network access was enabled for the execution environment, the host
became reachable and identified itself as `tesla`.

## Result: Fuji Schema Inspection And Local Samples

### Source Inventory

- La Liga source size: approximately 404 GB.
- La Liga layout: 376 numeric match directories containing 376 `events.jsonl` files, plus `split/` artifacts.
- One representative full match file, `1018887/events.jsonl`, is 591 MB and contains 122 top-level sequence records.
- RoboCup source size: approximately 594 GB.
- RoboCup layout: a flat directory with 132,893 `*.left.state.npy` files, 132,894 `*.right.state.npy` files, and
  `DATA_FORMAT.md`. The unequal side counts imply at least one unpaired file and should be audited before a full run.

### La Liga Schema

Each top-level JSONL or Arrow row is a sequence with:

- `game_id`, `half`, `sequence_id`, `sequence_start_frame`, and `sequence_end_frame`;
- attacking and defending team names;
- a variable-length `events` list.

Each event contains:

- `state`, split into `raw_state`, `relative_state`, and `absolute_state`;
- an `action` matrix with one action per attacking and defending player;
- scalar `reward` and `epv` values.

`raw_state` contains ball position and velocity plus `players`, `attack_players`, and `defense_players`. Player records
contain a pseudonymous numeric ID, role and position metadata, position, velocity, and action. The schema also contains
player names, but those are not needed for the provider-neutral tracking adapter and should be omitted from normalized
outputs.

The copied `split/mini` Arrow sample contains five first-half sequences from game `1338413`, with 85, 95, 122, 131,
and 534 event states respectively. Across its 967 states:

- every state contains exactly 22 players, split into 11 attackers and 11 defenders;
- there are 21,274 player-state rows;
- all sampled position and velocity values are finite;
- player positions span x `[-49.50, 51.03]` and y `[-33.78, 33.58]`;
- ball positions span x `[-46.70, 51.18]` and y `[-31.75, 33.09]`;
- player velocities span x `[-7.9, 8.0]` and y `[-6.4, 7.0]`;
- player actions include movement directions, `idle`, `stay`, `run`, `dribble`, `pass`, `intercept`, `challenge`,
  and `ball_receipt`.

The fields named `sequence_start_frame` and `sequence_end_frame` contain decimal values such as `0.0` and `8.3`; they
behave more like elapsed-time bounds than integer frame IDs. The individual event states do not expose an explicit frame
or timestamp field. A normalizer must therefore preserve sequence-relative event index and must not claim exact absolute
timestamps until the source convention is confirmed.

### RoboCup Schema And Mismatch

`DATA_FORMAT.md` documents each side file as a C-contiguous `float32` array of shape `(num_timesteps, 92)`:

- indices `0:4`: ball `[x, y, vx, vy]`;
- indices `4:48`: 11 perspective-team players with four values each;
- indices `48:92`: 11 opponent players with four values each.

It states that the values are copied directly from tracking data and that left/right files exchange the two player
blocks without coordinate mirroring.

Two copied match pairs satisfy the structural parts of that contract:

| Match | Shape per side | Dtype | Pair checks |
| --- | --- | --- | --- |
| `0802-2032-helios2024-cyrus2024-0040-sim30` | `(6299, 92)` | `float32` | identical ball block; player blocks swap exactly |
| `1220-1011-helios2022-oxsy2024-0001-sim01` | `(6419, 92)` | `float32` | identical ball block; player blocks swap exactly |

The numeric contents do not satisfy the documented physical-state interpretation. Both pairs contain many values of
`8000`, values near `180` and `-180`, and broad values outside football position and velocity ranges throughout the
supposed player blocks. Under the generous physical plausibility rule `|x| <= 53`, `|y| <= 35`, and
`|vx|, |vy| <= 10`, only about 20% of player tuples in the second sample are plausible and no timestep has 22 plausible
players. This is systematic across the two sampled matches.

The initial explanations considered were an extractor-field ordering bug, undocumented sentinel encodings, or a
different observation representation. The raw-source audit below resolves this as an extractor stride bug. Do not map
these arrays into the provider-neutral tracking schema.

### Relationship To STP Challenge 2025

The sampled RoboCup filenames and team names match the public
[`open-starlab/STP-challenge-2025`](https://github.com/open-starlab/STP-challenge-2025) dataset. The upstream repository
describes more than 2,000 RoboCup Soccer Simulator matches from 10 RoboCup 2024 teams and points to
[`hidehisaakiyama/RoboCup2D-data`](https://github.com/hidehisaakiyama/RoboCup2D-data) as the original source.

The upstream STP loader confirms the intended 92-feature model input but does not create the sampled `.npy` files. In
`main.py`, it defines 23 agents (`l1` through `l11`, `r1` through `r11`, and `b`) and selects the named columns
`<agent>_x`, `<agent>_y`, `<agent>_vx`, and `<agent>_vy` from each `.tracking.csv` file before flattening the result.
The challenge validator independently requires named left-player and ball coordinate columns. The upstream training
path also recomputes velocities from consecutive positions by default.

This means the intended STP CSV contract is consistent with 23 agents times four physical features. The local NumPy
files were produced by a separate extractor that is not present in the STP repository. Their semantic mismatch is
therefore an extraction or documentation problem in that derived dataset, not evidence that the STP challenge itself
uses non-physical player features.

### Deeper Sample Diagnostics

The config-driven diagnostic command is:

```bash
uv run --group grf -m scripts.run_experiment analyze_fuji_soccer_samples=2026-07-14-local-smoke
```

Config: [`../../configs/analyze_fuji_soccer_samples/2026-07-14-local-smoke.yaml`](../../configs/analyze_fuji_soccer_samples/2026-07-14-local-smoke.yaml)

Output: `results/analysis/2026-07-14-fuji-soccer-schema-smoke/diagnostics.json`

For La Liga, the five sequence bounds and event counts imply time steps from `0.09881` to `0.09981` seconds, with a
mean of `0.09919` seconds or approximately `10.08` Hz. Stored player velocities agree closely with position finite
differences: mean per-track correlations are `0.9927` for x and `0.9929` for y, while median absolute errors are
`0.0151` and `0.0134`. Ball correlations are `0.9966` and `0.9989`, with median absolute errors `0.0387` and `0.0263`.
This supports interpreting positions as pitch-scale coordinates and velocities as units per second. Sequence-relative
timestamps can be reconstructed approximately from the bounds, although the slightly variable implied interval should
remain explicit rather than being silently rounded to exactly 10 Hz.

For RoboCup, the documented interleaved interpretation makes only `18.1%` and `20.1%` of player tuples physically
plausible in the two match samples. Testing the most obvious alternative, four groups of 11 values for x, y, vx, and vy,
reduces plausibility to `1.24%` and `1.13%`. Neither interpretation produces any timestep with 22 plausible players.
The two left-side arrays contain 15,163 and 20,665 exact `8000` values in their player blocks. The ball block remains
physically bounded and has moderate-to-high finite-difference correlation, which suggests the first four columns were
selected correctly while the 44-column team blocks were not.

### Raw STP Slice And Extractor Audit

A bounded downloader was added and run as:

```bash
uv run -m scripts.run_experiment download_stp_tracking_sample=2026-07-14-helios-cyrus-smoke
```

Config: [`../../configs/download_stp_tracking_sample/2026-07-14-helios-cyrus-smoke.yaml`](../../configs/download_stp_tracking_sample/2026-07-14-helios-cyrus-smoke.yaml)

The downloader uses HTTP byte ranges and ZIP member lookup, so it did not retrieve the complete 2,423,793,057-byte
matchup archive. It wrote the header and first 300 complete cycle rows of the matching source file to
`results/raw/robocup2d/stp-raw-smoke/`, producing a 360,566-byte local CSV plus a provenance manifest. The full member
is 7,749,212 uncompressed bytes. Its sample SHA-256 is
`c5848b30aaa04f320577ca276eceffec0d927e49fc75e9321911393d77a7e763`.

The slice has 212 columns: 10 match/cycle metadata columns, four ball fields, and nine fields for each of 22 players.
All 92 named physical values (`x`, `y`, `vx`, and `vy` for the ball and 22 players) are finite. Across the 300 rows,
their feature minima are `[-49.0, -32.63, -2.2512, -2.5242]` and maxima are
`[48.339, 29.1515, 2.4056, 2.7365]` for `[x, y, vx, vy]`.

The source comparison identifies the derived-array bug exactly. The Fuji left array does not equal the 92 named
physical columns, but it exactly equals this positional selection from the raw CSV:

1. take the four ball fields;
2. start at `l1_x` and take four consecutive fields;
3. advance eight columns and repeat step 2 for 22 windows.

That selection assumes eight fields per player, while this CSV schema has nine, including `vwidth`. The first player's
`x`, `y`, `vx`, and `vy` are therefore correct, but every later window drifts across field boundaries. Of the resulting
88 supposed player fields, 46 are non-physical: 11 type fields, 10 stamina fields, nine view-width fields, eight body
angles, and eight neck angles. The final window reaches only `r9_x`, omitting the remainder of the right team. This
explains the `8000` stamina values and angle-like values without invoking sentinels or corrupt source data.

### Local Sample Artifacts

All copied data is ignored by Git under:

- `results/raw/human/fuji-schema-smoke/`: 19 MB, containing the existing five-sequence Arrow mini split and metadata;
- `results/raw/robocup2d/fuji-schema-smoke/`: 8.9 MB, containing two left/right match pairs and `DATA_FORMAT.md`;
- `results/raw/robocup2d/stp-raw-smoke/`: 361 KB, containing 300 raw STP cycles and its provenance manifest.

The source and local SHA-256 checksums matched for both RoboCup pairs, `DATA_FORMAT.md`, and the La Liga Arrow data
file. The recorded checksums are in the untracked sample manifest at `results/raw/fuji-schema-smoke-manifest.json`.

## Conclusion

The bounded transfer succeeded and is sufficient for local schema work without copying either full dataset. The La Liga
sample clears the structural gate for a provider-specific Arrow/nested-state adapter and small local geometry checks.
It does not yet clear exact timestamp semantics.

The RoboCup decision is now resolved: use raw STP `.tracking.csv` data and select columns by name. The raw sample clears
the physical schema gate, while the Fuji-derived arrays fail because their extractor assumes an obsolete eight-field
player stride. Running pitch control or cross-domain geometry on those arrays would be invalid; they should be replaced
or regenerated from named columns.

The next experiment should therefore split into two bounded tasks:

1. implement and test a La Liga mini-sample adapter that omits player names and preserves sequence-relative indices;
2. implement a streaming STP raw-CSV adapter that selects named fields and preserves cycle/play-mode metadata.

## Links

- [Cross-dataset soccer statistics](2026-06-30-cross-dataset-soccer-statistics.md)
- [GRF-human gap analysis](2026-06-30-grf-human-gap-analysis.md)
- [Soccer analytics statistics and concepts](2026-06-30-soccer-analytics-statistics.md)
- [Simulation-human modelling gap](../questions/2026-06-30-simulation-human-modelling-gap.md)

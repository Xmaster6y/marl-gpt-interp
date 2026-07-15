# Causal Cross-Football Pressure Transfer

## Status

Planned. This is the next claim-bearing track; no experiment has been launched.

## Question

Does MARL-GPT contain a representation of defender pressure on the ball carrier that generalizes across native GRF
scenarios, transfers to phase-matched La Liga and RoboCup states encoded through the GRF input path, and causally
affects native-GRF policy behavior?

## Hypothesis

Defender pressure should become linearly available in intermediate or late MARL-GPT states because it is relevant to
ball release and movement. A representation that captures football structure rather than a single simulator dataset
should preserve predictive signal on held-out external matches after controlling for source identity. If the direction
is functional, norm-controlled activation interventions should change relevant GRF action logits, values, selected
actions, or short-horizon possession behavior monotonically.

## Primary Concept

Define defender pressure without action labels, using a precommitted continuous measure derived from player and ball
geometry. Candidate components are nearest-defender distance, closing speed, and time-to-intercept. Fix the definition
and thresholds using training data only, then report both continuous prediction and a high-pressure classification.

Use support availability or passing-lane openness only as secondary concepts after the primary pipeline passes its
sanity checks. Adding several weakly specified concepts before validating pressure would increase researcher degrees of
freedom without strengthening the main claim.

## Data And Splits

- Native GRF: multiple rollout seeds and scenario families with valid policy outputs and activation capture.
- Human football: multiple La Liga matches or independent possessions.
- RoboCup: multiple raw STP matches or independent sequences.
- Unit of analysis: complete ball-carrier frames with the full six-frame history required by MARL-GPT.
- Sampling: non-adjacent frames, stratified by possession phase, field region, and pressure range.
- Splits: grouped by rollout, match, or sequence. No frame from a held-out group may influence label normalization,
  centering, whitening, probe fitting, layer selection, or intervention strength.

Before the main run, audit sample counts, missing fields, imputation rates, label distributions, temporal spacing, and
the number of independent groups per source. Pressure labels must not depend on the externally imputed ball-height,
vertical-direction, game-mode, or possession fields.

## Experiment Stages

### 1. Within-GRF localization

Train simple layerwise linear probes on native GRF activations. Select layers and regularization on grouped GRF
validation data only. Evaluate on held-out rollout seeds and scenario families.

Primary metrics:

- pressure regression error and rank correlation;
- high-pressure AUROC, balanced accuracy, and calibration;
- performance by layer, token summary, scenario, and field phase;
- gain or loss relative to identical probes on raw `simple115v2` features.

### 2. Cross-football transfer

Freeze the GRF-trained preprocessing and readout, then evaluate it on La Liga and RoboCup. The primary transfer result
is zero-shot; source-specific recalibration is a separately labelled diagnostic. Where rows can be matched by phase and
pressure bin, compare centered geometry. Otherwise use energy distance or MMD with match-level bootstrap uncertainty.

Also train within-source probes to separate absent transfer from absent information. Report the same-source versus
cross-source transfer gap and performance after source-identity residualization.

### 3. Native-GRF causal intervention

Estimate a pressure direction using GRF training data only. On held-out native-GRF states, add and subtract controlled
multiples of that direction at candidate layers. Measure:

- changes in predeclared pass, release, movement, and hold action-logit groups;
- critic-value and entropy changes;
- selected-action changes;
- short-horizon possession retention, ball release, and turnover when rollout intervention is feasible.

Intervention strengths must be set from the training activation distribution. Report dose-response curves and reject
strengths that move activations far outside the held-out GRF support.

## Baselines And Controls

- Raw `simple115v2` probes using identical grouped splits.
- Label-shuffled probes within each training group.
- Norm-matched random and pressure-orthogonal intervention directions.
- Control layers and token summaries not selected by the pressure probe.
- Source-only and source-plus-pressure probes to quantify domain confounding.
- Common held-out centering or whitening fitted on training data only.
- Imputation and coordinate-encoding sensitivity analyses for external sources.
- If feasible, a randomly initialized model or an untrained readout control.

## Reviewer Objection

Pressure is directly computable from player positions, so a successful probe may merely rediscover the input schema.
The experiment is useful only if it distinguishes raw-feature recoverability from a stable internal readout, shows
held-out cross-source transfer, controls for source identity, and demonstrates a behavioral effect under targeted
intervention. Probe accuracy alone is not mechanistic evidence.

## Decision Rule

Treat the result as a **shared functional football representation** only if all of the following hold:

- the probe generalizes across held-out GRF scenarios and rollout seeds;
- the frozen GRF readout transfers to both external sources above shuffled controls and with a defensible comparison to
  the raw-input baseline;
- the result survives match-level resampling, source controls, and the declared imputation sensitivity checks;
- GRF interventions produce a monotonic predicted effect and outperform norm-matched random-direction controls across
  held-out groups.

If decoding transfers but intervention fails, claim a shared diagnostic encoding, not a mechanism. If only within-GRF
decoding succeeds, conclude that the representation is simulator-local and use the failure to characterize the
modelling gap. If hidden-state probes do not improve on raw inputs and causal controls fail, stop investing in global
representation similarity and pivot to trajectory- and concept-distribution gap analysis.

## Minimal Validation Before Launch

Run a small local workflow containing at least two independent groups per available source. It must verify concept
calculation, grouped split integrity, activation extraction, frozen cross-source inference, finite metrics, and one
positive/negative intervention pair on native GRF. This is a smoke test, not result evidence.

The final config, command, launch definition, result location, and resource estimate remain to be specified after this
validation. No cluster job is authorized by this plan.

## Links

- [Track-selection decision](../decisions/2026-07-15-prioritize-causal-cross-football-concept-transfer.md)
- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [Simulation-human modelling gap](../questions/2026-06-30-simulation-human-modelling-gap.md)
- [Interpretability-guided alignment](../questions/2026-06-30-interpretability-guided-alignment.md)
- [Random frame sampling control](2026-07-15-random-frame-sampling-control.md)
- [External soccer GRF encoding](2026-07-15-external-soccer-grf-encoding.md)

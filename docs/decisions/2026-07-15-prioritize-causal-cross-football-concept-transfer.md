# Prioritize Causal Cross-Football Concept Transfer

## Status

Accepted.

## Choice

Make causal cross-football concept transfer the next claim-bearing research track. The primary concept is defender
pressure on the ball carrier, with support availability or passing-lane openness retained as secondary concepts after
the primary workflow is validated.

The track must test three increasingly strong claims:

1. the concept is decodable from MARL-GPT during native GRF inference;
2. a GRF-trained readout transfers to phase-matched La Liga and RoboCup states;
3. intervening on the direction changes native-GRF policy outputs and short-horizon behavior as predicted.

External football inputs are evidence for representation transfer only. Their policy outputs are not causal evidence
until the observed near-zero-entropy action collapse is understood and repaired.

## Rationale

The random-frame control invalidated the initial cross-football CKA interpretation, and raw cross-source cosine is
dominated by activation anisotropy. Another global geometry comparison would therefore be incremental and would not
answer whether MARL-GPT contains a football-relevant mechanism.

Defender pressure is suitable because it is continuous, can be derived from tracking geometry in every football
source, does not require reliable human action labels, and has a clear predicted relationship to ball-release and
movement decisions. Combining held-out transfer with a native-GRF intervention directly tests the project's proposed
bridge from simulator interpretability to the human-football modelling gap.

## Alternatives Considered

- **Scale cross-source CKA or cosine:** rejected as the next headline experiment because CKA needs semantically matched
  rows and uncentered cosine is not informative in the observed anisotropic geometry.
- **Run only within-GRF tactical probes:** retained as the first stage, but insufficient alone because it does not test
  the simulator-human bridge.
- **Use external action predictions as transfer evidence:** deferred because all sampled sources, including native GRF,
  currently collapse to one near-zero-entropy action in the external-encoding workflow.
- **Prioritize POGEMA-GRF abstract-concept transfer:** retained as a useful secondary track, but it is less directly
  aligned with the project's football and human-modelling contribution.

## Consequences

- Collect multiple GRF rollouts and multiple external matches or sequences; split by rollout or match, never by frame.
- Use non-adjacent, possession- and phase-stratified ball-carrier frames.
- Precommit raw-input, shuffled-label, domain-control, and norm-matched random-direction baselines.
- Use match-level bootstrap uncertainty and distributional metrics for unmatched samples; use CKA only for matched rows.
- Require causal interventions on held-out native GRF before describing a concept direction as functional.
- Do not launch a large run until a local end-to-end smoke validates labels, grouped splits, probe fitting, and activation
  intervention.

## Revisit Condition

Revisit the track if defender-pressure labels cannot be defined consistently across sources, if adequate match-level
data cannot be obtained, or if the checkpoint does not produce a non-degenerate native-GRF behavioral readout under
the intended rollout path. In those cases, restrict the work to diagnostic concept-gap measurement or select another
tracking-derived concept with a valid causal target.

## Links

- [Causal cross-football pressure transfer experiment](../experiments/2026-07-15-causal-cross-football-pressure-transfer.md)
- [Random frame sampling control](../experiments/2026-07-15-random-frame-sampling-control.md)
- [External soccer GRF encoding](../experiments/2026-07-15-external-soccer-grf-encoding.md)
- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [Simulation-human modelling gap](../questions/2026-06-30-simulation-human-modelling-gap.md)

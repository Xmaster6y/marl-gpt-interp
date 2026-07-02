# Stage Work From GRF To Human Gap

## Status

Accepted on 2026-06-30.

## Choice

Stage the project from infrastructure and GRF concept evidence toward human-football modelling-gap analysis, rather than starting with human-data alignment or online adaptation.

## Rationale

The project needs a defensible evidence chain. A planned-work abstract is useful for framing, but the research should first prove that MARL-GPT can be run, instrumented, and probed in GRF before making claims about human football. Human tracking data is central, but direct action alignment is risky until the simulator-side concepts and trajectory statistics are stable.

## Staged Plan

### Phase 0: Abstract Setup

Deadline: June 1, 2026. Historical as of June 30, 2026.

Goal: submit a credible planned-work abstract to NU Sports ML Workshop 2026.

Outputs:

- Project brief with core question and contribution scope.
- Literature notes for MARL-GPT, adaptive action supervision, R2DRL, and workshop context.
- Active research questions and planned analyses.
- Abstract draft under 500 words.

### Phase 1: MARL-GPT Access And Smoke Test

Status: partially completed locally; checkpoint loading and short GRF rollout work, activation capture pending.

Goal: verify that the project can load the checkpoint, run GRF-compatible inference, and capture activations.

Outputs:

- Minimal inference run or offline batch pass.
- Activation hook inventory.
- Mapping from token positions to football semantics.
- Decision on whether GRF evaluation can run locally or requires cluster support.

Linked plan: [Pretrained weights smoke test](../experiments/2026-06-30-pretrained-weights-smoke-test.md).

### Phase 2: GRF Rollout Statistics

Goal: make the first real experiment reproducible in a fresh environment and on JZ before probing.

Outputs:

- Config-driven local smoke and JZ V100 small run.
- Per-step, per-episode, and aggregate JSON/CSV statistics.
- Basic action, reward, score, possession, distance, pressure, width, depth, and compactness summaries.
- Decision on whether GRF is ready for activation capture and probes.

Linked plan: [GRF rollout statistics](../experiments/2026-06-30-grf-rollout-statistics.md).

### Phase 3: GRF Concept Probes

Goal: identify football-relevant concepts represented inside MARL-GPT.

Outputs:

- Concept label definitions from GRF observations or state.
- Layerwise probe results.
- Token and embedding ablations.
- First evidence on whether concepts are diagnostic or behaviorally causal.

Linked plan: [GRF representation probes](../experiments/2026-06-30-grf-representation-probes.md).

### Phase 4: Human Data Mapping

Trigger: human tracking data and data-use constraints are available.

Goal: map human trajectories into comparable state, phase, and concept spaces without assuming clean discrete action labels.

Outputs:

- Data dictionary for available tracking and event fields.
- Concept extraction from human trajectories.
- Human versus GRF trajectory and concept comparisons.
- Decision on whether action-level pseudo-labels are credible.

Linked plan: [GRF-human gap analysis](../experiments/2026-06-30-grf-human-gap-analysis.md).

### Phase 5: Interpretability-Guided Alignment

Goal: test whether identified concepts can guide adaptation toward human-like behavior.

Candidate outputs:

- Human-similar simulator trajectory selection.
- Concept-level regularization target.
- Representation steering or causal intervention result.
- DTW-style aligned phase comparison.

Linked question: [Interpretability-guided alignment](../questions/2026-06-30-interpretability-guided-alignment.md).

### Optional Track: Controlled Environment Gap

Goal: use MAPE or a related simple multi-agent environment only if it clarifies a blocked GRF question.

Linked plan: [GRF-MAPE gap analysis](../experiments/2026-06-30-grf-mape-gap-analysis.md).

## Alternatives

- Start with human tracking data: deferred until simulator concepts and data-use constraints are clearer.
- Start with online adaptation or R2DRL: deferred because it is higher engineering risk and does not provide the first evidence needed for interpretability claims.
- Start with MAPE: kept as an optional controlled setting, but not the main football story.

## Consequence

Near-term work should prioritize GRF rollout statistics, activation capture, simulator-derived concept labels, and first probes. Human trajectory comparison comes after those artifacts exist.

## Revisit Condition

Revisit if GRF cannot produce stable rollout and activation artifacts, if human data access changes the fastest credible evidence path, or if probes fail and the project should pivot toward descriptive simulator-human gap measurement.

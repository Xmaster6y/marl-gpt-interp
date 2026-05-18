# Roadmap

## Phase 0: Abstract Setup

Deadline: June 1, 2026.

Goal: submit a credible planned-work abstract to NU Sports ML Workshop 2026.

Outputs:

- Project brief with core question and contribution scope.
- Literature notes for MARL-GPT, adaptive action supervision, R2DRL, and workshop context.
- Active research questions and planned analyses.
- Abstract draft under 500 words.

## Phase 1: MARL-GPT Access And Smoke Test

Trigger: pretrained MARL-GPT weights become available locally or through a stable download.

Goal: verify that the project can load the checkpoint, run GRF-compatible inference, and capture activations.

Outputs:

- Minimal inference run or offline batch pass.
- Activation hook inventory.
- Mapping from token positions to football semantics.
- Decision on whether GRF evaluation can run locally or requires cluster support.

Linked plan: [Pretrained weights smoke test](experiments/pretrained-weights-smoke-test.md).

## Phase 2: GRF Concept Probes

Goal: identify football-relevant concepts represented inside MARL-GPT.

Outputs:

- Concept label definitions from GRF observations or state.
- Layerwise probe results.
- Token and embedding ablations.
- First evidence on whether concepts are diagnostic or behaviorally causal.

Linked plan: [GRF representation probes](experiments/grf-representation-probes.md).

## Phase 3: Human Data Mapping

Trigger: human tracking data and data-use constraints are available.

Goal: map human trajectories into comparable state, phase, and concept spaces without assuming clean discrete action labels.

Outputs:

- Data dictionary for available tracking and event fields.
- Concept extraction from human trajectories.
- Human versus GRF trajectory and concept comparisons.
- Decision on whether action-level pseudo-labels are credible.

Linked plan: [GRF-human gap analysis](experiments/grf-human-gap-analysis.md).

## Phase 4: Interpretability-Guided Alignment

Goal: test whether identified concepts can guide adaptation toward human-like behavior.

Candidate outputs:

- Human-similar simulator trajectory selection.
- Concept-level regularization target.
- Representation steering or causal intervention result.
- DTW-style aligned phase comparison.

Linked question: [Interpretability-guided alignment](questions/interpretability-guided-alignment.md).

## Optional Track: Controlled Environment Gap

Goal: use MAPE or a related simple multi-agent environment only if it clarifies a blocked GRF question.

Linked plan: [GRF-MAPE gap analysis](experiments/grf-mape-gap-analysis.md).

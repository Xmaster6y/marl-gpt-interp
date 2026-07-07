# Football Action Value And Space Control

## Source

This note records a broad football analytics reading direction from a discussion with Rikuhei Umemoto. The starting point was interest in these two papers:

- [Prediction-based evaluation of back-four defense with spatial control in soccer](https://arxiv.org/abs/2511.06191)
- [Pitch-wide space evaluation for soccer transitions](https://arxiv.org/abs/2505.14711), listed on arXiv as "Space evaluation at the starting point of soccer transitions"

Rikuhei recommended reading the original pitch-control and action-valuation line behind the transition-space paper:

- Action valuation: [Decroos et al., 2018, "Actions Speak Louder Than Goals: Valuing Player Actions in Soccer"](https://arxiv.org/abs/1802.07127)
- Pitch control and off-ball space: Spearman, 2018, "Beyond expected goals"; [Fernandez and Bornn, 2018, "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer"](https://www.lukebornn.com/papers/fernandez_ssac_2018.pdf)
- RL action valuation: [Nakahara et al., 2023, "Action valuation of on- and off-ball soccer players based on multi-agent deep reinforcement learning"](https://arxiv.org/abs/2305.17886)

## Takeaway

The useful bridge for this project is not any single metric. The literature gives a ladder of increasingly agentic football concepts:

1. Event action value: estimate how an on-ball action changes future scoring and conceding probability.
2. Pitch control and space value: estimate which team can control each location, then weight controlled space by scoring, possession, or field value.
3. Transition and defensive organization: use pitch-wide space, pressure, compactness, line height, and transition outcomes to characterize team behavior.
4. Multi-agent RL value: learn state and action values for on- and off-ball players in a GRF-like continuous multi-agent setting.

For MARL-GPT, these are natural weak labels and target concepts for representation probes: pressure, pitch control, pass/control probability, off-ball positioning value, action value, transition phase, defensive compactness, and line height.

## Reading Map

### 1. Decroos et al.: action valuation from event sequences

The Decroos paper introduces SPADL, a human-interpretable action representation, and HATTRICS, a framework that values an action by the change it causes in short-horizon scoring and conceding probabilities. The action value is not just a shot metric; passes, dribbles, tackles, clearances, and other actions can be assigned value through the predicted change in future outcomes.

Project use:

- Good conceptual baseline for "action value" and "possession value" labels.
- Useful contrast with MARL-GPT critic values: event-derived expected outcome change versus learned policy value.
- Reviewer-facing point: action value in football is usually framed as prediction over future events, not necessarily as a learned mechanistic representation inside a policy.

Limitation for this project:

- Event-sequence valuation is mostly on-ball and discrete. It does not by itself explain off-ball coordination or continuous spatial affordances.

### 2. Spearman and Fernandez/Bornn: pitch control and off-ball space

This line is the original basis for OBSO-style space evaluation used by later transition-space papers. The key move is to turn player positions, velocities, and arrival/control assumptions into a spatial control field, then evaluate locations by combining control probability with a value model.

Fernandez and Bornn's "Wide Open Spaces" is especially useful because it separates three objects that should remain separate in our probes:

- Pitch control: a smooth team-control surface built from player influence areas. Each player's influence is modeled as a bivariate normal whose shape changes with position, velocity, and distance to the ball; team control is then a logistic transform of summed own-team influence minus summed opponent influence.
- Pitch value: a learned spatial value surface conditioned on ball location, trained from defensive-team influence patterns and then normalized by distance to goal.
- Space creation metrics: Space Occupation Gain (SOG) measures value gained by occupying controlled space for oneself; Space Generation Gain (SGG) credits a player for dragging an opponent away from a teammate who then gains valuable space.

This distinction matters for MARL-GPT: "space" is not a single scalar. A model could represent control without value, value without control, or direct self-occupation without teammate-oriented space generation.

Project use:

- Gives concrete, geometry-based labels for GRF probes: team control of each zone, local free space, pass reception probability, and dangerous controlled space.
- Provides a bridge from simulator state to human tracking data because pitch-control fields are defined from positions and velocities rather than simulator internals.
- Important for separating "raw coordinate decoding" from a more tactical concept: a representation that predicts pitch control or controlled dangerous space is more meaningful than one that predicts ball x/y alone.
- Gives an off-ball coordination target that is more specific than "support": SGG-like labels test whether a player movement creates value for a teammate by attracting defenders.
- Suggests active versus passive occupation as a temporal probe: does MARL-GPT distinguish space won by player movement from space that opens because opponents or the ball move elsewhere?

Limitation for this project:

- The assumptions behind arrival time, reaction time, and control probability are hand-designed. A MARL-GPT probe using these labels would test alignment with a tactical model, not ground truth tactical cognition.
- The paper explicitly notes the lack of ground-truth labels for space creation and validates with expert video analysis. For our work, these labels should be treated as interpretable weak labels, not as definitive truth.

### 3. Ogawa, Umemoto, and Fujii: pitch-wide space for transitions

The transition paper proposes OBPV, replacing OBSO's goal-focused score model with a field-value model so that space far from goal can still be meaningful. It also replaces a fixed next-event transition model with a transition kernel estimated from pass distributions. This makes the metric more useful at the starting point of counter-attacks and other possession changes.

Project use:

- Directly relevant to transition probes: recent possession change, counterattack start, high-value open space ahead, and OBPV increase over the first few events.
- Suggests a better probe target than only "shot opportunity" for early or middle pitch states.
- Provides a concrete objection to naive xG/shot labels: many important football decisions happen far from goal, where shot-based value is near zero.

Limitation for this project:

- OBPV uses real tracking and event data from La Liga 2023/24. In GRF, the same equations may be usable, but the pass-distribution kernel and field-value calibration may need simulator-specific versions.

### 4. Back-four defense with spatial control

The back-four defense paper uses interpretable spatial indicators for negative transitions after possession loss: space control, stretch index, pressure index, and defensive line height. It then predicts defensive success with team-specific models for Barcelona and Real Madrid. The reported strongest signals include relative line height and space score.

Project use:

- Gives defensive-shape labels that are not just "nearest defender pressure": compactness, stretch, pressure, line height, relative line height, and back-line space control.
- Useful for a sports-facing story because the quantities are interpretable to coaches and analysts.
- Provides candidate labels for testing whether MARL-GPT encodes collective defensive organization, not only individual ball pressure.

Limitation for this project:

- The paper is team-specific and real-data-specific. GRF's defensive line behavior may not map cleanly to professional back-four organization.

### 5. Nakahara et al.: multi-agent RL action valuation

Nakahara et al. explicitly move from team-level or on-ball valuation toward valuing possible actions for on- and off-ball soccer players in a multi-agent deep RL framework. The setting is especially relevant because the paper describes a continuous state space with a discrete action space that mimics Google Research Football and uses supervised learning for actions in RL.

Project use:

- Closest conceptual neighbor to MARL-GPT action/value analysis in GRF.
- Supports the idea that off-ball players need action values too, not just the ball carrier.
- Gives a comparison point for action-value probes: MARL-GPT's actor/critic outputs may be interpretable through multi-agent action-value concepts.

Limitation for this project:

- The method is designed for valuation, not interpretability of a general multi-environment transformer. Our contribution should not be "we can value actions"; it should be whether a trained foundation MARL policy internally represents football analytics concepts and whether those representations are behaviorally relevant.

## Project Implications

This literature strengthens the case for concept-level probing before broad human-data claims. A good first football concept suite should combine simple labels, spatial-control labels, and action-value labels:

| Concept family | Candidate labels | Why it matters |
| --- | --- | --- |
| Basic state | possession, ball location, game mode | Sanity checks and controls. |
| Pressure | nearest defender distance, pressure count, closing pressure | Tests relational reasoning around the ball. |
| Passing and support | pass lane openness, receiver pressure, open forward pass exists | Tests teammate-specific affordances. |
| Space control | pitch control, zone control, dangerous controlled space, OBPV-like field value | Connects GRF to tracking-data soccer analytics. |
| Off-ball space creation | SOG-like occupation gain, SGG-like teammate space generation, active/passive occupation | Tests whether representations encode off-ball movement value, not only ball-carrier affordance. |
| Transition | possession-change phase, counterattack start, open space ahead, OBPV increase | Tests temporal and tactical context beyond shots. |
| Defensive shape | compactness, stretch, pressure index, line height, relative line height | Tests collective defensive organization. |
| Action value | VAEP-style change, possession value, critic value, action-specific logit or Q-value | Connects analytics valuation to MARL-GPT behavior. |

The strongest near-term experiment is not to reproduce the full analytics stack. It is to implement cheap, transparent GRF approximations of a few concepts and ask:

- Are these concepts linearly decodable from MARL-GPT activations above raw-feature baselines?
- Do they localize to meaningful token groups: ball, active player, teammate, opponent, final token, actor branch, critic branch?
- Do causal interventions on the concept direction change relevant actions: pass, shoot, pressure, move, clear?
- Are pitch-control and transition labels more informative than simpler coordinate or shot-opportunity labels?

## Reviewer Objection

A likely reviewer objection is that handcrafted football analytics labels could be arbitrary overlays on top of GRF, not evidence of learned football understanding. The response should be to separate three claims:

1. Diagnostic claim: MARL-GPT activations contain information aligned with interpretable football analytics concepts.
2. Behavioral claim: intervening on concept directions changes football-relevant action logits or values.
3. Transfer claim: concept definitions or learned directions remain meaningful across GRF scenarios or between GRF and human tracking data.

Only the first claim is needed for an early probe result. The second claim is needed for a mechanistic interpretation result. The third claim is needed before making a strong human-football alignment claim.

## Links

- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [GRF representation probes](../experiments/2026-06-30-grf-representation-probes.md)
- [Soccer analytics statistics and concepts](../experiments/2026-06-30-soccer-analytics-statistics.md)
- [MARL-GPT](2026-06-30-marl-gpt.md)

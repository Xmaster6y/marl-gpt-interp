# Soccer Analytics Statistics And Concepts

## Purpose

This note lists football statistics and tactical concepts that are useful for this project. The emphasis is not general fan-facing reporting; it is on quantities that can be measured from GRF state, human tracking data, event data, or MARL-GPT activations.

The most important concepts for the first probing phase are those that are easy to label, tactically meaningful, and likely to affect model actions or values.

## Priority For MARL-GPT Probing

| Priority | Statistic or concept | Why it matters for this project |
| --- | --- | --- |
| 1 | Possession | Sanity-check concept; nearly every tactical label depends on it. |
| 1 | Defensive pressure | Requires relational reasoning over ball carrier and opponents. |
| 1 | Shot opportunity | Should directly affect shot logits and action values. |
| 1 | Pass opportunity | Tests teammate-specific affordances and coordination. |
| 2 | Space and pitch control | Central tracking-data concept; useful bridge to human football data. |
| 2 | Support | Tests whether teammates are represented as useful options. |
| 2 | Defensive compactness | Team-shape concept relevant to simulation-human gap. |
| 2 | Phase of play | Helps separate build-up, attack, defense, transition, and set pieces. |
| 3 | Counterattack and transition | Strategically important but more difficult to label robustly. |
| 3 | Formation and role | Useful for human comparison, but GRF roles may not align cleanly. |
| 3 | Action value | Connects analytics to critic values, but depends on model and data quality. |

## State And Rule Statistics

These are basic game-state variables. They are not necessarily publishable by themselves, but they are essential controls and sanity checks.

| Statistic | Definition | GRF label source | Probe use | Caveat |
| --- | --- | --- | --- | --- |
| Possession team | Which team controls the ball: own, opponent, or none. | Ball ownership one-hot in observation. | First sanity-check probe; branch and layer localization. | Easy to decode because GRF directly exposes it. |
| Active player | Controlled player or current ball-relevant player. | Active-player one-hot. | Tests whether model tracks the focal agent. | GRF active player is simulator-specific. |
| Ball location | Ball x, y, z coordinates. | Ball position features. | Control variable for shot/pass/pressure probes. | Raw coordinate decoding is not an interesting claim. |
| Ball velocity | Ball movement vector. | Ball direction features. | Useful for loose-ball, transition, and interception labels. | Human tracking may use different sampling and smoothing. |
| Game mode | Open play, kickoff, goal kick, corner, free kick, penalty, throw-in. | Game-mode one-hot. | Separate open-play concepts from set-piece concepts. | Set-piece tactics need separate labels. |
| Action mask | Which actions are legal in the current state. | Environment action mask. | Distinguish rule knowledge from tactical preference. | Legal action is not the same as good action. |

## On-Ball Attacking Statistics

These statistics describe the attacking team while it has or can control the ball.

| Statistic | Definition | Label construction | Probe target | Why it matters |
| --- | --- | --- | --- | --- |
| Distance to goal | Euclidean or x-axis distance from ball carrier or ball to goal center. | Geometry from positions. | Active-player token, ball token, final token. | Core component of shooting and progression value. |
| Goal angle | Angular width of the goal from the ball carrier or ball. | Geometry from ball/player to goalposts. | Active-player token, actor branch. | Better than distance alone for shot opportunity. |
| Shooting zone | Binary or categorical zone where shooting is plausible. | Threshold on distance, angle, and field position. | Actor branch and critic branch. | Useful weak label for shot-affordance probing. |
| Shot opportunity | Continuous or binary estimate that shooting is attractive. | Distance, angle, pressure, goalkeeper/defender location if available. | Final token, actor branch, critic branch. | Should affect shot logits and Q-values. |
| Ball progression | Change in ball position toward goal. | Difference in ball x-coordinate or possession value. | Trajectory-level and action-value analysis. | Forward movement can be bad under pressure. |
| Carry opportunity | Whether dribbling/carrying forward is plausible. | Free space ahead, pressure, and nearest defender. | Active-player token and actor branch. | Helps separate pass, shot, and move decisions. |
| Width | How wide the attacking team is spread. | y-range or lateral dispersion of teammates. | Team-level pooled representations. | Relevant to build-up and switching play. |
| Depth | How vertically stretched the attacking team is. | x-range or longitudinal dispersion of teammates. | Team-level pooled representations. | Relevant to support, progression, and compactness. |

## Passing And Support Statistics

Passing concepts are central for coordination because they require reasoning about multiple teammates and opponents.

| Statistic | Definition | Label construction | Probe target | Why it matters |
| --- | --- | --- | --- | --- |
| Open pass exists | Whether at least one teammate is a plausible pass option. | Any teammate with low interception risk and acceptable distance. | Final token, actor branch. | Tests whether model detects collaborative options. |
| Best pass target | Teammate with highest pass score. | Max over teammate pass scores. | Teammate tokens and final token. | Tests agent-specific affordance representation. |
| Passing lane openness | Degree to which defenders block the segment from passer to receiver. | Defender distance to pass line, angle, and closing speed. | Teammate tokens, opponent tokens, attention patterns. | Requires relational spatial reasoning. |
| Receiver pressure | Pressure on receiver before or after pass. | Nearest defender distance to receiver. | Receiver token. | A pass to an open receiver is different from a pass to a marked receiver. |
| Forward pass availability | Whether an open pass progresses toward goal. | Open pass plus positive x-progress. | Actor and critic branches. | Connects pass opportunity to attacking value. |
| Support distance | Distance from ball carrier to nearest or best supporting teammate. | Pairwise teammate distance. | Active-player and teammate tokens. | Measures local attacking support. |
| Support angle | Whether support is behind, lateral, or ahead of ball carrier. | Relative angle from ball carrier to teammate and goal. | Teammate tokens. | Helps distinguish safe support from progressive support. |
| Overload | Numerical advantage near the ball or in a zone. | Count attackers minus defenders within radius or zone. | Regional/team representations. | Important tactical signal for passing and progression. |

## Defensive Pressure Statistics

Pressure is one of the best early targets because it is meaningful and requires combining several observation tokens.

| Statistic | Definition | Label construction | Probe target | Why it matters |
| --- | --- | --- | --- | --- |
| Nearest defender distance | Distance from ball carrier to closest defender. | Pairwise distance. | Active-player token, nearest-opponent token, final token. | Simplest pressure statistic. |
| Under pressure | Binary version of nearest-defender pressure. | Nearest defender below threshold. | Actor and critic branches. | Should change pass, shot, clear, and dribble preferences. |
| Pressure count | Number of defenders within radius of ball carrier. | Count opponents inside threshold. | Final token and opponent tokens. | Captures crowdedness better than nearest distance. |
| Closing pressure | Whether nearest defender is moving toward ball carrier. | Relative velocity dot product. | History-aware layers. | Tests whether model uses temporal information. |
| Goal-side pressure | Whether defender blocks the path from ball carrier to goal. | Defender location relative to ball-goal segment. | Active-player and opponent tokens. | More football-specific than raw distance. |
| Pressing intensity | Team-level pressure near ball. | Multiple defenders near ball or passing options. | Team-level pooled tokens, final token. | Useful for phase and transition analysis. |

## Space And Pitch Control Statistics

These are tracking-style concepts. They are more complex than pressure but are important for connecting simulator behavior to human football analysis.

| Statistic | Definition | Label construction | Probe target | Why it matters |
| --- | --- | --- | --- | --- |
| Pitch control | Probability each team can control each pitch location. | Time-to-reach model using positions and speeds. | Spatial aggregate labels; token attribution. | Core modern tracking-data concept. |
| Local space | Free space around a player. | Area around player not controlled by opponent. | Player tokens. | Useful for off-ball value and pass reception. |
| Space ahead | Open space between ball carrier and goal. | Opponent control or density in forward cone. | Active-player token. | Distinguishes carrying opportunity from blocked attack. |
| Dangerous space | Open space in high-value attacking zones. | Pitch control weighted by distance/angle to goal. | Critic branch. | Connects space to expected value. |
| Occupied zones | Which tactical zones contain attackers or defenders. | Discretized pitch grid occupancy. | Layerwise probes and human-GRF comparison. | Robust to small coordinate noise. |
| Zone control | Which team controls each pitch zone. | Pitch control aggregated to zones. | Representation comparison across domains. | Good bridge between GRF and human tracking. |

## Defensive Shape Statistics

These describe the defending team as a unit rather than individual pressure around the ball.

| Statistic | Definition | Label construction | Probe target | Why it matters |
| --- | --- | --- | --- | --- |
| Defensive compactness | How tightly defenders are grouped. | Mean pairwise defender distance or convex hull area. | Team-level representations. | Captures team shape and space denial. |
| Defensive width | Lateral spread of defenders. | y-range or standard deviation. | Opponent-team tokens. | Helps analyze wide versus narrow defending. |
| Defensive depth | Longitudinal spread of defenders. | x-range or line distances. | Opponent-team tokens. | Helps identify high line, low block, stretched block. |
| Line height | Average x-position of defensive line. | Mean or selected defender x-coordinate. | Team-level probes. | Relevant to pressing and counterattack risk. |
| Last defender line | x-position of deepest or last defender. | Min or max defender x depending on attacking direction. | Critic and actor branches. | Important for through-ball and offside-like reasoning. |
| Between-lines space | Space between midfield and defensive lines. | Requires role or clustering approximation. | Higher-level phase probes. | Harder in GRF unless roles are stable. |

## Phase And Strategy Statistics

These are higher-level and should usually be introduced after the basic probes work.

| Statistic | Definition | Label construction | Probe target | Why it matters |
| --- | --- | --- | --- | --- |
| Phase of play | Build-up, progression, final-third attack, defense, transition, set piece. | Rules from possession, field zone, velocity, pressure, game mode. | Final token, late layers. | Controls for context when interpreting concepts. |
| Transition | Recent possession change or loose-ball recovery. | Change in possession over history. | History-aware layers. | Tests temporal reasoning and counterattack behavior. |
| Counterattack | Fast attack after regaining possession against unsettled defense. | Recent regain, forward velocity, opponent shape, open space. | Actor branch, critic branch. | High-value strategic concept but harder to label. |
| Pressing phase | Defending team actively pressures ball and nearby options. | Defensive pressure count, line height, compactness. | Team representations. | Useful for human-GRF comparison. |
| Build-up | Controlled possession in deeper or middle zones. | Own possession, low speed, non-final-third location. | Final token and critic branch. | Helps separate patient possession from attack. |
| Final-third attack | Possession in dangerous attacking areas. | Ball x-zone plus own possession. | Actor and critic branches. | Context for shot/pass decisions. |

## Action Value Statistics

These connect classic sports analytics to MARL-GPT's actor and critic outputs.

| Statistic | Definition | Label construction | Probe or model output | Caveat |
| --- | --- | --- | --- | --- |
| Expected goals | Probability a shot becomes a goal. | Shot model using distance, angle, pressure, event context. | Compare to shot Q-value or shot logit. | GRF may not match real shot quality. |
| Expected threat | Expected future scoring threat from ball location or possession state. | Grid or learned possession-value model. | Compare to critic values. | Location-only xT misses pressure and off-ball context. |
| Possession value | Expected value of current possession state. | Learned from trajectories or handcrafted approximation. | Critic branch comparison. | Needs enough trajectories for reliable estimation. |
| VAEP-style action value | Change in scoring/conceding probability caused by action. | Event/action sequence model. | Compare with action-specific Q-values. | Requires clean action semantics. |
| Pass value | Expected value of a pass if completed, adjusted by completion risk. | Pass probability times receiver state value. | Pass affordance probes. | GRF action space may not expose target cleanly. |
| Carry value | Expected value from moving with ball. | Change in possession value after carry. | Move/dribble action logits and values. | Depends on action labeling. |
| Defensive action value | Value of pressing, intercepting, blocking, or recovering. | Change in opponent threat or possession recovery probability. | Defensive action values. | Often poorly captured by event-only data. |

## Traditional Aggregate Statistics

These are useful context but weaker targets for representation probing because they summarize outcomes over many events.

| Statistic | Definition | Use in this project | Caveat |
| --- | --- | --- | --- |
| Goals | Scored goals. | Outcome sanity check. | Sparse and noisy. |
| Shots | Number of shots. | Compare policy tendencies. | Shot volume ignores quality. |
| Shots on target | Shots requiring save or goal. | Basic attacking tendency. | Not a pure quality measure. |
| Pass completion | Completed passes divided by attempted passes. | Human-GRF comparison if actions are aligned. | Can reward safe non-progressive passing. |
| Progressive passes | Passes moving ball substantially toward goal. | Better than raw completion. | Needs consistent field orientation. |
| Carries | Ball movement by same player. | Useful with tracking or event data. | GRF discrete actions may not map exactly. |
| Turnovers | Losses of possession. | Risk and pressure analysis. | Possession definitions matter. |
| Recoveries | Regaining possession. | Defensive and transition analysis. | Hard to align across data sources. |

## Representation Statistics

These are not soccer statistics, but they are needed to evaluate whether MARL-GPT internally represents soccer concepts.

| Statistic | Definition | Use |
| --- | --- | --- |
| Probe accuracy or AUROC | How well a concept can be decoded from activations. | Basic evidence that a representation contains information. |
| Probe regression error | Error for continuous concepts such as distance, pressure, or value. | Used for geometry and value labels. |
| Layerwise localization | Probe performance by layer. | Shows where concepts become available. |
| Tokenwise localization | Probe performance by ball, teammate, opponent, active-player, or final token. | Shows which entity carries the concept. |
| Branch localization | Probe performance in shared, actor, and critic representations. | Tests whether concepts are action-facing or value-facing. |
| Intervention effect | Change in logits, Q-values, entropy, or selected action after ablation or patching. | Required for stronger mechanistic claims. |
| Domain gap | Difference between GRF and human trajectories for the same concept distribution. | Measures simulation-human modelling gap. |
| Concept calibration | Agreement between model value/logit shifts and concept labels. | Tests whether learned behavior uses concepts sensibly. |

## Minimal First Probe Set

The first probing batch should stay small enough to debug end to end.

| Concept | Label type | Primary metric | Expected behavioral relation |
| --- | --- | --- | --- |
| Possession | 3-class classification | Accuracy | Own possession should enable attacking actions. |
| Nearest defender pressure | Regression and binary classification | RMSE, AUROC | High pressure should reduce risky carry/shot choices. |
| Shot opportunity | Regression or binary classification | AUROC, correlation | High opportunity should raise shot logit or Q-value. |
| Open pass exists | Binary classification | AUROC | Open pass should raise pass-related action preference. |
| Defensive compactness | Regression | RMSE, correlation | Compact defense should reduce central progression value. |

## Interpretation Rules

- High probe performance alone means a concept is decodable, not necessarily used.
- A concept becomes more interesting if it is localized in meaningful tokens or later layers.
- A concept becomes mechanistic only if causal intervention changes relevant action logits, Q-values, or selected actions.
- Directly encoded GRF fields, such as possession and game mode, are sanity checks rather than final contributions.
- Human-football claims require checking whether the same concept definitions make sense outside GRF.

## Links

- [Project brief](../2026-06-30-project-brief.md)
- [Coordination representations in MARL-GPT](../questions/2026-06-30-coordination-representations-in-marl-gpt.md)
- [GRF representation probes](2026-06-30-grf-representation-probes.md)
- [MARL-GPT literature note](../literature/2026-06-30-marl-gpt.md)

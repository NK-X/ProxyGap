# Stage-One Reference Construct Adjudication V8

**Status:** engineering-validated post-run diagnostic; reference competence
construct insufficient; candidate-weight, formal and shaping runs prohibited.

## 1. Question and scope

V7 found that two of five fresh `ctrl_cost_weight = 0.5` policies passed the
prospectively frozen velocity-and-Gym-health gate. V8 asks why the other three
policies terminated and whether that gate adequately represents the intended
behaviour of stable quadrupedal forward locomotion.

No policy was retrained. The five saved 1M models were replayed deterministically
on the original evaluation seeds 51201-51220. Every simulator step was logged.
The resulting 100 episode summaries matched the frozen V6 endpoint exactly,
including zero difference in objective return, net progress, velocity and
torso-tilt RMS. Complete real-time videos were rendered for all five policies
on matched evaluation seed 51216.

## 2. High-z mechanism

All 37 unhealthy terminations were `high_z_excursion`; there were no low-z or
non-finite failures. Across those episodes, torso height increased by
`0.232-0.621` during the final simulated second. Terminal upward velocity was
`0.139-2.640` height units per second. Matched videos show the three affected
policies moving from ordinary torso heights to reared or strongly raised
postures before crossing the upper bound of `1.0`. The events are therefore not
explained by a harmless, stationary offset slightly above the threshold.

## 3. False-positive health case

The more consequential case is training seed 41204. It reached the 1,000-step
time limit in all 20 episodes and therefore passed the V6 Gym-health component.
However, the post-hoc trace audit found:

- `44.795%` of its recorded evaluation steps had torso tilt of at least 90
  degrees;
- `42.780%` of steps had torso height below `0.3`;
- 9 of 20 episodes spent a majority of their recorded steps inverted;
- in the matched video, the final frame shows torso height `0.27` and torso
  tilt approximately `156.2` degrees while the policy remains Gym-healthy.

The 90-degree and 0.3 descriptors are transparent post-hoc diagnostics, not
formal success thresholds. They nevertheless provide a decisive construct
counterexample: satisfying the finite-state and `[0.2,1.0]` torso-height rule
does not imply upright, stable quadrupedal locomotion.

## 4. Cause adjudication

The following facts are established:

1. the three gate failures are rapid high-z excursions;
2. at least one gate-passing policy exhibits sustained inverted, low-posture
   locomotion;
3. the default proxy reward has no direct torso-orientation or lateral-drift
   term;
4. the same PPO configuration produces materially different gait strategies
   across independent training seeds.

The supported interpretation is that stochastic PPO optimisation can converge
to different high-reward behaviours because the proxy and Gym-health rule omit
important aspects of the intended locomotion construct. This is stronger than
the earlier explanation of generic training instability alone.

The current evidence does not isolate the causal contribution of disabled
observation normalisation, network architecture or optimiser settings. It also
does not show a software defect in Ant-v5 or PPO. Observation normalisation may
alter reproducibility, but it cannot by itself add missing posture preferences
to the reward or evaluation construct.

## 5. Consequence for V7

The V7 arithmetic remains correct: two of five policies passed the rule that
was frozen before execution, so its operational label is `inconclusive`.
However, V8 demonstrates that this gate is construct-insufficient and must not
be used as scientific evidence that the two policies are competent reference
robots. The next gate proposed in V7 is therefore revised.

The healthy-z range must not be widened retrospectively. It remains a separate
Gymnasium termination event for reproducibility. It must no longer be described
as comprehensive robot health or used alone to certify baseline suitability.

## 6. Required next gate

Before a normalisation pilot or candidate-weight run, the project must freeze a
human-intent and baseline-suitability contract that:

1. keeps Gym health, posture stability, sustained inversion, forward
   effectiveness and lateral control as separate quantities;
2. states which quantities are eligibility constraints and which remain
   descriptive diagnostics;
3. defines episode-to-policy aggregation and practically meaningful thresholds
   before new outcomes are inspected;
4. preserves the default reward during stage-one detection rather than adding
   orientation or lateral penalties prematurely;
5. uses new development evidence to test any revised gate before consuming
   held-out formal seeds.

Only after that construct freeze should a one-factor observation-normalisation
pilot be considered. Candidate-weight, formal and shaping runs remain blocked.

## 7. Claim boundary

V8 supports a development claim that the current Ant-v5 proxy and height-only
health rule permit high-reward behaviours inconsistent with stable upright
quadrupedal locomotion. It is evidence of a reward-specification risk and
proxy-diagnostic divergence. It is not held-out confirmation of reward hacking
and does not test the predeclared cross-weight stage-one hypothesis.

## 8. Evidence

- machine-readable adjudication:
  `configs/stage1_reference_construct_adjudication_v8_20260814.json`;
- primary replay:
  `artifacts/analysis/stage1_reference_high_z_diagnostic_v8_20260814/high_z_diagnostic.json`;
- independent verification:
  `artifacts/analysis/stage1_reference_high_z_diagnostic_v8_20260814/independent_verification.json`;
- matched contact sheet:
  `artifacts/analysis/stage1_reference_high_z_diagnostic_v8_20260814/matched_video_contact_sheet.png`;
- complete matched videos: the `videos` directory under the same analysis root.

# Research Direction Changelog

## 18 August 2026 - Planar stop-to-lateral command transition

- Continued from the selected pre-pitch `reward-v1-foot-landing` checkpoint;
  no pitch-balance reward or pitch-model weights were used.
- Added two command observations, `(vx_command, vy_command)`, while preserving
  all 113 original observation columns during policy transfer.
- Recorded the audited positive-90-degree motor permutation
  `[6, 7, 0, 1, 2, 3, 4, 5]`; a direct-permutation probe and a distilled warm
  start were retained as negative development evidence because neither alone
  adapted the world-frame state.
- Trained the selected command-conditioned configuration with three seeds and
  four checkpoints per seed. The selected development policy is seed `42001`
  at one million nominal steps.
- Added a GPU-rendered, new-model-only video showing forward motion, a zero
  planar-velocity braking command, and positive-y translation without a yaw
  command.

## 17 August 2026 - Foot-landing and pitch-balance reward iterations

- Recorded `reward-v1-foot-landing`, which reduces torso x/y influence and
  adds four-foot grounded Vy/Vz shaping at a `0.03 m` height threshold.
- Recorded `reward-v2-pitch-balance`, which additionally balances positive and
  negative signed torso-pitch time between the first and fourth distinct foot
  landing.
- Retained the ineffective pitch-weight `0.1` run as calibration provenance;
  the accepted development version freezes weight `5.0` under new seeds.
- Added public, sanitised configuration/model/video hashes without committing
  generated models, full logs or MP4 files.
- Documented the improvement in the declared pitch-balance metric and the
  simultaneous increase in landing frequency and action roughness.

## 17 August 2026 - Versioned V2 handover

- Declared V2 as the canonical current direction.
- Retained V1 as historical exploratory evidence without deleting or rewriting
  its source, configurations or result indexes.
- Added `current/`, `legacy/` and `handoff/` navigation layers.
- Made specified gait operationalisation the next scientific gate.
- Clarified that future protocols and code may be public while newly generated
  raw outputs remain local pending evidence review.
- Preserved all existing executable paths to avoid breaking reproduction.

## 16 August 2026 - Development synthesis

- Reframed the project from a one-directional control-cost sweep to a
  default-reward construct audit followed by bounded mitigation development.
- Added posture, direction, path, action, termination and body-motion
  diagnostics.
- Recorded unresolved natural-gait and learned-control questions.

## Earlier work - V1 coefficient study

- Trained Ant-v5 PPO policies under several `ctrl_cost_weight` values.
- Added common rescoring and separate behavioural diagnostics.
- Retained the resulting material for provenance and retrospective analysis.
- The work is not treated as V2 formal evidence.

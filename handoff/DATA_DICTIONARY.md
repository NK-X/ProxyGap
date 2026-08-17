# Data and Evidence Dictionary

## Evidence levels

| Label | Meaning | Permitted use |
|---|---|---|
| `smoke` | Tiny engineering execution | Pipeline validation only |
| `development` | Data used to choose or reject designs | Mechanism exploration and design decisions |
| `retrospective` | Analysis defined after original execution | Descriptive or sensitivity evidence |
| `held_out_formal` | Frozen design evaluated on untouched training seeds | Confirmatory claim within the declared scope |

## Replication units

- `training_seed`: creates an independently trained policy and is the primary
  replication unit.
- `evaluation_seed`: creates a paired initial-condition/evaluation episode for
  a fixed policy; it is a nested repeated measurement, not another trained
  policy.
- `checkpoint`: a repeated measurement over training time, not an independent
  replicate.
- `video`: a qualitative audit of a prespecified episode, not a sample-size
  contribution.

## Core identifiers

| Field | Meaning |
|---|---|
| `config_id` | Immutable identifier of the executable experimental design |
| `condition_id` | Declared reward, constraint or mechanism condition |
| `training_seed` | PPO training randomness identifier |
| `evaluation_seed` | Evaluation reset and stochasticity identifier |
| `target_timesteps` | Intended training budget |
| `checkpoint_timesteps` | Saved policy-evaluation points |
| `model_sha256` | Integrity digest of a trained policy archive |
| `video_sha256` | Integrity digest of a rendered qualitative artifact |

## Main measure families

| Family | Examples | Interpretation |
|---|---|---|
| Proxy objective | condition return, base proxy return, common rescore | Optimiser-facing or comparator quantities; not human truth |
| Task | forward velocity, target tracking, net progress | Whether the commanded locomotion task is achieved |
| Direction and path | lateral velocity, direction error, path efficiency | Whether movement follows the intended route |
| Posture and body dynamics | tilt, vertical velocity, roll/pitch angular speed | Stability diagnostics |
| Contact and gait | contact sequence, flight fraction, take-offs | Coordination diagnostics; V2 definitions not yet frozen |
| Action and effort | action roughness, saturation, control effort | Command quality and proxy effort |
| Termination | unhealthy termination, truncation category | Environment-defined episode outcome |

Metric formulae, units and aggregation rules belong in versioned metric
contracts. A field with the same name must not silently change meaning between
versions.

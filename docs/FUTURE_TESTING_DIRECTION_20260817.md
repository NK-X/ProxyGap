# Future Testing Direction (17 August 2026)

## Decision

The next local experiment is one final bounded development replication of the
ordinary-exploration body-dynamics candidate. It does not retest the rejected
frequency-8 gSDE configuration, tune new reward weights or use reserved formal
training seeds.

## Question

Does the previously observed reduction in vertical and roll/pitch body motion
replicate in independently trained PPO policies when the only experimental
difference is whether the already calibrated body-dynamics shaping term is
enabled?

## Two conditions

| Condition | Shared reward package | Body-dynamics term | Exploration |
|---|---|---|---|
| `B0__G0_REP` | target speed, orientation, lateral velocity and action rate | absent | ordinary PPO Gaussian |
| `B1__G0_REP` | identical | fixed vertical and roll/pitch penalty | ordinary PPO Gaussian |

The body weights and scales are copied unchanged from the completed 16 August
matrix. This is a replication check, not a new coefficient search.

## Replication and evidence boundary

- Training seeds: `41601`, `41602`, `41603`.
- Evaluation seeds: `51601` to `51610`, paired across policies.
- Training budget: 1,000,000 steps per policy.
- Checkpoints: 250k, 500k, 750k and 1M.
- Reserved formal training seeds `42101` to `42105` remain untouched.
- The training seed/policy is the independent replication unit.
- Complete videos remain qualitative audit evidence and are not counted as
  additional replications.

## Predeclared interpretation

The candidate receives replication support only if the paired policy-level
contrasts show the expected direction in at least two of the three training
seed pairs for both primary body-rate measures:

1. RMS root vertical velocity; and
2. RMS root roll/pitch angular speed.

No-floor-contact fraction and prominent take-off count are supporting contact
diagnostics. Forward command tracking, unhealthy termination, path efficiency,
direction error, torso tilt, action roughness and saturation are guardrails and
must all be reported. A zero value on the strict all-domain compliance gate
does not erase continuous improvements, but it prevents a claim of complete
mitigation.

The experiment may support partial, body-specific mitigation. It cannot by
itself establish a biological gait, physical safety, terrain robustness or
formal held-out confirmation. The coefficients will not be changed after
viewing these results.

## Stopping and exclusion rules

- Complete all six training tasks unless a documented engineering failure
  prevents valid execution.
- Do not replace an inconvenient seed or extend only one condition.
- Exclude a run only for a prespecified technical failure, with the failed run
  retained and reported.
- Do not launch reserved formal seeds from this protocol.

## Publication boundary

This direction, the protocol, configuration, runner and tests are public. New
models, logs, videos, recovery files and unreviewed result summaries remain in
the local workspace. They require a separate evidence audit before any later
public release.

# Reward Iteration History (17 August 2026)

## Scope and naming

This record separates two user-directed reward iterations that were trained on
17 August 2026. The public sequence labels below describe the reward evolution;
they are distinct from the repository-wide V1/V2 research-direction labels.

| Public sequence label | Frozen configuration | Role |
|---|---|---|
| `reward-v1-foot-landing` | `configs/foot_landing_velocity_v1_20260817.json` | Reduced torso x/y influence and added four-foot landing Vy/Vz control |
| `reward-v2-pitch-balance` | `configs/pitch_balance_v3_calibrated_20260817.json` | Retained reward-v1 and added signed torso-pitch time balance |

The internal configuration name contains `v3` because a low-weight pitch
configuration was retained as a calibration attempt. It does not represent a
third public reward version.

## Shared base reward package

Both published iterations use the following package:

- target forward velocity: `1.0 m/s`;
- forward tracking weight: `0.5`;
- lateral-velocity weight: `0.025` with a zero target;
- cosine torso-orientation weight: `0.1`;
- action-rate weight: `0.2`;
- root vertical-velocity weight: `0.05`;
- root roll/pitch angular-speed weight: `0.05`;
- PPO `MlpPolicy`, two hidden layers of 64 units, three independent training
  seeds, and `1,000,000` nominal steps per policy.

The Ant action remains an eight-dimensional torque command. Reward terms are
computed from simulator state after the action has been applied.

## reward-v1-foot-landing

### Intervention

For each of four named ankle capsules, the distal sphere bottom is considered
grounded when its world height is at most `0.03 m`. While grounded, lateral and
vertical foot speeds are each penalised by

```text
tanh(((velocity - 0) / 1.0)^2)
```

with weight `0.025` per foot and direction. The maximum lateral and vertical
foot penalty budgets are therefore `0.1` each per environment step.

The frozen development run contained a matched no-foot-term condition and the
foot-term condition. There were six policies in total: two conditions by three
training seeds, each trained for one million nominal steps.

### Recorded representative result

- condition: `F1__FOOT_LANDING`;
- training seed: `41703`;
- representative rendered evaluation seed: `51710`;
- final-checkpoint evaluation mean over 10 episodes:
  - fixed-horizon velocity `0.955377 m/s`;
  - direction error `1.568335 deg`;
  - forward path efficiency `0.845783`;
  - normalised action roughness `0.002936`;
  - unhealthy terminations `0/10`;
- representative video replay:
  - velocity `0.956859 m/s`;
  - direction error `0.070409 deg`;
  - path efficiency `0.848140`;
  - no unhealthy termination.

This version did not optimise the signed-pitch balance objective. A later
post-hoc, read-only evaluation of its selected policy produced a weighted
pitch-event balance score of `0.500775`; that value was not part of its
original reward or selection rule.

## reward-v2-pitch-balance

### Landing event and reward

A new landing is the transition from foot height above `0.03 m` to height at
or below `0.03 m`. An event begins at the first new landing and ends when all
four distinct feet have produced a new landing. Initial grounded feet do not
count as new landings.

Signed torso pitch is reconstructed from the normalised MuJoCo torso
quaternion `(w, x, y, z)`:

```text
pitch = asin(clip(2 * (w*y - z*x), -1, 1))
```

Within an active event, `T_positive` and `T_negative` count environment steps
with pitch respectively above and below zero. Exact-zero or non-finite steps
are excluded from the denominator. When the fourth distinct foot lands, the
event receives

```text
balance_score = 1 - abs(T_positive - T_negative) / (T_positive + T_negative)
pitch_reward = 5.0 * balance_score
```

The reward is issued once at event completion. The score is one for equal
positive and negative time and zero when all signed time lies on one side.

### Weight calibration

The retained calibration configuration
`configs/pitch_balance_v2_20260817.json` used weight `0.1`. Its three endpoint
scores were `0.202668`, `0.206055` and `0.012348`, below the post-hoc reference
score `0.500775`. The pitch term contributed only `0.07` to `1.34` reward units
per episode against objective returns of roughly `1330` to `1410`, so it was
not accepted as the second reward version.

The calibrated configuration froze weight `5.0` before launching new training
seeds `41901`, `41902` and `41903`.

### Recorded representative result

- condition: `P2__PITCH_BALANCE_CAL`;
- selected training seed: `41903`;
- representative rendered evaluation seed: `51903`;
- final-checkpoint evaluation mean over 10 episodes:
  - weighted pitch-event balance score `0.757884`;
  - mean positive/negative active-event time `18.700 / 23.705 s`;
  - completed events `111.0` per episode;
  - fixed-horizon velocity `0.897005 m/s`;
  - direction error `5.628938 deg`;
  - forward path efficiency `0.841685`;
  - normalised action roughness `0.017853`;
  - unhealthy terminations `0/10`;
- representative video replay:
  - pitch-event score `0.755877`;
  - positive/negative time `18.7 / 23.6 s`;
  - velocity `0.918679 m/s`;
  - direction error `4.246303 deg`;
  - path efficiency `0.861094`;
  - no unhealthy termination.

All three new policies exceeded the post-hoc reference pitch score, with
endpoint scores `0.706724`, `0.709691` and `0.757884`.

## Interpretation boundary

The pitch reward improved its declared balance metric, but it also increased
landing-event frequency and action roughness. It did not improve every gait
dimension: speed and path efficiency were lower than the pre-pitch selected
policy, and mean torso tilt magnitude was higher. These development results do
not establish a natural gait, hardware transfer, biological fidelity or a
held-out formal comparison.

## Reproduction and publication boundary

The exact configurations, code and tests are Git-tracked. Generated model
archives, complete evaluation tables and MP4 files remain under the ignored
`artifacts/` tree. Their filenames and SHA-256 digests are recorded in
`results/development_20260817/reward_iterations/version_manifest.json` so they
can be verified or published separately without embedding machine-specific
absolute paths in the repository.

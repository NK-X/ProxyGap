# Curved-gait development, 18 August 2026

## Scope

This development trains the low-level walking effect required by a curve: the
torso translates while its forward axis yaws with the instantaneous curve
tangent. It deliberately does **not** train a route planner or positional path
tracker. Curve coordinates, waypoints, global route position and route-error
reward are absent from the policy and reward.

The source policy is the selected pre-pitch planar model:

- `artifacts/dev/planar_translation_transition_v3_20260818/runs/seed_42001/models/checkpoint_1000000.zip`;
- SHA-256 `e337df745896c0c8670ac50754499fece3a1c92a84efbe29803b53bfc43f79ee`;
- pitch-balance reward disabled.

## Interface and action meaning

The policy observation has 118 values. The preserved Ant proprioception and
previous-action history occupy 113 values. Five local commands are appended:

1. commanded forward speed in the body/tangent frame;
2. commanded lateral speed, fixed at zero here;
3. commanded yaw rate, `speed * curvature`;
4. sine of torso-heading error;
5. cosine of torso-heading error.

The output remains the eight Ant-v5 motor-control values. They are continuous
normalised actuator controls interpreted by the MuJoCo Ant motors, not joint
position or joint-velocity targets.

V4 rotates the root quaternion, root planar linear/angular vectors and external
body force/torque vectors into the instantaneous target-tangent frame before
inference. Joint states and the eight previous actions remain body-fixed. The
old forward policy therefore continues to see a canonical `+x` walking problem
as the world heading changes.

## Reward and curriculum

V4 preserves the pre-pitch gait package, including action-rate, vertical-body
velocity, roll/pitch angular velocity and the four feet's landing `Vy/Vz`
terms. Pitch-balance shaping remains zero.

The curve-specific terms are:

- world tangent-velocity tracking, weight `0.75`;
- cross-axis velocity penalty, weight `0.2`;
- bounded heading-alignment penalty, weight `1.5`;
- bounded yaw-rate tracking penalty, weight `1.0`.

Training uses no heading-error hard termination so that the inherited policy
can learn recovery from large initial yaw error. Evaluation retains the strict
rule: error above 20 degrees for five consecutive steps ends the episode. The
four curriculum endpoints are 102,400, 307,200, 614,400 and 1,024,000 steps,
ending at absolute curvature `0.35 m^-1`. Eight subprocess environments feed
each PPO learner.

## Version audit

| Version | Key change | Last inspected result | Decision |
|---|---|---|---|
| V1 | World-frame `vx/vy` plus yaw commands | Three completed seeds; all five non-straight profiles terminated in every final evaluation | Rejected: retained lateral translation rather than turning |
| V2 | Body-frame forward/lateral commands | Seed 43101 at 307,200 steps: mean curved heading RMS 14.08 degrees, cross-axis RMS 0.375 m/s, 5/5 terminations | Rejected |
| V3 | Bounded yaw losses; no training-time heading termination | Seed 43201 at 614,400 steps: mean curved heading RMS 13.50 degrees, cross-axis RMS 0.460 m/s, 5/5 terminations | Rejected: full training episodes were restored but arbitrary world yaw still had to be relearned |
| V4 | Target-tangent observation canonicalisation | Two 1,024,000-step seeds completed/recovered | Retained development result |

The attempted third V4 seed was stopped during environment startup after the
Windows memory/page-file limit was reached while three eight-environment jobs
were concurrent. It produced no valid candidate and is excluded rather than
counted as a failed policy.

## V4 final strict evaluations

Each strict evaluation is at 0.8 m/s for at most 600 steps. Values below are
descriptive development evidence, not a held-out formal claim.

| Seed | Profile | Steps | Heading RMS | Within 5 degrees | Cross-axis RMS |
|---:|---|---:|---:|---:|---:|
| 43301 | straight | 600 | 5.72 deg | 0.575 | 0.351 m/s |
| 43301 | gentle left | 600 | 5.91 deg | 0.617 | 0.378 m/s |
| 43301 | gentle right | 600 | 6.49 deg | 0.620 | 0.384 m/s |
| 43301 | medium left | 101 | 11.71 deg | 0.218 | 0.257 m/s |
| 43301 | medium right | 329 | 9.13 deg | 0.334 | 0.395 m/s |
| 43301 | S curve | 366 | 10.39 deg | 0.306 | 0.322 m/s |
| 43302 | straight | 600 | 5.40 deg | 0.662 | 0.337 m/s |
| 43302 | gentle left | 600 | 5.74 deg | 0.638 | 0.390 m/s |
| 43302 | gentle right | 497 | 6.72 deg | 0.616 | 0.427 m/s |
| 43302 | medium left | 114 | 9.77 deg | 0.219 | 0.297 m/s |
| 43302 | medium right | 274 | 12.01 deg | 0.124 | 0.392 m/s |
| 43302 | S curve | 600 | 9.74 deg | 0.312 | 0.346 m/s |

Seed 43301 has the slightly lower aggregate non-straight heading RMS
(`0.1523 rad` versus `0.1535 rad`) and cross-axis RMS (`0.347` versus `0.370
m/s`) with the same three strict terminations. Seed 43302 is used for the S
curve demonstration because it completes all 600 S-curve steps.

## Reproduction

```powershell
python scripts/run_curved_gait_training.py `
  --config configs/curved_gait_tangent_v4_canonical_frame_20260818.json `
  --seed 43302 --device cuda

python scripts/render_curved_gait_video.py `
  --config configs/curved_gait_tangent_v4_canonical_frame_20260818.json `
  --model artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43302/models/checkpoint_1024000.zip `
  --profile s_curve --curvature 0.25 --speed 0.8 --steps 300 `
  --evaluation-seed 53316 --camera-distance 16 --camera-azimuth 125 `
  --camera-elevation -60 --hide-target-path `
  --output artifacts/videos/curved_gait_v4_final_s_curve_fixed_camera.mp4
```

## Centre-start figure-eight route

`scripts/render_figure_eight_route_video.py` supplies a high-level route
controller outside the learned policy. The route consists of two tangent
circles of radius 4 m. The initial root position is exactly the centre
intersection `(0, 0)`, with the torso initialised along the shared `-Y`
tangent. The controller traverses the right circle counter-clockwise, returns
through the intersection, traverses the left circle clockwise, then stops on
return to the centre. Route coordinates and position error do not enter the
118-value policy observation and no route-position reward is enabled.

The retained V4 seed-43301 rollout used a fixed distant camera and four-times
playback:

```powershell
python scripts/render_figure_eight_route_video.py `
  --config configs/curved_gait_tangent_v4_canonical_frame_20260818.json `
  --model artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip `
  --radius 4 --speed 0.8 --lookahead 0.4 `
  --controller-mode pure_pursuit --yaw-feedback-gain 0.5 `
  --yaw-rate-limit 0.28 --max-steps 3800 --playback-speed 4 `
  --camera-distance 16 --camera-azimuth 135 --camera-elevation -62 `
  --output artifacts/videos/figure_eight_center_start_v4_fixed_camera.mp4
```

The rollout completed both loops in 2,233 steps (111.65 simulated seconds),
did not fall, achieved route-error RMS 0.175 m, and returned 0.452 m from the
exact centre. Its body-axis-to-path-tangent error remained 23.22 degrees RMS
(mean -21.60 degrees), so it is retained as a route-following demonstration,
not evidence that the torso-tangent requirement is solved. The MP4 SHA-256 is
`a592fce43bedfc06e564d96c91509ecaca44c29988e1a8495b65a8e70f530a4f`;
the adjacent JSON manifest contains the full controller and rollout record.

V5 long-horizon continuation and V6 coupled lateral/yaw continuation are
preserved as rejected development attempts. V5's final policy reached only
15.35% of this route in 3,800 steps. V6 improved strict local heading metrics
but still failed the closed-loop route because commanded lateral correction
did not reverse the policy's residual lateral drift. Neither model is used in
the video.

## Claim boundary

The result demonstrates local yaw-conditioned walking on flat-ground Ant-v5.
It does not yet establish robust medium-curvature control, commanded-speed
tracking, closed-loop route following, arbitrary trajectories, terrain
robustness or real-robot transfer. The remaining cross-axis and tangent-speed
errors should be improved before a higher-level path follower relies on this
policy.

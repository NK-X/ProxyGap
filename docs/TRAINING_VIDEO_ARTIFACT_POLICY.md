# Required training video artifact policy

Status: active for subsequent formal ProxyGap training runs.

Every completed formal training run must retain at least one representative final-checkpoint video beside its model, configuration, raw evaluation rows and execution record.

## Required selection rule

- Choose the representative evaluation seed before inspecting video quality or outcome.
- Use the final checkpoint selected by the experiment's predeclared checkpoint rule.
- Use deterministic action selection unless the frozen protocol explicitly studies stochastic policies.
- Do not replace a failed, unstable or visually unattractive rollout with a better-looking seed.
- Label any additional best-case or failure-case video as exploratory; it cannot replace the representative video.

## Required content

- At least 10 seconds of physical MuJoCo rollout; the fixed-map trainer currently records 45 seconds.
- A camera that follows the robot position.
- Adaptive camera elevation in valleys so heightfield terrain does not occlude the robot.
- Full-map inset with true elevation contours, start, goal, current position and trail.
- On-frame physical time, goal distance, best progress, support-foot count, torso tilt, contact-speed proxy and airborne/fall event state.
- No scripted joint animation; all joint actions must come from the saved policy checkpoint.

## Required provenance and QA

- Model, configuration, scene, height-array and video SHA-256 hashes.
- Training and evaluation seed identifiers.
- Friction and contact-dimension verification.
- Machine-readable per-step trace.
- Contact sheet containing start, intermediate and final frames.
- Full H.264 decode validation, including frame count, frame rate, dimensions and duration.
- Video-artifact status in `execution_record.json`.

The training model may remain technically valid if video encoding fails, but the run is not a complete deliverable until the required video artifact passes decoding and provenance checks.


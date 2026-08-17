# Planar Translation Transition (18 August 2026)

## Requested behaviour

The robot begins with a positive-x translation command. At a sudden lateral
request it first receives a zero planar-velocity command, must slow to the
declared stop threshold, and then receives a positive-y translation command.
No torso-yaw command is used. This work covers one fixed 90-degree change on a
flat plane; it does not yet cover arbitrary headings or z-axis steering.

## Starting policy and transfer

The starting point is the selected model before the pitch-balance reward was
added:

- condition: `F1__FOOT_LANDING`;
- training seed: `41703`;
- checkpoint: `1,000,000` nominal steps;
- SHA-256: `fba1413eb4b26f5517189aa03bcc86a7b7767eb8f2f84743a115ff916d1def94`;
- source observation/action dimensions: `113 / 8`;
- pitch-balance weight: `0.0`.

The new observation appends `(vx_command, vy_command)`, producing 115 inputs.
All equal-shaped policy/value tensors are copied exactly. The two expanded
first layers retain the first 113 source columns and initialise the two command
columns to zero. The measured initial deterministic-action transfer error was
`0.0` for all three full training seeds.

The audited positive-90-degree actuator mapping is:

```text
source order:       hip_4 ankle_4 hip_1 ankle_1 hip_2 ankle_2 hip_3 ankle_3
destination index:  6     7       0     1       2     3       4     5
```

Directly applying the permutation did not produce stable positive-y walking,
because the policy state remained in the unrotated world/body frame. A later
supervised symmetry warm start also preserved forward motion but suppressed
lateral motion after PPO fine-tuning. The mapping is therefore retained as an
audited physical prior, while the selected model learns state adaptation from
the command-conditioned environment.

## Selected environment and reward

The selected frozen configuration is
`configs/planar_translation_transition_v3_20260818.json`.

- command sequence: `(1,0) -> (0,0) -> (0,1) m/s`;
- training switch step: uniformly sampled from 160 through 320;
- evaluation switch step: 200;
- stop threshold: planar speed at or below `0.15 m/s` for three consecutive
  steps;
- maximum braking window: 60 steps (`3.0 s`);
- two-dimensional pseudo-Huber velocity-tracking weight/scale: `1.0 / 0.5`;
- cross-axis velocity weight/scale: `0.25 / 0.5`;
- torso-yaw penalty: weight `0.3`, scale `15 deg`, referenced to episode initial
  yaw;
- braking-speed weight/scale: `1.0 / 0.35`;
- preserved pre-pitch foot Vy/Vz, body-smoothness, action-rate, uprightness and
  control-cost terms;
- pitch-balance reward: disabled.

## Development calibration history

| Configuration | 250k pilot result | Decision |
|---|---|---|
| V1 | Lateral speed about `0.53 m/s`; no strict stops; yaw about `42 deg` | Signals too weak |
| V2 | Strict stops in 5/5 but collapsed to nearly stationary motion | Exponential tracking gradient vanished away from target |
| V3 | Lateral speed about `0.61 m/s`; 4/5 strict stops; no falls | Selected reward balance |
| V4 | Stronger non-saturating yaw constraint; only 2/5 strict stops | Not selected |
| V5 | Explicit 90-degree motor distillation; lateral speed about `0.02 m/s` | Direct motor prior did not solve state adaptation |

## Full training and selection

V3 trained seeds `42001`, `42002` and `42003` to one million nominal steps.
Each saved and evaluated checkpoints at 250k, 500k, 750k and 1M over ten fixed
evaluation seeds. There were 120 evaluation episodes in total and no training
task failures.

The selected model is seed `42001`, checkpoint `1,000,000`. Its ten-episode
means are:

- forward-phase velocity: `(vx, vy) = (0.57, 0.24) m/s`;
- lateral-phase velocity: `(vx, vy) = (0.12, 0.76) m/s`;
- strict consecutive-stop rate: `5/10`;
- near-stop rate (minimum braking speed at most `0.15 m/s`): `8/10`;
- mean yaw-error RMS: `12.8 deg`;
- falls: `0/10`;
- full fixed horizon: `10/10`.

The strict stop metric is deliberately reported separately from minimum speed:
several policies crossed the speed threshold but rebounded before maintaining
it for three steps.

## Selected deliverables

- model SHA-256:
  `e337df745896c0c8670ac50754499fece3a1c92a84efbe29803b53bfc43f79ee`;
- frozen configuration SHA-256:
  `f1d3ac14a08386424e43baa49bcff1ce01c90c5b0fc6c1bca255b10264a9e031`;
- final video evaluation seed: `52011`;
- final video: 500 frames, 20 fps, 25 seconds, GPU inference;
- final video SHA-256:
  `4c6375103e08c310545dfadedff71956e274b485b215581652a5e8465683a949`.

The rendered trajectory achieved the strict stop condition in `1.5 s`, then
entered the positive-y phase. Its phase means were `(0.54, 0.22) m/s` forward
and `(0.14, 0.76) m/s` lateral, with yaw-error RMS `12.2 deg` and no fall.

## Reproduction

```powershell
python scripts/run_planar_translation_transition.py `
  --config configs/planar_translation_transition_v3_20260818.json `
  --max-workers 3

python scripts/render_planar_transition_video.py `
  --model artifacts/dev/planar_translation_transition_v3_20260818/runs/seed_42001/models/checkpoint_1000000.zip `
  --training-seed 42001 --evaluation-seed 52011 `
  --target-timesteps 1000000 --max-steps 500 --device cuda `
  --output artifacts/dev/planar_translation_transition_v3_20260818/videos/FINAL_planar_transition_seed_42001_step_1000000_eval_52011.mp4
```

## Claim boundary

This is a development demonstration for one flat-ground, fixed-quarter-turn
translation transition. It does not establish arbitrary planar navigation,
torso-yaw steering, disturbance recovery, terrain robustness, natural gait,
real-robot transfer or held-out confirmatory performance.

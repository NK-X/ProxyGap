# Body-smoothness and gSDE mechanism matrix

**Status:** frozen authorised development protocol; formal launch prohibited.

## Question

The preceding 1M target-tracking policies reduced unhealthy termination and
policy-output roughness but retained approximately one-half of their steps with
no floor contact. This matrix asks whether that residual hopping is reduced by
(a) directly penalising bounded torso vertical and roll/pitch dynamics, (b)
temporally correlated state-dependent exploration, or (c) their combination.

## Construct boundary

This is a body-level smoothness experiment, not a biological-gait experiment.
It does not require crawl, trot, pace or bound. Contact patterns remain
diagnostics and are not retrospectively labelled natural or unnatural.

## Frozen factors

| Condition | Body-dynamics shaping | PPO exploration |
|---|---|---|
| B0__G0 | absent | ordinary PPO Gaussian exploration |
| B1__G0 | present | ordinary PPO Gaussian exploration |
| B0__G8 | absent | gSDE, resampled every 8 steps |
| B1__G8 | present | gSDE, resampled every 8 steps |

All conditions retain the same 1 m/s target-velocity reward, orientation and
lateral shaping, action-rate weight 0.2, default control cost, observation,
PPO network, optimiser and 1M-step budget.

The bounded body penalty is

\[
p_z(t)=\tanh\left[\left(\frac{v_z(t)}{1.0141}\right)^2\right],
\qquad
p_{\omega}(t)=\tanh\left[\left(
\frac{\sqrt{\omega_x(t)^2+\omega_y(t)^2}}{1.9893}
\right)^2\right],
\]

and contributes

\[
r_{\mathrm{body}}(t)=-0.05p_z(t)-0.05p_{\omega}(t).
\]

The scales are pooled 90th percentiles from six prior development endpoints,
60 evaluation episodes and 59,106 valid steps. The maximum combined penalty is
0.1 per step; its estimated mean under the calibration policies is 0.029 per
step. These values are development choices, not physical safety limits.

## Replication and evaluation

- Training seeds: 41501-41503, paired across all four conditions.
- Evaluation seeds: 51501-51510, paired across every policy checkpoint.
- Checkpoints: 250k, 500k, 750k and 1M.
- Independent replication unit: the training seed/policy.
- Primary disaggregated outcomes: commanded velocity, path efficiency,
  direction error, termination, action roughness, torso tilt, vertical velocity,
  roll/pitch angular velocity, no-floor-contact fraction, take-off count and raw
  MuJoCo impact diagnostics.

## Interpretation gate

Body shaping is mechanistically supported only when matched-seed endpoint
contrasts reduce vertical/angular dynamics or flight diagnostics without a
material collapse of target tracking. gSDE is supported only when its matched
contrast improves the same body outcomes or policy-output roughness. A better
single video is insufficient. No coefficient may be retuned from these results.

## Evidence basis

Gymnasium Ant-v5 uses direct torque actions and does not specify gait phase or
body-rate penalties. Raffin, Kober and Stulp (2022) show that independent
step-wise exploration can produce jerky robotic trajectories and propose smooth
state-dependent exploration. Aractingi et al. (2023) separately penalise action
differences and body/control quantities for quadruped locomotion. These sources
justify testing the mechanisms; they do not guarantee success in Ant-v5.

## References

Aractingi, M. *et al.* (2023) 'Controlling the Solo12 quadruped robot with deep
reinforcement learning', *Scientific Reports*, 13, 11945.

Farama Foundation (2026) 'Ant - Gymnasium documentation'. Available at:
https://gymnasium.farama.org/environments/mujoco/ant/.

Raffin, A., Kober, J. and Stulp, F. (2022) 'Smooth exploration for robotic
reinforcement learning', *Proceedings of Machine Learning Research*, 164,
pp. 1634-1644.

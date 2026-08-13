# ProxyGap v2 Metric Definitions

**Date:** 10 August 2026  
**Status:** implemented metric contract; `effort_distance_min` remains subject to the revision gate  
**Language:** British Academic English

## 1. Data grain and aggregation

The independent experimental unit is one trained `condition x training_seed` policy. An evaluation episode is a nested observation of that policy. Checkpoints are repeated observations of one evolving policy and are not independent replicates.

The required aggregation order is:

1. calculate step-level quantities;
2. aggregate steps to one episode;
3. average evaluation episodes within one trained policy and checkpoint;
4. calculate paired condition differences within each training seed;
5. summarise the seed-level differences without treating episodes or checkpoints as additional independent samples.

All rewards are dimensionless environment reward units. Positions and distances are reported in MuJoCo simulation length units. Torso angles are in radians. Actions are dimensionless values bounded to `[-1, 1]`.

## 2. Reward measures

### 2.1 Condition-specific training objective

For a control-cost weight `w`, the unshaped Ant objective is

```text
r_condition,t = r_forward,t + r_survive,t + r_contact,t - w * sum_j(a_t,j^2)
```

`condition_objective_return` is the episode sum of the reward actually supplied to PPO, including any frozen prospective intervention. It is the correct measure of what each condition optimised. It is not directly commensurable across conditions with different reward definitions.

The legacy field `proxy_return` remains an exact alias for formal-v1 compatibility. Prospective reports must use `condition_objective_return`.

### 2.2 Common counterfactual rescore

All evaluation trajectories are additionally rescored under the fixed default control-cost weight `w_common=0.5`:

```text
common_rescored_return
  = reward_forward_sum
  + reward_survive_sum
  + reward_contact_sum
  - 0.5 * cumulative_squared_action
```

The rescore excludes shaping additions. It enables a common numerical comparison of realised trajectories but does not constitute a ground-truth or `true_performance` objective.

### 2.3 Reward reconciliation

For every episode:

```text
base_proxy_return
  = reward_forward_sum
  + reward_survive_sum
  + reward_contact_sum
  - condition_ctrl_cost_weight * cumulative_squared_action
```

The absolute reconstruction error must not exceed `1e-4` for CSV-round-tripped data or `1e-8` for in-memory step checks. A larger error is a technical stop.

### 2.4 Prospective bounded intervention

The implemented candidate forms are:

```text
effort_penalty_t
  = -lambda_effort * tanh(squared_action_step / scale_effort)

orientation_penalty_t
  = -lambda_orientation * tanh(torso_tilt_rad / scale_orientation)
```

Effort-only, orientation-only and combined candidate types are supported. The four numeric scale and cap parameters are not yet frozen. Forward-reward reweighting is not a proposal-conformant v2 candidate.

## 3. Forward movement

### 3.1 Net forward progress

```text
net_forward_progress = final_x - initial_x
```

Larger values indicate greater net forward displacement. This is the primary diagnostic, not a scalar true reward. A policy may obtain positive progress before an unhealthy termination, so termination category and episode length must accompany the measure.

### 3.2 Net forward progress per step

```text
net_forward_progress_per_step = net_forward_progress / episode_length
```

This rate helps expose episode-length mediation. It remains a secondary diagnostic because short terminated episodes and full-length episodes may represent qualitatively different behaviours.

## 4. Action-based diagnostics

### 4.1 Cumulative squared action

```text
cumulative_squared_action = sum_t sum_j(a_t,j^2)
```

Lower values indicate smaller accumulated command magnitudes, conditional on the realised episode. This quantity is not mechanical work, electrical energy, torque expenditure or hardware wear. The legacy `control_effort` field is retained only as a formal-v1 alias.

### 4.2 Mean squared action per step

```text
mean_squared_action_per_step
  = cumulative_squared_action / episode_length
```

This measure reduces, but does not remove, episode-length dependence.

### 4.3 Action saturation rate

```text
action_saturation_rate
  = count(|a_t,j| >= 0.95) / count(all action components)
```

This is a prospective exploratory/protected diagnostic for repeated extreme commands. It may not be used post hoc to redefine the primary outcome.

### 4.4 Squared action per unit distance

```text
cumulative_squared_action_per_unit_distance
  = cumulative_squared_action / net_forward_progress
```

The ratio is defined only when `net_forward_progress > effort_distance_min`. Otherwise it is stored as `NaN`, while the numerator, denominator, `effort_per_distance_defined` flag and undefined proportion remain reported. Near-zero denominator sensitivity must be evaluated using the prespecified candidate values `0.05`, `0.10`, `0.25` and `0.50` simulation length units.

The operative `effort_distance_min` has not yet been approved. This metric therefore remains a revision-gate blocker rather than an already locked measure.

## 5. Termination and episode status

`unhealthy_termination` is true when Ant-v5 terminates because the state is outside its healthy definition. It is separated into mutually exclusive categories:

| Category | Operational definition |
|---|---|
| `low_z_collapse` | termination with torso height below `0.2` |
| `high_z_excursion` | termination with torso height above `1.0` |
| `non_finite_state` | termination with non-finite `qpos` or `qvel` |
| `other_unhealthy` | termination not explained by the preceding recorded categories |
| `none` | no unhealthy termination on the recorded step |

`time_limit_truncation` is logged independently when the 1,000-step TimeLimit is reached. The prospective report must not call all unhealthy terminations `falls`. The legacy field `fall` remains an alias of `unhealthy_termination` only to preserve formal-v1 readability.

Any non-finite state is a catastrophic technical/scientific event: the record is retained, the condition is stopped for audit, and the seed is not silently discarded.

## 6. Lateral movement

Let `d_y,t = |y_t-y_0|`.

```text
lateral_drift_final_abs = d_y,T
lateral_drift_mean_abs  = mean_t(d_y,t)
lateral_drift_max_abs   = max_t(d_y,t)
cumulative_lateral_path = sum_t |y_t-y_(t-1)|
```

Mean and maximum absolute offset are the primary lateral diagnostics. Final offset is secondary because outward-and-return trajectories can exploit it. Cumulative lateral path measures lateral movement rather than final displacement and is also secondary.

## 7. Torso-orientation diagnostics

For the normalised MuJoCo root quaternion `(w, x, y, z)`, vertical alignment is

```text
alignment = clip(1 - 2 * (x^2 + y^2), -1, 1)
torso_tilt_rad = acos(alignment)
```

An upright torso has an angle near zero. The episode summaries are:

```text
torso_tilt_mean = mean(theta_t)
torso_tilt_std  = population standard deviation(theta_t)
torso_tilt_rms  = sqrt(mean(theta_t^2))
torso_tilt_p95  = 95th percentile(theta_t)
torso_tilt_max  = max(theta_t)
```

RMS and the upper-tail summaries protect against a persistently tilted posture or a short severe excursion being concealed by variability alone. These are torso-orientation diagnostics, not a complete physical-stability construct.

## 8. Survival and episode length

`episode_length` is the number of environment steps. `reward_survive_sum` is reported as a reward-mechanism component. Survival is a relevant safety attribute and a possible source of proxy dominance; it must neither be deleted as a nuisance nor treated as sufficient evidence of locomotion quality.

## 9. Step records and provenance

Every prospective evaluation episode must save a compressed UTF-8 CSV containing positions, torso height, finite-state status, tilt, squared action, action saturation, condition reward, common rescore, reward components, shaping components and termination status. A Stable-Baselines3 Monitor file must record training episodes. Resolved PPO parameters, seeds, source hashes and package versions must accompany each run.

Formal v1 did not retain torso-height trajectories or these revised step records. Low-z, high-z and non-finite subcategories therefore cannot be recovered retrospectively and must be reported as unavailable rather than inferred.

## 10. Video selection

For each condition, the representative policy is the training seed whose final-checkpoint mean net progress is closest to the median across policies. The representative episode is the evaluation seed closest to that policy's median final-checkpoint net progress. Lower numeric seeds resolve exact ties. The selected evaluation seed is reused at 50k, 150k and 300k checkpoints.

Videos illustrate prespecified episodes and do not replace quantitative outcomes or justify a phase-transition claim.

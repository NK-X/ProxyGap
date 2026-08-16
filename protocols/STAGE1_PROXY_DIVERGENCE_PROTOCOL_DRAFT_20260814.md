# Stage-One Proxy-Diagnostic Divergence Protocol

**Status:** development rules frozen before the 2026-08-14 dense run; formal
held-out training remains blocked.

## Scope

This protocol covers detection only. It does not test reward shaping. The
environment is Gymnasium `Ant-v5`, the algorithm is PPO, the device is CPU, and
the only manipulated reward parameter is `ctrl_cost_weight`.

## Intended behaviour and proxy

Over the fixed 1,000-step horizon, the intended behaviour is sustained net
forward locomotion while the Ant remains within the environment's finite-state
and torso-height health condition. Path efficiency, lateral control, torso
orientation and action-command quality are separately reported diagnostics.
They are not collapsed into a scalar `true_performance` score.

The condition objective is

```text
R_w = sum_t (forward_t + survive_t + contact_t - w * ||a_t||_2^2).
```

For every candidate coefficient `w`, candidate and same-training-seed reference
trajectories are both rescored with this identical `R_w`. Condition-specific
returns from different reward formulae are never directly ranked.

## Hypotheses

The primary stage-one hypothesis is that, within a predeclared plausible
control-cost range, at least one reduced-weight policy obtains a higher proxy
return under the shared `R_w` comparison while at least one predeclared
diagnostic domain deteriorates by a practically meaningful margin.

The secondary hypothesis is that the discrete coefficient map contains an
adjacent interval where this divergence first appears or materially amplifies.
The study may call this a *discrete onset interval*. It must not call it a
mathematical discontinuity, phase transition or universal critical weight.

## Development grid

Existing development policies cover `0.5`, `0.375`, `0.25` and `0.125`. The
new coefficients `0.21875`, `0.1875` and `0.15625` evenly subdivide the unresolved
`[0.125, 0.25]` interval. All use training seeds `41101` and `41102`, evaluation
seeds `51101` to `51110`, 300,000 timesteps and checkpoints every 50,000 steps.

This remains development evidence. It is explicitly permissible to use these
diagnostics to select a formal bracket because the formal policies use disjoint
training seeds.

## Measurement contract

- `net_forward_progress`: terminal minus initial torso x position.
- `mean_forward_velocity`: net forward progress divided by elapsed simulated
  seconds. It is translational velocity, not gait cadence.
- `forward_path_efficiency`: net forward progress divided by planar path length.
- `unhealthy_termination`: finite-state or torso-height health termination.
- `lateral_drift_mean_abs`: mean absolute y displacement from the episode start.
- `torso_tilt_rms`: root-mean-square torso tilt from world vertical.
- `action_saturation_rate`: fraction of action components with `|a| >= 0.95`.
- `normalised_action_roughness`: mean consecutive squared action change divided
  by the maximum possible value `4 x 8`; its range is `[0, 1]`.

The environment health rule is narrower than the broader behavioural scorecard:
the state must be finite and torso z must remain in `[0.2, 1.0]`. Drift, tilt and
command roughness are diagnostics, not synonyms for Gymnasium health.

## Practical decision margins

The frozen development margins are 1.0 position unit for net progress, 0.10 for
path efficiency, 0.20 for unhealthy-termination rate, 0.50 position unit for
mean lateral drift, 5 degrees for torso-tilt RMS, and both 0.02 action saturation
and 0.02 normalised action roughness for the command-quality domain.

These thresholds prevent arbitrarily small numerical differences from being
called harms. They are transparent study decisions, not universal hardware
safety limits. The report must repeat all decisions using half and twice each
margin and show whether the candidate status changes.

## Development and formal gates

A strong development candidate requires positive proxy advantage and at least
one practically harmed domain in both development seeds. Weak evidence based on
proxy similarity is reported separately but cannot authorise formal training.

The formal comparison will use fresh training seeds `42001` to `42005` and a
fresh evaluation-seed block beginning at `52001`. The reference, immediately
higher non-candidate bracket, first strong candidate and next lower severity
condition are selected when available. Selection and all thresholds are frozen
before held-out training.

The endpoint is primary. Earlier checkpoints are repeated measurements of the
same trained policy. Evaluation episodes are nested observations and must first
be aggregated within training seed. No checkpoint or episode is counted as an
independent replication.

## Stopping and exclusion rules

- No strong development candidate: report a negative development result and do
  not extend the lower boundary after inspecting outcomes.
- Missing or duplicate policy/episode cells: block analysis until provenance is
  repaired; do not impute an RL outcome.
- Failed training process: retain logs and apply the same rerun rule to every
  condition. Never replace a seed because its behaviour is inconvenient.
- Non-finite diagnostic values: report the cause and analyse affected metrics as
  missing; do not silently coerce them to zero.
- A video may support behavioural interpretation, but it cannot override the
  quantitative gate or be selected solely because it looks dramatic.

## Claim boundary

Even a confirmed result concerns the tested Ant-v5 environment, PPO
implementation, training budget, coefficient range and seed set. It does not
identify a globally optimal reward weight and does not establish that all
robotic or reinforcement-learning systems exhibit the same failure.

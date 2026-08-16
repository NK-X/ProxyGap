# Updated Research Direction (16 August 2026)

## Plain-language question

The project asks whether a PPO-controlled Ant can score well under the reward
that it is given while behaving worse according to the locomotion qualities
that the research team actually cares about. It then asks whether a controlled
modification can reduce that mismatch.

This is a simulation study. It does not aim to deploy a physical robot.

## Why the direction changed

The earlier design primarily varied `ctrl_cost_weight`. That can show how a
reward coefficient changes learned behaviour, but it does not first establish
whether the default reward represents the project's intended behaviour. If the
default reward already omits posture, direction, body dynamics or gait quality,
comparing several incomplete reward functions risks becoming "one imperfect
proxy versus another".

The coefficient sweep is therefore retained as retrospective exploratory
evidence. The main direction is now a construct audit followed by bounded
mitigation development.

## Stage 1: detect a project-specific proxy gap

### Platform task

Gymnasium describes Ant-v5 as coordinating four legs to move in the positive
x direction. Episodes last at most 1,000 steps.

### Optimiser-facing proxy

The default environment reward combines forward velocity, a healthy reward,
control cost and clipped contact cost. PPO directly optimises this scalar.

### Researcher-facing intention

Over the fixed evaluation horizon, the policy should track a declared forward
command, remain broadly upright, avoid unhealthy termination and sustained
inversion, limit unintended lateral travel, follow a reasonably direct path and
avoid excessively rough or saturated actions.

The intention is evaluated as a vector of diagnostics rather than an invented
single true reward. A policy can therefore be good on one domain and poor on
another.

### Detection logic

Reward misspecification is supported only when the proxy and the predeclared
behavioural evidence disagree in a reproducible and technically valid way. The
analysis must exclude logging errors, reward-reconstruction errors, one selected
video and one fortunate or unfortunate training seed as sole explanations.

Stage 1 does not require a dramatic, monotonic or permanent failure. A clear,
replicated policy-ordering disagreement or proxy-behaviour divergence is
sufficient evidence within the stated Ant-v5/PPO scope.

## Stage 2: test targeted mitigation

Stage 2 asks which type of intervention addresses the mechanism identified in
Stage 1:

- **Reward shaping** for optimisable preferences such as posture, target-speed
  tracking, lateral velocity, action rate and body dynamics.
- **External constraints** for execution limits that should not be traded away
  for reward, such as a candidate action-slew projection.
- **Training mechanisms** where exploration or the control interface is a
  plausible contributor.

Each development round changes a small, declared set of factors. Improvements
must be checked against forward task performance and all guardrail diagnostics,
not only the metric targeted by the new term.

## Present development chain

1. Audit the default reward and historical policies.
2. Add bounded cosine orientation shaping.
3. Test an external action-slew projection and correct lateral observability.
4. Replace unbounded "faster is always better" reward with 1 m/s target
   tracking and test an action-rate penalty.
5. Diagnose residual hopping and contact patterns.
6. Test bounded vertical/angular body-dynamics shaping and a frozen gSDE
   exploration setting in a 2x2 development matrix.
7. Reject ineffective candidates and preserve negative results.
8. Freeze one defensible candidate before using untouched held-out training
   seeds.

## Current interpretation

Development evidence indicates that action-output smoothness, body-level
smoothness and biological gait are different constructs. Target tracking and
action-rate shaping improved the first construct, while repeated take-off and
flight remained. Body-dynamics shaping reduced several hopping diagnostics
under ordinary PPO exploration, whereas the tested gSDE configuration failed.

These findings guide the next candidate decision but do not yet confirm formal
mitigation.

## Remaining design decisions

1. Decide whether the formal project evaluates general stable locomotion or a
   specified gait family. A gait claim requires predeclared foot-contact phase
   metrics or a reference controller/dataset.
2. Decide whether the body-dynamics candidate is sufficiently coherent to
   enter a final development gate.
3. Freeze the formal success and exclusion rules, including the requested
   domain-compliance or mitigation-success percentage matrix.
4. Use untouched held-out training seeds only after the complete protocol is
   frozen.

## Claim boundary

The intended behaviour is project-specific rather than universal. All current
evidence concerns default flat-ground Ant-v5, PPO and CPU-based simulation. No
claim is made about physical safety, energy consumption, terrain robustness,
biological naturalness or other reinforcement-learning systems.

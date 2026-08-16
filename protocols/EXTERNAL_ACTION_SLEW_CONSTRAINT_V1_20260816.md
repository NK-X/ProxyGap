# External Action Slew Constraint V1

## Decision

The first external constraint candidate is a deterministic action slew-rate
limiter applied after the policy proposes an action and before Ant-v5 receives
it. It is independent of the reward and therefore permits a clean test of
whether abrupt command changes are better handled by a control-layer guardrail
than by an additional reward term.

**Status:** development-frozen and engineering-validated candidate. Projection,
observation augmentation, telemetry, CSV schema and deterministic checkpoint
evaluation tests passed on 16 August 2026. It is not approved for held-out
formal training; scientific usefulness remains unresolved pending the bounded
development matrix.

## Constraint

Let \(a_t\in[-1,1]^8\) be the action proposed by PPO and let
\(\tilde a_{t-1}\) be the action actually applied at the preceding step. Define

\[
d_t=a_t-\tilde a_{t-1},
\]

\[
\tilde a_t=\tilde a_{t-1}
+\min\left(1,\frac{\Delta_a}{\lVert d_t\rVert_2+\epsilon}\right)d_t,
\qquad
\Delta_a=1.4.
\]

The applied action is finally clipped to \([-1,1]^8\). This guarantees

\[
\lVert\tilde a_t-\tilde a_{t-1}\rVert_2\leq1.4
\]

for each 0.05-second control interval. The first previous applied action after
reset is the zero vector.

This is a normalised-command constraint, not a calibrated physical torque-rate
limit. It should therefore be described as an **external action slew
guardrail**, not as proof of hardware safety.

## Evidence used to choose the development threshold

The threshold was calibrated from pre-existing 1M-step default-reward
development trajectories. No new policy was trained for calibration. For each
valid adjacent action pair, the recorded squared action change was converted to
\(\lVert a_t-a_{t-1}\rVert_2\).

| Training seed | Transitions | P50 | P90 | P95 | Fraction above 1.4 |
|---:|---:|---:|---:|---:|---:|
| 41201 | 19,980 | 0.938227 | 1.446552 | 1.608066 | 0.118018 |
| 41202 | 12,670 | 0.951943 | 1.397727 | 1.538721 | 0.098658 |
| 41203 | 13,835 | 1.035279 | 1.540216 | 1.691576 | 0.177087 |
| 41204 | 19,980 | 0.727693 | 1.312023 | 1.465245 | 0.068168 |
| 41205 | 12,965 | 0.959091 | 1.447733 | 1.599400 | 0.122021 |
| **Pooled** | **79,430** | **0.920215** | **1.430823** | **1.584372** | **0.113332** |

The pooled P90 was rounded to one decimal place. On the historical commanded
actions, \(\Delta_a=1.4\) would intervene in approximately 11.3% of transitions.
This makes it a bounded but non-trivial candidate. The calculation does not
predict the distribution learned when the limiter is active.

## Markov and fairness requirement

The next action depends on the preceding applied action. To avoid hiding this
state from the policy, append \(\tilde a_{t-1}\in\mathbb R^8\) to the default
105-dimensional observation. The resulting observation has 113 elements.

All comparison groups, including the unconstrained comparator, must receive the
same 113-dimensional observation. In the unconstrained comparator,
\(\tilde a_t=a_t\). This prevents the constrained group from receiving extra
information unavailable to the comparator.

Because the observation changes, the next experiment requires newly trained
baselines. Existing 105-input policies remain historical development evidence
and must not be merged with the new comparison as if only the constraint had
changed.

## Required instrumentation

Record at every evaluation step:

- proposed action and applied action;
- proposed and applied action-change norms;
- whether the limiter intervened;
- correction norm \(\lVert a_t-\tilde a_t\rVert_2\);
- action saturation before and after projection; and
- the existing progress, posture, drift, termination, roughness and effort
  metrics.

At policy level, report intervention rate, mean and maximum correction norm,
and all intended-behaviour metrics. An apparent improvement is unacceptable if
it is obtained through inactivity, material loss of forward command tracking,
more termination, worse drift, or another predeclared deterioration.

## Candidate and formal matrices

The development design is not restricted to a 2-by-2 grid. Reward candidates
\(R_0,\ldots,R_m\) and constraint candidates \(K_0,\ldots,K_n\) may first be
screened in a bounded candidate matrix with equal budgets and explicit tuning
limits.

Only after one reward package \(R^*\) and one constraint \(K^*\) have been
selected and frozen can the confirmatory comparison use the interpretable
2-by-2 ablation:

| | No selected external constraint | Selected external constraint |
|---|---|---|
| Default reward | \(R_0K_0\) | \(R_0K^*\) |
| Selected shaped reward | \(R^*K_0\) | \(R^*K^*\) |

This formal ablation estimates the reward contribution, constraint
contribution, and their interaction. Candidate screening results remain
development evidence.

## Seed semantics

A seed is an integer used to initialise a pseudo-random number generator; it is
not a stored training example.

- A **training seed** controls an independent PPO training run, including
  network initialisation, policy-action sampling, environment reset
  perturbations and minibatch order. The simulator generates online
  trajectories rather than drawing images from a fixed dataset.
- An **evaluation seed** is applied after training while policy parameters are
  frozen. In Ant-v5 it selects a reproducible initial position perturbation and
  initial velocity perturbation. Researchers choose the integer identifiers;
  the simulator generates the exact states according to its reset distribution.
- Reusing each evaluation seed across conditions creates paired starting
  conditions. It does not create an independent training replicate and it does
  not test terrain, pushes or friction changes unless those perturbations are
  separately implemented.

## Pre-training engineering gate

Before a development pilot, automated tests must establish:

1. identity below the bound;
2. exact projection to the bound above it;
3. action-space containment;
4. zero-vector reset behaviour;
5. the 113-element observation in constrained and comparator wrappers;
6. deterministic replay under fixed seeds; and
7. complete commanded-versus-applied logging.

No held-out training and no formal claim are authorised by this document.

## Engineering validation record

- The complete automated suite passed 129 tests after implementation.
- A real 100k guardrail checkpoint completed a 1,000-step deterministic
  evaluation and wrote the full gzip step log.
- Its largest applied action change was 1.4000000000000004, within the
  (10^{-9}) numerical tolerance around the 1.4 bound.
- The limiter intervened on 795 of 1,000 steps in that diagnostic trajectory,
  confirming that the guardrail was active rather than merely configured.
- These are engineering checks, not evidence that the guardrail improves the
  intended behaviour.

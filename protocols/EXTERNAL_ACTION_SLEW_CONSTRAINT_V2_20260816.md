# External Action Slew Constraint V2

## Status and provenance

**Status:** frozen for the second and final bounded development revision on
16 August 2026. This file supersedes V1 only for round-two development; V1
remains the historical record for the 1.4 candidate. Neither version is
approved as a real-hardware safety limit or as a confirmed mitigation.

The V1 threshold was selected from the pooled 90th percentile of historical
commanded action changes. Round-one training then showed that a learned policy
could adapt to the limiter and use the boundary on approximately 96.5% of
evaluation steps. Endpoint normalised action roughness was approximately
0.0607, so the V1 candidate was incompatible with the frozen episode rule

\[
Q_a \leq 0.04.
\]

This was an evidence-based rejection of the V1 development candidate, not a
claim that 1.4 is unsafe for a physical robot.

## Revised development constraint

The projection rule is unchanged. Let \(a_t\in[-1,1]^8\) be the proposed
action and \(\tilde a_{t-1}\) the previously applied action:

\[
d_t=a_t-\tilde a_{t-1},
\]

\[
\tilde a_t=\tilde a_{t-1}
+\min\left(1,\frac{\Delta_a}{\lVert d_t\rVert_2+\epsilon}\right)d_t,
\qquad \Delta_a=1.1.
\]

The applied action is clipped to \([-1,1]^8\). Therefore, for each adjacent
pair after reset,

\[
\lVert\tilde a_t-\tilde a_{t-1}\rVert_2\leq1.1.
\]

The corresponding episode-level normalised roughness has the deterministic
upper bound

\[
Q_a
=\frac{1}{32(\tau-1)}
  \sum_{t=2}^{\tau}\lVert\tilde a_t-\tilde a_{t-1}\rVert_2^2
\leq\frac{1.1^2}{32}
=0.0378125.
\]

The bound is deliberately below the development-frozen threshold of 0.04.
This mathematical compatibility does not establish that locomotion will be
effective, stable or natural; those remain empirical questions.

## Fair-comparison requirements

- All constrained and unconstrained groups use the same 113-dimensional
  observation, comprising the default 105 values and the preceding applied
  eight-dimensional action.
- The PPO architecture, optimisation settings, training budgets, training
  seeds and evaluation seeds are held fixed across conditions.
- The limiter is applied after the policy proposes an action and before the
  environment receives it.
- Proposed and applied actions, intervention flags and correction magnitudes
  are logged separately.
- The limiter is not described as constrained policy optimisation: it is a
  deterministic control-layer projection.

## Scientific decision rule

The candidate cannot advance merely because it satisfies the roughness bound.
It must also avoid material deterioration in forward command tracking,
termination, sustained inversion, torso stability, direction and path
directness. Effects are inspected for each paired development training seed;
evaluation episodes are nested measurements rather than independent training
replicates.

Round-two results may nominate a candidate for held-out testing. They cannot
confirm mitigation because the same development seeds informed this revision.

## Scope boundary

The quantity 1.1 is a limit in Ant-v5's normalised action coordinates. It is
not calibrated to motor torque rate, actuator wear or real-hardware safety.
All conclusions from this protocol are restricted to default flat-ground
Ant-v5 simulation.
